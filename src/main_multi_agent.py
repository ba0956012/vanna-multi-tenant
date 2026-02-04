import os
import json
import logging
import threading
import multiprocessing
import random
from typing import Dict, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field
from vanna import Agent
from vanna.core.registry import ToolRegistry
from vanna.tools import RunSqlTool
from vanna.tools.agent_memory import (
    SaveQuestionToolArgsTool,
    SearchSavedCorrectToolUsesTool,
    SaveTextMemoryTool,
)
from vanna.servers.fastapi import VannaFastAPIServer
from vanna.integrations.azureopenai import AzureOpenAILlmService
from vanna.integrations.postgres import PostgresRunner
from vanna.integrations.chromadb.agent_memory import ChromaAgentMemory
from vanna.core.user import UserResolver, User, RequestContext
from vanna.core.system_prompt import SystemPromptBuilder
from db import PostgresRunnerPooled
from custom_tools import ListAllMemoriesTool
from templates import (
    get_admin_html,
    get_add_memory_html,
    get_detail_html,
    get_agents_management_html,
    get_create_agent_html,
)
import psycopg2

# Configure Logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d [%(process)d:%(thread)d] %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
logging.getLogger("vanna").setLevel(logging.DEBUG)
logging.getLogger("uvicorn").setLevel(logging.INFO)
logging.getLogger("uvicorn.access").setLevel(logging.INFO)

# Load environment variables
load_dotenv()

# Configuration - Azure OpenAI only
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
azure_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# Database connection pool configuration
USE_CONNECTION_POOL = os.getenv("USE_CONNECTION_POOL", "true").lower() == "true"
DB_POOL_MIN_CONN = int(os.getenv("DB_POOL_MIN_CONN", "2"))
DB_POOL_MAX_CONN = int(os.getenv("DB_POOL_MAX_CONN", "10"))

AGENTS_CONFIG_FILE = "./agents_config.json"
AGENT_DATA_DIR = "./agent_data"
BASE_AGENT_PORT = 8101  # Agent ports 從 8001 開始
VANNA_HOST = os.getenv("VANNA_HOST", "")

# === 全域儲存 ===
agent_configs: Dict[str, dict] = {}
agent_processes: Dict[str, multiprocessing.Process] = {}
agent_ports: Dict[str, int] = {}
lock = threading.Lock()


def ensure_agent_data_dir():
    if not os.path.exists(AGENT_DATA_DIR):
        os.makedirs(AGENT_DATA_DIR)


def save_agents_config():
    with open(AGENTS_CONFIG_FILE, "w", encoding="utf-8") as f:
        # 儲存設定和 port 對應
        save_data = {}
        for agent_id, config in agent_configs.items():
            save_data[agent_id] = {**config, "port": agent_ports.get(agent_id)}
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(agent_configs)} agent configs")


def load_agents_config() -> Dict[str, dict]:
    if os.path.exists(AGENTS_CONFIG_FILE):
        with open(AGENTS_CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def get_next_available_port() -> int:
    """取得下一個可用的 port"""
    used_ports = set(agent_ports.values())
    port = BASE_AGENT_PORT
    while port in used_ports:
        port += 1
    return port


# === 動態 System Prompt Builder ===
class DynamicSystemPromptBuilder(SystemPromptBuilder):
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt

    async def build_system_prompt(
        self, user: User, tool_schemas: list, context: dict = None
    ) -> str:
        prompt = self.system_prompt
        if tool_schemas:
            prompt += "\n\n## 可用工具\n"
            for tool in tool_schemas:
                if isinstance(tool, dict):
                    name = tool.get("name") or tool.get("function", {}).get("name", "unknown")
                    description = tool.get("description") or tool.get("function", {}).get("description", "")
                    prompt += f"\n### {name}\n{description}\n"
        return prompt


# === Agent Server 進程函數 ===
def run_agent_server(agent_id: str, config: dict, port: int):
    """在獨立進程中運行 agent server"""
    ensure_agent_data_dir()

    # 建立 LLM
    llm = AzureOpenAILlmService(
        model=azure_deployment_name,
        api_key=azure_api_key,
        azure_endpoint=azure_endpoint,
        api_version=azure_api_version,
    )

    # 建立 Memory
    persist_dir = f"{AGENT_DATA_DIR}/chroma_db_{agent_id}"
    memory = ChromaAgentMemory(
        persist_directory=persist_dir, collection_name=f"vanna_{agent_id}"
    )

    # 建立 DB 連線（可選擇使用連線池或每次新建連線）
    pg_user = config.get("postgres_user")
    pg_password = config.get("postgres_password")
    pg_host = config.get("postgres_host")
    pg_port = config.get("postgres_port") or "5432"
    pg_db = config.get("postgres_db")
    connection_string = f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_db}"

    # 根據配置選擇連線方式
    if USE_CONNECTION_POOL:
        logger.info(f"Agent '{agent_id}' using connection pool (min={DB_POOL_MIN_CONN}, max={DB_POOL_MAX_CONN})")
        db_tool = RunSqlTool(
            sql_runner=PostgresRunnerPooled(
                connection_string=connection_string,
                minconn=DB_POOL_MIN_CONN,
                maxconn=DB_POOL_MAX_CONN,
            )
        )
    else:
        logger.info(f"Agent '{agent_id}' using new connection per request")
        db_tool = RunSqlTool(
            sql_runner=PostgresRunner(connection_string=connection_string)
        )

    # 建立 Tools
    tools = ToolRegistry()
    tools.register_local_tool(db_tool, access_groups=["admin", "user"])
    # tools.register_local_tool(SaveQuestionToolArgsTool(), access_groups=["admin"])
    # tools.register_local_tool(SearchSavedCorrectToolUsesTool(), access_groups=["admin", "user"])
    # tools.register_local_tool(SaveTextMemoryTool(), access_groups=["admin", "user"])
    # tools.register_local_tool(VisualizeDataTool(), access_groups=["admin", "user"])
    tools.register_local_tool(ListAllMemoriesTool(), access_groups=["admin", "user"])

    # User Resolver
    class SimpleUserResolver(UserResolver):
        async def resolve_user(self, request_context: RequestContext) -> User:
            user_email = request_context.get_cookie("vanna_email") or "guest@example.com"
            group = "admin" if user_email in ["admin@example.com"] else "user"
            return User(id=user_email, email=user_email, group_memberships=[group])

    # 建立 Agent
    system_prompt = config.get("system_prompt", "你是一個數據分析助手")

    agent = Agent(
        llm_service=llm,
        tool_registry=tools,
        user_resolver=SimpleUserResolver(),
        agent_memory=memory,
        system_prompt_builder=DynamicSystemPromptBuilder(system_prompt),
    )

    # 啟動 Server
    # FastAPI 的 async 路由天生支援並發，uvicorn 預設使用 asyncio event loop
    # 可以同時處理多個請求，無需額外設定 workers
    logger.info(f"Starting agent '{agent_id}' on port {port}")
    server = VannaFastAPIServer(agent)
    server.run(host="0.0.0.0", port=port)


def start_agent(agent_id: str, config: dict, port: int = None):
    """啟動一個 agent server 進程"""
    if port is None:
        port = get_next_available_port()

    process = multiprocessing.Process(
        target=run_agent_server, args=(agent_id, config, port), daemon=True
    )
    process.start()

    agent_processes[agent_id] = process
    agent_ports[agent_id] = port
    logger.info(f"Agent '{agent_id}' started on port {port}, PID: {process.pid}")
    return port


def stop_agent(agent_id: str):
    """停止一個 agent server 進程"""
    if agent_id in agent_processes:
        process = agent_processes[agent_id]
        process.terminate()
        process.join(timeout=5)
        del agent_processes[agent_id]
        if agent_id in agent_ports:
            del agent_ports[agent_id]
        logger.info(f"Agent '{agent_id}' stopped")


# === FastAPI 管理 App ===
app = FastAPI(
    title="Vanna Multi-Agent Server",
    description="""
    ## 多 Agent 管理系統
    
    這個 API 提供完整的 Agent 生命週期管理功能，包括:
    - 建立,刪除,重啟 Agent
    - 管理 Agent 的 ChromaDB 記憶
    - 自動生成 Few-shot 訓練資料
    
    ### 頁面
    - **管理介面**: `/admin/agents` - Agent 列表管理
    - **記憶管理**: `/admin/memory` - ChromaDB 記憶管理
    - **API 文檔**: `/docs` - Swagger UI (本頁面)
    - **ReDoc**: `/redoc` - 替代文檔介面
    
    ### API 端點
    所有 API 端點都在 `/api` 路徑下
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# === 取得 Agent 的 ChromaDB Memory ===
def get_agent_memory(agent_id: str, create_if_not_exists: bool = False) -> ChromaAgentMemory:
    """取得指定 agent 的 ChromaDB memory
    
    Args:
        agent_id: Agent ID
        create_if_not_exists: 如果目錄不存在是否自動建立(預設 False)
    """
    persist_dir = f"{AGENT_DATA_DIR}/chroma_db_{agent_id}"
    
    if not os.path.exists(persist_dir):
        if create_if_not_exists:
            os.makedirs(persist_dir, exist_ok=True)
            logger.info(f"Created memory directory for agent '{agent_id}'")
        else:
            raise HTTPException(404, f"Agent '{agent_id}' memory not found")
    
    return ChromaAgentMemory(
        persist_directory=persist_dir, collection_name=f"vanna_{agent_id}"
    )


class AgentConfig(BaseModel):
    """Agent configuration model"""
    agent_id: str = Field(..., description="Agent unique ID", json_schema_extra={"example": "pos_sales_agent"})
    description: str = Field("", description="Agent description", json_schema_extra={"example": "POS Sales Analysis System"})
    postgres_user: str = Field(..., description="PostgreSQL username", json_schema_extra={"example": "postgres"})
    postgres_password: str = Field(..., description="PostgreSQL password")
    postgres_host: str = Field(..., description="PostgreSQL host", json_schema_extra={"example": "localhost"})
    postgres_port: str = Field("5432", description="PostgreSQL port", json_schema_extra={"example": "5432"})
    postgres_db: str = Field(..., description="PostgreSQL database name", json_schema_extra={"example": "pos_sales"})
    system_prompt: Optional[str] = Field("", description="Agent system prompt (留空則自動從資料庫生成)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "agent_id": "pos_sales_agent",
                "description": "POS Sales Analysis System",
                "postgres_user": "postgres",
                "postgres_password": "your_password",
                "postgres_host": "localhost",
                "postgres_port": "5432",
                "postgres_db": "pos_sales",
                "system_prompt": ""
            }
        }


class AddMemoryRequest(BaseModel):
    """新增記憶請求模型"""
    question: str = Field(..., description="問題描述", example="查詢今天的銷售總額")
    tool_name: str = Field(..., description="工具名稱", example="run_sql")
    args: dict = Field(
        default_factory=dict,
        description="工具參數",
        example={"sql": "SELECT SUM(total_amount) FROM pos_sale WHERE DATE(sale_date) = CURRENT_DATE"}
    )
    metadata: dict = Field(
        default_factory=dict,
        description="額外的 metadata",
        example={"category": "sales", "auto_generated": False}
    )
    
    class Config:
        schema_extra = {
            "example": {
                "question": "查詢今天的銷售總額",
                "tool_name": "run_sql",
                "args": {
                    "sql": "SELECT SUM(total_amount) FROM pos_sale WHERE DATE(sale_date) = CURRENT_DATE"
                },
                "metadata": {
                    "category": "sales",
                    "auto_generated": False
                }
            }
        }

@app.post("/api/agents", tags=["Agent Management"], summary="建立新 Agent")
async def register_agent(config: AgentConfig):
    """
    動態新增 agent(不需重啟服務)
    
    建立一個新的 Agent 實例，會自動:
    1. 驗證資料庫連線參數
    2. 如果未提供 system_prompt，自動從資料庫生成
    3. 分配可用的 port
    4. 啟動獨立的 Agent 進程
    5. 儲存設定到 agents_config.json
    
    Returns:
        - message: 建立結果訊息
        - port: 分配的 port 號
        - url: Agent 的訪問 URL
        - agents: 所有 Agent 列表
        - system_prompt_generated: 是否自動生成了 system_prompt
    """
    with lock:
        if config.agent_id in agent_configs:
            raise HTTPException(400, f"Agent '{config.agent_id}' already exists")

        config_dict = config.dict()

        # 驗證必要參數
        if not all([config.postgres_user, config.postgres_password, config.postgres_host, config.postgres_db]):
            raise HTTPException(400, "Missing required postgres connection parameters")

        # 如果沒有提供 system_prompt，自動從資料庫生成
        system_prompt_generated = False
        if not config_dict.get("system_prompt") or config_dict["system_prompt"].strip() == "":
            logger.info(f"Generating system prompt from database for agent '{config.agent_id}'")
            try:
                conn = psycopg2.connect(
                    host=config.postgres_host,
                    port=config.postgres_port,
                    user=config.postgres_user,
                    password=config.postgres_password,
                    dbname=config.postgres_db,
                    connect_timeout=10
                )
                
                config_dict["system_prompt"] = generate_system_prompt_from_db(conn, config.postgres_db)
                system_prompt_generated = True
                conn.close()
                logger.info(f"Successfully generated system prompt for agent '{config.agent_id}'")
            except Exception as e:
                logger.error(f"Failed to generate system prompt: {e}")
                raise HTTPException(500, f"無法連接資料庫或生成 system prompt: {e}")

        # 啟動 agent server
        port = start_agent(config.agent_id, config_dict)

        # 儲存設定
        agent_configs[config.agent_id] = config_dict
        save_agents_config()

    return {
        "message": f"Agent '{config.agent_id}' created",
        "port": port,
        "url": f"http://{VANNA_HOST}:{port}",
        "agents": list(agent_configs.keys()),
        "system_prompt_generated": system_prompt_generated,
    }


@app.get("/api/agents", tags=["Agent Management"], summary="列出所有 Agents")
async def list_agents():
    """
    列出所有已註冊的 agents
    
    Returns:
        - agents: Agent 列表，包含 ID,port,URL,運行狀態
    """
    agents_info = []
    for agent_id in agent_configs.keys():
        port = agent_ports.get(agent_id)
        process = agent_processes.get(agent_id)
        agents_info.append({
            "agent_id": agent_id,
            "port": port,
            "url": f"http://{VANNA_HOST}:{port}/api/vanna/v2/chat_sse" if port else None,
            "running": process.is_alive() if process else False,
        })
    return {"agents": agents_info}


@app.delete("/api/agents/{agent_id}", tags=["Agent Management"], summary="刪除 Agent")
async def remove_agent(agent_id: str, delete_memory: bool = False):
    """
    移除指定的 agent
    
    Args:
        agent_id: Agent ID
        delete_memory: 是否同時刪除 ChromaDB 記憶 (預設 False)
        
    Returns:
        - message: 刪除結果訊息
        - memory_deleted: 是否已刪除記憶
    """
    with lock:
        if agent_id not in agent_configs:
            raise HTTPException(404, f"Agent '{agent_id}' not found")

        stop_agent(agent_id)
        del agent_configs[agent_id]
        save_agents_config()
        
        # 刪除 ChromaDB 記憶
        memory_deleted = False
        if delete_memory:
            import shutil
            memory_dir = f"{AGENT_DATA_DIR}/chroma_db_{agent_id}"
            if os.path.exists(memory_dir):
                try:
                    shutil.rmtree(memory_dir)
                    memory_deleted = True
                    logger.info(f"Deleted memory for agent '{agent_id}'")
                except Exception as e:
                    logger.error(f"Failed to delete memory: {e}")

    return {
        "message": f"Agent '{agent_id}' removed",
        "memory_deleted": memory_deleted
    }


@app.post("/api/agents/{agent_id}/restart", tags=["Agent Management"], summary="重啟 Agent")
async def restart_agent_api(agent_id: str):
    """
    重啟指定的 agent
    
    會停止現有進程並使用相同的 port 重新啟動
    
    Args:
        agent_id: Agent ID
        
    Returns:
        - message: 重啟結果訊息
        - port: Agent 的 port 號
    """
    with lock:
        if agent_id not in agent_configs:
            raise HTTPException(404, f"Agent '{agent_id}' not found")

        config = agent_configs[agent_id]
        old_port = agent_ports.get(agent_id)

        stop_agent(agent_id)
        port = start_agent(agent_id, config, old_port)

    return {"message": f"Agent '{agent_id}' restarted", "port": port}


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/admin/agents")


@app.get("/admin/agents", response_class=HTMLResponse, include_in_schema=False)
async def agents_management_page(message: str = None):
    """Agent 管理頁面"""
    agents_info = []
    for agent_id in agent_configs.keys():
        port = agent_ports.get(agent_id)
        process = agent_processes.get(agent_id)
        config = agent_configs.get(agent_id, {})
        agents_info.append({
            "agent_id": agent_id,
            "description": config.get("description", ""),
            "port": port,
            "url": f"http://localhost:{port}" if port else "#",
            "running": process.is_alive() if process else False,
        })
    return get_agents_management_html(agents_info, message)


@app.get("/admin/agents/new", response_class=HTMLResponse, include_in_schema=False)
async def create_agent_page(message: str = None):
    """新增 Agent 頁面"""
    return get_create_agent_html(message)


@app.post("/admin/agents/new", response_class=HTMLResponse, include_in_schema=False)
async def create_agent_submit(
    agent_id: str = Form(...),
    description: str = Form(""),
    postgres_user: str = Form(...),
    postgres_password: str = Form(...),
    postgres_host: str = Form(...),
    postgres_port: str = Form("5432"),
    postgres_db: str = Form(...),
    system_prompt: str = Form(""),
):
    """處理新增 Agent 表單"""
    with lock:
        if agent_id in agent_configs:
            return get_create_agent_html(f"❌ Agent '{agent_id}' 已存在")

        config_dict = {
            "agent_id": agent_id,
            "description": description,
            "postgres_user": postgres_user,
            "postgres_password": postgres_password,
            "postgres_host": postgres_host,
            "postgres_port": postgres_port,
            "postgres_db": postgres_db,
            "system_prompt": system_prompt,
        }

        # 驗證必要參數
        if not all([postgres_user, postgres_password, postgres_host, postgres_db]):
            return get_create_agent_html("❌ 缺少必要的資料庫連線參數")

        try:
            # 如果沒有提供 system_prompt，自動從資料庫生成
            if not system_prompt or system_prompt.strip() == "":
                logger.info(f"Generating system prompt from database for agent '{agent_id}'")
                try:
                    conn = psycopg2.connect(
                        host=postgres_host,
                        port=postgres_port,
                        user=postgres_user,
                        password=postgres_password,
                        dbname=postgres_db,
                        connect_timeout=10
                    )
                    
                    config_dict["system_prompt"] = generate_system_prompt_from_db(conn, postgres_db)
                    conn.close()
                    logger.info(f"Successfully generated system prompt for agent '{agent_id}'")
                except Exception as e:
                    logger.error(f"Failed to generate system prompt: {e}")
                    return get_create_agent_html(f"❌ 無法連接資料庫或生成 system prompt: {e}")
            
            # 啟動 agent server
            port = start_agent(agent_id, config_dict)

            # 儲存設定
            agent_configs[agent_id] = config_dict
            save_agents_config()

            return RedirectResponse(f"/admin/agents?message=Agent '{agent_id}' created successfully (Port: {port})", status_code=302)
        except Exception as e:
            logger.error(f"Failed to create agent: {e}")
            return get_create_agent_html(f"❌ 建立失敗: {e}")


@app.get("/api")
async def api_root():
    return {
        "message": "Vanna Multi-Agent Server",
        "endpoints": {
            "list_agents": "GET /admin/agents",
            "create_agent": "POST /admin/agents",
            "remove_agent": "DELETE /admin/agents/{agent_id}",
            "restart_agent": "POST /admin/agents/{agent_id}/restart",
            "memory_ui": "GET /admin/memory",
        },
        "agents": [
            {"id": aid, "port": agent_ports.get(aid), "url": f"http://{VANNA_HOST}:{agent_ports.get(aid)}/api/vanna/v2/chat_sse"}
            for aid in agent_configs.keys()
        ],
    }


# === 記憶管理頁面 ===
@app.get("/admin/memory", response_class=HTMLResponse, include_in_schema=False)
async def memory_admin_page(agent_id: str = None, message: str = None):
    """記憶管理主頁面"""
    agents_info = []
    for aid in agent_configs.keys():
        port = agent_ports.get(aid)
        process = agent_processes.get(aid)
        agents_info.append({
            "agent_id": aid,
            "port": port,
            "running": process.is_alive() if process else False,
        })
    
    memories = None
    if agent_id and agent_id in agent_configs:
        try:
            memory = get_agent_memory(agent_id)
            collection = memory._get_collection()
            result = collection.get()
            
            memories = []
            if result["ids"]:
                for i, mid in enumerate(result["ids"]):
                    md = result["metadatas"][i]
                    memories.append({
                        "id": mid,
                        "question": md.get("question", ""),
                        "tool_name": md.get("tool_name", ""),
                        "timestamp": md.get("timestamp", ""),
                    })
                # 按時間排序
                memories.sort(key=lambda x: x["timestamp"] or "", reverse=True)
        except Exception as e:
            logger.error(f"Error loading memories: {e}")
            memories = []
    
    return get_admin_html(agents_info, agent_id, memories, message)


@app.get("/admin/memory/{agent_id}/add", response_class=HTMLResponse, include_in_schema=False)
async def add_memory_page(agent_id: str):
    """新增記憶頁面"""
    if agent_id not in agent_configs:
        raise HTTPException(404, f"Agent '{agent_id}' not found")
    return get_add_memory_html(agent_id)


@app.post("/admin/memory/{agent_id}/add", include_in_schema=False)
async def add_memory_submit(
    agent_id: str,
    question: str = Form(...),
    tool_name: str = Form(...),
    args_json: str = Form("{}"),
    metadata_json: str = Form("{}"),
):
    """新增記憶"""
    if agent_id not in agent_configs:
        raise HTTPException(404, f"Agent '{agent_id}' not found")
    
    try:
        args = json.loads(args_json or "{}")
        metadata = json.loads(metadata_json or "{}")
    except json.JSONDecodeError:
        raise HTTPException(400, "JSON 格式錯誤")
    
    memory = get_agent_memory(agent_id)
    context = RequestContext(user=User(id="admin"))
    
    await memory.save_tool_usage(
        question=question,
        tool_name=tool_name,
        args=args,
        context=context,
        success=True,
        metadata=metadata,
    )
    
    return RedirectResponse(f"/admin/memory?agent_id={agent_id}&message=記憶已新增", status_code=302)


@app.get("/admin/memory/{agent_id}/detail/{memory_id}", response_class=HTMLResponse, include_in_schema=False)
async def memory_detail_page(agent_id: str, memory_id: str):
    """記憶詳情頁面"""
    if agent_id not in agent_configs:
        raise HTTPException(404, f"Agent '{agent_id}' not found")
    
    memory = get_agent_memory(agent_id)
    collection = memory._get_collection()
    result = collection.get(ids=[memory_id])
    
    if not result["ids"]:
        raise HTTPException(404, "Memory not found")
    
    md = result["metadatas"][0]
    doc = result["documents"][0] if result["documents"] else ""
    
    memory_data = {
        "id": memory_id,
        "question": md.get("question", ""),
        "tool_name": md.get("tool_name", ""),
        "timestamp": md.get("timestamp", ""),
        "args_json": md.get("args_json", "{}"),
        "document": doc,
    }
    
    return get_detail_html(agent_id, memory_data)


@app.get("/admin/memory/{agent_id}/delete/{memory_id}", include_in_schema=False)
async def delete_memory(agent_id: str, memory_id: str):
    """刪除記憶"""
    if agent_id not in agent_configs:
        raise HTTPException(404, f"Agent '{agent_id}' not found")
    
    memory = get_agent_memory(agent_id)
    context = RequestContext(user=User(id="admin"))
    
    await memory.delete_by_id(context=context, memory_id=memory_id)
    
    return RedirectResponse(f"/admin/memory?agent_id={agent_id}&message=記憶已刪除", status_code=302)


# === Memory Management API ===
@app.get("/api/agents/{agent_id}/memories", tags=["Memory Management"], summary="列出 Agent 的所有記憶")
async def list_memories(agent_id: str, limit: int = 100):
    """
    列出指定 agent 的所有記憶
    
    Args:
        agent_id: Agent ID
        limit: 返回記憶數量限制 (預設 100)
        
    Returns:
        - memories: 記憶列表
        - total: 總記憶數量
    """
    if agent_id not in agent_configs:
        raise HTTPException(404, f"Agent '{agent_id}' not found")
    
    try:
        memory = get_agent_memory(agent_id)
        collection = memory._get_collection()
        result = collection.get()
        
        memories = []
        if result["ids"]:
            for i, mid in enumerate(result["ids"]):
                md = result["metadatas"][i]
                memories.append({
                    "id": mid,
                    "question": md.get("question", ""),
                    "tool_name": md.get("tool_name", ""),
                    "timestamp": md.get("timestamp", ""),
                    "args_json": md.get("args_json", "{}"),
                })
            # 按時間排序
            memories.sort(key=lambda x: x["timestamp"] or "", reverse=True)
            memories = memories[:limit]
        
        return {
            "agent_id": agent_id,
            "memories": memories,
            "total": len(result["ids"]) if result["ids"] else 0
        }
    except Exception as e:
        logger.error(f"Failed to list memories: {e}")
        raise HTTPException(500, f"Failed to list memories: {e}")


@app.post("/api/agents/{agent_id}/memories", tags=["Memory Management"], summary="新增記憶")
async def add_memory_api(agent_id: str, request: AddMemoryRequest):
    """
    為指定 agent 新增記憶
    
    Args:
        agent_id: Agent ID
        request: 記憶資料
        
    Returns:
        - message: 新增結果訊息
        
    Example:
        ```json
        {
            "question": "查詢今天的銷售總額",
            "tool_name": "run_sql",
            "args": {
                "sql": "SELECT SUM(total_amount) FROM pos_sale WHERE DATE(sale_date) = CURRENT_DATE"
            }
        }
        ```
    """
    if agent_id not in agent_configs:
        raise HTTPException(404, f"Agent '{agent_id}' not found")
    
    try:
        memory = get_agent_memory(agent_id, create_if_not_exists=True)
        context = RequestContext(user=User(id="admin"))
        
        await memory.save_tool_usage(
            question=request.question,
            tool_name=request.tool_name,
            args=request.args,
            context=context,
            success=True,
            metadata=request.metadata,
        )
        
        return {"message": f"Memory added to agent '{agent_id}'"}
    except Exception as e:
        logger.error(f"Failed to add memory: {e}")
        raise HTTPException(500, f"Failed to add memory: {e}")


@app.get("/api/agents/{agent_id}/memories/{memory_id}", tags=["Memory Management"], summary="取得記憶詳情")
async def get_memory_detail(agent_id: str, memory_id: str):
    """
    取得指定記憶的詳細資訊
    
    Args:
        agent_id: Agent ID
        memory_id: Memory ID
        
    Returns:
        記憶的完整資訊
    """
    if agent_id not in agent_configs:
        raise HTTPException(404, f"Agent '{agent_id}' not found")
    
    try:
        memory = get_agent_memory(agent_id)
        collection = memory._get_collection()
        result = collection.get(ids=[memory_id])
        
        if not result["ids"]:
            raise HTTPException(404, "Memory not found")
        
        md = result["metadatas"][0]
        doc = result["documents"][0] if result["documents"] else ""
        
        return {
            "id": memory_id,
            "question": md.get("question", ""),
            "tool_name": md.get("tool_name", ""),
            "timestamp": md.get("timestamp", ""),
            "args_json": md.get("args_json", "{}"),
            "document": doc,
            "metadata": md
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get memory detail: {e}")
        raise HTTPException(500, f"Failed to get memory detail: {e}")


@app.delete("/api/agents/{agent_id}/memories/{memory_id}", tags=["Memory Management"], summary="刪除記憶")
async def delete_memory_api(agent_id: str, memory_id: str):
    """
    刪除指定的記憶
    
    Args:
        agent_id: Agent ID
        memory_id: Memory ID
        
    Returns:
        - message: 刪除結果訊息
    """
    if agent_id not in agent_configs:
        raise HTTPException(404, f"Agent '{agent_id}' not found")
    
    try:
        memory = get_agent_memory(agent_id)
        context = RequestContext(user=User(id="admin"))
        
        await memory.delete_by_id(context=context, memory_id=memory_id)
        
        return {"message": f"Memory '{memory_id}' deleted from agent '{agent_id}'"}
    except Exception as e:
        logger.error(f"Failed to delete memory: {e}")
        raise HTTPException(500, f"Failed to delete memory: {e}")


# === Auto Generate System Prompt ===
def generate_system_prompt_from_db(conn, db_name: str) -> str:
    """從 PostgreSQL 資料庫自動生成 system prompt
    
    Args:
        conn: PostgreSQL 連接
        db_name: 資料庫名稱
        
    Returns:
        完整的 system prompt 字串
    """
    cur = conn.cursor()
    
    # 取得所有表
    cur.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
        ORDER BY table_name;
    """)
    tables = [r[0] for r in cur.fetchall()]
    
    # 建立 system prompt
    prompt_parts = []
    prompt_parts.append(f"你是一個專業的數據分析助手，專門協助分析 {db_name} 資料庫。")
    prompt_parts.append(f"\n## 資料庫結構 (PostgreSQL: {db_name})\n")
    prompt_parts.append("### 資料表\n")
    
    # 為每個表生成描述
    for idx, table in enumerate(tables, 1):
        # 取得欄位資訊
        cur.execute("""
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns 
            WHERE table_schema = 'public' AND table_name = %s
            ORDER BY ordinal_position;
        """, (table,))
        columns = cur.fetchall()
        
        # 取得主鍵
        cur.execute("""
            SELECT kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
            WHERE tc.constraint_type = 'PRIMARY KEY'
                AND tc.table_schema = 'public'
                AND tc.table_name = %s;
        """, (table,))
        pk_columns = [r[0] for r in cur.fetchall()]
        
        # 取得外鍵
        cur.execute("""
            SELECT
                kcu.column_name as from_column,
                ccu.table_name as to_table,
                ccu.column_name as to_column
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
            JOIN information_schema.constraint_column_usage ccu
                ON ccu.constraint_name = tc.constraint_name
            WHERE tc.constraint_type = 'FOREIGN KEY'
                AND tc.table_schema = 'public'
                AND tc.table_name = %s;
        """, (table,))
        fks = cur.fetchall()
        
        # 生成表描述（符合 extract_table_descriptions 格式）
        prompt_parts.append(f"{idx}. **{table}** - {table.replace('_', ' ').title()} 資料表")
        
        # 欄位描述
        for col_name, data_type, is_nullable, col_default in columns:
            col_desc = f"   - {col_name} ({data_type}"
            
            # 標記主鍵
            if col_name in pk_columns:
                col_desc += ", 主鍵"
            
            # 標記外鍵
            for fk_col, ref_table, ref_col in fks:
                if fk_col == col_name:
                    col_desc += f", 外鍵: 對應 {ref_table}.{ref_col}"
                    break
            
            col_desc += ")"
            
            # 標記可為空
            if is_nullable == 'YES':
                col_desc += " [可為空]"
            
            prompt_parts.append(col_desc)
        
        prompt_parts.append("")  # 空行
    
    # 加入工作流程說明
    prompt_parts.append("\n## 工作流程 (重要!)\n")
    prompt_parts.append("⚠️ **執行任何 SQL 查詢前，你必須先呼叫 search_saved_correct_tool_uses 搜尋相似問題!**\n")
    prompt_parts.append("1. 用戶提問")
    prompt_parts.append("2. 🔍 **先搜尋**: 呼叫 search_saved_correct_tool_uses(question=\"用戶的問題\")")
    prompt_parts.append("3. 參考搜尋結果中的 SQL 模式")
    prompt_parts.append("4. 執行 SQL: 呼叫 run_sql(sql=\"SELECT ...\")")
    # prompt_parts.append("5. 💾 **儲存成功結果**: 呼叫 save_question_tool_args(question=\"用戶的問題\", tool_name=\"run_sql\", args={\"sql\": \"SELECT ...\"})\n")
    # prompt_parts.append("   ⚠️ 注意: save_question_tool_args 必須包含三個參數: question, tool_name, args\n")
    
    # 加入回應風格
    prompt_parts.append("## 回應風格\n")
    prompt_parts.append("- 簡潔專業，使用繁體中文")
    # prompt_parts.append("- 執行查詢後，解釋結果的商業意義")
    # prompt_parts.append("- 主動建議視覺化圖表")
    
    return "\n".join(prompt_parts)


# === Auto Generate Fewshot API ===
def analyze_pg_database(conn):
    """分析 PostgreSQL 資料庫結構"""
    cur = conn.cursor()

    cur.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
        ORDER BY table_name;
    """)
    tables = [r[0] for r in cur.fetchall()]

    schema = {}
    for t in tables:
        cur.execute("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns 
            WHERE table_schema = 'public' AND table_name = %s
            ORDER BY ordinal_position;
        """, (t,))
        cols = cur.fetchall()

        cur.execute("""
            SELECT
                kcu.column_name as from_column,
                ccu.table_name as to_table,
                ccu.column_name as to_column
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
            JOIN information_schema.constraint_column_usage ccu
                ON ccu.constraint_name = tc.constraint_name
            WHERE tc.constraint_type = 'FOREIGN KEY'
                AND tc.table_schema = 'public'
                AND tc.table_name = %s;
        """, (t,))
        fks = cur.fetchall()

        schema[t] = {"columns": cols, "fks": fks}

    return tables, schema


def build_fk_graph(tables, schema):
    """建立 FK 關係圖"""
    graph = {t: [] for t in tables}
    for t in tables:
        for fk in schema[t]["fks"]:
            from_col, ref_table, to_col = fk
            if ref_table in graph:
                graph[t].append((ref_table, from_col, to_col))
                graph[ref_table].append((t, to_col, from_col))
    return graph


def bfs_join_tables(root, graph):
    """BFS 取得 JOIN 順序"""
    visited = set()
    queue = [root]
    order = []
    while queue:
        t = queue.pop(0)
        if t in visited:
            continue
        visited.add(t)
        order.append(t)
        for to_table, _, _ in graph[t]:
            if to_table not in visited:
                queue.append(to_table)
    return order


def get_pg_sample_row(conn, table):
    """取得範例資料"""
    cur = conn.cursor()
    try:
        cur.execute(f'SELECT * FROM "{table}" LIMIT 1;')
        row = cur.fetchone()
        if row is None:
            return None
        cols = [c[0] for c in cur.description]
        return dict(zip(cols, row))
    except Exception as e:
        logger.error(f"Error getting sample row from {table}: {e}")
        return None


def generate_fewshot_sql(root, join_order, schema, graph, sample_row):
    """生成 few-shot SQL"""
    aliases = {t: f"t{i}" for i, t in enumerate(join_order)}

    sql_parts = []
    sql_parts.append("SELECT " + ", ".join(f'{aliases[t]}.*' for t in join_order))
    sql_parts.append(f'FROM "{root}" {aliases[root]}')

    for t in join_order:
        if t == root:
            continue
        parent = None
        parent_fk = None
        for pt in join_order:
            if pt == t:
                break
            for to_table, from_col, to_col in graph[pt]:
                if to_table == t:
                    parent = pt
                    parent_fk = (from_col, to_col)
                    break
            if parent:
                break
        if parent:
            p_alias = aliases[parent]
            t_alias = aliases[t]
            from_col, to_col = parent_fk
            sql_parts.append(f'LEFT JOIN "{t}" {t_alias} ON {p_alias}."{from_col}" = {t_alias}."{to_col}"')

    # WHERE 條件
    exclude_cols = {"id", "created_at", "updated_at"}
    exclude_types = {"date", "timestamp with time zone", "timestamp without time zone"}
    
    where_cols = []
    for col in schema[root]["columns"]:
        name, ctype = col[0], (col[1] or "").lower()
        if name.lower() in exclude_cols or ctype in exclude_types:
            continue
        if sample_row and sample_row.get(name) is not None:
            where_cols.append((name, ctype))
    
    random.shuffle(where_cols)
    where_cols = where_cols[:2]
    
    if where_cols:
        conditions = []
        for name, ctype in where_cols:
            if "char" in ctype or "text" in ctype:
                conditions.append(f't0."{name}" LIKE \'%[{name}]%\'')
            else:
                conditions.append(f't0."{name}" = [{name}]')
        sql_parts.append("WHERE " + " AND ".join(conditions))

    sql_parts.append("LIMIT 100;")
    return "\n".join(sql_parts)


def extract_table_descriptions(system_prompt: str) -> dict:
    """從 system_prompt 提取表的描述"""
    import re
    descriptions = {}
    
    # 匹配 **table_name** - 描述 的格式
    pattern = r'\*\*(\w+)\*\*\s*-\s*([^\n]+)'
    matches = re.findall(pattern, system_prompt)
    
    for table_name, desc in matches:
        descriptions[table_name.lower()] = desc.strip()
    
    return descriptions


def generate_question(root_table, join_order, table_descriptions=None):
    """生成自然語言問題"""
    table_descriptions = table_descriptions or {}
    
    # 嘗試從描述中取得表的中文名稱
    root_desc = table_descriptions.get(root_table.lower(), root_table)
    
    if len(join_order) == 1:
        return f"查詢{root_desc}的資料"
    else:
        related_descs = []
        for t in join_order[1:3]:
            desc = table_descriptions.get(t.lower(), t)
            related_descs.append(desc)
        
        related = ", ".join(related_descs)
        if len(join_order) > 3:
            related += " 等"
        return f"查詢{root_desc}及關聯的{related}資料"


@app.post("/api/agents/{agent_id}/generate-fewshot", tags=["Training"], summary="自動生成 Few-shot")
async def generate_fewshot(agent_id: str):
    """
    自動生成 few-shot 訓練資料
    
    會自動:
    1. 連接 Agent 的 PostgreSQL 資料庫
    2. 分析資料庫結構(表,欄位,外鍵關係)
    3. 生成 JOIN SQL 查詢範例
    4. 從 system_prompt 提取表的中文描述
    5. 生成自然語言問題
    6. 儲存到 ChromaDB 作為 few-shot 範例
    
    Args:
        agent_id: Agent ID
        
    Returns:
        - message: 生成結果訊息
        - total_tables: 資料庫總表數
        - imported: 成功匯入的 few-shot 數量
        - fewshots: 生成的 few-shot 列表
    """
    if agent_id not in agent_configs:
        raise HTTPException(404, f"Agent '{agent_id}' not found")
    
    config = agent_configs[agent_id]
    
    # 連接資料庫
    pg_user = config.get("postgres_user")
    pg_password = config.get("postgres_password")
    pg_host = config.get("postgres_host")
    pg_port = config.get("postgres_port") or "5432"
    pg_db = config.get("postgres_db")
    
    logger.info(f"Connecting to {pg_host}:{pg_port}/{pg_db}")
    
    conn = None
    try:
        conn = psycopg2.connect(
            host=pg_host,
            port=pg_port,
            user=pg_user,
            password=pg_password,
            dbname=pg_db,
            connect_timeout=10
        )
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise HTTPException(500, f"資料庫連接失敗: {e}")
    
    try:
        tables, schema = analyze_pg_database(conn)
        logger.info(f"Found {len(tables)} tables")
        graph = build_fk_graph(tables, schema)
        
        # 從 system_prompt 提取表描述
        system_prompt = config.get("system_prompt", "")
        table_descriptions = extract_table_descriptions(system_prompt)
        logger.info(f"Extracted {len(table_descriptions)} table descriptions from prompt")
        
        fewshots = []
        for table in tables:
            try:
                sample = get_pg_sample_row(conn, table)
                if not sample:
                    logger.info(f"Skipping {table}: no data")
                    continue
                
                join_order = bfs_join_tables(table, graph)
                sql = generate_fewshot_sql(table, join_order, schema, graph, sample)
                question = generate_question(table, join_order, table_descriptions)
                
                fewshots.append({
                    "question": question,
                    "tool_name": "run_sql",
                    "sql": sql,
                    "table": table,
                })
                logger.info(f"Generated fewshot for {table}")
            except Exception as e:
                logger.error(f"Error generating fewshot for {table}: {e}")
                continue
        
        conn.close()
        conn = None
        
        logger.info(f"Generated {len(fewshots)} fewshots, importing to ChromaDB...")
        
        # 匯入到 ChromaDB
        memory = get_agent_memory(agent_id, create_if_not_exists=True)
        context = RequestContext(user=User(id="admin"))
        
        imported = 0
        for fs in fewshots:
            try:
                await memory.save_tool_usage(
                    question=fs["question"],
                    tool_name=fs["tool_name"],
                    args={"sql": fs["sql"]},
                    context=context,
                    success=True,
                    metadata={"table": fs["table"], "auto_generated": True},
                )
                imported += 1
            except Exception as e:
                logger.error(f"Failed to save fewshot for {fs['table']}: {e}")
        
        return {
            "message": f"成功生成 {imported} 筆 few-shot",
            "total_tables": len(tables),
            "imported": imported,
            "fewshots": fewshots,
        }
        
    except Exception as e:
        if conn:
            conn.close()
        logger.error(f"Generate fewshot failed: {e}")
        raise HTTPException(500, f"生成失敗: {e}")


@app.post("/api/agents/{agent_id}/generate-system-prompt", tags=["Training"], summary="自動生成 System Prompt")
async def generate_system_prompt_api(agent_id: str):
    """
    從資料庫自動生成 system prompt
    
    會自動分析資料庫結構，生成包含：
    - 所有表的描述（符合 extract_table_descriptions 格式）
    - 欄位資訊（類型、主鍵、外鍵）
    - 工作流程說明
    - 回應風格指引
    
    Args:
        agent_id: Agent ID
        
    Returns:
        - system_prompt: 生成的 system prompt
        - total_tables: 資料庫總表數
    """
    if agent_id not in agent_configs:
        raise HTTPException(404, f"Agent '{agent_id}' not found")
    
    config = agent_configs[agent_id]
    
    # 連接資料庫
    pg_user = config.get("postgres_user")
    pg_password = config.get("postgres_password")
    pg_host = config.get("postgres_host")
    pg_port = config.get("postgres_port") or "5432"
    pg_db = config.get("postgres_db")
    
    logger.info(f"Connecting to {pg_host}:{pg_port}/{pg_db}")
    
    conn = None
    try:
        conn = psycopg2.connect(
            host=pg_host,
            port=pg_port,
            user=pg_user,
            password=pg_password,
            dbname=pg_db,
            connect_timeout=10
        )
        
        # 生成 system prompt
        system_prompt = generate_system_prompt_from_db(conn, pg_db)
        
        # 計算表數量
        cur = conn.cursor()
        cur.execute("""
            SELECT COUNT(*) 
            FROM information_schema.tables 
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE';
        """)
        total_tables = cur.fetchone()[0]
        
        conn.close()
        
        return {
            "system_prompt": system_prompt,
            "total_tables": total_tables,
            "message": f"成功生成 system prompt，包含 {total_tables} 張表"
        }
        
    except Exception as e:
        if conn:
            conn.close()
        logger.error(f"Generate system prompt failed: {e}")
        raise HTTPException(500, f"生成失敗: {e}")


@app.on_event("startup")
async def startup():
    """啟動時載入已儲存的 agents"""
    saved_configs = load_agents_config()
    for agent_id, config in saved_configs.items():
        try:
            port = config.pop("port", None)  # 取出之前的 port
            agent_configs[agent_id] = config
            start_agent(agent_id, config, port)
        except Exception as e:
            logger.error(f"Failed to load agent '{agent_id}': {e}")

    logger.info(f"Loaded {len(agent_configs)} agents")


@app.on_event("shutdown")
async def shutdown():
    """關閉時停止所有 agents"""
    for agent_id in list(agent_processes.keys()):
        stop_agent(agent_id)


if __name__ == "__main__":
    import uvicorn

    print("=" * 50)
    print("Vanna Multi-Agent Server")
    print("=" * 50)
    print("管理 API: http://localhost:8100")
    print("Agent ports: 8101, 8102, 8103...")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8100)
