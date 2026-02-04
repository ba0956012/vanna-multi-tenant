import os
import json
import logging
import boto3
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse, Response
from pydantic import BaseModel, Field
import httpx
from templates_ecs import (
    get_admin_html,
    get_add_memory_html,
    get_detail_html,
    get_agents_management_html,
    get_create_agent_html,
    get_efs_browser_html,
)
from vanna.integrations.chromadb.agent_memory import ChromaAgentMemory
import psycopg2

# Configure Logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d [%(process)d:%(thread)d] %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# AWS Configuration
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "ap-northeast-1")
DYNAMODB_TABLE_NAME = os.getenv("DYNAMODB_TABLE_NAME", "vanna-agents")
ECS_CLUSTER_NAME = os.getenv("ECS_CLUSTER_NAME", "vanna-cluster")
ECS_TASK_DEFINITION_AGENT = os.getenv("ECS_TASK_DEFINITION_AGENT", "vanna-agent")

# 處理 SUBNET_IDS 和 SECURITY_GROUP_IDS
subnet_ids_str = os.getenv("SUBNET_IDS", "")
SUBNET_IDS = [s.strip() for s in subnet_ids_str.split(",") if s.strip()] if subnet_ids_str else []

security_group_ids_str = os.getenv("SECURITY_GROUP_IDS", "")
SECURITY_GROUP_IDS = [s.strip() for s in security_group_ids_str.split(",") if s.strip()] if security_group_ids_str else []

# 驗證必要的環境變數
if not SUBNET_IDS:
    logger.error("SUBNET_IDS environment variable is not set or empty!")
if not SECURITY_GROUP_IDS:
    logger.error("SECURITY_GROUP_IDS environment variable is not set or empty!")

# URL Configuration
BASE_DOMAIN = os.getenv("BASE_DOMAIN") or os.getenv("ALB_DNS_NAME", "localhost:8100")
USE_HTTPS = os.getenv("USE_HTTPS", "false").lower() == "true"
PROTOCOL = "https" if USE_HTTPS else "http"

# Agent data directory (EFS mount point)
AGENT_DATA_DIR = "/app/agent_data"

# AWS Clients
ecs = boto3.client("ecs", region_name=AWS_REGION)
dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
table = dynamodb.Table(DYNAMODB_TABLE_NAME)

# FastAPI App
app = FastAPI(
    title="Vanna Multi-Agent Manager (ECS)",
    description="""
    ## ECS 版本的多 Agent 管理系統
    
    這個版本使用 AWS ECS 來管理 Agents，而不是 multiprocessing。
    
    ### 主要差異
    - Agent 配置儲存在 DynamoDB (不是 JSON 檔案)
    - Agent 運行在獨立的 ECS Services (不是進程)
    - 自動重啟和健康檢查由 ECS 管理
    
    ### 頁面
    - **管理介面**: `/admin/agents` - Agent 列表管理
    - **記憶管理**: `/admin/memory` - ChromaDB 記憶管理
    - **API 文檔**: `/docs` - Swagger UI (本頁面)
    """,
    version="2.0.0-ecs",
    docs_url="/docs",
    redoc_url="/redoc",
)


# === URL Helper ===
class URLConfig:
    """URL 配置輔助類"""
    
    @staticmethod
    def get_agent_url(agent_id: str, path: str = "") -> str:
        """生成 agent 的完整 URL"""
        base = f"{PROTOCOL}://{BASE_DOMAIN}/agent/{agent_id}"
        return f"{base}{path}" if path else base
    
    @staticmethod
    def get_manager_url(path: str = "") -> str:
        """生成 manager 的完整 URL"""
        base = f"{PROTOCOL}://{BASE_DOMAIN}"
        return f"{base}{path}" if path else base


# === ECS Operations ===
async def start_agent_ecs(agent_id: str, config: dict) -> str:
    """啟動 Agent ECS Service"""
    service_name = f"vanna-agent-{agent_id}"
    
    try:
        # 建立 ECS Service
        # 注意: create_service 不支援 overrides，所以 AGENT_ID 需要從 service name 解析
        # 或者在 Task Definition 中設定
        response = ecs.create_service(
            cluster=ECS_CLUSTER_NAME,
            serviceName=service_name,
            taskDefinition=ECS_TASK_DEFINITION_AGENT,
            desiredCount=1,
            launchType="FARGATE",
            networkConfiguration={
                "awsvpcConfiguration": {
                    "subnets": SUBNET_IDS,
                    "securityGroups": SECURITY_GROUP_IDS,
                    "assignPublicIp": "ENABLED",
                }
            },
            healthCheckGracePeriodSeconds=60,
        )
        
        service_arn = response["service"]["serviceArn"]
        logger.info(f"Created ECS Service for agent '{agent_id}': {service_arn}")
        return service_arn
        
    except Exception as e:
        # 如果 service 已存在，嘗試更新
        if "already exists" in str(e).lower():
            logger.warning(f"Service '{service_name}' already exists, updating...")
            try:
                response = ecs.update_service(
                    cluster=ECS_CLUSTER_NAME,
                    service=service_name,
                    desiredCount=1,
                )
                return response["service"]["serviceArn"]
            except Exception as update_error:
                logger.error(f"Failed to update existing service: {update_error}")
                raise
        else:
            logger.error(f"Failed to create ECS Service: {e}")
            raise


async def stop_agent_ecs(agent_id: str) -> None:
    """停止 Agent ECS Service"""
    service_name = f"vanna-agent-{agent_id}"
    
    try:
        # 先設定 desiredCount = 0
        ecs.update_service(
            cluster=ECS_CLUSTER_NAME,
            service=service_name,
            desiredCount=0,
        )
        logger.info(f"Stopped ECS Service for agent '{agent_id}'")
        
        # 可選: 刪除 service (註解掉，保留 service 以便重啟)
        # ecs.delete_service(
        #     cluster=ECS_CLUSTER_NAME,
        #     service=service_name,
        #     force=True,
        # )
        
    except ecs.exceptions.ServiceNotFoundException:
        logger.warning(f"Service '{service_name}' not found")
    except Exception as e:
        logger.error(f"Failed to stop ECS Service: {e}")
        raise


async def get_agent_ips(agent_id: str) -> dict:
    """取得 Agent Task 的 Public IP 和 Private IP"""
    service_name = f"vanna-agent-{agent_id}"
    
    result = {
        "public_ip": None,
        "private_ip": None,
    }
    
    try:
        # 1. 取得 running tasks
        tasks_response = ecs.list_tasks(
            cluster=ECS_CLUSTER_NAME,
            serviceName=service_name,
            desiredStatus="RUNNING"
        )
        
        if not tasks_response.get("taskArns"):
            logger.debug(f"No running tasks found for service '{service_name}'")
            return result
        
        # 2. 取得 task 詳細資訊
        task_arn = tasks_response["taskArns"][0]
        logger.debug(f"Found task: {task_arn}")
        
        task_response = ecs.describe_tasks(
            cluster=ECS_CLUSTER_NAME,
            tasks=[task_arn]
        )
        
        if not task_response.get("tasks"):
            logger.debug(f"No task details found for {task_arn}")
            return result
        
        task = task_response["tasks"][0]
        task_status = task.get("lastStatus")
        logger.debug(f"Task status: {task_status}")
        
        # 檢查 Task 是否真的在 RUNNING 狀態
        if task_status != "RUNNING":
            logger.debug(f"Task is not RUNNING yet (status: {task_status})")
            return result
        
        # 3. 從 attachments 取得 ENI ID
        attachments = task.get("attachments", [])
        logger.debug(f"Task has {len(attachments)} attachments")
        
        for attachment in attachments:
            if attachment["type"] == "ElasticNetworkInterface":
                for detail in attachment.get("details", []):
                    if detail["name"] == "networkInterfaceId":
                        eni_id = detail["value"]
                        logger.debug(f"Found ENI: {eni_id}")
                        
                        # 4. 查詢 ENI 的 Public IP 和 Private IP
                        ec2 = boto3.client("ec2", region_name=AWS_REGION)
                        eni_response = ec2.describe_network_interfaces(
                            NetworkInterfaceIds=[eni_id]
                        )
                        
                        if eni_response.get("NetworkInterfaces"):
                            network_interface = eni_response["NetworkInterfaces"][0]
                            
                            # Private IP
                            private_ip = network_interface.get("PrivateIpAddress")
                            if private_ip:
                                result["private_ip"] = private_ip
                                logger.info(f"✅ Found private IP for {agent_id}: {private_ip}")
                            
                            # Public IP
                            association = network_interface.get("Association", {})
                            public_ip = association.get("PublicIp")
                            if public_ip:
                                result["public_ip"] = public_ip
                                logger.info(f"✅ Found public IP for {agent_id}: {public_ip}")
                            else:
                                logger.warning(f"ENI {eni_id} has no public IP association")
                            
                            return result
        
        logger.warning(f"No IPs found for agent '{agent_id}'")
        return result
    except Exception as e:
        logger.error(f"Failed to get agent IPs for '{agent_id}': {e}")
        import traceback
        logger.error(traceback.format_exc())
        return result


async def get_agent_status(agent_id: str) -> dict:
    """取得 Agent 的運行狀態和位置"""
    service_name = f"vanna-agent-{agent_id}"
    
    try:
        response = ecs.describe_services(
            cluster=ECS_CLUSTER_NAME,
            services=[service_name],
        )
        
        if not response["services"]:
            return {
                "running": False,
                "status": "not_found",
                "public_ip": None,
                "private_ip": None,
            }
        
        service = response["services"][0]
        running_count = service["runningCount"]
        desired_count = service["desiredCount"]
        
        # 取得 Public IP 和 Private IP
        ips = {"public_ip": None, "private_ip": None}
        if running_count > 0:
            ips = await get_agent_ips(agent_id)
        
        return {
            "running": running_count > 0,
            "status": service["status"],
            "running_count": running_count,
            "desired_count": desired_count,
            "public_ip": ips["public_ip"],
            "private_ip": ips["private_ip"],
        }
    except Exception as e:
        logger.error(f"Failed to get agent status: {e}")
        return {
            "running": False,
            "status": "error",
            "error": str(e),
            "public_ip": None,
            "private_ip": None,
        }


# === Pydantic Models ===
class AgentConfig(BaseModel):
    """Agent configuration model"""
    agent_id: str = Field(..., description="Agent unique ID")
    description: str = Field("", description="Agent description")
    postgres_user: str = Field(..., description="PostgreSQL username")
    postgres_password: str = Field(..., description="PostgreSQL password")
    postgres_host: str = Field(..., description="PostgreSQL host")
    postgres_port: str = Field("5432", description="PostgreSQL port")
    postgres_db: str = Field(..., description="PostgreSQL database name")
    system_prompt: Optional[str] = Field("", description="Agent system prompt")
    
    class Config:
        extra = "ignore"  # 忽略額外的欄位（例如舊的 port 欄位）


class AddMemoryRequest(BaseModel):
    """新增記憶請求模型"""
    question: str = Field(..., description="問題描述")
    tool_name: str = Field(..., description="工具名稱")
    args: dict = Field(default_factory=dict, description="工具參數")
    metadata: dict = Field(default_factory=dict, description="額外的 metadata")


# === Agent Management APIs ===
@app.post("/api/agents", tags=["Agent Management"], summary="建立新 Agent")
async def register_agent(config: AgentConfig, wait_for_running: bool = False):
    """
    動態新增 agent (使用 ECS Service)
    
    建立一個新的 Agent 實例，會自動:
    1. 驗證資料庫連線參數
    2. 如果未提供 system_prompt，自動從資料庫生成
    3. 儲存配置到 DynamoDB
    4. 建立 ECS Service
    5. ECS 會自動管理 agent 的生命週期
    
    Args:
        config: Agent 配置
        wait_for_running: 是否等待 Task 啟動完成 (預設 False，建議保持 False 以加快回應)
    """
    agent_id = config.agent_id
    
    # 1. 檢查是否已存在
    try:
        response = table.get_item(Key={"agent_id": agent_id})
        if "Item" in response and response["Item"].get("status") != "deleted":
            raise HTTPException(400, f"Agent '{agent_id}' already exists")
    except Exception as e:
        if "Item" not in str(e):
            logger.error(f"DynamoDB error: {e}")
    
    config_dict = config.dict()
    
    # 2. 驗證必要參數
    if not all([config.postgres_user, config.postgres_password, config.postgres_host, config.postgres_db]):
        raise HTTPException(400, "Missing required postgres connection parameters")
    
    # 3. 如果沒有提供 system_prompt，自動從資料庫生成
    system_prompt_generated = False
    if not config_dict.get("system_prompt") or config_dict["system_prompt"].strip() == "":
        logger.info(f"Generating system prompt from database for agent '{agent_id}'")
        try:
            conn = psycopg2.connect(
                host=config.postgres_host,
                port=config.postgres_port,
                user=config.postgres_user,
                password=config.postgres_password,
                dbname=config.postgres_db,
                connect_timeout=10,
            )
            
            config_dict["system_prompt"] = generate_system_prompt_from_db(conn, config.postgres_db)
            system_prompt_generated = True
            conn.close()
            
            # 檢查大小
            prompt_size = len(config_dict["system_prompt"].encode("utf-8"))
            logger.info(f"Generated system prompt size: {prompt_size:,} bytes ({prompt_size/1024:.1f} KB)")
            
            if prompt_size > 300_000:  # 300 KB 警告
                logger.warning(f"System prompt is large ({prompt_size/1024:.1f} KB), consider simplifying")
            
            logger.info(f"Successfully generated system prompt for agent '{agent_id}'")
        except Exception as e:
            logger.error(f"Failed to generate system prompt: {e}")
            raise HTTPException(500, f"無法連接資料庫或生成 system prompt: {e}")
    
    # 4. 儲存配置到 DynamoDB
    table.put_item(
        Item={
            "agent_id": agent_id,
            "config": config_dict,
            "status": "creating",
            "created_at": datetime.now().isoformat(),
        }
    )
    
    # 5. 建立 ECS Service
    try:
        service_arn = await start_agent_ecs(agent_id, config_dict)
        
        # 6. 更新 service_arn 和狀態
        table.update_item(
            Key={"agent_id": agent_id},
            UpdateExpression="SET service_arn = :arn, #status = :status",
            ExpressionAttributeNames={"#status": "status"},
            ExpressionAttributeValues={":arn": service_arn, ":status": "running"},
        )
        
        # 7. 如果需要等待 Task 啟動
        if wait_for_running:
            logger.info(f"Waiting for agent '{agent_id}' task to start...")
            import asyncio
            max_wait = 120  # 最多等待 2 分鐘
            wait_interval = 5  # 每 5 秒檢查一次
            elapsed = 0
            
            while elapsed < max_wait:
                await asyncio.sleep(wait_interval)
                elapsed += wait_interval
                
                status = await get_agent_status(agent_id)
                if status["running"] and status.get("public_ip"):
                    logger.info(f"Agent '{agent_id}' is now running at {status['public_ip']}")
                    return {
                        "message": f"Agent '{agent_id}' 已成功啟動！",
                        "agent_id": agent_id,
                        "service_arn": service_arn,
                        "status": "running",
                        "public_ip": status["public_ip"],
                        "url": f"http://{status['public_ip']}:8101",
                        "chat_url": f"http://{status['public_ip']}:8101/api/vanna/v2/chat_sse",
                        "system_prompt_generated": system_prompt_generated,
                    }
            
            # 超時但 Service 已建立
            logger.warning(f"Timeout waiting for agent '{agent_id}' to start, but service is created")
        
        return {
            "message": f"Agent '{agent_id}' 建立成功！ECS Service 正在部署中...",
            "agent_id": agent_id,
            "service_arn": service_arn,
            "status": "deploying",
            "note": "Task 啟動需要 1-2 分鐘，請稍後重新整理頁面查看 Public IP 和 URL",
            "system_prompt_generated": system_prompt_generated,
        }
    except Exception as e:
        # 失敗時更新狀態
        table.update_item(
            Key={"agent_id": agent_id},
            UpdateExpression="SET #status = :status, #error = :error",
            ExpressionAttributeNames={"#status": "status", "#error": "error"},
            ExpressionAttributeValues={":status": "error", ":error": str(e)},
        )
        raise HTTPException(500, f"Failed to create agent: {e}")


@app.get("/api/agents/{agent_id}", tags=["Agent Management"], summary="取得 Agent 詳細資訊")
async def get_agent_detail(agent_id: str, refresh: bool = True):
    """
    取得指定 Agent 的詳細資訊，包含 Public IP 和 URL
    
    用於建立 Agent 後輪詢查詢狀態和 IP
    
    Args:
        agent_id: Agent ID
        refresh: 是否即時查詢 ECS 狀態 (預設 True)
    """
    try:
        # 從 DynamoDB 取得配置
        response = table.get_item(Key={"agent_id": agent_id})
        if "Item" not in response:
            raise HTTPException(404, f"Agent '{agent_id}' not found")
        
        item = response["Item"]
        
        if refresh:
            # 即時查詢 ECS 狀態和 Public IP
            ecs_status = await get_agent_status(agent_id)
            public_ip = ecs_status.get("public_ip")
            private_ip = ecs_status.get("private_ip")
            running = ecs_status["running"]
            
            # 更新到 DynamoDB（快取）
            try:
                update_expr = "SET last_public_ip = :pub_ip, last_private_ip = :priv_ip, last_checked = :time, ecs_running = :running"
                expr_values = {
                    ":pub_ip": public_ip if public_ip else "N/A",
                    ":priv_ip": private_ip if private_ip else "N/A",
                    ":time": datetime.now().isoformat(),
                    ":running": running
                }
                
                table.update_item(
                    Key={"agent_id": agent_id},
                    UpdateExpression=update_expr,
                    ExpressionAttributeValues=expr_values
                )
            except Exception as e:
                logger.warning(f"Failed to cache IPs for {agent_id}: {e}")
            
            # 動態生成 URL（使用 Private IP）
            if private_ip and running:
                agent_url = f"http://{private_ip}:8101"
                chat_url = f"http://{private_ip}:8101/api/vanna/v2/chat_sse"
            else:
                agent_url = None
                chat_url = None
            
            return {
                "agent_id": agent_id,
                "description": item.get("config", {}).get("description", ""),
                "status": item.get("status", "unknown"),
                "running": running,
                "public_ip": public_ip,
                "private_ip": private_ip,
                "url": agent_url,
                "chat_url": chat_url,
                "service_arn": item.get("service_arn"),
                "created_at": item.get("created_at"),
                "ecs_status": ecs_status.get("status"),
                "running_count": ecs_status.get("running_count", 0),
                "desired_count": ecs_status.get("desired_count", 0),
                "refreshed": True,
            }
        else:
            # 快速模式：從 DynamoDB 讀取快取
            cached_public_ip = item.get("last_public_ip")
            cached_private_ip = item.get("last_private_ip")
            cached_running = item.get("ecs_running", False)
            last_checked = item.get("last_checked")
            
            if cached_private_ip and cached_private_ip != "N/A" and cached_running:
                agent_url = f"http://{cached_private_ip}:8101"
                chat_url = f"http://{cached_private_ip}:8101/api/vanna/v2/chat_sse"
            else:
                agent_url = None
                chat_url = None
            
            return {
                "agent_id": agent_id,
                "description": item.get("config", {}).get("description", ""),
                "status": item.get("status", "unknown"),
                "running": cached_running,
                "public_ip": cached_public_ip if cached_public_ip != "N/A" else None,
                "private_ip": cached_private_ip if cached_private_ip != "N/A" else None,
                "url": agent_url,
                "chat_url": chat_url,
                "service_arn": item.get("service_arn"),
                "created_at": item.get("created_at"),
                "last_checked": last_checked,
                "refreshed": False,
                "cached": True,
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agent detail: {e}")
        raise HTTPException(500, f"Failed to get agent detail: {e}")


@app.get("/api/agents", tags=["Agent Management"], summary="列出所有 Agents")
async def list_agents(refresh: bool = True):
    """
    列出所有已註冊的 agents
    
    Args:
        refresh: 是否即時查詢 ECS 狀態和 Public IP (預設 True)
                 - True: 查詢最新狀態並更新到 DynamoDB，較慢但資料最新
                 - False: 只從 DynamoDB 讀取快取的資料，快速
    
    Returns:
        - agents: Agent 列表，包含 ID, URL, 運行狀態, Public IP
    """
    try:
        # 從 DynamoDB 掃描所有 agents (排除已刪除)
        response = table.scan(
            FilterExpression="attribute_not_exists(#status) OR #status <> :deleted",
            ExpressionAttributeNames={"#status": "status"},
            ExpressionAttributeValues={":deleted": "deleted"},
        )
        
        agents_info = []
        for item in response["Items"]:
            agent_id = item["agent_id"]
            
            if refresh:
                # 即時查詢 ECS 狀態和 Public IP
                ecs_status = await get_agent_status(agent_id)
                public_ip = ecs_status.get("public_ip")
                private_ip = ecs_status.get("private_ip")
                running = ecs_status["running"]
                
                # 更新到 DynamoDB（快取）
                try:
                    update_expr = "SET last_public_ip = :pub_ip, last_private_ip = :priv_ip, last_checked = :time, ecs_running = :running"
                    expr_values = {
                        ":pub_ip": public_ip if public_ip else "N/A",
                        ":priv_ip": private_ip if private_ip else "N/A",
                        ":time": datetime.now().isoformat(),
                        ":running": running
                    }
                    
                    table.update_item(
                        Key={"agent_id": agent_id},
                        UpdateExpression=update_expr,
                        ExpressionAttributeValues=expr_values
                    )
                except Exception as e:
                    logger.warning(f"Failed to cache IPs for {agent_id}: {e}")
                
                # 生成 URL（使用 Private IP）
                if running:
                    # 直接 URL（使用 Private IP）
                    direct_url = f"http://{private_ip}:8101" if private_ip else None
                    direct_chat_url = f"http://{private_ip}:8101/api/vanna/v2/chat_sse" if private_ip else None
                    
                    # Proxy URL（通過 Manager）
                    proxy_url = f"{PROTOCOL}://{BASE_DOMAIN}/agent/{agent_id}"
                    proxy_chat_url = f"{PROTOCOL}://{BASE_DOMAIN}/agent/{agent_id}/api/vanna/v2/chat_sse"
                else:
                    direct_url = None
                    direct_chat_url = None
                    proxy_url = None
                    proxy_chat_url = None
                
                agents_info.append({
                    "agent_id": agent_id,
                    "description": item.get("config", {}).get("description", ""),
                    "status": item.get("status", "unknown"),
                    "running": running,
                    "public_ip": public_ip,
                    "private_ip": private_ip,
                    "url": proxy_url,  # 預設使用 proxy URL
                    "chat_url": proxy_chat_url,
                    "direct_url": direct_url,  # 也提供直接 URL
                    "direct_chat_url": direct_chat_url,
                })
            else:
                # 快速模式：從 DynamoDB 讀取快取的資料
                cached_public_ip = item.get("last_public_ip")
                cached_private_ip = item.get("last_private_ip")
                cached_running = item.get("ecs_running", False)
                last_checked = item.get("last_checked")
                
                if cached_running:
                    # 直接 URL（使用 Private IP）
                    direct_url = f"http://{cached_private_ip}:8101" if cached_private_ip and cached_private_ip != "N/A" else None
                    direct_chat_url = f"http://{cached_private_ip}:8101/api/vanna/v2/chat_sse" if cached_private_ip and cached_private_ip != "N/A" else None
                    
                    # Proxy URL
                    proxy_url = f"{PROTOCOL}://{BASE_DOMAIN}/agent/{agent_id}"
                    proxy_chat_url = f"{PROTOCOL}://{BASE_DOMAIN}/agent/{agent_id}/api/vanna/v2/chat_sse"
                else:
                    direct_url = None
                    direct_chat_url = None
                    proxy_url = None
                    proxy_chat_url = None
                
                agents_info.append({
                    "agent_id": agent_id,
                    "description": item.get("config", {}).get("description", ""),
                    "status": item.get("status", "unknown"),
                    "running": cached_running,
                    "public_ip": cached_public_ip if cached_public_ip != "N/A" else None,
                    "private_ip": cached_private_ip if cached_private_ip != "N/A" else None,
                    "url": proxy_url,
                    "chat_url": proxy_chat_url,
                    "direct_url": direct_url,
                    "direct_chat_url": direct_chat_url,
                    "last_checked": last_checked,
                    "cached": True
                })
        
        return {"agents": agents_info, "refreshed": refresh}
    except Exception as e:
        logger.error(f"Failed to list agents: {e}")
        raise HTTPException(500, f"Failed to list agents: {e}")


@app.post("/api/agents/{agent_id}/stop", tags=["Agent Management"], summary="停止 Agent")
async def stop_agent_api(agent_id: str):
    """
    停止指定的 agent（不刪除）
    
    將 ECS Service 的 desiredCount 設為 0，停止 Task 運行。
    Service 保留，可以隨時重新啟動。
    """
    # 1. 檢查是否存在
    try:
        response = table.get_item(Key={"agent_id": agent_id})
        if "Item" not in response:
            raise HTTPException(404, f"Agent '{agent_id}' not found")
    except Exception as e:
        logger.error(f"DynamoDB error: {e}")
        raise HTTPException(500, f"Failed to check agent: {e}")
    
    # 2. 停止 ECS Service
    try:
        await stop_agent_ecs(agent_id)
        
        # 3. 更新狀態
        table.update_item(
            Key={"agent_id": agent_id},
            UpdateExpression="SET #status = :status, stopped_at = :time",
            ExpressionAttributeNames={"#status": "status"},
            ExpressionAttributeValues={
                ":status": "stopped",
                ":time": datetime.now().isoformat(),
            },
        )
        
        return {"message": f"Agent '{agent_id}' stopped"}
    except Exception as e:
        logger.error(f"Failed to stop agent: {e}")
        raise HTTPException(500, f"Failed to stop agent: {e}")


@app.post("/api/agents/{agent_id}/start", tags=["Agent Management"], summary="啟動 Agent")
async def start_agent_api(agent_id: str):
    """
    啟動已停止的 agent
    
    將 ECS Service 的 desiredCount 設為 1，重新啟動 Task。
    """
    # 1. 檢查是否存在
    try:
        response = table.get_item(Key={"agent_id": agent_id})
        if "Item" not in response:
            raise HTTPException(404, f"Agent '{agent_id}' not found")
        
        config = response["Item"]["config"]
    except Exception as e:
        logger.error(f"DynamoDB error: {e}")
        raise HTTPException(500, f"Failed to check agent: {e}")
    
    # 2. 啟動 ECS Service
    service_name = f"vanna-agent-{agent_id}"
    
    try:
        # 嘗試更新現有 service
        ecs.update_service(
            cluster=ECS_CLUSTER_NAME,
            service=service_name,
            desiredCount=1,
        )
        logger.info(f"Started existing ECS Service for agent '{agent_id}'")
        
        # 3. 更新狀態
        table.update_item(
            Key={"agent_id": agent_id},
            UpdateExpression="SET #status = :status, started_at = :time",
            ExpressionAttributeNames={"#status": "status"},
            ExpressionAttributeValues={
                ":status": "running",
                ":time": datetime.now().isoformat(),
            },
        )
        
        return {"message": f"Agent '{agent_id}' started"}
    except ecs.exceptions.ServiceNotFoundException:
        # Service 不存在，建立新的
        logger.info(f"Service not found, creating new service for agent '{agent_id}'")
        try:
            service_arn = await start_agent_ecs(agent_id, config)
            
            table.update_item(
                Key={"agent_id": agent_id},
                UpdateExpression="SET service_arn = :arn, #status = :status, started_at = :time",
                ExpressionAttributeNames={"#status": "status"},
                ExpressionAttributeValues={
                    ":arn": service_arn,
                    ":status": "running",
                    ":time": datetime.now().isoformat(),
                },
            )
            
            return {"message": f"Agent '{agent_id}' started (new service created)", "service_arn": service_arn}
        except Exception as e:
            logger.error(f"Failed to create service: {e}")
            raise HTTPException(500, f"Failed to create service: {e}")
    except Exception as e:
        logger.error(f"Failed to start agent: {e}")
        raise HTTPException(500, f"Failed to start agent: {e}")


@app.delete("/api/agents/{agent_id}", tags=["Agent Management"], summary="刪除 Agent")
async def remove_agent(agent_id: str, delete_memory: bool = False, delete_service: bool = False):
    """
    移除指定的 agent
    
    Args:
        agent_id: Agent ID
        delete_memory: 是否同時刪除 ChromaDB 記憶 (預設 False)
        delete_service: 是否完全刪除 ECS Service (預設 False，只停止)
    """
    # 1. 檢查是否存在
    try:
        response = table.get_item(Key={"agent_id": agent_id})
        if "Item" not in response:
            raise HTTPException(404, f"Agent '{agent_id}' not found")
    except Exception as e:
        logger.error(f"DynamoDB error: {e}")
        raise HTTPException(500, f"Failed to check agent: {e}")
    
    # 2. 停止或刪除 ECS Service
    service_name = f"vanna-agent-{agent_id}"
    try:
        if delete_service:
            # 完全刪除 Service
            logger.info(f"Deleting ECS Service '{service_name}'")
            ecs.delete_service(
                cluster=ECS_CLUSTER_NAME,
                service=service_name,
                force=True,
            )
            logger.info(f"Deleted ECS Service for agent '{agent_id}'")
        else:
            # 只停止 Service（保留以便重啟）
            await stop_agent_ecs(agent_id)
    except ecs.exceptions.ServiceNotFoundException:
        logger.warning(f"Service '{service_name}' not found")
    except Exception as e:
        logger.error(f"Failed to stop/delete ECS Service: {e}")
        # 繼續執行，不中斷
    
    # 2.5. 如果需要刪除記憶，等待 Task 完全停止
    if delete_memory:
        logger.info(f"Waiting for agent '{agent_id}' tasks to stop before deleting EFS...")
        import asyncio
        max_wait = 90  # 增加到 90 秒
        wait_interval = 5
        elapsed = 0
        
        while elapsed < max_wait:
            try:
                # 檢查是否還有 running tasks
                logger.info(f"Checking tasks for service '{service_name}'...")
                tasks_response = ecs.list_tasks(
                    cluster=ECS_CLUSTER_NAME,
                    serviceName=service_name,
                    desiredStatus="RUNNING"
                )
                
                task_count = len(tasks_response.get("taskArns", []))
                logger.info(f"Found {task_count} running tasks")
                
                if task_count == 0:
                    logger.info(f"✅ All tasks stopped for agent '{agent_id}'")
                    break
                
                logger.info(f"⏳ Still have {task_count} running tasks, waiting {wait_interval} seconds...")
                await asyncio.sleep(wait_interval)
                elapsed += wait_interval
            except ecs.exceptions.ServiceNotFoundException:
                logger.info(f"✅ Service not found, assuming tasks are stopped")
                break
            except Exception as e:
                logger.warning(f"⚠️ Error checking tasks: {e}")
                import traceback
                logger.warning(traceback.format_exc())
                break
        
        if elapsed >= max_wait:
            logger.error(f"❌ Timeout waiting for tasks to stop after {max_wait} seconds!")
            logger.error(f"⚠️ Will try to delete anyway, but may fail due to busy resources")
        
        # 額外等待 10 秒讓檔案系統同步（增加到 10 秒）
        logger.info("Waiting 10 seconds for filesystem sync...")
        await asyncio.sleep(10)
        logger.info("✅ Filesystem sync wait complete")
        
        # 關閉 Manager 自己打開的檔案描述符
        logger.info(f"Closing any open file descriptors for agent '{agent_id}'...")
        try:
            persist_dir = f"{AGENT_DATA_DIR}/chroma_db_{agent_id}"
            current_pid = os.getpid()
            closed_count = 0
            
            fd_dir = f"/proc/{current_pid}/fd"
            if os.path.exists(fd_dir):
                for fd in os.listdir(fd_dir):
                    try:
                        link = os.readlink(os.path.join(fd_dir, fd))
                        if persist_dir in link:
                            # 關閉這個檔案描述符
                            try:
                                os.close(int(fd))
                                closed_count += 1
                                logger.info(f"Closed fd {fd}: {link}")
                            except Exception as e:
                                logger.warning(f"Cannot close fd {fd}: {e}")
                    except:
                        pass
            
            logger.info(f"✅ Closed {closed_count} file descriptors")
        except Exception as e:
            logger.warning(f"⚠️ Error closing file descriptors: {e}")
    
    # 3. 刪除 EFS 上的 ChromaDB 記憶
    memory_deleted = False
    if delete_memory:
        try:
            persist_dir = f"{AGENT_DATA_DIR}/chroma_db_{agent_id}"
            logger.info(f"Attempting to delete memory at: {persist_dir}")
            logger.info(f"AGENT_DATA_DIR: {AGENT_DATA_DIR}")
            logger.info(f"agent_id: {agent_id}")
            
            # 檢查目錄是否存在
            exists = os.path.exists(persist_dir)
            logger.info(f"Directory exists check: {exists}")
            
            if exists:
                logger.info(f"Directory exists, deleting: {persist_dir}")
                
                # 列出目錄內容
                try:
                    items = os.listdir(persist_dir)
                    logger.info(f"Directory contains {len(items)} items")
                except Exception as e:
                    logger.warning(f"Cannot list directory: {e}")
                
                # 在 NFS/EFS 上，使用 rm -rf 比 shutil.rmtree 更可靠
                import subprocess
                
                # 重試機制：最多嘗試 3 次
                max_retries = 3
                retry_delay = 10  # 每次重試前等待 10 秒
                
                for attempt in range(1, max_retries + 1):
                    logger.info(f"Deletion attempt {attempt}/{max_retries}")
                    
                    result = subprocess.run(
                        ["rm", "-rf", persist_dir],
                        capture_output=True,
                        text=True,
                        timeout=60
                    )
                    
                    logger.info(f"rm -rf return code: {result.returncode}")
                    if result.stdout:
                        logger.info(f"rm -rf stdout: {result.stdout}")
                    if result.stderr:
                        logger.info(f"rm -rf stderr: {result.stderr}")
                    
                    if result.returncode == 0:
                        # 再次檢查是否真的刪除了
                        still_exists = os.path.exists(persist_dir)
                        logger.info(f"After deletion, directory still exists: {still_exists}")
                        
                        if not still_exists:
                            logger.info(f"✅ Successfully deleted ChromaDB memory at {persist_dir}")
                            memory_deleted = True
                            break
                        else:
                            logger.warning(f"⚠️ rm -rf returned 0 but directory still exists")
                    else:
                        logger.warning(f"❌ Attempt {attempt} failed with return code {result.returncode}")
                    
                    # 如果不是最後一次嘗試，等待後重試
                    if attempt < max_retries:
                        logger.info(f"Waiting {retry_delay} seconds before retry...")
                        await asyncio.sleep(retry_delay)
                
                if not memory_deleted:
                    logger.error(f"❌ Failed to delete after {max_retries} attempts")
            else:
                logger.warning(f"⚠️ Memory directory not found: {persist_dir}")
                # 列出父目錄內容
                try:
                    parent_items = os.listdir(AGENT_DATA_DIR)
                    logger.info(f"Parent directory contains: {parent_items}")
                except Exception as e:
                    logger.warning(f"Cannot list parent directory: {e}")
        except Exception as e:
            logger.error(f"❌ Failed to delete memory: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # 繼續執行，不中斷
    else:
        logger.info(f"delete_memory=False, skipping EFS deletion for agent '{agent_id}'")
    
    # 4. 從 DynamoDB 刪除記錄
    try:
        table.delete_item(Key={"agent_id": agent_id})
        logger.info(f"Deleted DynamoDB record for agent '{agent_id}'")
    except Exception as e:
        logger.error(f"Failed to delete DynamoDB record: {e}")
        raise HTTPException(500, f"Failed to delete DynamoDB record: {e}")
    
    return {
        "message": f"Agent '{agent_id}' removed",
        "service_deleted": delete_service,
        "memory_deleted": memory_deleted,
        "dynamodb_deleted": True,
    }


@app.post("/api/agents/{agent_id}/restart", tags=["Agent Management"], summary="重啟 Agent")
async def restart_agent_api(agent_id: str):
    """
    重啟指定的 agent
    
    會停止現有 ECS Service 並重新啟動
    """
    # 1. 檢查是否存在
    try:
        response = table.get_item(Key={"agent_id": agent_id})
        if "Item" not in response:
            raise HTTPException(404, f"Agent '{agent_id}' not found")
        
        config = response["Item"]["config"]
    except Exception as e:
        logger.error(f"DynamoDB error: {e}")
        raise HTTPException(500, f"Failed to check agent: {e}")
    
    # 2. 停止舊 service
    try:
        await stop_agent_ecs(agent_id)
    except Exception as e:
        logger.warning(f"Failed to stop old service: {e}")
    
    # 3. 啟動新 service
    try:
        service_arn = await start_agent_ecs(agent_id, config)
        
        # 4. 更新狀態
        table.update_item(
            Key={"agent_id": agent_id},
            UpdateExpression="SET service_arn = :arn, #status = :status, restarted_at = :time",
            ExpressionAttributeNames={"#status": "status"},
            ExpressionAttributeValues={
                ":arn": service_arn,
                ":status": "running",
                ":time": datetime.now().isoformat(),
            },
        )
        
        return {"message": f"Agent '{agent_id}' restarted", "service_arn": service_arn}
    except Exception as e:
        logger.error(f"Failed to restart agent: {e}")
        raise HTTPException(500, f"Failed to restart agent: {e}")


# === Web UI Routes ===
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/admin/agents")


@app.get("/admin/agents", response_class=HTMLResponse, include_in_schema=False)
async def agents_management_page(message: str = None):
    """Agent 管理頁面"""
    try:
        response = table.scan(
            FilterExpression="attribute_not_exists(#status) OR #status <> :deleted",
            ExpressionAttributeNames={"#status": "status"},
            ExpressionAttributeValues={":deleted": "deleted"},
        )
        
        agents_info = []
        for item in response["Items"]:
            agent_id = item["agent_id"]
            ecs_status = await get_agent_status(agent_id)
            
            # 動態生成 URL（使用 Private IP）
            public_ip = ecs_status.get("public_ip")
            private_ip = ecs_status.get("private_ip")
            if private_ip and ecs_status["running"]:
                agent_url = f"http://{private_ip}:8101"
            else:
                agent_url = None
            
            agents_info.append({
                "agent_id": agent_id,
                "description": item.get("config", {}).get("description", ""),
                "running": ecs_status["running"],
                "public_ip": public_ip,
                "private_ip": private_ip,
                "url": agent_url,
            })
        
        return get_agents_management_html(agents_info, message)
    except Exception as e:
        logger.error(f"Failed to load agents page: {e}")
        return f"<html><body><h1>Error</h1><p>{e}</p></body></html>"


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
    # 檢查是否已存在
    try:
        response = table.get_item(Key={"agent_id": agent_id})
        if "Item" in response and response["Item"].get("status") != "deleted":
            return get_create_agent_html(f"❌ Agent '{agent_id}' 已存在")
    except Exception as e:
        logger.error(f"DynamoDB error: {e}")
        return get_create_agent_html(f"❌ 檢查失敗: {e}")
    
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
        system_prompt_generated = False
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
                system_prompt_generated = True
                conn.close()
                logger.info(f"Successfully generated system prompt for agent '{agent_id}'")
            except Exception as e:
                logger.error(f"Failed to generate system prompt: {e}")
                return get_create_agent_html(f"❌ 無法連接資料庫或生成 system prompt: {e}")
        
        # 儲存配置到 DynamoDB
        table.put_item(
            Item={
                "agent_id": agent_id,
                "config": config_dict,
                "status": "creating",
                "created_at": datetime.now().isoformat(),
            }
        )
        
        # 建立 ECS Service
        service_arn = await start_agent_ecs(agent_id, config_dict)
        
        # 更新 service_arn 和狀態
        table.update_item(
            Key={"agent_id": agent_id},
            UpdateExpression="SET service_arn = :arn, #status = :status",
            ExpressionAttributeNames={"#status": "status"},
            ExpressionAttributeValues={":arn": service_arn, ":status": "running"},
        )
        
        msg = f"✅ Agent '{agent_id}' 建立成功！ECS Service 正在部署中..."
        if system_prompt_generated:
            msg += " (已自動生成 system prompt)"
        
        return RedirectResponse(f"/admin/agents?message={msg}", status_code=302)
    except Exception as e:
        logger.error(f"Failed to create agent: {e}")
        # 失敗時更新狀態
        try:
            table.update_item(
                Key={"agent_id": agent_id},
                UpdateExpression="SET #status = :status, #error = :error",
                ExpressionAttributeNames={"#status": "status", "#error": "error"},
                ExpressionAttributeValues={":status": "error", ":error": str(e)},
            )
        except:
            pass
        return get_create_agent_html(f"❌ 建立失敗: {e}")


# === Memory Management Web UI ===
@app.get("/admin/memory", response_class=HTMLResponse, include_in_schema=False)
async def memory_admin_page(agent_id: str = None, message: str = None):
    """記憶管理主頁面"""
    # 取得所有 agents
    try:
        response = table.scan(
            FilterExpression="attribute_not_exists(#status) OR #status <> :deleted",
            ExpressionAttributeNames={"#status": "status"},
            ExpressionAttributeValues={":deleted": "deleted"},
        )
        
        agents_info = []
        for item in response["Items"]:
            aid = item["agent_id"]
            ecs_status = await get_agent_status(aid)
            agents_info.append({
                "agent_id": aid,
                "port": None,  # ECS 版本沒有固定 port
                "running": ecs_status["running"],
            })
    except Exception as e:
        logger.error(f"Failed to load agents: {e}")
        agents_info = []
    
    # 如果選擇了 agent，載入記憶
    memories = None
    if agent_id:
        # 檢查 agent 是否存在
        try:
            response = table.get_item(Key={"agent_id": agent_id})
            if "Item" not in response:
                return get_admin_html(agents_info, agent_id, [], f"❌ Agent '{agent_id}' not found")
        except Exception as e:
            logger.error(f"DynamoDB error: {e}")
            return get_admin_html(agents_info, agent_id, [], f"❌ 檢查失敗: {e}")
        
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
    # 檢查 agent 是否存在
    try:
        response = table.get_item(Key={"agent_id": agent_id})
        if "Item" not in response:
            raise HTTPException(404, f"Agent '{agent_id}' not found")
    except Exception as e:
        logger.error(f"DynamoDB error: {e}")
        raise HTTPException(500, f"Failed to check agent: {e}")
    
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
    # 檢查 agent 是否存在
    try:
        response = table.get_item(Key={"agent_id": agent_id})
        if "Item" not in response:
            raise HTTPException(404, f"Agent '{agent_id}' not found")
    except Exception as e:
        logger.error(f"DynamoDB error: {e}")
        raise HTTPException(500, f"Failed to check agent: {e}")
    
    try:
        args = json.loads(args_json or "{}")
        metadata = json.loads(metadata_json or "{}")
    except json.JSONDecodeError:
        raise HTTPException(400, "JSON 格式錯誤")
    
    memory = get_agent_memory(agent_id, create_if_not_exists=True)
    from vanna.core.user import User, RequestContext
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
    # 檢查 agent 是否存在
    try:
        response = table.get_item(Key={"agent_id": agent_id})
        if "Item" not in response:
            raise HTTPException(404, f"Agent '{agent_id}' not found")
    except Exception as e:
        logger.error(f"DynamoDB error: {e}")
        raise HTTPException(500, f"Failed to check agent: {e}")
    
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
    # 檢查 agent 是否存在
    try:
        response = table.get_item(Key={"agent_id": agent_id})
        if "Item" not in response:
            raise HTTPException(404, f"Agent '{agent_id}' not found")
    except Exception as e:
        logger.error(f"DynamoDB error: {e}")
        raise HTTPException(500, f"Failed to check agent: {e}")
    
    memory = get_agent_memory(agent_id)
    from vanna.core.user import User, RequestContext
    context = RequestContext(user=User(id="admin"))
    
    await memory.delete_by_id(context=context, memory_id=memory_id)
    
    return RedirectResponse(f"/admin/memory?agent_id={agent_id}&message=記憶已刪除", status_code=302)


# === Memory Helper Function ===
def get_agent_memory(agent_id: str, create_if_not_exists: bool = False) -> ChromaAgentMemory:
    """取得指定 agent 的 ChromaDB memory"""
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
    # 檢查 agent 是否存在
    try:
        response = table.get_item(Key={"agent_id": agent_id})
        if "Item" not in response:
            raise HTTPException(404, f"Agent '{agent_id}' not found")
    except Exception as e:
        logger.error(f"DynamoDB error: {e}")
        raise HTTPException(500, f"Failed to check agent: {e}")
    
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
    except HTTPException:
        raise
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
    # 檢查 agent 是否存在
    try:
        response = table.get_item(Key={"agent_id": agent_id})
        if "Item" not in response:
            raise HTTPException(404, f"Agent '{agent_id}' not found")
    except Exception as e:
        logger.error(f"DynamoDB error: {e}")
        raise HTTPException(500, f"Failed to check agent: {e}")
    
    try:
        memory = get_agent_memory(agent_id, create_if_not_exists=True)
        from vanna.core.user import User, RequestContext
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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add memory: {e}")
        raise HTTPException(500, f"Failed to add memory: {e}")


@app.get("/api/agents/{agent_id}/memories/{memory_id}", tags=["Memory Management"], summary="取得記憶詳情")
async def get_memory_detail_api(agent_id: str, memory_id: str):
    """
    取得指定記憶的詳細資訊
    
    Args:
        agent_id: Agent ID
        memory_id: Memory ID
        
    Returns:
        記憶的完整資訊
    """
    # 檢查 agent 是否存在
    try:
        response = table.get_item(Key={"agent_id": agent_id})
        if "Item" not in response:
            raise HTTPException(404, f"Agent '{agent_id}' not found")
    except Exception as e:
        logger.error(f"DynamoDB error: {e}")
        raise HTTPException(500, f"Failed to check agent: {e}")
    
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
    # 檢查 agent 是否存在
    try:
        response = table.get_item(Key={"agent_id": agent_id})
        if "Item" not in response:
            raise HTTPException(404, f"Agent '{agent_id}' not found")
    except Exception as e:
        logger.error(f"DynamoDB error: {e}")
        raise HTTPException(500, f"Failed to check agent: {e}")
    
    try:
        memory = get_agent_memory(agent_id)
        from vanna.core.user import User, RequestContext
        context = RequestContext(user=User(id="admin"))
        
        await memory.delete_by_id(context=context, memory_id=memory_id)
        
        return {"message": f"Memory '{memory_id}' deleted from agent '{agent_id}'"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete memory: {e}")
        raise HTTPException(500, f"Failed to delete memory: {e}")


# === Training / Auto-generation APIs ===
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
    # 檢查 agent 是否存在並取得配置
    try:
        response = table.get_item(Key={"agent_id": agent_id})
        if "Item" not in response:
            raise HTTPException(404, f"Agent '{agent_id}' not found")
        agent_config = response["Item"]["config"]
    except Exception as e:
        logger.error(f"DynamoDB error: {e}")
        raise HTTPException(500, f"Failed to check agent: {e}")
    
    # 連接資料庫
    pg_user = agent_config.get("postgres_user")
    pg_password = agent_config.get("postgres_password")
    pg_host = agent_config.get("postgres_host")
    pg_port = agent_config.get("postgres_port") or "5432"
    pg_db = agent_config.get("postgres_db")
    
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
        # 使用 auto_generate_fewshot_pg.py 的函數
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        from auto_generate_fewshot_pg import (
            analyze_database,
            build_fk_graph,
            generate_fewshot_for_table,
        )
        
        tables, schema = analyze_database(conn)
        logger.info(f"Found {len(tables)} tables")
        graph = build_fk_graph(tables, schema)
        
        fewshots = []
        for tbl in tables:
            try:
                fs = generate_fewshot_for_table(tbl, conn, schema, graph, pg_db)
                if fs:
                    fewshots.append(fs)
                    logger.info(f"Generated fewshot for {tbl}")
            except Exception as e:
                logger.error(f"Error generating fewshot for {tbl}: {e}")
                continue
        
        conn.close()
        conn = None
        
        logger.info(f"Generated {len(fewshots)} fewshots, importing to ChromaDB...")
        
        # 匯入到 ChromaDB
        memory = get_agent_memory(agent_id, create_if_not_exists=True)
        from vanna.core.user import User, RequestContext
        context = RequestContext(user=User(id="admin"))
        
        imported = 0
        for fs in fewshots:
            try:
                args = json.loads(fs["args_json"])
                await memory.save_tool_usage(
                    question=fs["question"],
                    tool_name=fs["tool_name"],
                    args=args,
                    context=context,
                    success=True,
                    metadata={"db_id": fs["db_id"], "auto_generated": True},
                )
                imported += 1
            except Exception as e:
                logger.error(f"Failed to save fewshot: {e}")
        
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
    - 所有表的描述
    - 欄位資訊（類型、主鍵、外鍵）
    - 工作流程說明
    - 回應風格指引
    
    Args:
        agent_id: Agent ID
        
    Returns:
        - system_prompt: 生成的 system prompt
        - total_tables: 資料庫總表數
    """
    # 檢查 agent 是否存在並取得配置
    try:
        response = table.get_item(Key={"agent_id": agent_id})
        if "Item" not in response:
            raise HTTPException(404, f"Agent '{agent_id}' not found")
        agent_config = response["Item"]["config"]
    except Exception as e:
        logger.error(f"DynamoDB error: {e}")
        raise HTTPException(500, f"Failed to check agent: {e}")
    
    # 連接資料庫
    pg_user = agent_config.get("postgres_user")
    pg_password = agent_config.get("postgres_password")
    pg_host = agent_config.get("postgres_host")
    pg_port = agent_config.get("postgres_port") or "5432"
    pg_db = agent_config.get("postgres_db")
    
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
        
        # 更新到 DynamoDB
        table.update_item(
            Key={"agent_id": agent_id},
            UpdateExpression="SET config.system_prompt = :prompt, system_prompt_generated_at = :time",
            ExpressionAttributeValues={
                ":prompt": system_prompt,
                ":time": datetime.now().isoformat(),
            },
        )
        
        return {
            "message": "System prompt 生成成功",
            "system_prompt": system_prompt,
            "total_tables": total_tables,
        }
        
    except Exception as e:
        if conn:
            conn.close()
        logger.error(f"Generate system prompt failed: {e}")
        raise HTTPException(500, f"生成失敗: {e}")


# === System Prompt Generation (保持不變) ===
def generate_system_prompt_from_db(conn, db_name: str) -> str:
    """從 PostgreSQL 資料庫自動生成 system prompt"""
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
        
        # 生成表描述
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
    
    # 加入回應風格
    prompt_parts.append("\n## 回應風格\n")
    prompt_parts.append("- 簡潔專業，使用繁體中文")
    
    return "\n".join(prompt_parts)


@app.get("/api/efs/list", tags=["EFS Management"], summary="列出 EFS 上的檔案")
async def list_efs_contents(path: str = ""):
    """
    列出 EFS 上的檔案和目錄
    
    Args:
        path: 相對於 agent_data 的路徑 (例如: "" 或 "chroma_db_pos_test")
    
    Returns:
        - path: 當前路徑
        - items: 檔案和目錄列表
    """
    try:
        target_path = os.path.join(AGENT_DATA_DIR, path) if path else AGENT_DATA_DIR
        
        if not os.path.exists(target_path):
            raise HTTPException(404, f"Path not found: {target_path}")
        
        items = []
        for item in os.listdir(target_path):
            item_path = os.path.join(target_path, item)
            stat = os.stat(item_path)
            
            items.append({
                "name": item,
                "type": "directory" if os.path.isdir(item_path) else "file",
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })
        
        # 排序：目錄在前，然後按名稱
        items.sort(key=lambda x: (x["type"] != "directory", x["name"]))
        
        return {
            "path": target_path,
            "relative_path": path,
            "total_items": len(items),
            "items": items,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list EFS contents: {e}")
        raise HTTPException(500, f"Failed to list EFS contents: {e}")


@app.get("/api/efs/disk-usage", tags=["EFS Management"], summary="查看 EFS 磁碟使用量")
async def get_efs_disk_usage():
    """
    查看 EFS 的磁碟使用量
    
    Returns:
        - total_size: 總大小 (bytes)
        - agents: 各 Agent 的磁碟使用量
    """
    try:
        import subprocess
        
        # 取得總大小
        result = subprocess.run(
            ["du", "-sb", AGENT_DATA_DIR],
            capture_output=True,
            text=True,
        )
        total_size = int(result.stdout.split()[0]) if result.returncode == 0 else 0
        
        # 取得各 Agent 的大小
        agents = []
        if os.path.exists(AGENT_DATA_DIR):
            for item in os.listdir(AGENT_DATA_DIR):
                if item.startswith("chroma_db_"):
                    agent_id = item.replace("chroma_db_", "")
                    item_path = os.path.join(AGENT_DATA_DIR, item)
                    
                    result = subprocess.run(
                        ["du", "-sb", item_path],
                        capture_output=True,
                        text=True,
                    )
                    size = int(result.stdout.split()[0]) if result.returncode == 0 else 0
                    
                    agents.append({
                        "agent_id": agent_id,
                        "size_bytes": size,
                        "size_mb": round(size / 1024 / 1024, 2),
                    })
        
        # 排序：大小由大到小
        agents.sort(key=lambda x: x["size_bytes"], reverse=True)
        
        return {
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / 1024 / 1024, 2),
            "total_size_gb": round(total_size / 1024 / 1024 / 1024, 2),
            "agents": agents,
        }
    except Exception as e:
        logger.error(f"Failed to get disk usage: {e}")
        raise HTTPException(500, f"Failed to get disk usage: {e}")


@app.post("/api/efs/test-delete", tags=["EFS Management"], summary="測試 EFS 刪除權限")
async def test_efs_delete(agent_id: str):
    """
    測試是否能刪除指定 Agent 的 EFS 目錄
    
    這是一個診斷 API，用於測試權限和路徑
    """
    try:
        import subprocess
        persist_dir = f"{AGENT_DATA_DIR}/chroma_db_{agent_id}"
        
        # 檢查目錄是否存在
        if not os.path.exists(persist_dir):
            return {
                "success": False,
                "error": "Directory not found",
                "path": persist_dir,
                "agent_data_dir": AGENT_DATA_DIR,
                "exists": False,
            }
        
        # 檢查權限
        can_read = os.access(persist_dir, os.R_OK)
        can_write = os.access(persist_dir, os.W_OK)
        can_execute = os.access(persist_dir, os.X_OK)
        
        # 取得目錄資訊
        stat_info = os.stat(persist_dir)
        
        # 嘗試刪除（使用 rm -rf）
        try:
            result = subprocess.run(
                ["rm", "-rf", persist_dir],
                capture_output=True,
                text=True,
                timeout=60
            )
            deleted = (result.returncode == 0)
            error = result.stderr if result.returncode != 0 else None
        except Exception as e:
            deleted = False
            error = str(e)
        
        return {
            "success": deleted,
            "path": persist_dir,
            "agent_data_dir": AGENT_DATA_DIR,
            "exists_before": True,
            "exists_after": os.path.exists(persist_dir),
            "permissions": {
                "read": can_read,
                "write": can_write,
                "execute": can_execute,
            },
            "stat": {
                "uid": stat_info.st_uid,
                "gid": stat_info.st_gid,
                "mode": oct(stat_info.st_mode),
            },
            "error": error,
        }
    except Exception as e:
        logger.error(f"Test delete failed: {e}")
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


@app.get("/api/efs/check-locks", tags=["EFS Management"], summary="檢查哪些進程在使用 EFS 目錄")
async def check_efs_locks(agent_id: str):
    """
    檢查哪些進程正在使用指定 Agent 的 EFS 目錄
    
    使用 /proc 檔案系統來診斷檔案鎖定問題
    """
    try:
        import subprocess
        import glob
        persist_dir = f"{AGENT_DATA_DIR}/chroma_db_{agent_id}"
        
        if not os.path.exists(persist_dir):
            return {
                "error": "Directory not found",
                "path": persist_dir,
            }
        
        # 1. 列出所有 .nfs* 檔案
        nfs_files = []
        for root, dirs, files in os.walk(persist_dir):
            for file in files:
                if file.startswith('.nfs'):
                    nfs_files.append(os.path.join(root, file))
        
        # 2. 檢查當前進程（Manager）打開的檔案
        current_pid = os.getpid()
        manager_open_files = []
        
        try:
            fd_dir = f"/proc/{current_pid}/fd"
            if os.path.exists(fd_dir):
                for fd in os.listdir(fd_dir):
                    try:
                        link = os.readlink(os.path.join(fd_dir, fd))
                        if persist_dir in link:
                            manager_open_files.append({
                                "fd": fd,
                                "path": link,
                            })
                    except:
                        pass
        except Exception as e:
            logger.warning(f"Cannot read /proc/{current_pid}/fd: {e}")
        
        # 3. 檢查所有進程
        all_processes_using = []
        try:
            for pid_dir in glob.glob("/proc/[0-9]*"):
                try:
                    pid = os.path.basename(pid_dir)
                    fd_dir = f"{pid_dir}/fd"
                    
                    if not os.path.exists(fd_dir):
                        continue
                    
                    for fd in os.listdir(fd_dir):
                        try:
                            link = os.readlink(os.path.join(fd_dir, fd))
                            if persist_dir in link:
                                # 讀取進程資訊
                                try:
                                    with open(f"{pid_dir}/cmdline", 'r') as f:
                                        cmdline = f.read().replace('\x00', ' ').strip()
                                except:
                                    cmdline = "unknown"
                                
                                all_processes_using.append({
                                    "pid": pid,
                                    "fd": fd,
                                    "path": link,
                                    "cmdline": cmdline,
                                })
                        except:
                            pass
                except:
                    pass
        except Exception as e:
            logger.warning(f"Cannot scan /proc: {e}")
        
        # 4. 使用 Python 的 psutil（如果有的話）
        psutil_info = None
        try:
            import psutil
            current_process = psutil.Process(current_pid)
            open_files = current_process.open_files()
            psutil_info = {
                "open_files_count": len(open_files),
                "open_files": [
                    {"path": f.path, "fd": f.fd}
                    for f in open_files
                    if persist_dir in f.path
                ]
            }
        except ImportError:
            psutil_info = {"error": "psutil not installed"}
        except Exception as e:
            psutil_info = {"error": str(e)}
        
        return {
            "path": persist_dir,
            "current_pid": current_pid,
            "nfs_silly_rename_files": {
                "count": len(nfs_files),
                "files": nfs_files[:20],  # 只顯示前 20 個
            },
            "manager_open_files": {
                "count": len(manager_open_files),
                "files": manager_open_files,
            },
            "all_processes_using": {
                "count": len(all_processes_using),
                "processes": all_processes_using,
            },
            "psutil_info": psutil_info,
        }
    except Exception as e:
        logger.error(f"Check locks failed: {e}")
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


@app.get("/api/debug/check-agents", tags=["Debug"], summary="檢查 DynamoDB 和 EFS 的 Agent 資料")
async def check_agents_data():
    """
    檢查 DynamoDB 和 EFS 中的 Agent 資料是否一致
    
    用於診斷 agent_id 命名問題
    """
    try:
        # 1. 從 DynamoDB 取得所有 agents
        response = table.scan(
            FilterExpression="attribute_not_exists(#status) OR #status <> :deleted",
            ExpressionAttributeNames={"#status": "status"},
            ExpressionAttributeValues={":deleted": "deleted"},
        )
        
        dynamodb_agents = []
        for item in response["Items"]:
            dynamodb_agents.append({
                "agent_id": item["agent_id"],
                "status": item.get("status", "unknown"),
                "created_at": item.get("created_at"),
            })
        
        # 2. 從 EFS 取得所有 chroma_db 目錄
        efs_directories = []
        if os.path.exists(AGENT_DATA_DIR):
            for item in os.listdir(AGENT_DATA_DIR):
                if item.startswith("chroma_db_"):
                    agent_id_from_dir = item.replace("chroma_db_", "")
                    item_path = os.path.join(AGENT_DATA_DIR, item)
                    stat_info = os.stat(item_path)
                    
                    efs_directories.append({
                        "directory_name": item,
                        "agent_id_extracted": agent_id_from_dir,
                        "path": item_path,
                        "modified": datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
                    })
        
        # 3. 比對
        dynamodb_ids = set(a["agent_id"] for a in dynamodb_agents)
        efs_ids = set(d["agent_id_extracted"] for d in efs_directories)
        
        # 找出不一致的
        only_in_dynamodb = dynamodb_ids - efs_ids
        only_in_efs = efs_ids - dynamodb_ids
        in_both = dynamodb_ids & efs_ids
        
        return {
            "dynamodb_agents": dynamodb_agents,
            "efs_directories": efs_directories,
            "summary": {
                "total_in_dynamodb": len(dynamodb_agents),
                "total_in_efs": len(efs_directories),
                "in_both": list(in_both),
                "only_in_dynamodb": list(only_in_dynamodb),
                "only_in_efs": list(only_in_efs),
            },
            "issues": {
                "orphaned_efs_directories": list(only_in_efs),  # EFS 有但 DynamoDB 沒有
                "missing_efs_directories": list(only_in_dynamodb),  # DynamoDB 有但 EFS 沒有
            }
        }
    except Exception as e:
        logger.error(f"Check agents data failed: {e}")
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


@app.post("/api/debug/cleanup-orphaned-efs", tags=["Debug"], summary="清理孤兒 EFS 目錄")
async def cleanup_orphaned_efs(dry_run: bool = True):
    """
    清理 EFS 上的孤兒目錄（DynamoDB 中不存在的目錄）
    
    Args:
        dry_run: 是否只是模擬（不真正刪除），預設 True
    
    Returns:
        清理結果
    """
    try:
        # 1. 檢查哪些是孤兒目錄
        check_result = await check_agents_data()
        orphaned = check_result["issues"]["orphaned_efs_directories"]
        
        if not orphaned:
            return {
                "message": "No orphaned directories found",
                "orphaned_count": 0,
                "deleted": [],
            }
        
        # 2. 刪除孤兒目錄
        deleted = []
        errors = []
        
        for agent_id in orphaned:
            persist_dir = f"{AGENT_DATA_DIR}/chroma_db_{agent_id}"
            
            try:
                if dry_run:
                    # 只是模擬，不真正刪除
                    logger.info(f"[DRY RUN] Would delete: {persist_dir}")
                    deleted.append({
                        "agent_id": agent_id,
                        "path": persist_dir,
                        "dry_run": True,
                    })
                else:
                    # 真正刪除 - 使用 rm -rf（在 NFS/EFS 上更可靠）
                    import subprocess
                    if os.path.exists(persist_dir):
                        result = subprocess.run(
                            ["rm", "-rf", persist_dir],
                            capture_output=True,
                            text=True,
                            timeout=60
                        )
                        if result.returncode == 0:
                            logger.info(f"Deleted orphaned directory: {persist_dir}")
                            deleted.append({
                                "agent_id": agent_id,
                                "path": persist_dir,
                                "deleted": True,
                            })
                        else:
                            raise Exception(f"rm -rf failed: {result.stderr}")
                    else:
                        logger.warning(f"Directory not found: {persist_dir}")
            except Exception as e:
                logger.error(f"Failed to delete {persist_dir}: {e}")
                errors.append({
                    "agent_id": agent_id,
                    "path": persist_dir,
                    "error": str(e),
                })
        
        return {
            "message": f"{'[DRY RUN] Would delete' if dry_run else 'Deleted'} {len(deleted)} orphaned directories",
            "dry_run": dry_run,
            "orphaned_count": len(orphaned),
            "deleted": deleted,
            "errors": errors,
        }
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


@app.get("/admin/efs", response_class=HTMLResponse, include_in_schema=False)
async def efs_browser_page(path: str = ""):
    """EFS 檔案瀏覽器頁面"""
    try:
        # 取得檔案列表
        target_path = os.path.join(AGENT_DATA_DIR, path) if path else AGENT_DATA_DIR
        
        if not os.path.exists(target_path):
            return f"<html><body><h1>Error</h1><p>Path not found: {target_path}</p></body></html>"
        
        items = []
        for item in os.listdir(target_path):
            item_path = os.path.join(target_path, item)
            stat = os.stat(item_path)
            
            items.append({
                "name": item,
                "type": "directory" if os.path.isdir(item_path) else "file",
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            })
        
        # 排序：目錄在前，然後按名稱
        items.sort(key=lambda x: (x["type"] != "directory", x["name"]))
        
        # 只在根目錄顯示磁碟使用量
        disk_usage = None
        if not path:
            disk_usage_data = await get_efs_disk_usage()
            disk_usage = disk_usage_data
        
        return get_efs_browser_html(path, items, disk_usage)
    except Exception as e:
        logger.error(f"Failed to load EFS browser page: {e}")
        return f"<html><body><h1>Error</h1><p>{e}</p></body></html>"


@app.on_event("startup")
async def startup():
    """Manager 啟動時，只同步狀態，不重啟 agents"""
    logger.info("=" * 50)
    logger.info("Vanna Multi-Agent Manager (ECS) Starting...")
    logger.info("=" * 50)
    logger.info(f"Base Domain: {BASE_DOMAIN}")
    logger.info(f"Protocol: {PROTOCOL}")
    logger.info(f"DynamoDB Table: {DYNAMODB_TABLE_NAME}")
    logger.info(f"ECS Cluster: {ECS_CLUSTER_NAME}")
    logger.info("=" * 50)
    
    # 同步 DynamoDB 和 ECS 狀態
    try:
        response = table.scan(
            FilterExpression="#status = :running",
            ExpressionAttributeNames={"#status": "status"},
            ExpressionAttributeValues={":running": "running"},
        )
        
        for item in response["Items"]:
            agent_id = item["agent_id"]
            ecs_status = await get_agent_status(agent_id)
            
            if not ecs_status["running"]:
                logger.warning(f"Agent '{agent_id}' is marked as running but ECS service is not running")
                # 更新狀態
                table.update_item(
                    Key={"agent_id": agent_id},
                    UpdateExpression="SET #status = :status",
                    ExpressionAttributeNames={"#status": "status"},
                    ExpressionAttributeValues={":status": "stopped"},
                )
        
        logger.info("Manager startup complete - agents status synchronized")
    except Exception as e:
        logger.error(f"Failed to sync agent status: {e}")


# === Agent API Proxy ===
@app.api_route("/agent/{agent_id}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"], tags=["Agent Proxy"])
async def proxy_to_agent(agent_id: str, path: str, request: Request):
    """
    代理請求到指定的 Agent
    
    統一入口：所有對 Agent 的請求都通過 Manager 轉發
    
    範例：
    - GET  /agent/pos_test_dtfull_salesid/api/vanna/v2/chat_sse
    - POST /agent/pos_test_dtfull_salesid/api/vanna/v2/chat
    """
    # 1. 取得 Agent 的 IP
    ecs_status = await get_agent_status(agent_id)
    
    if not ecs_status["running"]:
        raise HTTPException(503, f"Agent '{agent_id}' is not running")
    
    public_ip = ecs_status.get("public_ip")
    private_ip = ecs_status.get("private_ip")
    
    # 優先使用 Private IP（如果 Manager 和 Agent 在同一個 VPC）
    # 否則使用 Public IP
    target_ip = private_ip if private_ip else public_ip
    
    if not target_ip:
        raise HTTPException(503, f"Agent '{agent_id}' has no accessible IP address")
    
    # 2. 建立目標 URL
    target_url = f"http://{target_ip}:8101/{path}"
    
    # 3. 取得請求的 query parameters
    query_string = str(request.url.query)
    if query_string:
        target_url += f"?{query_string}"
    
    logger.info(f"Proxying {request.method} request to: {target_url}")
    
    # 4. 轉發請求
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            # 讀取請求 body
            body = await request.body()
            
            # 複製 headers（排除 host）
            headers = dict(request.headers)
            headers.pop("host", None)
            
            # 發送請求
            response = await client.request(
                method=request.method,
                url=target_url,
                content=body,
                headers=headers,
            )
            
            # 5. 回傳回應
            # 檢查是否為 SSE (Server-Sent Events)
            content_type = response.headers.get("content-type", "")
            if "text/event-stream" in content_type:
                # SSE 需要串流回傳
                async def stream_response():
                    async for chunk in response.aiter_bytes():
                        yield chunk
                
                return StreamingResponse(
                    stream_response(),
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=content_type,
                )
            else:
                # 一般回應
                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=content_type,
                )
        
        except httpx.TimeoutException:
            raise HTTPException(504, f"Request to agent '{agent_id}' timed out")
        except httpx.RequestError as e:
            logger.error(f"Failed to proxy request to agent '{agent_id}': {e}")
            raise HTTPException(502, f"Failed to connect to agent '{agent_id}': {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 50)
    print("Vanna Multi-Agent Manager (ECS)")
    print("=" * 50)
    print(f"管理 API: {PROTOCOL}://{BASE_DOMAIN}")
    print(f"Swagger UI: {PROTOCOL}://{BASE_DOMAIN}/docs")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8100)
