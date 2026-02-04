import os
import logging
import boto3
from dotenv import load_dotenv
from vanna import Agent
from vanna.core.registry import ToolRegistry
from vanna.tools import RunSqlTool
from custom_tools import ListAllMemoriesTool
from vanna.servers.fastapi import VannaFastAPIServer
from vanna.integrations.azureopenai import AzureOpenAILlmService
from vanna.integrations.chromadb.agent_memory import ChromaAgentMemory
from vanna.core.user import UserResolver, User, RequestContext
from vanna.core.system_prompt import SystemPromptBuilder
from db import PostgresRunnerPooled

# Configure Logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d [%(process)d:%(thread)d] %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
logging.getLogger("vanna").setLevel(logging.DEBUG)

# Load environment variables
load_dotenv()


# === Dynamic System Prompt Builder ===
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


# === Simple User Resolver ===
class SimpleUserResolver(UserResolver):
    async def resolve_user(self, request_context: RequestContext) -> User:
        user_email = request_context.get_cookie("vanna_email") or "guest@example.com"
        group = "admin" if user_email in ["admin@example.com"] else "user"
        return User(id=user_email, email=user_email, group_memberships=[group])


# === Main Agent Setup ===
def main():
    # 1. 取得 AGENT_ID
    agent_id = os.getenv("AGENT_ID")
    
    # 如果沒有 AGENT_ID 環境變數，嘗試從 ECS metadata 取得 service name
    if not agent_id:
        try:
            import requests
            metadata_uri = os.getenv("ECS_CONTAINER_METADATA_URI_V4")
            if metadata_uri:
                # 取得 task metadata
                task_metadata = requests.get(f"{metadata_uri}/task").json()
                task_arn = task_metadata.get("TaskARN", "")
                cluster_arn = task_metadata.get("Cluster", "")
                
                logger.info(f"Task ARN: {task_arn}")
                logger.info(f"Cluster: {cluster_arn}")
                
                # 從 task ARN 提取 cluster 和 task ID
                if task_arn and cluster_arn:
                    import boto3
                    ecs = boto3.client("ecs", region_name=os.getenv("AWS_DEFAULT_REGION", "ap-northeast-1"))
                    
                    # 查詢 task 所屬的 service
                    response = ecs.describe_tasks(
                        cluster=cluster_arn,
                        tasks=[task_arn]
                    )
                    
                    if response.get("tasks"):
                        task = response["tasks"][0]
                        # 從 task 的 group 欄位取得 service name
                        # group 格式: "service:vanna-agent-{agent_id}"
                        group = task.get("group", "")
                        if group.startswith("service:vanna-agent-"):
                            agent_id = group.replace("service:vanna-agent-", "")
                            logger.info(f"✅ Extracted agent_id from task group: {agent_id}")
        except Exception as e:
            logger.warning(f"Failed to extract agent_id from ECS metadata: {e}")
            import traceback
            logger.warning(traceback.format_exc())
    
    if not agent_id:
        raise ValueError("AGENT_ID environment variable is required or must be extractable from ECS service name")
    
    logger.info("=" * 50)
    logger.info(f"Starting Vanna Agent: {agent_id}")
    logger.info("=" * 50)
    
    # 2. 從 DynamoDB 載入配置
    aws_region = os.getenv("AWS_DEFAULT_REGION", "ap-northeast-1")
    dynamodb_table_name = os.getenv("DYNAMODB_TABLE_NAME", "vanna-agents")
    
    dynamodb = boto3.resource("dynamodb", region_name=aws_region)
    table = dynamodb.Table(dynamodb_table_name)
    
    try:
        response = table.get_item(Key={"agent_id": agent_id})
        if "Item" not in response:
            raise ValueError(f"Agent '{agent_id}' not found in DynamoDB table '{dynamodb_table_name}'")
        
        config = response["Item"]["config"]
        logger.info(f"✅ Loaded config for agent '{agent_id}' from DynamoDB")
    except Exception as e:
        logger.error(f"❌ Failed to load config from DynamoDB: {e}")
        raise
    
    # 3. Azure OpenAI 配置
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    azure_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    
    if not all([azure_api_key, azure_endpoint, azure_api_version, azure_deployment_name]):
        raise ValueError("Missing Azure OpenAI configuration in environment variables")
    
    # 4. 建立 LLM
    llm = AzureOpenAILlmService(
        model=azure_deployment_name,
        api_key=azure_api_key,
        azure_endpoint=azure_endpoint,
        api_version=azure_api_version,
    )
    logger.info("✅ Created Azure OpenAI LLM service")
    
    # 5. 建立 Memory (EFS 路徑)
    persist_dir = f"/app/agent_data/chroma_db_{agent_id}"
    os.makedirs(persist_dir, exist_ok=True)
    
    memory = ChromaAgentMemory(
        persist_directory=persist_dir,
        collection_name=f"vanna_{agent_id}",
    )
    logger.info(f"✅ Created ChromaDB memory at {persist_dir}")
    
    # 6. 建立 DB 連線
    pg_user = config.get("postgres_user")
    pg_password = config.get("postgres_password")
    pg_host = config.get("postgres_host")
    pg_port = config.get("postgres_port", "5432")
    pg_db = config.get("postgres_db")
    
    connection_string = f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_db}"
    
    # 連線池配置
    use_connection_pool = os.getenv("USE_CONNECTION_POOL", "true").lower() == "true"
    db_pool_min_conn = int(os.getenv("DB_POOL_MIN_CONN", "2"))
    db_pool_max_conn = int(os.getenv("DB_POOL_MAX_CONN", "10"))
    
    if use_connection_pool:
        logger.info(f"Using connection pool (min={db_pool_min_conn}, max={db_pool_max_conn})")
        db_tool = RunSqlTool(
            sql_runner=PostgresRunnerPooled(
                connection_string=connection_string,
                minconn=db_pool_min_conn,
                maxconn=db_pool_max_conn,
            )
        )
    else:
        logger.info("Using new connection per request")
        from vanna.integrations.postgres import PostgresRunner
        db_tool = RunSqlTool(
            sql_runner=PostgresRunner(connection_string=connection_string)
        )
    
    logger.info(f"✅ Created PostgreSQL connection to {pg_host}:{pg_port}/{pg_db}")
    
    # 7. 建立 Tools
    tools = ToolRegistry()
    tools.register_local_tool(db_tool, access_groups=["admin", "user"])
    tools.register_local_tool(ListAllMemoriesTool(), access_groups=["admin", "user"])
    logger.info("✅ Registered tools")
    
    # 8. 建立 Agent
    system_prompt = config.get("system_prompt", "你是一個數據分析助手")
    
    agent = Agent(
        llm_service=llm,
        tool_registry=tools,
        user_resolver=SimpleUserResolver(),
        agent_memory=memory,
        system_prompt_builder=DynamicSystemPromptBuilder(system_prompt),
    )
    logger.info("✅ Created Vanna Agent")
    
    # 9. 建立 FastAPI Server
    server = VannaFastAPIServer(agent)
    app = server.create_app()
    
    # 10. 新增健康檢查 endpoint
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "agent_id": agent_id,
            "postgres_db": pg_db,
        }
    
    logger.info("✅ Added health check endpoint")
    
    # 11. 啟動 Server
    logger.info("=" * 50)
    logger.info(f"Agent '{agent_id}' is ready!")
    logger.info(f"Listening on http://0.0.0.0:8101")
    logger.info("=" * 50)
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8101)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"❌ Failed to start agent: {e}")
        raise
