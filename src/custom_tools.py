"""
自訂工具:列出所有訓練記憶體
"""
from typing import Any, Dict
from pydantic import BaseModel, Field
from vanna.core.tool import Tool, ToolContext, ToolResult


class ListAllMemoriesParams(BaseModel):
    """列出所有記憶體的參數"""
    limit: int = Field(
        default=50,
        description="要顯示的記憶體數量上限"
    )


class ListAllMemoriesTool(Tool):
    """列出所有訓練記憶體的工具"""
    
    @property
    def name(self) -> str:
        return "list_all_memories"
    
    @property
    def description(self) -> str:
        return "列出所有訓練記憶體,包括問題-SQL配對。可以指定要顯示的數量。"
    
    def get_args_schema(self) -> type[BaseModel]:
        return ListAllMemoriesParams
    
    async def execute(
        self,
        context: ToolContext,
        args: ListAllMemoriesParams
    ) -> ToolResult:
        """執行列出記憶體的操作"""
        try:
            # 從 context 取得 agent_memory
            agent_memory = context.agent_memory
            
            # 取得 tool memories
            memories = await agent_memory.get_recent_memories(
                context=context,
                limit=args.limit
            )
            
            # 格式化輸出
            result_text = f"找到 {len(memories)} 筆訓練記憶體:\n\n"
            
            for i, mem in enumerate(memories, 1):
                result_text += f"{i}. 問題: {mem.question}\n"
                result_text += f"   工具: {mem.tool_name}\n"
                
                if mem.args and 'sql' in mem.args:
                    sql = mem.args['sql']
                    # 顯示前 200 字元的 SQL
                    if len(sql) > 200:
                        result_text += f"   SQL: {sql[:200]}...\n"
                    else:
                        result_text += f"   SQL: {sql}\n"
                
                result_text += f"   時間: {mem.timestamp}\n"
                result_text += "\n"
            
            return ToolResult(
                success=True,
                result_for_llm=result_text
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"列出記憶體時發生錯誤: {str(e)}"
            )
