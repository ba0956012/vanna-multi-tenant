"""
PostgreSQL Runner with Connection Pooling

提供連線池功能的 PostgreSQL 執行器，適用於高併發場景。
相比每次查詢建立新連線，連線池可以大幅提升效能。

環境變數:
    USE_CONNECTION_POOL: true/false - 是否使用連線池
    DB_POOL_MIN_CONN: 最小連線數 (預設 2)
    DB_POOL_MAX_CONN: 最大連線數 (預設 10)
"""

import logging
import time
import threading
from typing import Optional
import pandas as pd
from vanna.capabilities.sql_runner import SqlRunner, RunSqlToolArgs
from vanna.core.tool import ToolContext

logger = logging.getLogger(__name__)


class PostgresRunnerPooled(SqlRunner):
    """PostgreSQL implementation with connection pooling for concurrent requests."""

    def __init__(
        self,
        connection_string: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = 5432,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        minconn: int = 2,
        maxconn: int = 10,
        **kwargs,
    ):
        """
        Initialize with PostgreSQL connection pool.
        
        Args:
            connection_string: PostgreSQL connection string (優先使用)
            host: Database host
            port: Database port (default: 5432)
            database: Database name
            user: Database user
            password: Database password
            minconn: Minimum connections in pool (default: 2)
            maxconn: Maximum connections in pool (default: 10)
        """
        try:
            import psycopg2
            import psycopg2.extras
            from psycopg2 import pool as pg_pool

            self.psycopg2 = psycopg2
            self.extras = psycopg2.extras
        except ImportError as e:
            raise ImportError(
                "psycopg2 package is required. Install with: pip install psycopg2-binary"
            ) from e

        # Create connection pool
        if connection_string:
            self.connection_pool = pg_pool.ThreadedConnectionPool(
                minconn, maxconn, connection_string, **kwargs
            )
        elif host and database and user:
            self.connection_pool = pg_pool.ThreadedConnectionPool(
                minconn,
                maxconn,
                host=host,
                port=port,
                database=database,
                user=user,
                password=password,
                **kwargs,
            )
        else:
            raise ValueError(
                "Either provide connection_string OR (host, database, and user) parameters"
            )
        
        logger.info(f"Connection pool created (min={minconn}, max={maxconn})")

    async def run_sql(self, args: RunSqlToolArgs, context: ToolContext) -> pd.DataFrame:
        """
        Execute SQL query using connection from pool.
        
        Args:
            args: SQL query arguments containing the SQL string
            context: Tool execution context
            
        Returns:
            DataFrame with query results or affected row count
        """
        thread_id = threading.get_ident()
        start_time = time.time()
        
        logger.debug(f"[Thread-{thread_id}] Getting connection from pool...")
        
        # Get connection from pool
        conn = self.connection_pool.getconn()
        conn_time = time.time() - start_time
        
        logger.debug(f"[Thread-{thread_id}] Connection acquired ({conn_time*1000:.2f}ms)")

        try:
            cursor = conn.cursor(cursor_factory=self.extras.RealDictCursor)

            try:
                sql_preview = args.sql[:100].replace('\n', ' ')
                logger.info(f"[Thread-{thread_id}] Executing SQL: {sql_preview}...")
                
                query_start = time.time()
                cursor.execute(args.sql)
                query_time = time.time() - query_start
                
                query_type = args.sql.strip().upper().split()[0]

                if query_type == "SELECT":
                    rows = cursor.fetchall()
                    logger.info(
                        f"[Thread-{thread_id}] SQL completed "
                        f"({query_time*1000:.2f}ms, rows: {len(rows) if rows else 0})"
                    )
                    
                    if not rows:
                        return pd.DataFrame()
                    results_data = [dict(row) for row in rows]
                    return pd.DataFrame(results_data)
                else:
                    conn.commit()
                    rows_affected = cursor.rowcount
                    logger.info(
                        f"[Thread-{thread_id}] SQL completed "
                        f"({query_time*1000:.2f}ms, affected: {rows_affected})"
                    )
                    return pd.DataFrame({"rows_affected": [rows_affected]})
            finally:
                cursor.close()
        finally:
            # Return connection to pool (don't close it!)
            self.connection_pool.putconn(conn)
            total_time = time.time() - start_time
            logger.debug(f"[Thread-{thread_id}] Connection returned ({total_time*1000:.2f}ms total)")

    def close_pool(self):
        """Close all connections in the pool."""
        if hasattr(self, "connection_pool"):
            self.connection_pool.closeall()
            logger.info("Connection pool closed")
