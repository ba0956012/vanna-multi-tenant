#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Few-shot 自動生成（PostgreSQL 版本）- MASK + LIMIT FIELDS
=========================================================

特性總覽：
1. 使用 PostgreSQL information_schema 取得 FK 關係
2. 只使用 FK 建 JOIN graph（無向）→ BFS → JOIN Route
3. SELECT 一律：t0.*, t1.*, t2.* ...
4. WHERE 規則：
   - 永遠只使用 root table（t0）欄位
   - 最多自動挑選 2–3 個欄位
   - 排除 id / created_at / updated_at / timestamp 類型
   - TEXT/VARCHAR → LIKE '%[column]%'
   - 非 TEXT → = [column]
5. WHERE 一律使用 MASK（placeholder），不使用實際值
6. SQL 會先用 PostgreSQL execute 驗證，錯誤即跳過
7. 每張表產生 1 筆 few-shot
8. 輸出可直接匯入 ChromaDB 記憶
"""

import json
import os
import argparse
import random
import psycopg2
from pathlib import Path


# =====================================================
#  取得 PostgreSQL Schema
# =====================================================

def analyze_database(conn):
    cur = conn.cursor()

    # 取得所有 public schema 的表
    cur.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
        ORDER BY table_name;
    """)
    tables = [r[0] for r in cur.fetchall()]

    schema = {}
    for t in tables:
        # 取得欄位資訊
        cur.execute("""
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns 
            WHERE table_schema = 'public' AND table_name = %s
            ORDER BY ordinal_position;
        """, (t,))
        cols = cur.fetchall()

        # 取得 FK 資訊
        cur.execute("""
            SELECT
                kcu.column_name as from_column,
                ccu.table_name as to_table,
                ccu.column_name as to_column
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
                AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage ccu
                ON ccu.constraint_name = tc.constraint_name
                AND ccu.table_schema = tc.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY'
                AND tc.table_schema = 'public'
                AND tc.table_name = %s;
        """, (t,))
        fks = cur.fetchall()

        schema[t] = {"columns": cols, "fks": fks}

    return tables, schema


# =====================================================
#  建 FK Graph（無向）
# =====================================================

def build_fk_graph(tables, schema):
    graph = {t: [] for t in tables}

    for t in tables:
        for fk in schema[t]["fks"]:
            from_col, ref_table, to_col = fk

            if ref_table in graph:
                graph[t].append((ref_table, from_col, to_col))
                graph[ref_table].append((t, to_col, from_col))

    return graph


# =====================================================
#  BFS JOIN 順序
# =====================================================

def bfs_join_tables(root, graph):
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


# =====================================================
#  取得 sample row（僅用來判斷哪些欄位非 NULL）
# =====================================================

def get_sample_row(conn, table):
    cur = conn.cursor()
    try:
        cur.execute(f'SELECT * FROM "{table}" LIMIT 1;')
        row = cur.fetchone()
        if row is None:
            return None
        cols = [c[0] for c in cur.description]
        return dict(zip(cols, row))
    except Exception as e:
        print(f"  ⚠️ 取得 sample row 失敗: {e}")
        return None


# =====================================================
#  WHERE 欄位選擇與 MASK 規則
# =====================================================

EXCLUDE_COLUMN_NAMES = {"id", "created_at", "updated_at"}
EXCLUDE_TYPES = {"date", "timestamp with time zone", "timestamp without time zone"}


def select_where_columns(schema_cols, sample_row, max_fields=3):
    candidates = []
    for col in schema_cols:
        name = col[0]
        ctype = (col[1] or "").lower()

        if name.lower() in EXCLUDE_COLUMN_NAMES:
            continue
        if ctype in EXCLUDE_TYPES:
            continue
        if sample_row.get(name) is None:
            continue

        candidates.append(col)

    random.shuffle(candidates)
    return candidates[:max_fields]


def build_where_clause(schema_cols, sample_row, max_fields=3):
    selected = select_where_columns(schema_cols, sample_row, max_fields)
    if not selected:
        return ""

    parts = []
    for col in selected:
        name = col[0]
        ctype = (col[1] or "").lower()
        val = sample_row.get(name)

        if "char" in ctype or "text" in ctype:
            # 使用 placeholder，不加引號避免 SQL 語法錯誤
            parts.append(f't0."{name}" LIKE \'%[{name}]%\'')
        else:
            parts.append(f't0."{name}" = [{name}]')

    return "WHERE " + " AND ".join(parts)


# =====================================================
#  JOIN SQL 生成
# =====================================================

def generate_join_sql(root, join_order, schema, graph, sample_row, max_where_fields=3):
    aliases = {t: f"t{i}" for i, t in enumerate(join_order)}

    sql = []
    sql.append("SELECT " + ", ".join(f'{aliases[t]}.*' for t in join_order))
    sql.append(f'FROM "{root}" {aliases[root]}')

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

        if not parent:
            continue

        p_alias = aliases[parent]
        t_alias = aliases[t]
        from_col, to_col = parent_fk

        sql.append(
            f'LEFT JOIN "{t}" {t_alias} ON {p_alias}."{from_col}" = {t_alias}."{to_col}"'
        )

    where_sql = build_where_clause(schema[root]["columns"], sample_row, max_where_fields)
    if where_sql:
        sql.append(where_sql)

    sql.append("LIMIT 200;")
    return "\n".join(sql)


# =====================================================
#  SQL 驗證
# =====================================================

def validate_sql(conn, sql):
    try:
        cur = conn.cursor()
        # 用 EXPLAIN 驗證 SQL 語法，不實際執行
        cur.execute(f"EXPLAIN {sql}")
        cur.close()
        return True
    except Exception as e:
        print(f"  ❌ SQL 驗證失敗: {e}")
        # Rollback 以避免 transaction 中止，讓後續查詢可以繼續
        conn.rollback()
        return False


# =====================================================
#  Schema 描述
# =====================================================

def generate_schema_description(schema):
    lines = ["/* Database Schema */\n"]
    for t, info in schema.items():
        lines.append(f"-- {t}")
        lines.append(f"CREATE TABLE {t} (")
        col_defs = []
        for c in info["columns"]:
            col_defs.append(f"  {c[0]} {c[1]}")
        lines.append(",\n".join(col_defs))
        lines.append(");\n")

        if info["fks"]:
            lines.append("/* FOREIGN KEYS:")
            for fk in info["fks"]:
                lines.append(f" * {t}.{fk[0]} -> {fk[1]}.{fk[2]}")
            lines.append(" */\n")

    return "\n".join(lines)


# =====================================================
#  Few-shot 生成
# =====================================================

def generate_fewshot_for_table(table, conn, schema, graph, db_name):
    print(f"🧩 Table: {table}")

    sample = get_sample_row(conn, table)
    if not sample:
        print("  ⚠️ 無資料，跳過")
        return None

    join_order = bfs_join_tables(table, graph)
    sql = generate_join_sql(table, join_order, schema, graph, sample)

    # 移除 MASK 來驗證 SQL
    # 根據欄位類型使用不同的測試值
    test_sql = sql
    for col in schema[table]["columns"]:
        name = col[0]
        ctype = (col[1] or "").lower()
        
        # 根據資料類型選擇測試值
        if "char" in ctype or "text" in ctype:
            # LIKE '%[name]%' → LIKE '%test%'
            test_sql = test_sql.replace(f"'%[{name}]%'", "'%test%'")
        elif "int" in ctype or "serial" in ctype:
            # = [name] → = 1（整數類型）
            test_sql = test_sql.replace(f"[{name}]", "1")
        elif "numeric" in ctype or "decimal" in ctype or "float" in ctype or "double" in ctype or "real" in ctype:
            # = [name] → = 1.0（數值類型）
            test_sql = test_sql.replace(f"[{name}]", "1.0")
        elif "bool" in ctype:
            # = [name] → = true（布林類型）
            test_sql = test_sql.replace(f"[{name}]", "true")
        elif "date" in ctype or "time" in ctype:
            # = [name] → = '2024-01-01'（日期時間類型）
            test_sql = test_sql.replace(f"[{name}]", "'2024-01-01'")
        else:
            # 其他類型預設用字串
            test_sql = test_sql.replace(f"[{name}]", "'test'")

    if not validate_sql(conn, test_sql):
        return None

    # 生成問題描述
    question = generate_question_from_sql(table, join_order, schema)

    return {
        "question": question,
        "tool_name": "run_sql",
        "args_json": json.dumps({"sql": sql}),
        "db_id": db_name,
    }


def generate_question_from_sql(root_table, join_order, schema):
    """根據 SQL 結構生成自然語言問題"""
    if len(join_order) == 1:
        return f"查詢 {root_table} 的資料"
    else:
        related = ", ".join(join_order[1:])
        return f"查詢 {root_table} 及其關聯的 {related} 資料"


# =====================================================
#  匯出到 ChromaDB 記憶
# =====================================================

def export_to_chromadb_memory(fewshots, agent_id, agent_data_dir):
    """將 few-shot 匯入到指定 agent 的 ChromaDB"""
    from vanna.integrations.chromadb.agent_memory import ChromaAgentMemory
    from vanna.core.user import User, RequestContext
    import asyncio

    persist_dir = f"{agent_data_dir}/chroma_db_{agent_id}"
    memory = ChromaAgentMemory(
        persist_directory=persist_dir,
        collection_name=f"vanna_{agent_id}"
    )

    context = RequestContext(user=User(id="admin"))

    async def save_all():
        for fs in fewshots:
            args = json.loads(fs["args_json"])
            await memory.save_tool_usage(
                question=fs["question"],
                tool_name=fs["tool_name"],
                args=args,
                context=context,
                success=True,
                metadata={"db_id": fs["db_id"], "auto_generated": True},
            )
            print(f"  ✓ {fs['question'][:50]}...")

    asyncio.run(save_all())


# =====================================================
#  main
# =====================================================

def main():
    parser = argparse.ArgumentParser(description="PostgreSQL Few-shot 自動生成")
    parser.add_argument("--host", type=str, required=True, help="PostgreSQL host")
    parser.add_argument("--port", type=str, default="5432", help="PostgreSQL port")
    parser.add_argument("--user", type=str, required=True, help="PostgreSQL user")
    parser.add_argument("--password", type=str, required=True, help="PostgreSQL password")
    parser.add_argument("--database", type=str, required=True, help="PostgreSQL database")
    parser.add_argument("--agent_id", type=str, help="Agent ID (用於匯入 ChromaDB)")
    parser.add_argument("--agent_data_dir", type=str, default="./agent_data", help="Agent data 目錄")
    parser.add_argument("--output", type=str, help="輸出 JSON 檔案路徑")
    args = parser.parse_args()

    # 連接資料庫
    conn_string = f"postgresql://{args.user}:{args.password}@{args.host}:{args.port}/{args.database}"
    print(f"🔗 連接資料庫: {args.host}:{args.port}/{args.database}")

    try:
        conn = psycopg2.connect(conn_string)
    except Exception as e:
        print(f"❌ 連接失敗: {e}")
        return

    tables, schema = analyze_database(conn)
    print(f"📊 找到 {len(tables)} 張表: {', '.join(tables)}")

    graph = build_fk_graph(tables, schema)

    fewshots = []
    for t in tables:
        fs = generate_fewshot_for_table(t, conn, schema, graph, args.database)
        if fs:
            fewshots.append(fs)

    conn.close()

    print(f"\n✅ 完成，共產生 {len(fewshots)} 筆 few-shot")

    # 輸出到 JSON 檔案
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(fewshots, f, ensure_ascii=False, indent=2)
        print(f"📄 輸出位置：{out_path}")

    # 匯入到 ChromaDB
    if args.agent_id:
        print(f"\n📥 匯入到 Agent: {args.agent_id}")
        export_to_chromadb_memory(fewshots, args.agent_id, args.agent_data_dir)
        print("✅ 匯入完成")


if __name__ == "__main__":
    main()
