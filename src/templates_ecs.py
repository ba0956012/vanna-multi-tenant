"""HTML 模板 - ECS 版本"""


def get_agents_management_html(agents_info: list, message: str = None):
    """Agent 管理頁面 (ECS 版本)"""
    message_html = ""
    if message:
        msg_type = "success" if "成功" in message else "error"
        bg_color = "#d4edda" if msg_type == "success" else "#f8d7da"
        message_html = f'<div style="background:{bg_color};padding:10px;margin:10px 0;border-radius:5px;">{message}</div>'

    agents_rows = ""
    for a in agents_info:
        status = "🟢 運行中" if a.get("running") else "🔴 已停止"
        url = a.get("url") or "#"
        public_ip = a.get("public_ip") or "N/A"
        private_ip = a.get("private_ip") or "N/A"
        
        # 如果沒有 URL，顯示「未啟動」
        link_html = f'<a href="{url}" target="_blank">開啟</a>' if url != "#" else '<span style="color:#999">未啟動</span>'
        
        # 根據運行狀態顯示不同的操作按鈕
        if a.get("running"):
            action_buttons = f'''
                {link_html} |
                <a href="/admin/memory?agent_id={a["agent_id"]}">記憶</a> |
                <button onclick="stopAgent('{a["agent_id"]}')" class="btn-sm btn-warning">停止</button> |
                <button onclick="restartAgent('{a["agent_id"]}')" class="btn-sm">重啟</button> |
                <button onclick="deleteAgent('{a["agent_id"]}')" class="btn-sm btn-danger">刪除</button>
            '''
        else:
            action_buttons = f'''
                <span style="color:#999">未啟動</span> |
                <a href="/admin/memory?agent_id={a["agent_id"]}">記憶</a> |
                <button onclick="startAgent('{a["agent_id"]}')" class="btn-sm btn-success">啟動</button> |
                <button onclick="deleteAgent('{a["agent_id"]}')" class="btn-sm btn-danger">刪除</button>
            '''
        
        agents_rows += f"""
        <tr>
            <td>{a["agent_id"]}</td>
            <td>{a.get("description", "")}</td>
            <td>{public_ip}</td>
            <td>{private_ip}</td>
            <td>{status}</td>
            <td>{action_buttons}</td>
        </tr>"""

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Agent 管理 - ECS</title>
        <meta charset="utf-8">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            h1 {{ color: #333; }}
            h2 {{ color: #666; border-bottom: 1px solid #eee; padding-bottom: 10px; margin-top: 30px; }}
            table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
            th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background: #f8f9fa; font-weight: bold; }}
            tr:hover {{ background: #f5f5f5; }}
            .btn {{ padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; text-decoration: none; display: inline-block; }}
            .btn:hover {{ background: #0056b3; }}
            .btn-sm {{ padding: 5px 10px; font-size: 12px; background: #6c757d; color: white; border: none; border-radius: 3px; cursor: pointer; }}
            .btn-sm:hover {{ background: #5a6268; }}
            .btn-success {{ background: #28a745; }}
            .btn-success:hover {{ background: #218838; }}
            .btn-warning {{ background: #ffc107; color: #212529; }}
            .btn-warning:hover {{ background: #e0a800; }}
            .btn-danger {{ background: #dc3545; }}
            .btn-danger:hover {{ background: #c82333; }}
            a {{ color: #007bff; text-decoration: none; }}
            a:hover {{ text-decoration: underline; }}
            .badge {{ display: inline-block; padding: 3px 8px; border-radius: 3px; font-size: 12px; }}
            .badge-ecs {{ background: #17a2b8; color: white; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 Agent 管理 <span class="badge badge-ecs">ECS</span></h1>
            <p>
                <a href="/admin/memory" class="btn">📝 記憶管理</a>
                <a href="/admin/efs" class="btn">📂 EFS 瀏覽器</a>
                <a href="/admin/agents/new" class="btn">➕ 新增 Agent</a>
                <a href="/docs" class="btn" target="_blank">📚 API 文檔</a>
            </p>
            
            {message_html}
            
            <h2>Agent 列表</h2>
            <table>
                <tr>
                    <th>Agent ID</th>
                    <th>描述</th>
                    <th>Public IP</th>
                    <th>Private IP</th>
                    <th>狀態</th>
                    <th>操作</th>
                </tr>
                {agents_rows if agents_rows else "<tr><td colspan='6'>尚無 Agent，點擊上方「新增 Agent」開始</td></tr>"}
            </table>
            
            <div style="margin-top:30px; padding:15px; background:#e7f3ff; border-radius:5px;">
                <h3 style="margin-top:0;">💡 使用說明</h3>
                <ul>
                    <li>每個 Agent 會自動部署為獨立的 ECS Service</li>
                    <li>Public IP 是動態分配的，重啟後會變更</li>
                    <li>Private IP 用於 VPC 內部通訊</li>
                </ul>
            </div>
        </div>
        
        <script>
        async function startAgent(agentId) {{
            if (!confirm('確定要啟動 ' + agentId + '?')) return;
            
            try {{
                const response = await fetch('/api/agents/' + agentId + '/start', {{
                    method: 'POST'
                }});
                const data = await response.json();
                
                if (response.ok) {{
                    alert('✅ ' + data.message + '\\n\\n請等待 1-2 分鐘讓 ECS Task 啟動');
                    location.reload();
                }} else {{
                    alert('❌ 錯誤: ' + (data.detail || '未知錯誤'));
                }}
            }} catch (e) {{
                alert('❌ 請求失敗: ' + e.message);
            }}
        }}
        
        async function stopAgent(agentId) {{
            if (!confirm('確定要停止 ' + agentId + '?\\n\\n這會停止 ECS Task，但保留 Service')) return;
            
            try {{
                const response = await fetch('/api/agents/' + agentId + '/stop', {{
                    method: 'POST'
                }});
                const data = await response.json();
                
                if (response.ok) {{
                    alert('✅ ' + data.message);
                    location.reload();
                }} else {{
                    alert('❌ 錯誤: ' + (data.detail || '未知錯誤'));
                }}
            }} catch (e) {{
                alert('❌ 請求失敗: ' + e.message);
            }}
        }}
        
        async function restartAgent(agentId) {{
            if (!confirm('確定要重啟 ' + agentId + '?\\n\\n這會重新部署 ECS Service')) return;
            
            try {{
                const response = await fetch('/api/agents/' + agentId + '/restart', {{
                    method: 'POST'
                }});
                const data = await response.json();
                
                if (response.ok) {{
                    alert('✅ ' + data.message + '\\n\\n請等待 1-2 分鐘讓 ECS Task 啟動');
                    location.reload();
                }} else {{
                    alert('❌ 錯誤: ' + (data.detail || '未知錯誤'));
                }}
            }} catch (e) {{
                alert('❌ 請求失敗: ' + e.message);
            }}
        }}
        
        async function deleteAgent(agentId) {{
            if (!confirm('確定要刪除 ' + agentId + '?')) {{
                return;
            }}
            
            const deleteService = confirm('是否完全刪除 ECS Service？\\n\\n確定 = 完全刪除\\n取消 = 只停止（可重啟）');
            const deleteMemory = confirm('是否同時刪除 ChromaDB 記憶？\\n\\n確定 = 刪除記憶\\n取消 = 保留記憶');
            
            try {{
                const response = await fetch('/api/agents/' + agentId + '?delete_memory=' + deleteMemory + '&delete_service=' + deleteService, {{
                    method: 'DELETE'
                }});
                const data = await response.json();
                
                if (response.ok) {{
                    alert('✅ ' + data.message);
                    location.reload();
                }} else {{
                    alert('❌ 錯誤: ' + (data.detail || '未知錯誤'));
                }}
            }} catch (e) {{
                alert('❌ 請求失敗: ' + e.message);
            }}
        }}
        </script>
    </body>
    </html>
    """


def get_create_agent_html(message: str = None):
    """新增 Agent 頁面 (ECS 版本)"""
    message_html = ""
    if message:
        msg_type = "success" if "成功" in message else "error"
        bg_color = "#d4edda" if msg_type == "success" else "#f8d7da"
        message_html = f'<div style="background:{bg_color};padding:10px;margin:10px 0;border-radius:5px;">{message}</div>'

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>新增 Agent - ECS</title>
        <meta charset="utf-8">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .container {{ max-width: 900px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
            h1 {{ color: #333; }}
            .form-group {{ margin: 15px 0; }}
            label {{ display: block; margin-bottom: 5px; font-weight: bold; }}
            input, textarea {{ width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; }}
            textarea {{ height: 300px; font-family: monospace; font-size: 13px; }}
            .form-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }}
            button {{ padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; margin-right: 10px; }}
            button:hover {{ background: #0056b3; }}
            a {{ color: #007bff; }}
            .hint {{ color: #666; font-size: 12px; margin-top: 5px; }}
            .info-box {{ background: #e7f3ff; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>➕ 新增 Agent (ECS)</h1>
            <p><a href="/admin/agents">← 返回列表</a></p>
            
            {message_html}
            
            <div class="info-box">
                <strong>📌 注意事項：</strong>
                <ul style="margin:10px 0;">
                    <li>建立後會自動部署為 ECS Service</li>
                    <li>首次啟動需要 1-2 分鐘</li>
                    <li>System Prompt 留空會自動從資料庫生成</li>
                </ul>
            </div>
            
            <form method="post" action="/admin/agents/new">
                <div class="form-group">
                    <label>Agent ID *</label>
                    <input type="text" name="agent_id" required placeholder="例如: pos_sales_agent">
                    <div class="hint">唯一識別碼，只能使用英文、數字、底線</div>
                </div>
                
                <div class="form-group">
                    <label>描述</label>
                    <input type="text" name="description" placeholder="例如: POS 銷售分析系統">
                </div>
                
                <div class="form-row">
                    <div class="form-group">
                        <label>PostgreSQL Host *</label>
                        <input type="text" name="postgres_host" required placeholder="例如: 0.0.0.0">
                    </div>
                    <div class="form-group">
                        <label>PostgreSQL Port</label>
                        <input type="text" name="postgres_port" value="5432" placeholder="5432">
                    </div>
                </div>
                
                <div class="form-row">
                    <div class="form-group">
                        <label>PostgreSQL User *</label>
                        <input type="text" name="postgres_user" required placeholder="postgres">
                    </div>
                    <div class="form-group">
                        <label>PostgreSQL Password *</label>
                        <input type="password" name="postgres_password" required>
                    </div>
                </div>
                
                <div class="form-group">
                    <label>PostgreSQL Database *</label>
                    <input type="text" name="postgres_db" required placeholder="例如: pos_sales">
                </div>
                
                <div class="form-group">
                    <label>System Prompt (選填，留空則自動從資料庫生成)</label>
                    <textarea name="system_prompt" placeholder="留空則自動從資料庫結構生成 system prompt...&#10;&#10;或手動輸入:&#10;你是一個專業的數據分析助手..."></textarea>
                    <div class="hint">定義 Agent 的角色、資料庫結構說明、工作流程等。留空則系統會自動分析資料庫結構並生成。</div>
                </div>
                
                <button type="submit">建立 Agent (會自動部署到 ECS)</button>
                <a href="/admin/agents">取消</a>
            </form>
        </div>
    </body>
    </html>
    """


def get_efs_browser_html(path: str, items: list, disk_usage: dict = None):
    """EFS 檔案瀏覽器頁面"""
    
    # 麵包屑導航
    breadcrumb = '<a href="/admin/efs">EFS Root</a>'
    if path:
        parts = path.split('/')
        current_path = ""
        for part in parts:
            if part:
                current_path += f"{part}/"
                breadcrumb += f' / <a href="/admin/efs?path={current_path.rstrip("/")}">{part}</a>'
    
    # 檔案列表
    items_html = ""
    if not items:
        items_html = '<tr><td colspan="4" style="text-align:center;color:#999;">目錄是空的</td></tr>'
    else:
        for item in items:
            icon = "📁" if item["type"] == "directory" else "📄"
            size_str = f"{item['size']:,} bytes" if item["type"] == "file" else "-"
            
            if item["type"] == "directory":
                link_path = f"{path}/{item['name']}" if path else item["name"]
                name_html = f'<a href="/admin/efs?path={link_path}">{icon} {item["name"]}</a>'
            else:
                name_html = f'{icon} {item["name"]}'
            
            items_html += f"""
            <tr>
                <td>{name_html}</td>
                <td>{item["type"]}</td>
                <td>{size_str}</td>
                <td>{item["modified"]}</td>
            </tr>"""
    
    # 磁碟使用量
    disk_usage_html = ""
    if disk_usage:
        total_gb = disk_usage.get("total_size_gb", 0)
        total_mb = disk_usage.get("total_size_mb", 0)
        
        agents_rows = ""
        for agent in disk_usage.get("agents", []):
            agents_rows += f"""
            <tr>
                <td>{agent["agent_id"]}</td>
                <td>{agent["size_mb"]} MB</td>
                <td>
                    <a href="/admin/efs?path=chroma_db_{agent['agent_id']}">查看</a>
                </td>
            </tr>"""
        
        disk_usage_html = f"""
        <div style="margin-top:30px; padding:15px; background:#f8f9fa; border-radius:5px;">
            <h3 style="margin-top:0;">💾 磁碟使用量</h3>
            <p><strong>總大小：</strong>{total_gb:.2f} GB ({total_mb:.1f} MB)</p>
            
            {f'''
            <table style="margin-top:10px;">
                <tr>
                    <th>Agent ID</th>
                    <th>大小</th>
                    <th>操作</th>
                </tr>
                {agents_rows}
            </table>
            ''' if agents_rows else '<p style="color:#999;">尚無 Agent 資料</p>'}
        </div>"""
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>EFS 檔案瀏覽器</title>
        <meta charset="utf-8">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            h1 {{ color: #333; }}
            .breadcrumb {{ background: #e9ecef; padding: 10px 15px; border-radius: 5px; margin: 15px 0; }}
            .breadcrumb a {{ color: #007bff; text-decoration: none; }}
            .breadcrumb a:hover {{ text-decoration: underline; }}
            table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
            th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background: #f8f9fa; font-weight: bold; }}
            tr:hover {{ background: #f5f5f5; }}
            a {{ color: #007bff; text-decoration: none; }}
            a:hover {{ text-decoration: underline; }}
            .btn {{ padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; text-decoration: none; display: inline-block; margin-right: 10px; }}
            .btn:hover {{ background: #0056b3; }}
            .info-box {{ background: #e7f3ff; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📂 EFS 檔案瀏覽器</h1>
            <p>
                <a href="/admin/agents" class="btn">← 返回 Agent 管理</a>
                <a href="/admin/memory" class="btn">📝 記憶管理</a>
                <button onclick="location.reload()" class="btn">🔄 重新整理</button>
            </p>
            
            <div class="breadcrumb">
                <strong>當前路徑：</strong> {breadcrumb}
            </div>
            
            <h2>📋 檔案列表</h2>
            <table>
                <tr>
                    <th>名稱</th>
                    <th>類型</th>
                    <th>大小</th>
                    <th>修改時間</th>
                </tr>
                {items_html}
            </table>
            
            {disk_usage_html}
            
            <div class="info-box">
                <h3 style="margin-top:0;">💡 說明</h3>
                <ul>
                    <li>點擊目錄名稱可以進入該目錄</li>
                    <li>每個 Agent 的 ChromaDB 資料儲存在 <code>chroma_db_{{agent_id}}</code> 目錄</li>
                    <li>刪除 Agent 時可以選擇是否同時刪除 EFS 上的記憶資料</li>
                    <li>EFS 是共享儲存，所有 ECS Tasks 都可以存取</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """


# 其他共用的 templates 可以從原本的 templates.py import
from templates import get_admin_html, get_add_memory_html, get_detail_html
