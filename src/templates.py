"""HTML Templates - Vanna Multi-Tenant Management UI"""

# Vanna 風格的配色
# Primary: #4F46E5 (Indigo)
# Background: linear-gradient(to bottom, #e7e1cf, #ffffff)
# Card: white with shadow

BASE_STYLE = """
<style>
    * { box-sizing: border-box; }
    body { 
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        margin: 0;
        padding: 20px;
        background: linear-gradient(to bottom, #f5f3ef, #ffffff);
        min-height: 100vh;
        color: #1f2937;
    }
    .container { 
        max-width: 1200px; 
        margin: 0 auto;
    }
    .card {
        background: white;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        padding: 24px;
        margin-bottom: 20px;
    }
    h1 { 
        font-size: 28px;
        font-weight: 700;
        color: #111827;
        margin: 0 0 8px 0;
    }
    h2 { 
        font-size: 18px;
        font-weight: 600;
        color: #374151;
        margin: 24px 0 16px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid #e5e7eb;
    }
    .subtitle {
        color: #6b7280;
        font-size: 14px;
        margin-bottom: 24px;
    }
    
    /* Navigation */
    .nav { 
        display: flex;
        gap: 12px;
        margin-bottom: 24px;
        flex-wrap: wrap;
    }
    .nav a { 
        padding: 10px 16px;
        background: #4f46e5;
        color: white;
        border-radius: 8px;
        text-decoration: none;
        font-size: 14px;
        font-weight: 500;
        transition: background 0.2s;
    }
    .nav a:hover { background: #4338ca; }
    .nav a.secondary {
        background: #6b7280;
    }
    .nav a.secondary:hover { background: #4b5563; }
    
    /* Table */
    table { 
        width: 100%; 
        border-collapse: collapse;
    }
    th, td { 
        padding: 12px 16px;
        text-align: left;
        border-bottom: 1px solid #e5e7eb;
    }
    th { 
        background: #f9fafb;
        font-weight: 600;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #6b7280;
    }
    tr:hover { background: #f9fafb; }
    td { font-size: 14px; }
    
    /* Status Badge */
    .status {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 500;
    }
    .status.running {
        background: #d1fae5;
        color: #065f46;
    }
    .status.stopped {
        background: #fee2e2;
        color: #991b1b;
    }
    .status-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
    }
    .status.running .status-dot { background: #10b981; }
    .status.stopped .status-dot { background: #ef4444; }
    
    /* Buttons */
    .btn {
        display: inline-flex;
        align-items: center;
        padding: 8px 16px;
        border-radius: 6px;
        font-size: 14px;
        font-weight: 500;
        border: none;
        cursor: pointer;
        text-decoration: none;
        transition: all 0.2s;
    }
    .btn-primary {
        background: #4f46e5;
        color: white;
    }
    .btn-primary:hover { background: #4338ca; }
    .btn-secondary {
        background: #f3f4f6;
        color: #374151;
    }
    .btn-secondary:hover { background: #e5e7eb; }
    .btn-danger {
        background: #fee2e2;
        color: #991b1b;
    }
    .btn-danger:hover { background: #fecaca; }
    .btn-sm {
        padding: 4px 10px;
        font-size: 12px;
    }
    
    /* Links */
    a { color: #4f46e5; text-decoration: none; }
    a:hover { text-decoration: underline; }
    .action-link {
        color: #6b7280;
        font-size: 13px;
    }
    .action-link:hover { color: #4f46e5; }
    
    /* Forms */
    .form-group { margin-bottom: 20px; }
    label { 
        display: block;
        font-size: 14px;
        font-weight: 500;
        color: #374151;
        margin-bottom: 6px;
    }
    input, textarea, select { 
        width: 100%;
        padding: 10px 12px;
        border: 1px solid #d1d5db;
        border-radius: 6px;
        font-size: 14px;
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    input:focus, textarea:focus, select:focus {
        outline: none;
        border-color: #4f46e5;
        box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
    }
    textarea { 
        min-height: 120px;
        font-family: 'SF Mono', Monaco, monospace;
        font-size: 13px;
    }
    .hint {
        font-size: 12px;
        color: #6b7280;
        margin-top: 4px;
    }
    .form-row {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 16px;
    }
    
    /* Alert */
    .alert {
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 20px;
        font-size: 14px;
    }
    .alert-success {
        background: #d1fae5;
        color: #065f46;
        border: 1px solid #a7f3d0;
    }
    .alert-error {
        background: #fee2e2;
        color: #991b1b;
        border: 1px solid #fecaca;
    }
    .alert-info {
        background: #dbeafe;
        color: #1e40af;
        border: 1px solid #bfdbfe;
    }
    
    /* Code */
    pre {
        background: #1f2937;
        color: #f9fafb;
        padding: 16px;
        border-radius: 8px;
        overflow-x: auto;
        font-size: 13px;
        font-family: 'SF Mono', Monaco, monospace;
    }
    
    /* Empty State */
    .empty-state {
        text-align: center;
        padding: 48px 24px;
        color: #6b7280;
    }
    .empty-state p {
        margin: 8px 0;
    }
</style>
"""


def get_admin_html(
    agents_info: list,
    current_agent: str = None,
    memories: list = None,
    message: str = None,
):
    """Memory management page"""
    agents_options = "".join(
        [
            f'<option value="{a["agent_id"]}" {"selected" if a["agent_id"] == current_agent else ""}>{a["agent_id"]}</option>'
            for a in agents_info
        ]
    )

    memories_html = ""
    if memories:
        for m in memories:
            memories_html += f"""
            <tr>
                <td style="max-width:300px;overflow:hidden;text-overflow:ellipsis;">{m.get("question", "")[:100]}</td>
                <td><code>{m.get("tool_name", "")}</code></td>
                <td style="color:#6b7280;font-size:13px;">{m.get("timestamp", "")[:19] if m.get("timestamp") else ""}</td>
                <td>
                    <a href="/admin/memory/{current_agent}/detail/{m["id"]}" class="action-link">View</a>
                    <span style="color:#d1d5db;margin:0 4px;">|</span>
                    <a href="/admin/memory/{current_agent}/delete/{m["id"]}" class="action-link" onclick="return confirm('Delete this memory?')">Delete</a>
                </td>
            </tr>"""

    message_html = ""
    if message:
        alert_type = "success" if "success" in message.lower() or "成功" in message else "error"
        message_html = f'<div class="alert alert-{alert_type}">{message}</div>'

    memory_section = ""
    if current_agent:
        memory_section = f"""
            <h2>Memories - {current_agent}</h2>
            <div style="margin-bottom:16px;">
                <a href="/admin/memory/{current_agent}/add" class="btn btn-primary btn-sm">Add Memory</a>
                <button onclick="generateFewshot('{current_agent}')" id="generateBtn" class="btn btn-secondary btn-sm" style="margin-left:8px;">
                    Auto Generate Few-shot
                </button>
            </div>
            <div id="generateResult" class="alert alert-info" style="display:none;"></div>
            <table>
                <thead>
                    <tr>
                        <th>Question</th>
                        <th>Tool</th>
                        <th>Timestamp</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
                    {memories_html if memories_html else '<tr><td colspan="4" class="empty-state">No memories found</td></tr>'}
                </tbody>
            </table>
            
            <script>
            async function generateFewshot(agentId) {{
                const btn = document.getElementById('generateBtn');
                const resultDiv = document.getElementById('generateResult');
                
                btn.disabled = true;
                btn.textContent = 'Generating...';
                resultDiv.style.display = 'block';
                resultDiv.className = 'alert alert-info';
                resultDiv.innerHTML = 'Analyzing database schema and generating few-shot examples...';
                
                try {{
                    const response = await fetch('/api/agents/' + agentId + '/generate-fewshot', {{ method: 'POST' }});
                    const data = await response.json();
                    
                    if (response.ok) {{
                        resultDiv.className = 'alert alert-success';
                        resultDiv.innerHTML = data.message + ' (' + data.total_tables + ' tables, ' + data.imported + ' imported)';
                        setTimeout(() => location.reload(), 2000);
                    }} else {{
                        resultDiv.className = 'alert alert-error';
                        resultDiv.innerHTML = 'Error: ' + (data.detail || 'Unknown error');
                    }}
                }} catch (e) {{
                    resultDiv.className = 'alert alert-error';
                    resultDiv.innerHTML = 'Request failed: ' + e.message;
                }}
                
                btn.disabled = false;
                btn.textContent = 'Auto Generate Few-shot';
            }}
            </script>
        """

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <title>Memory Management - Vanna Multi-Tenant</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        {BASE_STYLE}
    </head>
    <body>
        <div class="container">
            <div class="card">
                <h1>Memory Management</h1>
                <p class="subtitle">Manage agent memories and few-shot examples</p>
                
                <div class="nav">
                    <a href="/admin/agents">Agents</a>
                    <a href="/admin/memory" class="secondary">Memories</a>
                    <a href="/docs" target="_blank" class="secondary">API Docs</a>
                </div>
                
                {message_html}
                
                <form method="get" action="/admin/memory" style="margin-bottom:24px;">
                    <label>Select Agent</label>
                    <select name="agent_id" onchange="this.form.submit()" style="max-width:300px;">
                        <option value="">-- Select Agent --</option>
                        {agents_options}
                    </select>
                </form>
                
                {memory_section}
            </div>
        </div>
    </body>
    </html>
    """


def get_add_memory_html(agent_id: str):
    """Add memory page"""
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <title>Add Memory - {agent_id}</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        {BASE_STYLE}
    </head>
    <body>
        <div class="container">
            <div class="card" style="max-width:700px;">
                <h1>Add Memory</h1>
                <p class="subtitle">Agent: {agent_id}</p>
                
                <form method="post" action="/admin/memory/{agent_id}/add">
                    <div class="form-group">
                        <label>Question</label>
                        <input type="text" name="question" required placeholder="e.g., What is today's total sales?">
                    </div>
                    <div class="form-group">
                        <label>Tool Name</label>
                        <input type="text" name="tool_name" value="run_sql" required>
                    </div>
                    <div class="form-group">
                        <label>Arguments (JSON)</label>
                        <textarea name="args_json" placeholder='{{"sql": "SELECT SUM(amount) FROM sales"}}'></textarea>
                    </div>
                    <div class="form-group">
                        <label>Metadata (JSON, optional)</label>
                        <textarea name="metadata_json" placeholder='{{}}' style="min-height:80px;"></textarea>
                    </div>
                    <div style="display:flex;gap:12px;">
                        <button type="submit" class="btn btn-primary">Save Memory</button>
                        <a href="/admin/memory?agent_id={agent_id}" class="btn btn-secondary">Cancel</a>
                    </div>
                </form>
            </div>
        </div>
    </body>
    </html>
    """


def get_detail_html(agent_id: str, memory_data: dict):
    """Memory detail page"""
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <title>Memory Detail - {agent_id}</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        {BASE_STYLE}
    </head>
    <body>
        <div class="container">
            <div class="card" style="max-width:800px;">
                <p><a href="/admin/memory?agent_id={agent_id}">&larr; Back to list</a></p>
                <h1>Memory Detail</h1>
                
                <div class="form-group">
                    <label>ID</label>
                    <div style="padding:10px;background:#f9fafb;border-radius:6px;font-family:monospace;font-size:13px;">
                        {memory_data.get("id", "")}
                    </div>
                </div>
                <div class="form-group">
                    <label>Question</label>
                    <div style="padding:10px;background:#f9fafb;border-radius:6px;">
                        {memory_data.get("question", "")}
                    </div>
                </div>
                <div class="form-group">
                    <label>Tool Name</label>
                    <div style="padding:10px;background:#f9fafb;border-radius:6px;">
                        <code>{memory_data.get("tool_name", "")}</code>
                    </div>
                </div>
                <div class="form-group">
                    <label>Timestamp</label>
                    <div style="padding:10px;background:#f9fafb;border-radius:6px;color:#6b7280;">
                        {memory_data.get("timestamp", "")}
                    </div>
                </div>
                <div class="form-group">
                    <label>Arguments</label>
                    <pre>{memory_data.get("args_json", "{}")}</pre>
                </div>
                <div class="form-group">
                    <label>Document</label>
                    <pre>{memory_data.get("document", "")}</pre>
                </div>
                
                <div style="margin-top:24px;">
                    <a href="/admin/memory/{agent_id}/delete/{memory_data.get("id")}" 
                       class="btn btn-danger" 
                       onclick="return confirm('Delete this memory?')">
                        Delete Memory
                    </a>
                </div>
            </div>
        </div>
    </body>
    </html>
    """


def get_agents_management_html(agents_info: list, message: str = None):
    """Agent management page"""
    message_html = ""
    if message:
        alert_type = "success" if "success" in message.lower() or "成功" in message else "error"
        message_html = f'<div class="alert alert-{alert_type}">{message}</div>'

    agents_rows = ""
    for a in agents_info:
        status_class = "running" if a.get("running") else "stopped"
        status_text = "Running" if a.get("running") else "Stopped"
        url = a.get("url", "#")
        port_or_ip = a.get("public_ip") or a.get("port") or "N/A"
        
        agents_rows += f"""
        <tr>
            <td><strong>{a["agent_id"]}</strong></td>
            <td style="color:#6b7280;">{a.get("description", "-")}</td>
            <td><code>{port_or_ip}</code></td>
            <td>
                <span class="status {status_class}">
                    <span class="status-dot"></span>
                    {status_text}
                </span>
            </td>
            <td>
                <a href="{url}" target="_blank" class="action-link">Open</a>
                <span style="color:#d1d5db;margin:0 4px;">|</span>
                <a href="/admin/memory?agent_id={a["agent_id"]}" class="action-link">Memories</a>
                <span style="color:#d1d5db;margin:0 4px;">|</span>
                <button onclick="restartAgent('{a["agent_id"]}')" class="btn btn-secondary btn-sm">Restart</button>
                <button onclick="deleteAgent('{a["agent_id"]}')" class="btn btn-danger btn-sm" style="margin-left:4px;">Delete</button>
            </td>
        </tr>"""

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <title>Agent Management - Vanna Multi-Tenant</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        {BASE_STYLE}
    </head>
    <body>
        <div class="container">
            <div class="card">
                <h1>Vanna Multi-Tenant</h1>
                <p class="subtitle">Manage your SQL AI agents</p>
                
                <div class="nav">
                    <a href="/admin/agents">Agents</a>
                    <a href="/admin/memory" class="secondary">Memories</a>
                    <a href="/admin/agents/new" class="secondary">New Agent</a>
                    <a href="/docs" target="_blank" class="secondary">API Docs</a>
                </div>
                
                {message_html}
                
                <h2>Agents</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Agent ID</th>
                            <th>Description</th>
                            <th>Port / IP</th>
                            <th>Status</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {agents_rows if agents_rows else '<tr><td colspan="5" class="empty-state"><p>No agents yet</p><p><a href="/admin/agents/new">Create your first agent</a></p></td></tr>'}
                    </tbody>
                </table>
            </div>
        </div>
        
        <script>
        async function restartAgent(agentId) {{
            if (!confirm('Restart agent "' + agentId + '"?')) return;
            
            try {{
                const response = await fetch('/api/agents/' + agentId + '/restart', {{ method: 'POST' }});
                const data = await response.json();
                
                if (response.ok) {{
                    alert(data.message);
                    location.reload();
                }} else {{
                    alert('Error: ' + (data.detail || 'Unknown error'));
                }}
            }} catch (e) {{
                alert('Request failed: ' + e.message);
            }}
        }}
        
        async function deleteAgent(agentId) {{
            if (!confirm('Delete agent "' + agentId + '"?')) return;
            
            const deleteMemory = confirm('Also delete ChromaDB memories?\\n\\nOK = Delete memories\\nCancel = Keep memories');
            
            try {{
                const response = await fetch('/api/agents/' + agentId + '?delete_memory=' + deleteMemory, {{ method: 'DELETE' }});
                const data = await response.json();
                
                if (response.ok) {{
                    alert(data.message);
                    location.reload();
                }} else {{
                    alert('Error: ' + (data.detail || 'Unknown error'));
                }}
            }} catch (e) {{
                alert('Request failed: ' + e.message);
            }}
        }}
        </script>
    </body>
    </html>
    """


def get_create_agent_html(message: str = None):
    """Create agent page"""
    message_html = ""
    if message:
        alert_type = "success" if "success" in message.lower() or "成功" in message else "error"
        message_html = f'<div class="alert alert-{alert_type}">{message}</div>'

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <title>Create Agent - Vanna Multi-Tenant</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        {BASE_STYLE}
    </head>
    <body>
        <div class="container">
            <div class="card" style="max-width:700px;">
                <p><a href="/admin/agents">&larr; Back to agents</a></p>
                <h1>Create Agent</h1>
                <p class="subtitle">Configure a new SQL AI agent</p>
                
                {message_html}
                
                <form method="post" action="/admin/agents/new">
                    <div class="form-group">
                        <label>Agent ID *</label>
                        <input type="text" name="agent_id" required placeholder="e.g., sales_agent">
                        <p class="hint">Unique identifier. Use letters, numbers, and underscores only.</p>
                    </div>
                    
                    <div class="form-group">
                        <label>Description</label>
                        <input type="text" name="description" placeholder="e.g., Sales Analytics Agent">
                    </div>
                    
                    <h2>Database Connection</h2>
                    
                    <div class="form-row">
                        <div class="form-group">
                            <label>Host *</label>
                            <input type="text" name="postgres_host" required placeholder="localhost">
                        </div>
                        <div class="form-group">
                            <label>Port</label>
                            <input type="text" name="postgres_port" value="5432">
                        </div>
                    </div>
                    
                    <div class="form-row">
                        <div class="form-group">
                            <label>Username *</label>
                            <input type="text" name="postgres_user" required placeholder="postgres">
                        </div>
                        <div class="form-group">
                            <label>Password *</label>
                            <input type="password" name="postgres_password" required>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label>Database *</label>
                        <input type="text" name="postgres_db" required placeholder="my_database">
                    </div>
                    
                    <h2>System Prompt</h2>
                    
                    <div class="form-group">
                        <label>Custom System Prompt (optional)</label>
                        <textarea name="system_prompt" style="min-height:200px;" placeholder="Leave empty to auto-generate from database schema..."></textarea>
                        <p class="hint">Define the agent's role and behavior. If left empty, the system will analyze your database schema and generate an appropriate prompt.</p>
                    </div>
                    
                    <div style="display:flex;gap:12px;margin-top:24px;">
                        <button type="submit" class="btn btn-primary">Create Agent</button>
                        <a href="/admin/agents" class="btn btn-secondary">Cancel</a>
                    </div>
                </form>
            </div>
        </div>
    </body>
    </html>
    """
