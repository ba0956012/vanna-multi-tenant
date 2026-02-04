# Multi Vanna - 單一 Docker 版本
# 使用 multiprocessing 在一個容器內管理多個 Agent

FROM python:3.13-slim

WORKDIR /app

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 複製 requirements.txt
COPY requirements.txt .

# 安裝 Python 依賴
RUN pip install --no-cache-dir -r requirements.txt

# 複製程式碼
COPY src/main_multi_agent.py .
COPY src/custom_tools.py .
COPY src/templates.py .
COPY src/db/ ./db/
COPY static/ ./static/

# 建立 agent_data 目錄
RUN mkdir -p /app/agent_data

# 暴露端口
EXPOSE 8100

# 啟動服務
CMD ["python", "main_multi_agent.py"]
