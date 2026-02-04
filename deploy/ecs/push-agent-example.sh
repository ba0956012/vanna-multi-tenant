#!/bin/bash
# 建置並推送 Agent Image 到 ECR
# 使用方式: ./push-agent.sh

set -e

# ============================================================
# 配置（請根據你的環境修改）
# ============================================================
AWS_PROFILE=""
AWS_REGION="ap-northeast-1"
AWS_ACCOUNT_ID="123456789"
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE_NAME="vanna-agent"

echo "=========================================="
echo "建置並推送 Agent Image"
echo "=========================================="
echo "Registry: ${ECR_REGISTRY}"
echo "Image: ${IMAGE_NAME}:latest"
echo "=========================================="

# 切換到專案根目錄
cd "$(dirname "$0")/../.."

# 登入 ECR
echo ""
echo "步驟 1: 登入 ECR..."
aws ecr get-login-password --region ${AWS_REGION} --profile ${AWS_PROFILE} | \
  docker login --username AWS --password-stdin ${ECR_REGISTRY}

# 建置 Image
echo ""
echo "步驟 2: 建置 Agent Image..."
docker build --platform linux/amd64 -f docker/Dockerfile.agent -t vanna-agent:latest .

# 標記 Image
echo ""
echo "步驟 3: 標記 Image..."
docker tag vanna-agent:latest ${ECR_REGISTRY}/${IMAGE_NAME}:latest

# 推送到 ECR
echo ""
echo "步驟 4: 推送到 ECR..."
docker push ${ECR_REGISTRY}/${IMAGE_NAME}:latest

# 完成
echo ""
echo "=========================================="
echo "✅ 推送完成！"
echo "=========================================="
echo ""
echo "Image URI: ${ECR_REGISTRY}/${IMAGE_NAME}:latest"
echo ""
echo "下一步: 新建立的 Agent 會自動使用最新 Image"
echo "現有的 Agent 需要重新部署才會更新"
echo ""
