#!/bin/bash
# 建置並推送 Manager Image 到 ECR
# 使用方式: ./push-manager.sh

set -e

# ============================================================
# 配置（請根據你的環境修改）
# ============================================================
AWS_PROFILE=""
AWS_REGION="ap-northeast-1"
AWS_ACCOUNT_ID="123456789"
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE_NAME="vanna-manager"

echo "=========================================="
echo "建置並推送 Manager Image"
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
echo "步驟 2: 建置 Manager Image..."
docker build --platform linux/amd64 -f docker/Dockerfile.manager -t vanna-manager:latest .

# 標記 Image
echo ""
echo "步驟 3: 標記 Image..."
docker tag vanna-manager:latest ${ECR_REGISTRY}/${IMAGE_NAME}:latest

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
echo "下一步: 在 ECS Console 更新 Manager Service"
echo "  1. 進入 ECS Cluster"
echo "  2. 選擇 Manager Service"
echo "  3. Update Service → Force new deployment"
echo ""
