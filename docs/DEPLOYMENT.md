# ECS 部署指南

## 前置準備

### 需要建立的 AWS 資源

#### 1. ECR Repositories (儲存 Docker Images)
```bash
# 建立 Manager Repository
aws ecr create-repository \
  --repository-name vanna-manager \
  --region ap-northeast-1

# 建立 Agent Repository
aws ecr create-repository \
  --repository-name vanna-agent \
  --region ap-northeast-1
```

#### 2. ECS Cluster
```bash
aws ecs create-cluster \
  --cluster-name vanna-cluster \
  --region ap-northeast-1
```

#### 3. DynamoDB Table
```bash
aws dynamodb create-table \
  --table-name vanna-agents \
  --attribute-definitions AttributeName=agent_id,AttributeType=S \
  --key-schema AttributeName=agent_id,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST \
  --region ap-northeast-1
```

#### 4. EFS (共享 ChromaDB 資料)
```bash
# 在 AWS Console 建立 EFS
# 1. 進入 EFS Console
# 2. 點擊「建立檔案系統」
# 3. 選擇與 ECS 相同的 VPC
# 4. 記下 File System ID (例如: fs-12345678)
```

#### 5. IAM Task Role

**什麼是 Task Role？**
- Task Role 是給**容器內的應用程式**使用的 IAM 角色
- 你的 Python 程式呼叫 `boto3.client()` 時，會自動使用這個角色的權限
- 不需要 AWS_ACCESS_KEY_ID、不需要 MFA、不需要 assume role

```bash
# 步驟 1: 建立 Trust Policy
cat > ecs-task-trust-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ecs-tasks.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

# 步驟 2: 建立 Task Role
aws iam create-role \
  --role-name VannaECSTaskRole \
  --assume-role-policy-document file://ecs-task-trust-policy.json

# 步驟 3: 建立 Policy
cat > vanna-task-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "dynamodb:GetItem",
        "dynamodb:PutItem",
        "dynamodb:UpdateItem",
        "dynamodb:DeleteItem",
        "dynamodb:Scan",
        "dynamodb:Query"
      ],
      "Resource": "arn:aws:dynamodb:*:*:table/vanna-agents"
    },
    {
      "Effect": "Allow",
      "Action": [
        "ecs:CreateService",
        "ecs:UpdateService",
        "ecs:DeleteService",
        "ecs:DescribeServices",
        "ecs:ListServices",
        "ecs:ListTasks",
        "ecs:RunTask",
        "ecs:StopTask",
        "ecs:DescribeTasks"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "ec2:DescribeNetworkInterfaces"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "iam:PassRole"
      ],
      "Resource": "*"
    }
  ]
}
EOF

# 步驟 4: 附加 Policy
aws iam put-role-policy \
  --role-name VannaECSTaskRole \
  --policy-name VannaTaskPolicy \
  --policy-document file://vanna-task-policy.json

# 步驟 5: 附加 Task Execution Policy
aws iam attach-role-policy \
  --role-name VannaECSTaskRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
```

## 步驟 1: 建置和推送 Docker Images

```bash
# 進入專案目錄
cd multi-vanna

# 登入 ECR (替換 YOUR_ACCOUNT_ID)
aws ecr get-login-password --region ap-northeast-1 | \
  docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.ap-northeast-1.amazonaws.com

# 建置並推送 Manager
docker build -f docker/Dockerfile.manager -t vanna-manager:latest .
docker tag vanna-manager:latest YOUR_ACCOUNT_ID.dkr.ecr.ap-northeast-1.amazonaws.com/vanna-manager:latest
docker push YOUR_ACCOUNT_ID.dkr.ecr.ap-northeast-1.amazonaws.com/vanna-manager:latest

# 建置並推送 Agent
docker build -f docker/Dockerfile.agent -t vanna-agent:latest .
docker tag vanna-agent:latest YOUR_ACCOUNT_ID.dkr.ecr.ap-northeast-1.amazonaws.com/vanna-agent:latest
docker push YOUR_ACCOUNT_ID.dkr.ecr.ap-northeast-1.amazonaws.com/vanna-agent:latest
```

## 步驟 2: 建立 CloudWatch Log Groups

```bash
aws logs create-log-group --log-group-name /ecs/vanna-manager --region ap-northeast-1
aws logs create-log-group --log-group-name /ecs/vanna-agent --region ap-northeast-1
```

## 步驟 3: 註冊 Task Definitions

修改 `deploy/ecs/task-definition-manager.json` 和 `deploy/ecs/task-definition-agent.json`：
- 替換 `YOUR_ACCOUNT_ID` 為你的 AWS Account ID
- 替換 `YOUR_EFS_ID` 為你的 EFS File System ID
- 填入 Azure OpenAI 配置

```bash
aws ecs register-task-definition --cli-input-json file://deploy/ecs/task-definition-manager.json
aws ecs register-task-definition --cli-input-json file://deploy/ecs/task-definition-agent.json
```

## 步驟 4: 部署 Manager Service

```bash
# 查詢你的 Subnet 和 Security Group
aws ec2 describe-subnets --query 'Subnets[*].[SubnetId,VpcId,AvailabilityZone]' --output table
aws ec2 describe-security-groups --query 'SecurityGroups[*].[GroupId,GroupName]' --output table

# 建立 Service (替換 subnet 和 security group)
aws ecs create-service \
  --cluster vanna-cluster \
  --service-name vanna-manager \
  --task-definition vanna-manager \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"
```

## 步驟 5: 測試

```bash
# 取得 Manager 的 Public IP
TASK_ARN=$(aws ecs list-tasks --cluster vanna-cluster --service-name vanna-manager --query 'taskArns[0]' --output text)
aws ecs describe-tasks --cluster vanna-cluster --tasks $TASK_ARN --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' --output text

# 用 ENI ID 查詢 Public IP
aws ec2 describe-network-interfaces --network-interface-ids eni-xxx --query 'NetworkInterfaces[0].Association.PublicIp' --output text

# 測試 API
curl http://<PUBLIC_IP>:8100/api/agents

# 開啟管理介面
open http://<PUBLIC_IP>:8100/admin/agents
```

## 透過 Manager 建立 Agent

```bash
curl -X POST http://<MANAGER_IP>:8100/api/agents \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "my_agent",
    "description": "My SQL Agent",
    "postgres_user": "postgres",
    "postgres_password": "your_password",
    "postgres_host": "your_db_host",
    "postgres_port": "5432",
    "postgres_db": "your_database"
  }'
```

Manager 會自動：
1. 驗證配置
2. 儲存到 DynamoDB
3. 建立 ECS Service
4. Agent 自動啟動

## 成本估算

| 資源 | 規格 | 預估成本 |
|------|------|----------|
| ECS Fargate | 0.5 vCPU, 1GB | ~$15/月/容器 |
| EFS | 按使用量 | ~$0.30/GB/月 |
| DynamoDB | 按需計費 | ~$5/月 |
| **總計** | 1 Manager + 3 Agents | **~$70-80/月** |

## 常見問題

### Q: 為什麼 ECS 上沒有權限問題？
使用 IAM Task Role，容器啟動時 AWS 自動注入臨時 credentials，boto3 會自動使用。

### Q: 如何更新環境變數？
1. 修改 task-definition JSON
2. 重新註冊: `aws ecs register-task-definition ...`
3. 強制更新: `aws ecs update-service --cluster vanna-cluster --service vanna-manager --force-new-deployment`

### Q: 如何查看 Logs？
```bash
aws logs tail /ecs/vanna-manager --follow
aws logs tail /ecs/vanna-agent --follow
```
