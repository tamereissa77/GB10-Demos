# Quick Reference Guide

## 🚀 Quick Start Commands

```bash
# Navigate to project
cd d:\vscodes\GB10\multi-agent-chatbot

# Download models (first time only)
chmod +x model_download.sh
./model_download.sh

# Start all services
docker compose -f docker-compose.yml -f docker-compose-models.yml up -d --build

# Check status
docker ps

# View logs
docker logs -f backend
docker logs -f frontend

# Stop all services
docker compose -f docker-compose.yml -f docker-compose-models.yml down

# Complete cleanup
docker compose -f docker-compose.yml -f docker-compose-models.yml down -v
```

## 🌐 Access Points

| Service | URL | Description |
|---------|-----|-------------|
| Frontend UI | http://localhost:3000 | Main chat interface |
| Backend API | http://localhost:8000 | API endpoints |
| API Docs | http://localhost:8000/docs | Swagger documentation |
| PostgreSQL | localhost:5432 | Database |
| Milvus | localhost:19530 | Vector database |

## 🐳 Container Names

| Container | Purpose | Port |
|-----------|---------|------|
| `frontend` | Next.js UI | 3000 |
| `backend` | FastAPI server | 8000 |
| `gpt-oss-120b` | Supervisor agent | 8000 (internal) |
| `deepseek-coder` | Code generation | 8000 (internal) |
| `qwen2.5-vl` | Vision understanding | 8000 (internal) |
| `qwen3-embedding` | Document embeddings | 8000 (internal) |
| `postgres` | Database | 5432 |
| `milvus-standalone` | Vector DB | 19530 |
| `milvus-etcd` | Milvus metadata | 2379 |
| `milvus-minio` | Object storage | 9000 |

## 📊 Useful Docker Commands

```bash
# View all containers
docker ps -a

# View container logs
docker logs <container-name>

# Follow logs in real-time
docker logs -f <container-name>

# View last 100 lines
docker logs --tail 100 <container-name>

# Restart a specific container
docker restart <container-name>

# Stop a specific container
docker stop <container-name>

# Execute command in container
docker exec -it <container-name> bash

# View container resource usage
docker stats

# Remove stopped containers
docker container prune

# Remove unused images
docker image prune

# View networks
docker network ls

# Inspect container
docker inspect <container-name>
```

## 🔍 Debugging Commands

```bash
# Check GPU availability
nvidia-smi

# Check GPU in Docker
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi

# Check disk space
df -h

# Check memory usage
free -h

# Check Docker disk usage
docker system df

# View Docker events
docker events

# Check container health
docker ps --filter "health=unhealthy"

# Inspect specific container
docker inspect <container-name> | grep -A 10 Health
```

## 📝 Log Locations

```bash
# Backend logs
docker logs backend

# Frontend logs
docker logs frontend

# Model server logs
docker logs gpt-oss-120b
docker logs deepseek-coder
docker logs qwen2.5-vl
docker logs qwen3-embedding

# Database logs
docker logs postgres

# Milvus logs
docker logs milvus-standalone
```

## 🛠️ Common Issues & Fixes

### Container Won't Start

```bash
# Check logs
docker logs <container-name>

# Restart container
docker restart <container-name>

# Rebuild and restart
docker compose -f docker-compose.yml -f docker-compose-models.yml up -d --build --force-recreate
```

### Out of Memory

```bash
# Check memory
free -h

# Flush cache (DGX Spark)
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'

# Check Docker memory
docker stats
```

### Port Already in Use

```bash
# Find process using port
lsof -i :3000
lsof -i :8000

# Kill process
kill -9 <PID>

# Or change port in docker-compose.yml
```

### GPU Not Available

```bash
# Check GPU
nvidia-smi

# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi

# Restart Docker
sudo systemctl restart docker
```

## 📦 Model Information

| Model | Size | Purpose | Container |
|-------|------|---------|-----------|
| gpt-oss-120B | ~63GB | Supervisor agent | gpt-oss-120b |
| Deepseek-Coder-6.7B | ~7GB | Code generation | deepseek-coder |
| Qwen3-Embedding-4B | ~4GB | Document embeddings | qwen3-embedding |
| Qwen2.5-VL-7B | Downloaded on-demand | Vision understanding | qwen2.5-vl |

## 🔄 Update & Maintenance

```bash
# Pull latest changes
git pull origin main

# Rebuild containers
docker compose -f docker-compose.yml -f docker-compose-models.yml up -d --build

# Clean up old images
docker image prune -a

# Backup database
docker exec postgres pg_dump -U chatbot_user chatbot > backup.sql

# Restore database
docker exec -i postgres psql -U chatbot_user chatbot < backup.sql
```

## 🧪 Testing Endpoints

```bash
# Health check
curl http://localhost:8000/health

# List conversations
curl http://localhost:8000/conversations

# Test model endpoint
curl http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "conversation_id": "test"}'
```

## 📊 Monitoring

```bash
# Real-time container status
watch 'docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"'

# Real-time resource usage
watch 'docker stats --no-stream'

# Real-time GPU usage
watch -n 1 nvidia-smi

# Disk usage
watch 'df -h'
```

## 🔐 Environment Variables

Key environment variables in `docker-compose.yml`:

```yaml
# Backend
POSTGRES_HOST=postgres
POSTGRES_DB=chatbot
POSTGRES_USER=chatbot_user
POSTGRES_PASSWORD=chatbot_password
ETCD_ENDPOINTS=etcd:2379
MINIO_ADDRESS=minio:9000
MILVUS_ADDRESS=milvus:19530
MODELS=gpt-oss-120b

# PostgreSQL
POSTGRES_DB=chatbot
POSTGRES_USER=chatbot_user
POSTGRES_PASSWORD=chatbot_password

# MinIO
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
```

## 📁 Important Files

```
multi-agent-chatbot/
├── docker-compose.yml              # Main services configuration
├── docker-compose-models.yml       # Model servers configuration
├── Dockerfile.llamacpp             # llama.cpp server image
├── model_download.sh               # Model download script
├── backend/
│   ├── main.py                     # FastAPI application
│   ├── agent.py                    # Agent orchestration
│   ├── client.py                   # LLM clients
│   ├── vector_store.py             # Milvus integration
│   └── tools/mcp_servers/          # MCP server implementations
├── frontend/
│   └── src/
│       ├── app/                    # Next.js pages
│       └── components/             # React components
└── models/                         # Downloaded model files
```

## 🎯 Sample Prompts

### General (Supervisor)
- "What is machine learning?"
- "Explain the difference between AI and ML"

### Coding
- "Write a Python function to sort a list"
- "Create a REST API with FastAPI"
- "Debug this code: [paste code]"

### RAG (after uploading documents)
- "Summarize the uploaded document"
- "What are the key points in the document?"
- "Find information about [topic] in the document"

### Vision (with image upload)
- "What objects are in this image?"
- "Describe this picture in detail"
- "What colors are dominant in this image?"

## 🔗 Useful Links

- [Original Repository](https://github.com/NVIDIA/dgx-spark-playbooks)
- [Docker Documentation](https://docs.docker.com/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Next.js Documentation](https://nextjs.org/docs)
- [Milvus Documentation](https://milvus.io/docs)

## 💡 Tips

1. **First startup is slow** - Models need to load into memory
2. **Use `watch` commands** - Monitor status in real-time
3. **Check logs first** - Most issues are visible in logs
4. **GPU memory is shared** - Close other GPU applications
5. **Models are cached** - Subsequent starts are faster
6. **Use smaller models** - If memory constrained, use gpt-oss-20b
7. **SSH tunneling** - For remote access, tunnel ports 3000 and 8000

## 🆘 Getting Help

1. Check logs: `docker logs <container-name>`
2. Check health: `docker ps`
3. Check resources: `docker stats` and `nvidia-smi`
4. Review setup guides: `SETUP_GUIDE.md` and `WINDOWS_SETUP.md`
5. Check original repo: [GitHub Issues](https://github.com/NVIDIA/dgx-spark-playbooks/issues)

---

**Quick Help**: For most issues, try restarting containers:
```bash
docker compose -f docker-compose.yml -f docker-compose-models.yml restart
```
