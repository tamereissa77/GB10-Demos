# Dell GB10 Deployment Guide

## 🎯 Optimized for Dell GB10 Machine

This guide ensures the multi-agent chatbot runs perfectly on your Dell GB10 machine.

---

## 📋 Dell GB10 Specifications

### Expected Hardware
- **GPU**: NVIDIA GPU with unified memory architecture
- **Memory**: 128GB unified memory (CPU + GPU shared)
- **Architecture**: ARM64 or x86_64
- **OS**: Linux (Ubuntu 22.04 or similar)
- **Docker**: With NVIDIA Container Toolkit

---

## ✅ Pre-Deployment Checklist

### 1. Verify System Requirements

```bash
# Check GPU
nvidia-smi

# Check memory
free -h

# Check disk space (need 200GB+)
df -h

# Check Docker
docker --version
docker compose version

# Check NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

### 2. Verify Architecture

```bash
# Check CPU architecture
uname -m

# Should return: aarch64 (ARM64) or x86_64
```

---

## 🔧 Configuration Adjustments

### Architecture-Specific Settings

The configuration is already optimized for Dell GB10, but verify:

#### For ARM64 (aarch64):
- ��� Milvus image: `milvusdb/milvus:v2.5.15-20250718-3a3b374f-gpu-arm64`
- ✅ This is already configured in `docker-compose.yml`

#### For x86_64:
If your Dell GB10 is x86_64, update Milvus image:

```bash
# Edit docker-compose.yml
# Change line 119 from:
# image: milvusdb/milvus:v2.5.15-20250718-3a3b374f-gpu-arm64
# To:
# image: milvusdb/milvus:v2.5.15-20250718-3a3b374f-gpu
```

---

## 🚀 Deployment Steps

### Step 1: Transfer Files to Dell GB10

**From your Windows machine:**

```powershell
# Using SCP
scp -r "d:\vscodes\GB10\multi-agent-chatbot" username@dell-gb10-ip:~/

# Or using Git on Dell GB10
ssh username@dell-gb10-ip
git clone https://github.com/tamereissa77/GB10-multiagent-chatbot.git
cd GB10-multiagent-chatbot
```

### Step 2: Configure Docker Permissions

```bash
# On Dell GB10
sudo usermod -aG docker $USER
newgrp docker

# Verify
docker ps
```

### Step 3: Download Models

```bash
cd ~/GB10-multiagent-chatbot  # or ~/multi-agent-chatbot
chmod +x model_download.sh
./model_download.sh
```

**Models to download (~74GB total):**
- gpt-oss-120B: ~63GB
- Deepseek-Coder: ~7GB
- Qwen3-Embedding: ~4GB

**Time**: 30 minutes to 2 hours depending on network

### Step 4: Verify GPU Access

```bash
# Ensure no other GPU workloads
nvidia-smi

# Check GPU memory is available
nvidia-smi --query-gpu=memory.free --format=csv
```

### Step 5: Start Services

```bash
# Start all services
docker compose -f docker-compose.yml -f docker-compose-models.yml up -d --build

# Monitor startup
docker compose -f docker-compose.yml -f docker-compose-models.yml logs -f
```

**Expected startup time**: 10-20 minutes

### Step 6: Monitor Container Health

```bash
# Watch containers become healthy
watch 'docker ps --format "table {{.Names}}\t{{.Status}}"'

# Or check specific container
docker logs -f backend
docker logs -f gpt-oss-120b
```

### Step 7: Verify Deployment

```bash
# Check all containers are running
docker ps

# Test backend health
curl http://localhost:8000/health

# Expected response: {"status":"healthy"}
```

---

## 🎮 Access the Application

### Option 1: SSH Tunnel (Recommended)

From your Windows machine:

```bash
ssh -L 3000:localhost:3000 -L 8000:localhost:8000 username@dell-gb10-ip
```

Then open: **http://localhost:3000**

### Option 2: Direct Access

If Dell GB10 has GUI:
```
http://localhost:3000
```

### Option 3: Network Access

Configure firewall and access via IP:

```bash
# On Dell GB10
sudo ufw allow 3000/tcp
sudo ufw allow 8000/tcp
```

Then from any machine: **http://dell-gb10-ip:3000**

---

## 🔍 Container Status Reference

### Expected Containers (10 total)

| Container | Status | Port | Purpose |
|-----------|--------|------|---------|
| frontend | healthy | 3000 | Next.js UI |
| backend | running | 8000 | FastAPI server |
| gpt-oss-120b | running | 8000 | Supervisor agent |
| deepseek-coder | running | 8000 | Code generation |
| qwen2.5-vl | healthy | 8000 | Vision agent |
| qwen3-embedding | running | 8000 | Embeddings |
| postgres | healthy | 5432 | Database |
| milvus-standalone | healthy | 19530 | Vector DB |
| milvus-etcd | running | 2379 | Metadata |
| milvus-minio | healthy | 9000 | Object storage |

### Check Container Status

```bash
# All containers
docker ps

# Specific container logs
docker logs <container-name>

# Follow logs
docker logs -f <container-name>

# Container resource usage
docker stats
```

---

## 💾 Memory Management for Dell GB10

### Unified Memory Architecture

Dell GB10 uses unified memory (CPU + GPU share 128GB):

```bash
# Monitor memory usage
watch -n 1 'free -h && echo "---" && nvidia-smi'

# If memory issues occur, flush cache
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

### Memory Usage Breakdown

| Component | Memory Usage |
|-----------|--------------|
| gpt-oss-120b | ~63GB |
| qwen2.5-vl | ~14GB |
| deepseek-coder | ~7GB |
| qwen3-embedding | ~4GB |
| Milvus + Services | ~10GB |
| System overhead | ~10GB |
| **Total** | **~108GB / 128GB** |

**Buffer**: ~20GB remaining for operations

---

## ⚙️ Performance Optimization

### 1. GPU Utilization

```bash
# Monitor GPU usage
nvidia-smi dmon -s u

# Check GPU processes
nvidia-smi pmon
```

### 2. Container Resource Limits

If needed, add resource limits to `docker-compose.yml`:

```yaml
services:
  backend:
    deploy:
      resources:
        limits:
          memory: 4G
```

### 3. Model Loading Optimization

Models load on first request. To pre-warm:

```bash
# Test each model endpoint
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "conversation_id": "test"}'
```

---

## 🐛 Troubleshooting

### Issue 1: Container Won't Start

```bash
# Check logs
docker logs <container-name>

# Common fixes
docker compose -f docker-compose.yml -f docker-compose-models.yml restart
docker system prune -a
```

### Issue 2: Out of Memory

```bash
# Check memory
free -h
nvidia-smi

# Flush cache
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'

# Consider using gpt-oss-20b instead of 120b
# See SETUP_GUIDE.md for instructions
```

### Issue 3: GPU Not Detected

```bash
# Verify NVIDIA drivers
nvidia-smi

# Verify Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi

# Restart Docker
sudo systemctl restart docker
```

### Issue 4: Milvus Architecture Mismatch

```bash
# Check your architecture
uname -m

# If x86_64, update docker-compose.yml:
# Change Milvus image to: milvusdb/milvus:v2.5.15-20250718-3a3b374f-gpu
```

### Issue 5: Port Conflicts

```bash
# Check if ports are in use
netstat -tulpn | grep -E '3000|8000|5432|19530'

# Kill conflicting processes or change ports in docker-compose.yml
```

---

## 🔄 Maintenance Commands

### Start/Stop Services

```bash
# Start
docker compose -f docker-compose.yml -f docker-compose-models.yml up -d

# Stop
docker compose -f docker-compose.yml -f docker-compose-models.yml down

# Restart
docker compose -f docker-compose.yml -f docker-compose-models.yml restart

# Rebuild
docker compose -f docker-compose.yml -f docker-compose-models.yml up -d --build
```

### Update Application

```bash
# Pull latest changes
git pull origin main

# Rebuild and restart
docker compose -f docker-compose.yml -f docker-compose-models.yml up -d --build
```

### Clean Up

```bash
# Stop and remove containers
docker compose -f docker-compose.yml -f docker-compose-models.yml down

# Remove volumes (complete cleanup)
docker compose -f docker-compose.yml -f docker-compose-models.yml down -v

# Remove unused Docker resources
docker system prune -a
```

---

## 📊 Performance Benchmarks

### Expected Performance on Dell GB10

| Metric | Value |
|--------|-------|
| First token latency (120B) | 2-3s |
| Tokens/sec (120B) | 10-15 |
| First token latency (6.7B) | 0.5s |
| Tokens/sec (6.7B) | 30-50 |
| Document embedding | 0.3s |
| Image understanding | 1-2s |

---

## 🔐 Security Recommendations

### For Production Deployment

1. **Change Default Passwords**

Edit `docker-compose.yml`:
```yaml
environment:
  - POSTGRES_PASSWORD=your_secure_password_here
  - MINIO_SECRET_KEY=your_secure_key_here
```

2. **Enable Firewall**

```bash
sudo ufw enable
sudo ufw allow from trusted_ip to any port 3000
sudo ufw allow from trusted_ip to any port 8000
```

3. **Use HTTPS**

Set up reverse proxy (nginx) with SSL certificates.

---

## ✅ Deployment Verification Checklist

- [ ] All 10 containers running
- [ ] All containers show "healthy" or "Up" status
- [ ] Frontend accessible at http://localhost:3000
- [ ] Backend health check passes: `curl http://localhost:8000/health`
- [ ] GPU memory usage ~108GB (check with `nvidia-smi`)
- [ ] Can send test message and receive response
- [ ] All agents (coding, RAG, vision) respond correctly
- [ ] Document upload works
- [ ] No error messages in logs

---

## 📞 Support

### Quick Diagnostics

```bash
# Full system check
echo "=== Docker Status ===" && docker ps && \
echo "=== GPU Status ===" && nvidia-smi && \
echo "=== Memory Status ===" && free -h && \
echo "=== Disk Status ===" && df -h
```

### Log Collection

```bash
# Collect all logs
docker compose -f docker-compose.yml -f docker-compose-models.yml logs > deployment-logs.txt

# Share logs for troubleshooting
```

---

## 🎯 Success Criteria

Your deployment is successful when:

✅ All containers are running and healthy
✅ Frontend loads with Dell branding
✅ Can send messages and receive responses
✅ All four agent types work (general, code, RAG, vision)
✅ GPU memory usage is stable (~108GB)
✅ No critical errors in logs

---

## 📝 Notes

- **First startup**: Takes 10-20 minutes as models load
- **Subsequent starts**: Much faster (2-5 minutes)
- **Model loading**: Happens on first request to each agent
- **Memory**: Monitor with `nvidia-smi` and `free -h`
- **Updates**: Pull from Git and rebuild containers

---

**Optimized for**: Dell GB10
**Architecture**: ARM64/x86_64 compatible
**Memory**: 128GB unified memory
**Status**: Production Ready
**Last Updated**: January 2025
