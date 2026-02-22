# DGX Spark Quick Start Card

## 🎯 For DGX Spark Deployment

---

## Step 1: Transfer Files to DGX Spark

**From Windows (PowerShell):**
```powershell
scp -r "d:\vscodes\GB10\multi-agent-chatbot" username@dgx-spark-ip:~/
```

**Or use Git on DGX Spark:**
```bash
git clone https://github.com/NVIDIA/dgx-spark-playbooks.git
cd dgx-spark-playbooks/nvidia/multi-agent-chatbot/assets
```

---

## Step 2: SSH into DGX Spark

```bash
ssh username@dgx-spark-ip
cd ~/multi-agent-chatbot
```

---

## Step 3: Configure Docker

```bash
# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Verify
docker ps
```

---

## Step 4: Download Models

```bash
# Make executable
chmod +x model_download.sh

# Download (~74GB, 30min-2hrs)
./model_download.sh
```

**Models being downloaded:**
- gpt-oss-120B: ~63GB
- Deepseek-Coder: ~7GB
- Qwen3-Embedding: ~4GB

---

## Step 5: Start Services

```bash
# Ensure no other GPU workloads
nvidia-smi

# Start everything (10-20min)
docker compose -f docker-compose.yml -f docker-compose-models.yml up -d --build
```

---

## Step 6: Monitor Status

```bash
# Watch containers become healthy
watch 'docker ps --format "table {{.Names}}\t{{.Status}}"'

# Press Ctrl+C when all are healthy
```

**Expected containers (10 total):**
- frontend, backend
- gpt-oss-120b, deepseek-coder, qwen2.5-vl, qwen3-embedding
- postgres, milvus-standalone, milvus-etcd, milvus-minio

---

## Step 7: Access the UI

**Option A: SSH Tunnel (Recommended)**

From Windows:
```bash
ssh -L 3000:localhost:3000 -L 8000:localhost:8000 username@dgx-spark-ip
```

Then open: **http://localhost:3000**

**Option B: Direct IP**

Configure firewall on DGX Spark:
```bash
sudo ufw allow 3000/tcp
sudo ufw allow 8000/tcp
```

Then open: **http://dgx-spark-ip:3000**

---

## ✅ Verify Installation

```bash
# Check containers
docker ps

# Check backend health
curl http://localhost:8000/health

# Check GPU usage
nvidia-smi

# Check memory
free -h
```

---

## 🎮 Test the Agents

1. **General**: "What is machine learning?"
2. **Coding**: "Write a Python function to reverse a string"
3. **RAG**: Upload PDF → "Summarize this document"
4. **Vision**: Upload image → "What's in this image?"

---

## 🔧 Useful Commands

```bash
# View logs
docker logs -f backend

# Restart services
docker compose -f docker-compose.yml -f docker-compose-models.yml restart

# Stop services
docker compose -f docker-compose.yml -f docker-compose-models.yml down

# Monitor GPU
watch -n 1 nvidia-smi

# Monitor containers
docker stats
```

---

## 🐛 Quick Troubleshooting

**Out of memory:**
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

**Container won't start:**
```bash
docker logs <container-name>
docker restart <container-name>
```

**Slow performance:**
```bash
nvidia-smi  # Check GPU usage
ps aux | grep python  # Check for other processes
```

---

## 📚 Full Documentation

For detailed instructions, see:
- **DGX_SPARK_INSTALL.md** - Complete guide
- **QUICK_REFERENCE.md** - Command reference
- **README.md** - Architecture overview

---

## ⏱️ Time Expectations

- File Transfer: 5-15 min
- Docker Setup: 5-10 min
- Model Download: 30 min - 2 hrs
- Container Build: 10-20 min
- First Startup: 5-10 min

**Total: 1-4 hours** (mostly automated)

---

## 💡 Pro Tips

1. ✅ Ensure no other GPU workloads before starting
2. ✅ Use SSH tunnel for secure remote access
3. ✅ Monitor GPU memory with `nvidia-smi`
4. ✅ First startup is slower (models loading)
5. ✅ Check logs if issues: `docker logs <container>`

---

## 🎯 Success Criteria

- ✅ All 10 containers healthy
- ✅ Frontend loads at localhost:3000
- ✅ Backend responds to health check
- ✅ All agents respond to queries
- ✅ GPU memory ~120GB / 128GB

---

## 📞 Need Help?

1. Check logs: `docker logs <container-name>`
2. Review: **DGX_SPARK_INSTALL.md**
3. Check GPU: `nvidia-smi`
4. Check memory: `free -h`

---

**Status**: Ready for DGX Spark Deployment
**Memory**: ~120GB / 128GB
**Storage**: ~200GB required
