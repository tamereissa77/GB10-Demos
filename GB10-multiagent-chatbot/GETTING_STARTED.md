# Getting Started Checklist

Welcome! This checklist will guide you through setting up the NVIDIA Multi-Agent Chatbot system.

## 📋 Pre-Installation Checklist

### ✅ Step 1: Verify Your Environment

**Choose your platform:**

- [ ] **Option A: Linux/DGX Spark** → Follow [SETUP_GUIDE.md](SETUP_GUIDE.md)
- [ ] **Option B: Windows** → Follow [WINDOWS_SETUP.md](WINDOWS_SETUP.md)
- [ ] **Option C: Cloud/Remote** → See [WINDOWS_SETUP.md](WINDOWS_SETUP.md) Option 3

### ✅ Step 2: Check Hardware Requirements

- [ ] NVIDIA GPU available
- [ ] At least 64GB RAM (128GB recommended)
- [ ] At least 200GB free disk space
- [ ] GPU drivers installed

**Verify GPU:**
```bash
nvidia-smi
```

### ✅ Step 3: Install Prerequisites

#### For Linux/DGX Spark:
- [ ] Docker installed
- [ ] Docker Compose installed
- [ ] NVIDIA Container Toolkit installed
- [ ] Git installed

**Quick test:**
```bash
docker --version
docker compose version
nvidia-smi
git --version
```

#### For Windows:
- [ ] WSL2 installed
- [ ] Docker Desktop installed (with WSL2 backend)
- [ ] NVIDIA drivers installed on Windows
- [ ] Ubuntu distribution in WSL2

**Quick test (in WSL2):**
```bash
docker --version
nvidia-smi
```

### ✅ Step 4: Configure Docker

- [ ] Docker daemon running
- [ ] User added to docker group (Linux)
- [ ] GPU accessible in Docker

**Test GPU in Docker:**
```bash
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

If this works, you're ready! ✨

## 🚀 Installation Checklist

### ✅ Step 5: Get the Code

- [ ] Navigate to project directory

```bash
cd d:\vscodes\GB10\multi-agent-chatbot
```

**Or for WSL2:**
```bash
cd /mnt/d/vscodes/GB10/multi-agent-chatbot
```

### ✅ Step 6: Download Models

- [ ] Make script executable
- [ ] Run download script
- [ ] Wait for completion (30min - 2hrs)

```bash
chmod +x model_download.sh
./model_download.sh
```

**What's downloading:**
- gpt-oss-120B: ~63GB (Supervisor agent)
- Deepseek-Coder: ~7GB (Code generation)
- Qwen3-Embedding: ~4GB (Document embeddings)

**Total: ~74GB**

### ✅ Step 7: Start Services

- [ ] Run docker compose command
- [ ] Wait for build (10-20 minutes)
- [ ] Verify all containers are running

```bash
docker compose -f docker-compose.yml -f docker-compose-models.yml up -d --build
```

### ✅ Step 8: Verify Installation

- [ ] Check container status
- [ ] Wait for all containers to be "healthy"

```bash
docker ps --format "table {{.Names}}\t{{.Status}}"
```

**Expected containers:**
- ✅ frontend (healthy)
- ✅ backend (healthy)
- ✅ gpt-oss-120b (running)
- ✅ deepseek-coder (running)
- ✅ qwen2.5-vl (healthy)
- ✅ qwen3-embedding (running)
- ✅ postgres (healthy)
- ✅ milvus-standalone (healthy)
- ✅ milvus-etcd (healthy)
- ✅ milvus-minio (healthy)

### ✅ Step 9: Access the Application

- [ ] Open browser
- [ ] Navigate to http://localhost:3000
- [ ] See the chat interface

**For remote access:**
```bash
ssh -L 3000:localhost:3000 -L 8000:localhost:8000 user@remote-ip
```

### ✅ Step 10: Test the System

- [ ] Try a general question
- [ ] Test code generation
- [ ] Upload a document and test RAG
- [ ] Upload an image and test vision

**Sample prompts:**

1. **General**: "What is machine learning?"
2. **Coding**: "Write a Python function to reverse a string"
3. **RAG**: Upload PDF → "Summarize this document"
4. **Vision**: Upload image → "What's in this image?"

## 🎉 Success Criteria

You've successfully installed the system if:

- ✅ All containers are running and healthy
- ✅ Frontend loads at http://localhost:3000
- ✅ You can send messages and get responses
- ✅ Different agents respond to different types of queries

## 🐛 Troubleshooting Checklist

### If containers won't start:

- [ ] Check Docker is running: `docker ps`
- [ ] Check logs: `docker logs <container-name>`
- [ ] Verify GPU access: `nvidia-smi`
- [ ] Check disk space: `df -h`
- [ ] Check memory: `free -h`

### If frontend won't load:

- [ ] Check backend is running: `curl http://localhost:8000/health`
- [ ] Check frontend logs: `docker logs frontend`
- [ ] Check browser console for errors
- [ ] Try different browser

### If getting errors:

- [ ] Check all containers are healthy: `docker ps`
- [ ] Review logs: `docker logs backend`
- [ ] Check GPU memory: `nvidia-smi`
- [ ] Restart containers: `docker compose restart`

### If out of memory:

- [ ] Close other applications
- [ ] Check GPU memory: `nvidia-smi`
- [ ] Consider using smaller model (gpt-oss-20b)
- [ ] Flush cache: `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'`

## 📚 Next Steps

Once everything is working:

1. **Explore the UI**
   - [ ] Try different sample prompts
   - [ ] Upload various documents
   - [ ] Test image understanding
   - [ ] Explore conversation history

2. **Learn the System**
   - [ ] Read [SETUP_GUIDE.md](SETUP_GUIDE.md) for details
   - [ ] Review [QUICK_REFERENCE.md](QUICK_REFERENCE.md) for commands
   - [ ] Check [README.md](README.md) for architecture

3. **Customize**
   - [ ] Modify agent prompts in `backend/prompts.py`
   - [ ] Add custom MCP servers in `backend/tools/mcp_servers/`
   - [ ] Customize UI in `frontend/src/components/`

4. **Optimize**
   - [ ] Adjust model parameters in `docker-compose-models.yml`
   - [ ] Configure memory allocation
   - [ ] Tune performance settings

## 🆘 Need Help?

### Quick Fixes

**Most common issues can be fixed by:**
```bash
# Restart all containers
docker compose -f docker-compose.yml -f docker-compose-models.yml restart

# Or rebuild everything
docker compose -f docker-compose.yml -f docker-compose-models.yml up -d --build --force-recreate
```

### Documentation

- **Setup Issues**: [SETUP_GUIDE.md](SETUP_GUIDE.md)
- **Windows Issues**: [WINDOWS_SETUP.md](WINDOWS_SETUP.md)
- **Commands**: [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
- **Architecture**: [README.md](README.md)

### Support Resources

- Check container logs: `docker logs <container-name>`
- Review [QUICK_REFERENCE.md](QUICK_REFERENCE.md) troubleshooting section
- Check [original repository](https://github.com/NVIDIA/dgx-spark-playbooks/issues)

## 📊 Installation Time Estimates

| Step | Time | Notes |
|------|------|-------|
| Prerequisites | 30-60 min | If not already installed |
| Model Download | 30 min - 2 hrs | Depends on internet speed |
| Docker Build | 10-20 min | First time only |
| Container Startup | 5-10 min | Models loading into memory |
| **Total** | **1-4 hours** | Mostly waiting for downloads |

## ✨ Tips for Success

1. **Be Patient**: First startup takes time as models load
2. **Check Logs**: Most issues are visible in container logs
3. **Monitor Resources**: Keep an eye on GPU and RAM usage
4. **Start Small**: Test with simple prompts first
5. **Read Docs**: The guides have detailed troubleshooting

## 🎯 Quick Command Reference

```bash
# Check status
docker ps

# View logs
docker logs -f backend

# Restart services
docker compose -f docker-compose.yml -f docker-compose-models.yml restart

# Stop everything
docker compose -f docker-compose.yml -f docker-compose-models.yml down

# Complete cleanup
docker compose -f docker-compose.yml -f docker-compose-models.yml down -v
rm -rf models/
```

## 📝 Installation Log

Keep track of your installation:

```
Date Started: _______________
Platform: [ ] Linux  [ ] Windows/WSL2  [ ] Cloud
GPU Model: _______________
RAM: _______________
Issues Encountered: _______________
Resolution: _______________
Date Completed: _______________
```

---

**Ready to start?** Begin with Step 1 above! 🚀

**Questions?** Check the documentation or container logs.

**Success?** Start exploring the multi-agent system! 🎉
