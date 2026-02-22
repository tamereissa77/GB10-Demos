# NVIDIA Multi-Agent Chatbot - Complete Setup Guide

This is a complete implementation of the NVIDIA DGX Spark Multi-Agent Chatbot system. This guide will help you set up and run the application.

## 📋 Overview

This multi-agent chatbot system includes:
- **Supervisor Agent**: Powered by gpt-oss-120B, orchestrates specialized downstream agents
- **Coding Agent**: Uses Deepseek-Coder for code generation tasks
- **RAG Agent**: Retrieval-Augmented Generation for document-based queries
- **Vision Agent**: Image understanding using Qwen2.5-VL
- **Full-stack application**: FastAPI backend + Next.js frontend
- **Vector Database**: Milvus for document embeddings
- **Model Serving**: llama.cpp and TensorRT-LLM servers

## 🎯 What You'll Build

A fully functional multi-agent chatbot system with:
- Web-based UI accessible at http://localhost:3000
- Multiple specialized AI agents working together
- Document upload and RAG capabilities
- Image understanding capabilities
- GPU-accelerated inference

## 📦 Prerequisites

### Hardware Requirements
- **NVIDIA DGX Spark** (or compatible NVIDIA GPU with sufficient VRAM)
- ~120GB of unified memory for default configuration
- Sufficient disk space (~80GB for models)

### Software Requirements
- **Docker** and **Docker Compose**
- **NVIDIA Container Toolkit** (for GPU support in Docker)
- **Git**
- **Linux/Unix environment** (WSL2 on Windows, or native Linux)

### Network Requirements
- Internet connection for downloading models (~74GB total)
- Open ports: 3000 (frontend), 8000 (backend), 5432 (postgres), 19530 (milvus)

## 🚀 Quick Start

### Step 1: Configure Docker Permissions

Ensure you can run Docker without sudo:

```bash
# Test Docker access
docker ps

# If permission denied, add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

### Step 2: Navigate to Project Directory

```bash
cd d:\vscodes\GB10\multi-agent-chatbot
```

### Step 3: Download Models

This will download:
- gpt-oss-120B (~63GB) - Supervisor agent
- Deepseek-Coder-6.7B (~7GB) - Coding agent
- Qwen3-Embedding-4B (~4GB) - Embeddings

```bash
chmod +x model_download.sh
./model_download.sh
```

⏱️ **Expected time**: 30 minutes to 2 hours depending on network speed

### Step 4: Start All Services

```bash
docker compose -f docker-compose.yml -f docker-compose-models.yml up -d --build
```

This will:
1. Build the llama.cpp server image
2. Start all model servers
3. Start the backend API server
4. Start the frontend UI
5. Start supporting services (PostgreSQL, Milvus, etcd, MinIO)

⏱️ **Expected time**: 10-20 minutes

### Step 5: Monitor Container Status

Wait for all containers to become healthy:

```bash
watch 'docker ps --format "table {{.ID}}\t{{.Names}}\t{{.Status}}"'
```

Press `Ctrl+C` to exit the watch command once all containers show "healthy" status.

### Step 6: Access the Application

Open your browser and navigate to:
```
http://localhost:3000
```

**For Remote Access via SSH:**
If running on a remote machine, create SSH tunnels:
```bash
ssh -L 3000:localhost:3000 -L 8000:localhost:8000 username@remote-ip
```

### Step 7: Try Sample Prompts

#### General Queries (Supervisor Agent)
- "What is the capital of France?"
- "Explain quantum computing in simple terms"

#### Coding Agent
- "Write a Python function to calculate fibonacci numbers"
- "Create a REST API endpoint using FastAPI"

#### RAG Agent
1. First, upload a PDF document:
   - Download the [NVIDIA Blackwell Whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)
   - Click "Upload Documents" in the left sidebar
   - Select the downloaded PDF
   - Check the box in "Select Sources"

2. Then ask questions about the document:
   - "What are the key features of the Blackwell architecture?"
   - "Summarize the main improvements in this GPU"

#### Vision Agent
- Upload an image and ask: "What do you see in this image?"
- "Describe the objects in this picture"

## 📁 Project Structure

```
multi-agent-chatbot/
├── backend/                    # FastAPI backend
│   ├── agent.py               # Agent orchestration logic
│   ├── client.py              # LLM client implementations
│   ├── main.py                # FastAPI application
│   ├── vector_store.py        # Milvus integration
│   ├── tools/                 # MCP servers
│   │   └── mcp_servers/
│   │       ├── code_generation.py
│   │       ├── rag.py
│   │       └── image_understanding.py
│   └── Dockerfile
├── frontend/                   # Next.js frontend
│   ├── src/
│   │   ├── app/
│   │   └── components/
│   └── Dockerfile
├── docker-compose.yml         # Main services
├── docker-compose-models.yml  # Model servers
├── Dockerfile.llamacpp        # llama.cpp server image
├── model_download.sh          # Model download script
└── models/                    # Downloaded models (created after Step 3)
```

## 🔧 Configuration Options

### Using Smaller Models (gpt-oss-20B instead of 120B)

If you have memory constraints, you can use the smaller 20B model:

1. **Edit `model_download.sh`**:
   - Comment out the three gpt-oss-120b download lines
   - Uncomment the gpt-oss-20b download line

2. **Edit `docker-compose-models.yml`**:
   - Comment out the `gpt-oss-120b` service block
   - Uncomment the `gpt-oss-20b` service block

3. **Edit `docker-compose.yml`**:
   - Change `MODELS=gpt-oss-120b` to `MODELS=gpt-oss-20b`

### Customizing Model Parameters

Edit `docker-compose-models.yml` to adjust:
- `-n`: Context length
- `--n-gpu-layers`: Number of layers to offload to GPU
- `--port`: Service port

## 🛠️ Troubleshooting

### Container Health Issues

Check container logs:
```bash
docker logs <container-name>
```

Common containers:
- `backend` - Backend API server
- `gpt-oss-120b` - Supervisor model
- `deepseek-coder` - Coding agent
- `qwen2.5-vl` - Vision agent
- `milvus-standalone` - Vector database

### Memory Issues

If you encounter OOM errors:

1. **Flush buffer cache** (DGX Spark UMA):
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

2. **Switch to smaller model** (see Configuration Options above)

3. **Check GPU memory**:
```bash
nvidia-smi
```

### Port Conflicts

If ports are already in use, modify the port mappings in `docker-compose.yml`:
```yaml
ports:
  - "3001:3000"  # Change host port (left side)
```

### Model Download Issues

If downloads fail or are interrupted:
- The script supports resume with `curl -C -`
- Simply re-run `./model_download.sh`
- For gated models, ensure you have HuggingFace access token

### Cannot Access Frontend

1. Verify backend is running:
```bash
curl http://localhost:8000/health
```

2. Check frontend logs:
```bash
docker logs frontend
```

3. Ensure all dependencies are healthy:
```bash
docker ps --filter "health=unhealthy"
```

## 🧹 Cleanup

### Stop All Services
```bash
docker compose -f docker-compose.yml -f docker-compose-models.yml down
```

### Remove Volumes (Complete Cleanup)
```bash
docker compose -f docker-compose.yml -f docker-compose-models.yml down -v
docker volume rm multi-agent-chatbot_postgres_data
```

### Remove Downloaded Models
```bash
rm -rf models/
```

## 🔄 Updating the Application

To update to the latest version:

```bash
# Pull latest changes
git pull origin main

# Rebuild containers
docker compose -f docker-compose.yml -f docker-compose-models.yml up -d --build
```

## 📊 System Architecture

### Services Overview

1. **Frontend (Next.js)**: Port 3000
   - User interface
   - Document upload
   - Chat interface

2. **Backend (FastAPI)**: Port 8000
   - Agent orchestration
   - API endpoints
   - MCP server management

3. **Model Servers**:
   - `gpt-oss-120b`: Supervisor agent (llama.cpp)
   - `deepseek-coder`: Code generation (llama.cpp)
   - `qwen3-embedding`: Document embeddings (llama.cpp)
   - `qwen2.5-vl`: Vision understanding (TensorRT-LLM)

4. **Supporting Services**:
   - PostgreSQL: Conversation storage
   - Milvus: Vector database for RAG
   - etcd: Milvus metadata
   - MinIO: Object storage for Milvus

## 🎓 Next Steps

1. **Experiment with different prompts** to understand agent capabilities
2. **Upload your own documents** for RAG queries
3. **Add custom MCP servers** in `backend/tools/mcp_servers/`
4. **Modify agent prompts** in `backend/prompts.py`
5. **Customize the UI** in `frontend/src/components/`

## 📚 Additional Resources

- [NVIDIA DGX Spark Documentation](https://docs.nvidia.com/dgx/)
- [Model Context Protocol (MCP)](https://modelcontextprotocol.io/)
- [llama.cpp Documentation](https://github.com/ggerganov/llama.cpp)
- [TensorRT-LLM Documentation](https://github.com/NVIDIA/TensorRT-LLM)
- [Milvus Documentation](https://milvus.io/docs)

## ⚠️ Important Notes

- This system uses ~120GB of memory by default
- Ensure no other GPU workloads are running
- Model downloads are large - ensure sufficient bandwidth and storage
- First startup takes longer due to model loading
- Some models may take 1-2 minutes to become ready

## 🐛 Reporting Issues

If you encounter issues:
1. Check the troubleshooting section above
2. Review container logs
3. Verify system requirements
4. Check the [original repository](https://github.com/NVIDIA/dgx-spark-playbooks) for updates

## 📝 License

This project is licensed under Apache 2.0. See the original repository for details.

---

**Last Updated**: January 2025
**Version**: Based on NVIDIA DGX Spark Playbooks
