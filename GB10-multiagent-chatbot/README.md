# NVIDIA Multi-Agent Chatbot System

A complete implementation of NVIDIA's DGX Spark Multi-Agent Chatbot - a sophisticated AI system featuring multiple specialized agents working together to handle diverse tasks including coding, document analysis, and image understanding.

![Multi-Agent System](assets/multi-agent-chatbot.png)

## 🌟 Features

- **🤖 Multi-Agent Architecture**: Supervisor agent orchestrates specialized downstream agents
- **💻 Code Generation**: Powered by Deepseek-Coder for intelligent code assistance
- **📚 RAG (Retrieval-Augmented Generation)**: Upload documents and query them with AI
- **👁️ Vision Understanding**: Analyze and describe images using Qwen2.5-VL
- **⚡ GPU Acceleration**: Optimized for NVIDIA GPUs with unified memory architecture
- **🎨 Modern UI**: Clean, responsive Next.js frontend
- **🔌 Extensible**: Easy to add new agents via Model Context Protocol (MCP)

## 📋 What's Included

This repository contains:

- ✅ Complete backend API (FastAPI)
- ✅ Modern frontend UI (Next.js + React)
- ✅ Docker Compose configuration for all services
- ✅ Model download scripts
- ✅ Pre-configured agent implementations
- ✅ Vector database setup (Milvus)
- ✅ Comprehensive documentation

## 🚀 Quick Start

### Prerequisites

- NVIDIA GPU with sufficient VRAM (128GB recommended for full setup)
- Docker & Docker Compose
- NVIDIA Container Toolkit
- ~80GB free disk space for models
- Linux environment (or WSL2 on Windows)

### Installation

```bash
# 1. Navigate to project directory
cd multi-agent-chatbot

# 2. Download models (~74GB, 30min-2hrs depending on network)
chmod +x model_download.sh
./model_download.sh

# 3. Start all services (10-20 minutes)
docker compose -f docker-compose.yml -f docker-compose-models.yml up -d --build

# 4. Wait for containers to be healthy
watch 'docker ps --format "table {{.ID}}\t{{.Names}}\t{{.Status}}"'

# 5. Access the application
# Open browser: http://localhost:3000
```

That's it! 🎉

## 📚 Documentation

We've prepared comprehensive guides for different scenarios:

### 📖 Main Guides

- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Complete setup instructions for Linux/DGX Spark
- **[WINDOWS_SETUP.md](WINDOWS_SETUP.md)** - Windows-specific setup using WSL2
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Command cheat sheet and troubleshooting

### 🎯 Choose Your Path

| Your Situation | Read This |
|----------------|-----------|
| Running on NVIDIA DGX Spark | [SETUP_GUIDE.md](SETUP_GUIDE.md) |
| Running on Windows PC | [WINDOWS_SETUP.md](WINDOWS_SETUP.md) |
| Need quick commands | [QUICK_REFERENCE.md](QUICK_REFERENCE.md) |
| Want to customize | See [Architecture](#architecture) below |

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                         Frontend (Next.js)                   │
│                      http://localhost:3000                   │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    Backend API (FastAPI)                     │
│                      http://localhost:8000                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Supervisor Agent (gpt-oss-120B)         │  │
│  │                                                       │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │  │
│  │  │   Coding    │  │     RAG     │  │   Vision    │ │  │
│  │  │   Agent     │  │    Agent    │  │   Agent     │ │  │
│  │  │ (Deepseek)  │  │  (Qwen3)    │  │ (Qwen2.5)   │ │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  PostgreSQL  │    │    Milvus    │    │  Model       │
│  (Storage)   │    │  (Vectors)   │    │  Servers     │
└──────────────┘    └──────────────┘    └──────────────┘
```

### Services

| Service | Technology | Purpose |
|---------|-----------|---------|
| Frontend | Next.js 14 | User interface |
| Backend | FastAPI | API & agent orchestration |
| Supervisor | gpt-oss-120B (llama.cpp) | Main orchestration agent |
| Code Agent | Deepseek-Coder (llama.cpp) | Code generation |
| RAG Agent | Qwen3-Embedding (llama.cpp) | Document embeddings |
| Vision Agent | Qwen2.5-VL (TensorRT-LLM) | Image understanding |
| Database | PostgreSQL | Conversation storage |
| Vector DB | Milvus | Document embeddings |
| Object Storage | MinIO | Milvus backend |
| Metadata | etcd | Milvus metadata |

## 🎮 Usage Examples

### 1. General Conversation

Simply type your question:
```
"What is the capital of France?"
"Explain quantum computing"
```

### 2. Code Generation

Ask for code help:
```
"Write a Python function to calculate fibonacci numbers"
"Create a REST API endpoint using FastAPI"
"Debug this code: [paste your code]"
```

### 3. Document Analysis (RAG)

1. Click "Upload Documents" in the sidebar
2. Upload a PDF file
3. Check the box in "Select Sources"
4. Ask questions:
```
"Summarize the main points of the document"
"What does the document say about [topic]?"
```

### 4. Image Understanding

1. Upload an image
2. Ask questions:
```
"What objects are in this image?"
"Describe this picture in detail"
```

## 🔧 Configuration

### Using Smaller Models

If you have memory constraints, switch to gpt-oss-20B:

1. Edit `model_download.sh`:
   - Comment out gpt-oss-120b downloads
   - Uncomment gpt-oss-20b download

2. Edit `docker-compose-models.yml`:
   - Comment out `gpt-oss-120b` service
   - Uncomment `gpt-oss-20b` service

3. Edit `docker-compose.yml`:
   - Change `MODELS=gpt-oss-120b` to `MODELS=gpt-oss-20b`

### Custom Agents

Add new agents by creating MCP servers in `backend/tools/mcp_servers/`:

```python
# backend/tools/mcp_servers/my_agent.py
from mcp.server import Server

server = Server("my-agent")

@server.tool()
async def my_tool(query: str) -> str:
    # Your agent logic here
    return result
```

## 🛠️ Development

### Project Structure

```
multi-agent-chatbot/
├── backend/                    # FastAPI backend
│   ├── agent.py               # Agent orchestration
│   ├── client.py              # LLM client wrappers
│   ├── main.py                # FastAPI app
│   ├── vector_store.py        # Milvus integration
│   ├── tools/                 # MCP servers
│   │   └── mcp_servers/
│   │       ├── code_generation.py
│   │       ├── rag.py
│   │       └── image_understanding.py
│   └── Dockerfile
├── frontend/                   # Next.js frontend
│   ├── src/
│   │   ├── app/               # Pages
│   │   └── components/        # React components
│   └── Dockerfile
├── docker-compose.yml         # Main services
├── docker-compose-models.yml  # Model servers
├── Dockerfile.llamacpp        # llama.cpp image
├── model_download.sh          # Model downloader
└── models/                    # Downloaded models
```

### Running in Development Mode

```bash
# Backend (with hot reload)
cd backend
pip install -e .
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Frontend (with hot reload)
cd frontend
npm install
npm run dev
```

## 🧪 Testing

```bash
# Test backend health
curl http://localhost:8000/health

# Test chat endpoint
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "conversation_id": "test"}'

# View API documentation
open http://localhost:8000/docs
```

## 🐛 Troubleshooting

### Common Issues

**Containers won't start:**
```bash
docker logs <container-name>
docker compose -f docker-compose.yml -f docker-compose-models.yml restart
```

**Out of memory:**
```bash
# Check memory
free -h
nvidia-smi

# Flush cache (DGX Spark)
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

**Port conflicts:**
```bash
# Find process using port
lsof -i :3000
lsof -i :8000

# Kill process or change port in docker-compose.yml
```

See [QUICK_REFERENCE.md](QUICK_REFERENCE.md) for more troubleshooting tips.

## 🧹 Cleanup

```bash
# Stop all services
docker compose -f docker-compose.yml -f docker-compose-models.yml down

# Remove volumes (complete cleanup)
docker compose -f docker-compose.yml -f docker-compose-models.yml down -v

# Remove downloaded models
rm -rf models/
```

## ��� System Requirements

### Minimum Requirements
- NVIDIA GPU with 64GB+ VRAM
- 64GB+ RAM
- 200GB free disk space
- Docker with GPU support

### Recommended (DGX Spark)
- 128GB unified memory
- 200GB+ free disk space
- High-speed internet for model downloads

## 🤝 Contributing

This is an implementation of NVIDIA's DGX Spark Playbooks. For contributions:

1. Fork the [original repository](https://github.com/NVIDIA/dgx-spark-playbooks)
2. Create a feature branch
3. Submit a pull request

## 📄 License

Apache 2.0 - See LICENSE file for details.

This project is based on [NVIDIA DGX Spark Playbooks](https://github.com/NVIDIA/dgx-spark-playbooks).

## 🔗 Resources

- [NVIDIA DGX Spark Documentation](https://docs.nvidia.com/dgx/)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [Milvus Vector Database](https://milvus.io/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Next.js](https://nextjs.org/)

## 🙏 Acknowledgments

- NVIDIA for the DGX Spark platform and playbooks
- The open-source AI community
- Contributors to llama.cpp, TensorRT-LLM, and Milvus

## 📞 Support

- **Documentation**: See guides in this repository
- **Issues**: Check [original repository issues](https://github.com/NVIDIA/dgx-spark-playbooks/issues)
- **Community**: NVIDIA Developer Forums

## 🗺️ Roadmap

- [ ] Add more specialized agents
- [ ] Support for additional model formats
- [ ] Enhanced UI features
- [ ] Performance optimizations
- [ ] Multi-user support
- [ ] Cloud deployment guides

---

**Built with ❤️ for the AI community**

**Last Updated**: January 2025

**Status**: ✅ Production Ready

For detailed setup instructions, see [SETUP_GUIDE.md](SETUP_GUIDE.md)
