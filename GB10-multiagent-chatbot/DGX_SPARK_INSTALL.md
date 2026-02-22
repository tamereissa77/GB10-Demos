# DGX Spark Installation Guide

This guide is specifically for installing the NVIDIA Multi-Agent Chatbot on your **DGX Spark machine**.

## 🎯 Overview

You'll deploy a complete multi-agent chatbot system on your DGX Spark with:
- Supervisor agent (gpt-oss-120B) orchestrating specialized agents
- Code generation, RAG, and vision understanding capabilities
- Web UI accessible from your browser
- All running locally on your DGX Spark hardware

## ⚡ Quick Installation

### Prerequisites Check

On your DGX Spark, verify:

```bash
# Check Docker
docker --version
docker compose version

# Check GPU
nvidia-smi

# Check available memory (need 128GB)
free -h

# Check disk space (need 200GB+)
df -h
```

## 📥 Step 1: Transfer Files to DGX Spark

You have the files on your Windows machine at `d:\vscodes\GB10\multi-agent-chatbot\`

**Option A: Using SCP (Recommended)**

From your Windows machine (PowerShell or WSL):

```bash
# Replace with your DGX Spark details
scp -r "d:\vscodes\GB10\multi-agent-chatbot" username@dgx-spark-ip:~/
```

**Option B: Using Git**

On your DGX Spark:

```bash
cd ~
git clone https://github.com/NVIDIA/dgx-spark-playbooks.git
cd dgx-spark-playbooks/nvidia/multi-agent-chatbot/assets
```

**Option C: Using SFTP/FileZilla**

Use an SFTP client to transfer the `multi-agent-chatbot` folder to your DGX Spark home directory.

## 🚀 Step 2: Configure Docker Permissions

SSH into your DGX Spark and run:

```bash
# Test Docker access
docker ps

# If permission denied, add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Verify
docker ps
```

## 📦 Step 3: Download Models

Navigate to the project directory:

```bash
cd ~/multi-agent-chatbot
# Or if using git clone:
# cd ~/dgx-spark-playbooks/nvidia/multi-agent-chatbot/assets
```

Make the script executable and run it:

```bash
chmod +x model_download.sh
./model_download.sh
```

**What's downloading:**
- gpt-oss-120B: ~63GB (Supervisor agent)
- Deepseek-Coder-6.7B: ~7GB (Code generation)
- Qwen3-Embedding-4B: ~4GB (Document embeddings)

**Total: ~74GB**

⏱️ **Time**: 30 minutes to 2 hours depending on network speed

**Monitor progress**: The script shows download progress for each file.

## 🐳 Step 4: Start All Services

Ensure no other GPU workloads are running:

```bash
# Check GPU usage
nvidia-smi

# If other processes are using GPU, stop them first
```

Start the services:

```bash
docker compose -f docker-compose.yml -f docker-compose-models.yml up -d --build
```

⏱️ **Time**: 10-20 minutes for first build

**What's happening:**
1. Building llama.cpp server image
2. Pulling required Docker images
3. Starting all containers
4. Loading models into GPU memory

## 📊 Step 5: Monitor Container Status

Watch containers become healthy:

```bash
watch 'docker ps --format "table {{.ID}}\t{{.Names}}\t{{.Status}}"'
```

Press `Ctrl+C` to exit when all show "healthy" or "Up" status.

**Expected containers:**
- ✅ frontend (healthy)
- ✅ backend (healthy)
- ✅ gpt-oss-120b (Up)
- ✅ deepseek-coder (Up)
- ✅ qwen2.5-vl (healthy)
- ✅ qwen3-embedding (Up)
- ✅ postgres (healthy)
- ✅ milvus-standalone (healthy)
- ✅ milvus-etcd (healthy)
- ✅ milvus-minio (healthy)

## 🌐 Step 6: Access the Application

### Option A: Direct Access (if you have GUI on DGX Spark)

Open browser on DGX Spark:
```
http://localhost:3000
```

### Option B: SSH Tunnel (Recommended for remote access)

From your Windows machine, create SSH tunnel:

```bash
ssh -L 3000:localhost:3000 -L 8000:localhost:8000 username@dgx-spark-ip
```

Then open browser on your Windows machine:
```
http://localhost:3000
```

### Option C: Direct IP Access (if firewall allows)

Open browser on any machine:
```
http://dgx-spark-ip:3000
```

**Note**: You may need to configure firewall rules on DGX Spark:
```bash
sudo ufw allow 3000/tcp
sudo ufw allow 8000/tcp
```

## ✅ Step 7: Verify Installation

### Test the System

1. **General Query** (Supervisor Agent):
   - Type: "What is machine learning?"
   - Should get a comprehensive response

2. **Code Generation** (Coding Agent):
   - Type: "Write a Python function to calculate factorial"
   - Should generate working code

3. **Document Analysis** (RAG Agent):
   - Click "Upload Documents" in sidebar
   - Upload a PDF (e.g., [NVIDIA Blackwell Whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf))
   - Check the box in "Select Sources"
   - Ask: "Summarize the key features"

4. **Image Understanding** (Vision Agent):
   - Upload an image
   - Ask: "What do you see in this image?"

### Check System Health

```bash
# Check all containers
docker ps

# Check backend health
curl http://localhost:8000/health

# Check GPU usage
nvidia-smi

# Check logs if issues
docker logs backend
docker logs frontend
```

## 🎮 Usage Tips for DGX Spark

### Memory Management

DGX Spark has 128GB unified memory. This setup uses ~120GB:

```bash
# Monitor memory usage
watch -n 1 'free -h && echo "---" && nvidia-smi'

# If memory issues, flush cache
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

### Performance Optimization

```bash
# Check GPU utilization
nvidia-smi dmon -s u

# Monitor container resources
docker stats

# View detailed GPU info
nvidia-smi -q
```

### Using Smaller Model (if needed)

If you need to free up memory, switch to gpt-oss-20B:

1. **Stop services**:
   ```bash
   docker compose -f docker-compose.yml -f docker-compose-models.yml down
   ```

2. **Edit `model_download.sh`**:
   - Comment out the three gpt-oss-120b lines
   - Uncomment the gpt-oss-20b line
   - Run: `./model_download.sh`

3. **Edit `docker-compose-models.yml`**:
   - Comment out the `gpt-oss-120b` service block
   - Uncomment the `gpt-oss-20b` service block

4. **Edit `docker-compose.yml`**:
   - Change `MODELS=gpt-oss-120b` to `MODELS=gpt-oss-20b`

5. **Restart**:
   ```bash
   docker compose -f docker-compose.yml -f docker-compose-models.yml up -d --build
   ```

## 🔧 DGX Spark Specific Commands

### System Information

```bash
# Check DGX Spark system info
cat /etc/dgx-release

# Check CUDA version
nvcc --version

# Check driver version
nvidia-smi --query-gpu=driver_version --format=csv

# Check memory architecture
numactl --hardware
```

### Maintenance

```bash
# Restart all services
docker compose -f docker-compose.yml -f docker-compose-models.yml restart

# View all logs
docker compose -f docker-compose.yml -f docker-compose-models.yml logs -f

# Stop all services
docker compose -f docker-compose.yml -f docker-compose-models.yml down

# Complete cleanup
docker compose -f docker-compose.yml -f docker-compose-models.yml down -v
docker system prune -a
```

## 🐛 Troubleshooting

### Container Won't Start

```bash
# Check logs
docker logs <container-name>

# Check GPU availability
nvidia-smi

# Restart specific container
docker restart <container-name>

# Rebuild everything
docker compose -f docker-compose.yml -f docker-compose-models.yml up -d --build --force-recreate
```

### Out of Memory

```bash
# Check memory
free -h
nvidia-smi

# Flush cache (DGX Spark UMA)
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'

# Check what's using memory
ps aux --sort=-%mem | head -20

# Consider using smaller model (gpt-oss-20b)
```

### Model Loading Slow

```bash
# Check GPU utilization
nvidia-smi

# Check container logs
docker logs gpt-oss-120b

# First load is always slower (models loading into memory)
# Subsequent starts are faster
```

### Network Issues

```bash
# Check if ports are available
netstat -tulpn | grep -E '3000|8000'

# Check firewall
sudo ufw status

# Allow ports if needed
sudo ufw allow 3000/tcp
sudo ufw allow 8000/tcp
```

### Permission Issues

```bash
# Fix Docker permissions
sudo chmod 666 /var/run/docker.sock

# Or ensure user is in docker group
groups $USER
sudo usermod -aG docker $USER
newgrp docker
```

## 📊 Monitoring Dashboard

Create a monitoring script:

```bash
# Create monitor.sh
cat > monitor.sh << 'EOF'
#!/bin/bash
while true; do
    clear
    echo "=== DGX Spark Multi-Agent Chatbot Status ==="
    echo ""
    echo "=== GPU Status ==="
    nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader
    echo ""
    echo "=== Container Status ==="
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    echo ""
    echo "=== Memory Usage ==="
    free -h | grep -E 'Mem|Swap'
    echo ""
    echo "Press Ctrl+C to exit"
    sleep 5
done
EOF

chmod +x monitor.sh
./monitor.sh
```

## 🔐 Security Considerations

### For Production Deployment

1. **Change default passwords** in `docker-compose.yml`:
   ```yaml
   POSTGRES_PASSWORD=your_secure_password
   MINIO_SECRET_KEY=your_secure_key
   ```

2. **Enable firewall**:
   ```bash
   sudo ufw enable
   sudo ufw allow from trusted_ip to any port 3000
   sudo ufw allow from trusted_ip to any port 8000
   ```

3. **Use HTTPS** (add reverse proxy like nginx)

4. **Regular updates**:
   ```bash
   docker compose pull
   docker compose up -d
   ```

## 📈 Performance Benchmarks

Expected performance on DGX Spark:

| Agent | First Token | Tokens/sec | Memory Usage |
|-------|-------------|------------|--------------|
| Supervisor (120B) | ~2-3s | 10-15 | ~63GB |
| Coding (6.7B) | ~0.5s | 30-50 | ~7GB |
| RAG (4B) | ~0.3s | 40-60 | ~4GB |
| Vision (7B) | ~1-2s | 20-30 | ~14GB |

## 🎓 Next Steps

Once running successfully:

1. **Explore Agents**: Try different types of queries
2. **Upload Documents**: Test RAG with your own PDFs
3. **Customize Prompts**: Edit `backend/prompts.py`
4. **Add New Agents**: Create MCP servers in `backend/tools/mcp_servers/`
5. **Monitor Performance**: Use the monitoring dashboard
6. **Optimize Settings**: Adjust model parameters in `docker-compose-models.yml`

## 📞 Support

### Quick Commands

```bash
# Health check
curl http://localhost:8000/health

# View API docs
curl http://localhost:8000/docs

# Check logs
docker logs -f backend

# Restart everything
docker compose -f docker-compose.yml -f docker-compose-models.yml restart
```

### Resources

- [DGX Spark Documentation](https://docs.nvidia.com/dgx/)
- [Original Repository](https://github.com/NVIDIA/dgx-spark-playbooks)
- [SETUP_GUIDE.md](SETUP_GUIDE.md) - Detailed reference
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Command cheat sheet

## ✅ Installation Checklist

- [ ] SSH access to DGX Spark
- [ ] Docker permissions configured
- [ ] Models downloaded (~74GB)
- [ ] All containers running and healthy
- [ ] Frontend accessible (http://localhost:3000)
- [ ] Backend healthy (curl http://localhost:8000/health)
- [ ] All agents responding to queries
- [ ] GPU memory usage acceptable (~120GB)

## 🎉 Success!

Your DGX Spark is now running a sophisticated multi-agent AI system!

**Access**: http://localhost:3000 (or via SSH tunnel)

**Monitor**: `docker ps` and `nvidia-smi`

**Logs**: `docker logs <container-name>`

---

**Optimized for**: NVIDIA DGX Spark
**Memory Usage**: ~120GB / 128GB
**Status**: Production Ready
