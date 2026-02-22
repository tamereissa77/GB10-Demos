# Windows Setup Guide for NVIDIA Multi-Agent Chatbot

This guide is specifically for setting up the NVIDIA Multi-Agent Chatbot on Windows systems.

## ⚠️ Important Note

This application is designed for **NVIDIA DGX Spark** running Linux. To run it on Windows, you have two main options:

## Option 1: Using WSL2 (Recommended)

Windows Subsystem for Linux 2 (WSL2) with GPU support is the recommended approach.

### Prerequisites

1. **Windows 11** or **Windows 10** (version 21H2 or higher)
2. **NVIDIA GPU** with latest drivers
3. **WSL2** installed and configured
4. **Docker Desktop** with WSL2 backend

### Step-by-Step Setup

#### 1. Install WSL2

Open PowerShell as Administrator:

```powershell
# Enable WSL
wsl --install

# Or if already installed, update to WSL2
wsl --set-default-version 2

# Install Ubuntu (recommended)
wsl --install -d Ubuntu-22.04
```

Restart your computer if prompted.

#### 2. Install NVIDIA CUDA on WSL2

Inside WSL2 Ubuntu terminal:

```bash
# Update package list
sudo apt update && sudo apt upgrade -y

# Install NVIDIA CUDA Toolkit for WSL
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-3

# Verify CUDA installation
nvidia-smi
```

#### 3. Install Docker Desktop

1. Download [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)
2. Install with WSL2 backend enabled
3. In Docker Desktop settings:
   - Go to **Settings** → **General**
   - Enable "Use the WSL 2 based engine"
   - Go to **Settings** → **Resources** → **WSL Integration**
   - Enable integration with your Ubuntu distribution

#### 4. Install NVIDIA Container Toolkit in WSL2

In WSL2 Ubuntu terminal:

```bash
# Add NVIDIA Container Toolkit repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install NVIDIA Container Toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

#### 5. Access Your Project Files

In WSL2, navigate to your Windows files:

```bash
# Windows drives are mounted under /mnt/
cd /mnt/d/vscodes/GB10/multi-agent-chatbot
```

#### 6. Follow the Main Setup Guide

Now follow the steps in `SETUP_GUIDE.md` starting from Step 3 (Download Models).

### WSL2 Tips

**File Performance**: For better performance, consider copying the project to WSL2's native filesystem:

```bash
# Copy to WSL2 home directory
cp -r /mnt/d/vscodes/GB10/multi-agent-chatbot ~/multi-agent-chatbot
cd ~/multi-agent-chatbot
```

**Accessing from Windows**: WSL2 filesystem is accessible from Windows at:
```
\\wsl$\Ubuntu-22.04\home\<username>\
```

**Memory Allocation**: Configure WSL2 memory in `%USERPROFILE%\.wslconfig`:

```ini
[wsl2]
memory=64GB
processors=8
swap=32GB
```

## Option 2: Using a Linux VM

If WSL2 doesn't work for your setup, use a Linux VM:

### Using VirtualBox or VMware

1. **Install virtualization software**:
   - [VirtualBox](https://www.virtualbox.org/) (Free)
   - [VMware Workstation](https://www.vmware.com/products/workstation-pro.html)

2. **Create Ubuntu VM**:
   - Download [Ubuntu 22.04 LTS](https://ubuntu.com/download/desktop)
   - Allocate at least:
     - 64GB RAM (more if possible)
     - 200GB disk space
     - All CPU cores
   - Enable GPU passthrough (if supported)

3. **Install NVIDIA drivers in VM**:
   ```bash
   sudo apt update
   sudo apt install nvidia-driver-535
   sudo reboot
   ```

4. **Install Docker and NVIDIA Container Toolkit** (see WSL2 instructions above)

5. **Transfer project files** to VM and follow main setup guide

## Option 3: Cloud/Remote Linux Server

Deploy on a cloud GPU instance:

### Recommended Providers

1. **NVIDIA NGC** - Native support for NVIDIA tools
2. **AWS EC2** - G5 or P4 instances
3. **Google Cloud** - A2 instances
4. **Azure** - NC-series VMs
5. **Lambda Labs** - GPU cloud instances

### Setup Steps

1. **Launch GPU instance** with:
   - Ubuntu 22.04
   - NVIDIA GPU (A100, H100, or similar)
   - At least 128GB RAM
   - 200GB+ storage

2. **SSH into instance**:
   ```bash
   ssh -L 3000:localhost:3000 -L 8000:localhost:8000 user@instance-ip
   ```

3. **Install dependencies**:
   ```bash
   # Update system
   sudo apt update && sudo apt upgrade -y
   
   # Install Docker
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo usermod -aG docker $USER
   
   # Install NVIDIA Container Toolkit
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
       sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
       sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   
   # Install Git
   sudo apt install git -y
   ```

4. **Clone and setup**:
   ```bash
   git clone <your-repo-url>
   cd multi-agent-chatbot
   # Follow main setup guide
   ```

## Windows-Specific Considerations

### Path Conversions

When working with paths in WSL2:

**Windows Path**: `D:\vscodes\GB10\multi-agent-chatbot`
**WSL2 Path**: `/mnt/d/vscodes/GB10/multi-agent-chatbot`

### Line Endings

Ensure scripts use Unix line endings (LF, not CRLF):

```bash
# Convert line endings
sudo apt install dos2unix
dos2unix model_download.sh
```

### Firewall

Windows Firewall may block Docker ports. Add exceptions:

```powershell
# Run in PowerShell as Administrator
New-NetFirewallRule -DisplayName "Docker Frontend" -Direction Inbound -LocalPort 3000 -Protocol TCP -Action Allow
New-NetFirewallRule -DisplayName "Docker Backend" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
```

### Performance Tips

1. **Use WSL2 native filesystem** for better I/O performance
2. **Allocate sufficient memory** to WSL2 (see .wslconfig above)
3. **Close unnecessary Windows applications** to free up RAM
4. **Disable Windows Search indexing** for WSL2 directories

## Troubleshooting Windows-Specific Issues

### Docker Desktop Not Starting

1. Ensure Hyper-V is enabled:
   ```powershell
   # Run as Administrator
   Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V -All
   ```

2. Ensure WSL2 is properly installed:
   ```powershell
   wsl --status
   wsl --update
   ```

### GPU Not Detected in WSL2

1. Update NVIDIA drivers on Windows (not in WSL2)
2. Verify GPU is visible in WSL2:
   ```bash
   nvidia-smi
   ```

3. If not visible, reinstall NVIDIA drivers on Windows

### Out of Memory in WSL2

1. Increase WSL2 memory allocation (see .wslconfig above)
2. Restart WSL2:
   ```powershell
   wsl --shutdown
   wsl
   ```

### Permission Denied Errors

```bash
# Fix Docker socket permissions
sudo chmod 666 /var/run/docker.sock

# Or add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

### Slow File Access

If accessing files from `/mnt/` is slow:

```bash
# Copy project to WSL2 native filesystem
cp -r /mnt/d/vscodes/GB10/multi-agent-chatbot ~/
cd ~/multi-agent-chatbot
```

## Verification Checklist

Before running the application, verify:

- [ ] WSL2 is installed and running
- [ ] Docker Desktop is running with WSL2 backend
- [ ] NVIDIA drivers are installed on Windows
- [ ] `nvidia-smi` works in WSL2
- [ ] Docker can access GPU: `docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi`
- [ ] Sufficient disk space (200GB+)
- [ ] Sufficient RAM allocated to WSL2 (64GB+)

## Quick Start Command Summary

Once WSL2 and Docker are set up:

```bash
# Navigate to project
cd /mnt/d/vscodes/GB10/multi-agent-chatbot

# Make script executable
chmod +x model_download.sh

# Download models
./model_download.sh

# Start services
docker compose -f docker-compose.yml -f docker-compose-models.yml up -d --build

# Monitor status
watch 'docker ps --format "table {{.ID}}\t{{.Names}}\t{{.Status}}"'

# Access at http://localhost:3000
```

## Additional Resources

- [WSL2 Installation Guide](https://docs.microsoft.com/en-us/windows/wsl/install)
- [NVIDIA CUDA on WSL2](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
- [Docker Desktop WSL2 Backend](https://docs.docker.com/desktop/windows/wsl/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

## Support

For Windows-specific issues:
1. Check Docker Desktop logs
2. Check WSL2 logs: `wsl --status`
3. Verify GPU access: `nvidia-smi` in WSL2
4. Check Windows Event Viewer for system errors

---

**Note**: This application is optimized for NVIDIA DGX Spark. Running on Windows may have limitations in performance and compatibility. For production use, a native Linux environment is recommended.
