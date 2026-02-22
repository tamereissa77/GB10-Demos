#!/bin/bash
#
# Dell GB10 Deployment Script
# Automated deployment for Dell GB10 Multi-Agent Chatbot
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Check if running on Dell GB10
check_system() {
    print_header "Checking System Requirements"
    
    # Check GPU
    if command -v nvidia-smi &> /dev/null; then
        print_success "NVIDIA GPU detected"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    else
        print_error "NVIDIA GPU not detected"
        exit 1
    fi
    
    # Check Docker
    if command -v docker &> /dev/null; then
        print_success "Docker installed: $(docker --version)"
    else
        print_error "Docker not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if docker compose version &> /dev/null; then
        print_success "Docker Compose installed: $(docker compose version)"
    else
        print_error "Docker Compose not installed"
        exit 1
    fi
    
    # Check architecture
    ARCH=$(uname -m)
    print_info "Architecture: $ARCH"
    
    # Check memory
    TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
    print_info "Total Memory: ${TOTAL_MEM}GB"
    
    if [ "$TOTAL_MEM" -lt 64 ]; then
        print_warning "Memory is less than 64GB. Consider using smaller models."
    fi
    
    # Check disk space
    DISK_SPACE=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    print_info "Available Disk Space: ${DISK_SPACE}GB"
    
    if [ "$DISK_SPACE" -lt 200 ]; then
        print_warning "Less than 200GB free space. May not be enough for models."
    fi
    
    echo ""
}

# Check Docker permissions
check_docker_permissions() {
    print_header "Checking Docker Permissions"
    
    if docker ps &> /dev/null; then
        print_success "Docker permissions OK"
    else
        print_warning "Docker permission denied. Adding user to docker group..."
        sudo usermod -aG docker $USER
        print_info "Please log out and log back in, then run this script again"
        exit 0
    fi
    
    echo ""
}

# Download models
download_models() {
    print_header "Downloading Models"
    
    if [ -f "model_download.sh" ]; then
        chmod +x model_download.sh
        
        # Check if models already exist
        if [ -d "models" ] && [ "$(ls -A models/*.gguf 2>/dev/null)" ]; then
            print_warning "Models directory exists with GGUF files"
            read -p "Do you want to re-download models? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                print_info "Skipping model download"
                return
            fi
        fi
        
        print_info "Starting model download (~74GB, this may take 30min-2hrs)..."
        ./model_download.sh
        print_success "Models downloaded successfully"
    else
        print_error "model_download.sh not found"
        exit 1
    fi
    
    echo ""
}

# Check GPU availability
check_gpu() {
    print_header "Checking GPU Availability"
    
    # Check if GPU is free
    GPU_PROCESSES=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
    
    if [ "$GPU_PROCESSES" -gt 0 ]; then
        print_warning "GPU is currently in use by $GPU_PROCESSES process(es)"
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 0
        fi
    else
        print_success "GPU is available"
    fi
    
    echo ""
}

# Start services
start_services() {
    print_header "Starting Docker Services"
    
    print_info "Building and starting containers..."
    docker compose -f docker-compose.yml -f docker-compose-models.yml up -d --build
    
    print_success "Containers started"
    echo ""
}

# Monitor startup
monitor_startup() {
    print_header "Monitoring Container Startup"
    
    print_info "Waiting for containers to become healthy..."
    print_info "This may take 10-20 minutes for first startup"
    echo ""
    
    # Wait for containers
    sleep 10
    
    # Show container status
    docker ps --format "table {{.Names}}\t{{.Status}}"
    
    echo ""
    print_info "To monitor logs in real-time, run:"
    echo "  docker compose -f docker-compose.yml -f docker-compose-models.yml logs -f"
    echo ""
}

# Verify deployment
verify_deployment() {
    print_header "Verifying Deployment"
    
    # Wait a bit for services to start
    sleep 5
    
    # Check backend health
    print_info "Checking backend health..."
    if curl -s http://localhost:8001/health > /dev/null 2>&1; then
        print_success "Backend is healthy"
    else
        print_warning "Backend not responding yet (may still be starting)"
    fi
    
    # Check frontend
    print_info "Checking frontend..."
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        print_success "Frontend is accessible"
    else
        print_warning "Frontend not responding yet (may still be starting)"
    fi
    
    # Check GPU usage
    print_info "GPU Memory Usage:"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
    
    echo ""
}

# Print access information
print_access_info() {
    print_header "Access Information"
    
    echo -e "${GREEN}✓ Deployment Complete!${NC}"
    echo ""
    echo "Access the application:"
    echo "  • Local: http://localhost:3000"
    echo "  • Network: http://$(hostname -I | awk '{print $1}'):3000"
    echo ""
    echo "API Documentation:"
    echo "  • http://localhost:8001/docs"
    echo ""
    echo "Useful commands:"
    echo "  �� View logs: docker compose -f docker-compose.yml -f docker-compose-models.yml logs -f"
    echo "  • Check status: docker ps"
    echo "  • Check GPU: nvidia-smi"
    echo "  • Stop services: docker compose -f docker-compose.yml -f docker-compose-models.yml down"
    echo ""
    print_info "Note: Models load on first request, so first responses may be slower"
    echo ""
}

# Main deployment flow
main() {
    clear
    print_header "Dell GB10 Multi-Agent Chatbot Deployment"
    echo ""
    
    # Run checks
    check_system
    check_docker_permissions
    check_gpu
    
    # Ask for confirmation
    read -p "Proceed with deployment? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Deployment cancelled"
        exit 0
    fi
    
    # Download models
    download_models
    
    # Start services
    start_services
    
    # Monitor startup
    monitor_startup
    
    # Verify deployment
    verify_deployment
    
    # Print access info
    print_access_info
}

# Run main function
main
