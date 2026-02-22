#!/bin/bash
#
# Dell GB10 Health Check Script
# Monitors the health of all services
#

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

clear

print_header "Dell GB10 Multi-Agent Chatbot Health Check"
echo ""

# Check Docker containers
print_header "Container Status"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "NAME|frontend|backend|gpt-oss|deepseek|qwen|postgres|milvus|etcd|minio"
echo ""

# Count running containers
RUNNING=$(docker ps --filter "name=frontend|backend|gpt-oss|deepseek|qwen|postgres|milvus|etcd|minio" --format "{{.Names}}" | wc -l)
echo "Running containers: $RUNNING / 10"
echo ""

# Check backend health
print_header "Backend Health"
if curl -s http://localhost:8001/health > /dev/null 2>&1; then
    HEALTH=$(curl -s http://localhost:8001/health)
    print_success "Backend: $HEALTH"
else
    print_error "Backend: Not responding"
fi
echo ""

# Check frontend
print_header "Frontend Status"
if curl -s http://localhost:3000 > /dev/null 2>&1; then
    print_success "Frontend: Accessible"
else
    print_error "Frontend: Not accessible"
fi
echo ""

# Check GPU
print_header "GPU Status"
nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader
echo ""

# Check memory
print_header "System Memory"
free -h | grep -E "Mem|Swap"
echo ""

# Check disk space
print_header "Disk Space"
df -h | grep -E "Filesystem|/$|/home"
echo ""

# Check container logs for errors
print_header "Recent Errors (last 10 lines)"
ERROR_COUNT=0

for container in frontend backend gpt-oss-120b deepseek-coder qwen2.5-vl qwen3-embedding; do
    if docker ps --format "{{.Names}}" | grep -q "^${container}$"; then
        ERRORS=$(docker logs --tail 10 $container 2>&1 | grep -i "error" | wc -l)
        if [ "$ERRORS" -gt 0 ]; then
            print_warning "$container: $ERRORS error(s) in last 10 lines"
            ERROR_COUNT=$((ERROR_COUNT + ERRORS))
        fi
    fi
done

if [ "$ERROR_COUNT" -eq 0 ]; then
    print_success "No recent errors detected"
fi
echo ""

# Overall health summary
print_header "Health Summary"

HEALTH_SCORE=0

# Check containers (40 points)
if [ "$RUNNING" -eq 10 ]; then
    HEALTH_SCORE=$((HEALTH_SCORE + 40))
    print_success "All containers running"
elif [ "$RUNNING" -ge 8 ]; then
    HEALTH_SCORE=$((HEALTH_SCORE + 30))
    print_warning "Most containers running ($RUNNING/10)"
else
    print_error "Many containers not running ($RUNNING/10)"
fi

# Check backend (30 points)
if curl -s http://localhost:8001/health > /dev/null 2>&1; then
    HEALTH_SCORE=$((HEALTH_SCORE + 30))
    print_success "Backend healthy"
else
    print_error "Backend not healthy"
fi

# Check frontend (20 points)
if curl -s http://localhost:3000 > /dev/null 2>&1; then
    HEALTH_SCORE=$((HEALTH_SCORE + 20))
    print_success "Frontend accessible"
else
    print_error "Frontend not accessible"
fi

# Check GPU (10 points)
if nvidia-smi > /dev/null 2>&1; then
    HEALTH_SCORE=$((HEALTH_SCORE + 10))
    print_success "GPU accessible"
else
    print_error "GPU not accessible"
fi

echo ""
echo -e "Overall Health Score: ${GREEN}${HEALTH_SCORE}/100${NC}"

if [ "$HEALTH_SCORE" -ge 90 ]; then
    echo -e "${GREEN}Status: Excellent ✓${NC}"
elif [ "$HEALTH_SCORE" -ge 70 ]; then
    echo -e "${YELLOW}Status: Good (minor issues)${NC}"
elif [ "$HEALTH_SCORE" -ge 50 ]; then
    echo -e "${YELLOW}Status: Fair (needs attention)${NC}"
else
    echo -e "${RED}Status: Poor (requires troubleshooting)${NC}"
fi

echo ""
print_header "Quick Actions"
echo "View logs:    docker compose -f docker-compose.yml -f docker-compose-models.yml logs -f"
echo "Restart:      docker compose -f docker-compose.yml -f docker-compose-models.yml restart"
echo "Stop:         docker compose -f docker-compose.yml -f docker-compose-models.yml down"
echo "GPU monitor:  watch -n 1 nvidia-smi"
echo ""
