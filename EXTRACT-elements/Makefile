# ============================================================
# Makefile — Arabic Financial Document Extraction Platform
# ============================================================
# Usage:
#   make ingestion      — Build & start the full stack
#   make stop           — Stop all services
#   make restart        — Restart all services
#   make logs           — Tail logs from all services
#   make status         — Show service status
#   make clean          — Stop and remove all containers + volumes
#   make build          — Build all images without starting
#   make rebuild        — Force rebuild and restart
# ============================================================

COMPOSE = docker compose
PROJECT = extraction

.PHONY: ingestion stop restart logs status clean build rebuild \
        build-app build-txt2kg build-ollama build-embeddings \
        logs-app logs-txt2kg logs-ollama logs-kg logs-pg \
        pull shell-app shell-pg shell-arango

# ==============================================================
# PRIMARY TARGETS
# ==============================================================

## Build and start the full ingestion pipeline (all services)
ingestion:
	@echo "=============================================="
	@echo "  Starting Full Ingestion Pipeline"
	@echo "=============================================="
	@echo ""
	@echo "  Services:"
	@echo "    • Streamlit App (Chandra VLM)  → :8501"
	@echo "    • txt2kg Visualization          → :3001"
	@echo "    • PostgreSQL                    → :5432"
	@echo "    • ArangoDB (Knowledge Graph)    → :8529"
	@echo "    • Ollama (LLM)                  → :11434"
	@echo "    • Qdrant (Vector DB)            → :6333"
	@echo "    • Sentence Transformers         → :8000"
	@echo "    • MinIO (Object Store)          → :9000/:9001"
	@echo ""
	$(COMPOSE) up --build -d
	@echo ""
	@echo "=============================================="
	@echo "  All services starting..."
	@echo ""
	@echo "  Open in browser:"
	@echo "    📊 Document Extractor:  http://localhost:8501"
	@echo "    🔗 Knowledge Graph UI:  http://localhost:3001"
	@echo "    🗄️  ArangoDB Console:   http://localhost:8529"
	@echo "    🔍 Qdrant Dashboard:    http://localhost:6333/dashboard"
	@echo "    📦 MinIO Console:       http://localhost:9001"
	@echo ""
	@echo "  Note: First run downloads model weights (~10-15 GB total)"
	@echo "=============================================="

## Stop all services
stop:
	@echo "Stopping all services..."
	$(COMPOSE) down

## Restart all services
restart:
	@echo "Restarting all services..."
	$(COMPOSE) restart

## Show service status
status:
	$(COMPOSE) ps

## Tail logs from all services
logs:
	$(COMPOSE) logs -f --tail=50

## Stop and remove everything (containers, volumes, images)
clean:
	@echo "WARNING: This will delete all data (volumes)!"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	$(COMPOSE) down -v --rmi local
	@echo "Cleaned up."

# ==============================================================
# BUILD TARGETS
# ==============================================================

## Build all images without starting
build:
	$(COMPOSE) build

## Force rebuild all images and restart
rebuild:
	$(COMPOSE) up --build --force-recreate -d

## Build individual services
build-app:
	$(COMPOSE) build arabic-table-extractor

build-txt2kg:
	$(COMPOSE) build txt2kg-frontend

build-ollama:
	$(COMPOSE) build ollama

build-embeddings:
	$(COMPOSE) build sentence-transformers

# ==============================================================
# LOG TARGETS
# ==============================================================

## Tail logs for specific services
logs-app:
	$(COMPOSE) logs -f --tail=100 arabic-table-extractor

logs-txt2kg:
	$(COMPOSE) logs -f --tail=100 txt2kg-frontend

logs-ollama:
	$(COMPOSE) logs -f --tail=100 ollama

logs-kg:
	$(COMPOSE) logs -f --tail=100 arangodb

logs-pg:
	$(COMPOSE) logs -f --tail=100 postgres

logs-vectors:
	$(COMPOSE) logs -f --tail=100 qdrant

logs-minio:
	$(COMPOSE) logs -f --tail=100 minio

# ==============================================================
# SHELL ACCESS
# ==============================================================

## Open a shell in the main app container
shell-app:
	docker exec -it arabic-table-extractor /bin/bash

## Open psql in the PostgreSQL container
shell-pg:
	docker exec -it extraction-postgres psql -U extract_user -d extraction_db

## Open arangosh in the ArangoDB container
shell-arango:
	docker exec -it extraction-arangodb arangosh --server.authentication false --server.database extraction_kg

# ==============================================================
# UTILITY TARGETS
# ==============================================================

## Pull latest base images
pull:
	$(COMPOSE) pull postgres arangodb qdrant minio

## Flush GPU memory (DGX Spark / GB10)
flush-gpu:
	sudo sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
	@echo "GPU memory cache flushed."

## Show disk usage of Docker volumes
disk:
	docker system df -v | head -40

## Push code to GitHub
push:
	git add -A && git commit -m "Update" && git push origin main

# ==============================================================
# HELP
# ==============================================================

## Show this help
help:
	@echo ""
	@echo "Available targets:"
	@echo ""
	@echo "  make ingestion     — Build & start the full stack"
	@echo "  make stop          — Stop all services"
	@echo "  make restart       — Restart all services"
	@echo "  make status        — Show service status"
	@echo "  make logs          — Tail all logs"
	@echo "  make clean         — Remove everything (containers + data)"
	@echo ""
	@echo "  make build         — Build all images"
	@echo "  make rebuild       — Force rebuild + restart"
	@echo "  make build-app     — Build only the Streamlit app"
	@echo "  make build-txt2kg  — Build only the txt2kg frontend"
	@echo ""
	@echo "  make logs-app      — Logs: Streamlit app"
	@echo "  make logs-txt2kg   — Logs: txt2kg frontend"
	@echo "  make logs-ollama   — Logs: Ollama LLM"
	@echo "  make logs-kg       — Logs: ArangoDB"
	@echo "  make logs-pg       — Logs: PostgreSQL"
	@echo ""
	@echo "  make shell-app     — Shell into app container"
	@echo "  make shell-pg      — psql into PostgreSQL"
	@echo "  make shell-arango  — arangosh into ArangoDB"
	@echo ""
	@echo "  make flush-gpu     — Flush GPU memory cache"
	@echo "  make disk          — Show Docker disk usage"
	@echo ""

.DEFAULT_GOAL := help
