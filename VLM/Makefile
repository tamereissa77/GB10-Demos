# Port Definitions
FRONTEND_PORT=5173
CHAT_UI_PORT=5174
BACKEND_PORT=8000
GRAPH_DB_PORT=8529
VECTOR_DB_PORT=6333
NIM_PORT=8010

.PHONY: help ingestion retrieval ingestion-down retrieval-down stop logs ps clean

help:
	@echo "GraphRAG Control Panel"
	@echo "-----------------------"
	@echo "make ingestion       - Start ONLY ingestion containers (DBs + backend + ollama + init + extraction UI)"
	@echo "make retrieval       - Start ONLY retrieval containers (DBs + nim + chat API + chat UI)"
	@echo "make ingestion-down  - Stop only ingestion containers (keeps shared DBs running)"
	@echo "make retrieval-down  - Stop only retrieval containers (keeps shared DBs running)"
	@echo "make stop            - Stop all services (both workflows)"
	@echo "make logs            - Show logs for all running services"
	@echo "make ps              - Show status of all services"
	@echo "make clean           - Stop all and remove volumes"

# Start only ingestion-related services.
# Note: We explicitly list services to avoid accidentally starting retrieval-only containers
# and to avoid tearing down shared DBs when switching workflows.
ingestion:
	@echo "🚀 Starting Ingestion Workflow (DBs + Ollama + OCR Backend + Extraction UI + Arango init)..."
	docker compose up -d arangodb qdrant
	docker compose up -d ollama backend arangodb-init extractionview
	@echo "✅ Ingestion services are up."
	@echo "🔗 Backend API: http://localhost:$(BACKEND_PORT)"
	@echo "🔗 Extraction UI: http://localhost:$(FRONTEND_PORT)"

ingestion-down:
	@echo "🛑 Stopping ingestion services (keeping databases running)..."
	docker compose stop ollama backend arangodb-init extractionview

# Start only retrieval-related services.
retrieval:
	@echo "🚀 Starting Retrieval Workflow (DBs + NIM + Chat UI + Chat API)..."
	docker compose up -d arangodb qdrant
	docker compose up -d nim-llm chatview txt2kg-app
	@echo "✅ Retrieval services are up."
	@echo "🔗 Chat UI: http://localhost:$(CHAT_UI_PORT)"
	@echo "🔗 Chat API: http://localhost:3001"

retrieval-down:
	@echo "🛑 Stopping retrieval services (keeping databases running)..."
	docker compose stop nim-llm chatview txt2kg-app

stop:
	@echo "🛑 Stopping all services..."
	docker compose --profile ingestion --profile retrieval down

logs:
	docker compose --profile ingestion --profile retrieval logs -f

ps:
	docker compose --profile ingestion --profile retrieval ps

clean:
	@echo "🧹 Cleaning up all services and volumes..."
	docker compose --profile ingestion --profile retrieval down -v
