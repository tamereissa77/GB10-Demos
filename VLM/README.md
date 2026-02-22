# VLM-GraphRAG: Intelligent Document Processing & Knowledge Retrieval

A powerful, hybrid GraphRAG (Retrieval-Augmented Generation) system designed for intelligent document processing. It combines state-of-the-art Arabic OCR with Knowledge Graph extraction and semantic vector search to provide high-accuracy answers over complex documents.

---

## 🚀 Architecture Overview

The system is split into two independent workflows — **Ingestion** and **Retrieval** — that can run separately to optimize GPU VRAM usage on a single machine.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INGESTION WORKFLOW                          │
│                        (make ingestion)                            │
│                                                                     │
│  ┌─────────��────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │ Extraction UI │───▶│ FastAPI OCR  │───▶│  Chandra VLM Model   │  │
│  │  :5173        │    │ Backend :8000│    │  (Arabic OCR)        │  │
│  └──────────────┘    └──────┬───────┘    └──────────────────────┘  │
│                             │                                       │
│                             ▼                                       │
│                    ┌──────────────┐    ┌──────────────┐            │
│                    │  txt2kg-app  │───▶│ Ollama LLM   │            │
│                    │  :3001       │    │ llama3.1:8b   │            │
│                    └──────┬───────┘    │ :11434        │            │
│                           │           └────────────��─┘             │
│                    ┌──────┴───────┐                                 │
│                    ▼              ▼                                  │
│             ┌───────────┐  ┌──────────┐                            │
│             │ ArangoDB  │  │  Qdrant  │                            │
│             │ Graph DB  │  │ Vector DB│                            │
│             │ :8529     │  │ :6333    │                            │
│             └───────────┘  └──────────┘                            │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        RETRIEVAL WORKFLOW                           │
│                        (make retrieval)                             │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │   Chat UI    │───▶│  txt2kg-app  │───▶│   NVIDIA NIM LLM    │  │
│  │   :5174      │    │  :3001       │    │  Llama-3.1-8B        │  │
│  └──────────────┘    └──────┬───────┘    │  :8010               │  │
│                             │            └──────────────────────┘  │
│                    ┌────────┴────────┐                              │
│                    ▼                 ▼                               │
│             ┌───────────┐     ┌──────────┐                         │
│             │ ArangoDB  │     │  Qdrant  │                         │
│             │ Graph DB  │     │ Vector DB│                         │
│             │ :8529     │     │ :6333    │                         │
│             └───────────┘     └──────────┘                         │
└─────────────────────────────────────────────────────────────────────┘
```

### Why Two Workflows?

Running OCR + triple extraction and LLM inference simultaneously requires significant GPU VRAM. By splitting into independent workflows, you can:

- Run **ingestion** to process documents (OCR + KG extraction) using Ollama
- Then switch to **retrieval** to query the knowledge graph using NVIDIA NIM
- Shared databases (ArangoDB, Qdrant) persist across workflow switches

---

## 🛠 Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Extraction UI** | React 19, Vite, TypeScript | Document upload & OCR result viewer |
| **Chat UI** | React 19, Vite, TypeScript | GraphRAG Q&A interface |
| **OCR Backend** | FastAPI (Python), Chandra VLM | Arabic OCR with layout analysis |
| **RAG Engine** | Next.js, LangChain.js | Triple extraction, embeddings, GraphRAG queries |
| **Graph Database** | ArangoDB | Entity-relationship storage (knowledge graph) |
| **Vector Database** | Qdrant | Semantic text chunk retrieval |
| **Ingestion LLM** | Ollama (llama3.1:8b) | Triple extraction from OCR text |
| **Retrieval LLM** | NVIDIA NIM (Llama-3.1-8B-Instruct) | Answer generation from graph + vector context |
| **Reverse Proxy** | Nginx | Routes API calls from UI containers to backends |

---

## ✨ Key Features

- **Advanced Arabic OCR**: Real-time streaming OCR results using the Chandra model with page-by-page progress
- **Automatic Knowledge Graph Extraction**: Discovers entities and relationships (triples) directly from OCR text using Ollama
- **Hybrid GraphRAG**: Combines the precision of Knowledge Graphs (ArangoDB) with the flexibility of Vector Search (Qdrant)
- **Source Transparency**: Every chat answer cites the specific text chunks and graph triples used
- **Separate UIs**: Dedicated Extraction UI (port 5173) and Chat UI (port 5174) — each built with the correct mode baked in at compile time
- **Backend Health Polling**: The Extraction UI polls `/api/health` and shows a readiness indicator; the upload button is disabled until the OCR model is fully loaded
- **Robust NDJSON Streaming**: The upload component properly buffers incomplete JSON lines across network chunks, handling large Arabic text with Unicode correctly
- **Configurable Timeouts**: Triple extraction timeout is configurable via environment variables (default 300s, was 90s)
- **Independent Workflow Control**: Start/stop ingestion and retrieval independently without tearing down shared databases
- **Hardware Accelerated**: Optimized for NVIDIA GPUs with CUDA support

---

## 📦 Prerequisites

- **Docker** and **Docker Compose** v2.x+
- **NVIDIA GPU** with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- **NVIDIA NGC API Key** (for NIM container images)
- **make** (standard on Linux)

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/tamereissa77/GB10-GraphRAG-Arabic.git
cd VLM
```

### 2. Configure Environment Variables

Create a `.env` file or export your NGC key:

```bash
export NGC_API_KEY="your-nvapi-key"
```

Optional tuning (can also be set in `.env`):

```bash
# Triple extraction timeout per page (default: 300 seconds)
export TRIPLE_EXTRACT_TIMEOUT_SECONDS=300

# Connection timeout for triple extraction (default: 30 seconds)
export TRIPLE_EXTRACT_CONNECT_TIMEOUT_SECONDS=30
```

### 3. Build the Containers

Build both UI containers (they compile different React bundles):

```bash
docker compose build extractionview chatview backend
```

### 4. Start a Workflow

See all available commands:

```bash
make help
```

Output:

```
GraphRAG Control Panel
-----------------------
make ingestion       - Start ONLY ingestion containers (DBs + backend + ollama + init + extraction UI)
make retrieval       - Start ONLY retrieval containers (DBs + nim + chat API + chat UI)
make ingestion-down  - Stop only ingestion containers (keeps shared DBs running)
make retrieval-down  - Stop only retrieval containers (keeps shared DBs running)
make stop            - Stop all services (both workflows)
make logs            - Show logs for all running services
make ps              - Show status of all services
make clean           - Stop all and remove volumes
```

---

## 📖 Usage

### Workflow 1: Document Ingestion (OCR + Knowledge Graph Extraction)

```bash
make ingestion
```

**Starts**: ArangoDB, Qdrant, Ollama, OCR Backend, ArangoDB-init, Extraction UI

| Service | URL |
|---------|-----|
| Extraction UI | http://localhost:5173 |
| OCR Backend API | http://localhost:8000 |
| ArangoDB UI | http://localhost:8529 |
| Qdrant Dashboard | http://localhost:6333/dashboard |
| txt2kg-app | http://localhost:3001 |

**Steps**:

1. Open the **Extraction UI** at `http://localhost:5173`
2. Wait for the backend readiness indicator to turn green (🟢 Backend ready ✅)
   - The OCR model loads in the background; the UI polls `/api/health` every 3 seconds
   - The upload button is disabled until the model is ready
3. Upload a PDF or image file (supports Arabic, English, and mixed layouts)
4. Watch the live streaming OCR results — pages appear progressively with a progress bar
5. After each page's OCR completes, triples are automatically extracted via Ollama and stored in ArangoDB
6. Text chunks are embedded into Qdrant for vector search

**Stop ingestion** (keeps databases running for retrieval):

```bash
make ingestion-down
```

### Workflow 2: Chat & Retrieval (GraphRAG Q&A)

```bash
make retrieval
```

**Starts**: ArangoDB, Qdrant, NVIDIA NIM LLM, txt2kg-app, Chat UI

| Service | URL |
|---------|-----|
| Chat UI | http://localhost:5174 |
| txt2kg-app (full UI) | http://localhost:3001 |
| NIM LLM API | http://localhost:8010 |
| ArangoDB UI | http://localhost:8529 |
| Qdrant Dashboard | http://localhost:6333/dashboard |

**Steps**:

1. Open the **Chat UI** at `http://localhost:5174`
2. Ask questions about your ingested documents (e.g., *"What are the total assets?"*)
3. The system performs a hybrid search:
   - **Vector search** in Qdrant for semantically similar text chunks
   - **Graph search** in ArangoDB for related entity-relationship triples
4. Both contexts are sent to NVIDIA NIM (Llama-3.1-8B-Instruct) for answer synthesis
5. The response includes the answer plus source citations (text chunks + graph triples)

**Stop retrieval** (keeps databases running):

```bash
make retrieval-down
```

### Switching Between Workflows

Databases (ArangoDB, Qdrant) are shared and persist across workflow switches:

```bash
# Process documents
make ingestion
# ... upload and process documents ...
make ingestion-down

# Switch to chat
make retrieval
# ... query your knowledge graph ...
make retrieval-down

# Full teardown
make stop    # stops all containers
make clean   # stops all + removes volumes (data loss!)
```

---

## 🐳 Container Architecture

### Services by Workflow

| Container | Profile | Port | Description |
|-----------|---------|------|-------------|
| `backend` | ingestion | 8000 | FastAPI OCR server (Chandra model) |
| `extractionview` | ingestion | 5173 | React UI for document upload & OCR results |
| `ollama` | ingestion | 11434 | Local LLM for triple extraction |
| `arangodb-init` | ingestion | — | One-shot: creates `txt2kg` database |
| `chatview` | retrieval | 5174 | React UI for GraphRAG chat |
| `nim-llm` | retrieval | 8010 | NVIDIA NIM Llama-3.1-8B for answer generation |
| `txt2kg-app` | both | 3001 | Next.js RAG engine (extraction + query APIs) |
| `arangodb` | always | 8529 | Graph database (no profile, always starts) |
| `qdrant` | always | 6333 | Vector database (no profile, always starts) |

### Separate Nginx Configurations

The two UI containers use different nginx configs to avoid upstream resolution errors:

- **`extractionview`** uses `nginx.conf` — proxies `/api/` → `backend:8000`
- **`chatview`** uses `nginx-chat.conf` — proxies `/chat-api/` → `txt2kg-app:3000`

This prevents the chat container from crashing when the backend service isn't running (and vice versa).

### Build-Time Mode Selection

The React frontend uses `VITE_APP_MODE` (passed as a Docker build arg) to conditionally render:

- **`extraction` mode**: Shows `FileUpload` + `ResultViewer` components
- **`chat` mode**: Shows `GraphRAGChat` component

Since Vite environment variables are baked into the JavaScript bundle at build time, each UI container is built with its own mode.

---

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NGC_API_KEY` | — | NVIDIA NGC API key for NIM container |
| `NVIDIA_API_KEY` | — | NVIDIA API key for txt2kg-app |
| `TRIPLE_EXTRACT_TIMEOUT_SECONDS` | `300` | Read timeout (seconds) for triple extraction per page |
| `TRIPLE_EXTRACT_CONNECT_TIMEOUT_SECONDS` | `30` | Connection timeout (seconds) for triple extraction |
| `HF_TOKEN` | — | HuggingFace token for Chandra model download |

### Backend Health Endpoint

The OCR backend exposes `GET /api/health` which returns:

```json
// Model loading
{"ready": false, "status": "loading", "message": "OCR model is loading..."}

// Model ready
{"ready": true, "status": "ok"}

// Model failed
{"ready": false, "status": "error", "error": "..."}
```

The Extraction UI polls this endpoint and disables the upload button until `ready: true`.

---

## 📂 Project Structure

```
VLM/
├── Makefile                    # Workflow commands (ingestion, retrieval, stop, etc.)
├── docker-compose.yml          # All service definitions with profiles
├── README.md                   # This file
│
├── backend/                    # FastAPI OCR Backend
│   ├── Dockerfile
│   ├── main.py                 # API endpoints (/api/upload, /api/health, /api/graphrag)
│   ├── ocr_service.py          # Chandra OCR + triple extraction logic
│   └── requirements.txt
│
├── frontend/                   # React Frontend (shared codebase, two build modes)
│   ├── Dockerfile              # Multi-stage build with VITE_APP_MODE + NGINX_CONF args
│   ├── nginx.conf              # Nginx config for extraction UI (proxies to backend)
│   ├── nginx-chat.conf         # Nginx config for chat UI (proxies to txt2kg-app)
│   ├── package.json
│   ├── vite.config.ts
│   └── src/
│       ├── App.tsx             # Conditional rendering based on VITE_APP_MODE
│       └── components/
│           ├── FileUpload.tsx   # Upload with health polling + NDJSON streaming
│           ├── FileUpload.css   # Includes backend status indicator styles
│           ├── ResultViewer.tsx  # OCR result display (text + HTML)
│           ├── ResultViewer.css
│           ├── GraphRAGChat.tsx  # Chat interface calling /chat-api/api/graphrag-query
│           └── GraphRAGChat.css
│
├── txt2kg_integration/         # Next.js Knowledge Graph Engine
│   ├── deploy/
│   │   ├── app/                # Dockerfile + init scripts
│   │   ├── compose/            # Additional compose files
│   │   └── services/           # Ollama, sentence-transformers, etc.
│   ├── frontend/               # Next.js app with full UI + API routes
│   │   ├── app/
│   │   │   ├── api/            # API routes (extract-triples, graphrag-query, etc.)
│   │   │   ├── rag/            # RAG search page
│   │   │   └── page.tsx        # Main page
│   │   ├── components/         # UI components
│   │   └── lib/                # Service libraries (arangodb, qdrant, embeddings, etc.)
│   └── scripts/                # Benchmarking and GNN scripts
│
└── hf_cache/                   # HuggingFace model cache (mounted into backend)
```

---

## 🔄 Recent Changes

### Workflow Separation
- **Makefile** now starts only the exact containers needed per workflow (no profile-based `down` that tears down shared DBs)
- Added `make ingestion-down` and `make retrieval-down` to stop workflow-specific containers while keeping databases running
- `make stop`, `make logs`, `make ps`, `make clean` operate on all containers directly

### Separate UI Containers
- **`extractionview`** (port 5173) — Extraction/upload UI with its own nginx config proxying to `backend:8000`
- **`chatview`** (port 5174) — Chat UI with its own nginx config proxying to `txt2kg-app:3000`
- Each container is built with a different `VITE_APP_MODE` build arg so the correct React components are rendered
- Separate nginx configs prevent container crashes when upstream services aren't running

### Backend Improvements
- **Background model loading**: The OCR model now loads in a background thread; the FastAPI server starts immediately and is reachable for health checks
- **`/api/health` endpoint**: Reports model loading status (loading/ready/error)
- **Upload guard**: Returns HTTP 503 if the model isn't ready when an upload is attempted
- **Configurable triple extraction timeout**: Increased from 90s to 300s (configurable via `TRIPLE_EXTRACT_TIMEOUT_SECONDS`)

### Frontend Improvements
- **Health polling**: The Extraction UI polls `/api/health` every 3 seconds and shows a readiness indicator (�� loading / 🟢 ready)
- **Upload button disabled** until backend model is ready
- **Robust NDJSON streaming**: Proper line buffering across network chunks — fixes JSON parse errors with large Arabic text containing Unicode
- **Error handling**: Non-OK HTTP responses are caught and displayed as user-friendly error messages instead of JSON parse spam

### Configuration Fixes
- **NIM model name** corrected from `gpt-oss-20b` to `meta/llama-3.1-8b-instruct` (matching the actual model served by NIM)
- **GraphRAG chat endpoint** corrected from `/api/graphrag` to `/chat-api/api/graphrag-query` (matching the actual Next.js API route)
- **`client_max_body_size`** moved to nginx server scope for reliable large file uploads

---

## 🐛 Troubleshooting

### Backend returns 502 Bad Gateway
The OCR model takes time to load (~1-2 minutes). Wait for the health indicator in the UI to turn green, or check manually:
```bash
curl http://localhost:8000/api/health
```

### Chat returns "Could not generate LLM answer"
NIM LLM may still be loading. Check its status:
```bash
curl http://localhost:8010/v1/models
```

### Triple extraction times out
Increase the timeout:
```bash
TRIPLE_EXTRACT_TIMEOUT_SECONDS=600 make ingestion
```

### Chat UI shows extraction interface
The UI containers need to be rebuilt with the correct build args:
```bash
docker compose build --no-cache extractionview chatview
```

### Container crashes with "host not found in upstream"
This happens when a UI container's nginx config references a service that isn't running. The fix is already applied — each UI has its own nginx config. Rebuild if using old images:
```bash
docker compose build --no-cache chatview
```

---

## 📄 License

See [LICENSE](txt2kg_integration/LICENSE) for details.

---

Developed by TAMER ABDELFATTAH.
