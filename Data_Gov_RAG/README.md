# 🔐 Data Governance RAG Demo

A fully interactive **Retrieval-Augmented Generation (RAG)** demo that showcases **data governance and access control** over both **unstructured documents** and **structured database tables**. The application demonstrates how an AI assistant can enforce role-based access policies in real time, combining hybrid retrieval from ChromaDB and PostgreSQL with LLM-powered natural language answers.

> Built with **NVIDIA Embeddings**, **Ollama (Llama 3.2)**, **ChromaDB**, **PostgreSQL**, and **Streamlit**.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Quick Start](#-quick-start)
- [Configuration](#-configuration)
- [Access Control Model](#-access-control-model)
- [Demo Walkthrough](#-demo-walkthrough)
- [Customization](#-customization)
- [Troubleshooting](#-troubleshooting)

---

## 🧭 Overview

Traditional RAG systems retrieve information without considering **who** is asking. This demo adds a full **governance layer** that controls what data each user can access, based on their group membership and inherited roles.

**Key concept:** Users are assigned to **groups**, groups have **roles**, and both documents and database tables specify which roles can access them. The AI assistant enforces these policies at query time — users only receive answers from data they are authorized to see.

### What makes this demo unique

| Traditional RAG | This Demo |
|----------------|-----------|
| Single data source | Hybrid: Documents + Database |
| No access control | Role-based, group-inherited ACL |
| Static configuration | Live editing of users, groups, documents, and table access |
| Black-box retrieval | Full governance logs showing filters, SQL queries, and sources |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface (Streamlit)               │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │ User/Group   │  │ Document     │  │ Table Access           │ │
│  │ Manager      │  │ Manager      │  │ Manager                │ │
│  └──────┬───────┘  └──────┬───────┘  └────────┬───────────────┘ │
│         │                 │                    │                 │
│  ┌──────▼─────────────────▼────────────────────▼───────────────┐ │
│  │              Hybrid RAG Agent (Chat Interface)              │ │
│  │                                                             │ │
│  │  1. Check user roles (via group inheritance)                │ │
│  │  2. Retrieve docs from ChromaDB (role-filtered)             │ │
│  │  3. Generate SQL for accessible tables → query PostgreSQL   │ │
│  │  4. Merge contexts → Generate answer via LLM               │ │
│  └────────┬──────────────────────────────┬─────────────────────┘ │
│           │                              │                       │
│   ┌───────▼───────┐              ┌───────▼──────────┐            │
│   │   ChromaDB    │              │   PostgreSQL     │            │
│   │ (Documents)   │              │ (Structured Data)│            │
│   └───────────────┘              └──────────────────┘            │
│                                                                  │
│   ┌──────────────────────────────────────────────────┐           │
│   │      NVIDIA Embeddings (nv-embedqa-e5-v5)        │           │
│   └──────────────────────────────────────────────────┘           │
│   ┌──────────────────────────────────────────────────┐           │
│   │      Ollama LLM (llama3.2:3b) — Local/Private    │           │
│   └──────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

### Core Capabilities

- **🔐 Group-Based Access Control** — Users inherit roles from their assigned group. Access is checked against both documents and database tables.
- **🤖 Hybrid RAG Agent** — Retrieves from both ChromaDB (unstructured documents) and PostgreSQL (structured data) in a single query.
- **📊 Governance Logs** — Full transparency: see which filters were applied, what SQL was generated, and which data sources contributed to each answer.
- **🔄 Live Re-indexing** — Add, edit, or delete documents and re-embed them into the vector store without restarting.

### Management Interfaces (Sidebar)

| Manager | Capabilities |
|---------|-------------|
| 🧑‍💼 **User Manager** | Add, rename, and delete users |
| 👥 **Group Manager** | Create, edit, and delete groups with role assignments |
| 📝 **Document Manager** | Add, edit, and delete documents with metadata |
| 🗄️ **Table Access Manager** | Edit which roles can access each database table |

### Dashboard

- **📂 Document Access Cards** — Rich cards showing file name, classification, owner, allowed roles, and access status per user.
- **🗄️ Database Table Cards** — Same card style for database tables, including live row count.
- **🔐 Privilege Matrix** — Expandable table showing Group × Data Source access, with the current group highlighted.

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **UI** | Streamlit | Interactive web interface |
| **Embeddings** | NVIDIA NV-EmbedQA-E5-v5 | Document vectorization via NVIDIA API |
| **LLM** | Ollama (Llama 3.2 3B) | Local, private language model for generation and SQL synthesis |
| **Vector Store** | ChromaDB | In-memory document retrieval with metadata filtering |
| **Database** | PostgreSQL 16 | Structured data storage (employee records) |
| **Orchestration** | Docker Compose | Multi-container deployment |

---

## 📁 Project Structure

```
Data_Gov_RAG/
├── app.py                  # Main Streamlit application (hybrid agent + UI)
├── docker-compose.yml      # 3 services: ollama, postgres, data-gov-rag
├── Dockerfile              # Python 3.12 image with all dependencies
├── Makefile                # Convenience commands (build, run, stop, logs)
├── .env                    # NVIDIA_API_KEY
├── init.sql                # PostgreSQL seed: employee_salaries table
├── documents.json          # Document store (editable via UI)
├── groups.json             # Group→roles mapping (editable via UI)
├── table_access.json       # Table-level ACL (editable via UI)
└── .dockerignore           # Docker build exclusions
```

### Key Data Files

| File | Purpose | Editable via UI? |
|------|---------|:---:|
| `documents.json` | Stores document content, metadata, and allowed roles | ✅ |
| `groups.json` | Maps group names to lists of roles | ✅ |
| `table_access.json` | Maps table names to access rules (roles, classification) | ✅ |
| `init.sql` | Seeds the PostgreSQL database on first start | ❌ (one-time) |

---

## 📋 Prerequisites

- **Docker** and **Docker Compose** (v2+)
- **NVIDIA GPU** with drivers installed (for Ollama)
- **NVIDIA API Key** for embeddings — get one at [build.nvidia.com](https://build.nvidia.com)
- At least **8GB RAM** available for the containers

---

## 🚀 Quick Start

### 1. Clone and configure

```bash
cd /path/to/Data_Gov_RAG

# Set your NVIDIA API key
echo "NVIDIA_API_KEY=nvapi-your-key-here" > .env
```

### 2. Build and start

```bash
# Using Make
make build
make run

# Or directly with Docker Compose
docker compose up -d --build
```

This starts three containers:

| Container | Port | Service |
|-----------|------|---------|
| `data-gov-rag` | `8501` | Streamlit application |
| `llama` | `30000` | Ollama LLM server |
| `gov-postgres` | `5432` | PostgreSQL database |

### 3. Pull the LLM model (first time only)

```bash
docker exec llama ollama pull llama3.2:3b
```

### 4. Open the app

Navigate to **http://localhost:8501** in your browser.

### 5. Click "Re-index Documents" in the sidebar

This embeds the documents into ChromaDB using NVIDIA Embeddings.

---

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|------------|---------|
| `NVIDIA_API_KEY` | NVIDIA API key for embeddings | Set in `.env` |
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://govdemo:govdemo123@postgres:5432/governance` |
| `OPENAI_API_BASE` | Ollama endpoint (OpenAI-compatible) | `http://llama:11434/v1` |

### Predefined Roles

The system uses four roles: `intern`, `hr`, `finance`, `admin`.

These can be assigned to groups in any combination. You can add more roles by modifying the `ALL_ROLES` list in `app.py`.

---

## 🔑 Access Control Model

### Inheritance Chain

```
👤 User → 👥 Group → 🔑 Roles → 📄 Documents / 🗄️ Tables
```

1. **Users** are assigned to one **group**
2. **Groups** contain one or more **roles**
3. **Documents** and **tables** specify which **roles** can access them
4. At query time, the system checks if **any** of the user's inherited roles match the resource's allowed roles

### Default Configuration

**Groups:**

| Group | Roles | Description |
|-------|-------|-------------|
| HR Team | `hr`, `admin` | Full access to HR and admin resources |
| Finance Team | `finance` | Access to financial data only |
| General Staff | `intern` | Access to public resources only |
| Executive | `hr`, `finance`, `admin` | Full access to all resources |

**Documents:**

| Document | Classification | Allowed Roles |
|----------|---------------|---------------|
| Payroll.pdf | 🔴 Confidential | `hr`, `admin` |
| Finance_Report.pdf | 🟡 Internal | `finance`, `admin` |
| General_Announce.pdf | 🟢 Public | `intern`, `hr`, `finance`, `admin` |

**Database Tables:**

| Table | Classification | Allowed Roles |
|-------|---------------|---------------|
| employee_salaries | 🔴 Confidential | `hr`, `admin` |

### Access Example

| User | Group | Inherited Roles | Payroll.pdf | Finance_Report.pdf | employee_salaries |
|------|-------|----------------|:-----------:|:------------------:|:-----------------:|
| Alice | HR Team | `hr`, `admin` | ✅ | ❌ | ✅ |
| Bob | Finance Team | `finance` | ❌ | ✅ | ❌ |
| Charlie | General Staff | `intern` | ❌ | ❌ | ❌ |

---

## 🎮 Demo Walkthrough

### Scenario 1: Role-Based Document Access

1. Select **Alice** (HR Team) as the active user
2. Observe the File Access Dashboard — Payroll.pdf shows **ACCESS GRANTED**
3. Switch to **Charlie** (General Staff) — Payroll.pdf now shows **ACCESS DENIED**
4. Ask in chat: *"What is the CEO salary?"* — Alice gets an answer; Charlie does not

### Scenario 2: Hybrid Document + Database Query

1. As **Alice** (HR Team), ask: *"What is Ahmed's salary and benefits?"*
2. Check **Governance Logs** — you'll see:
   - **📄 Document Retrieval:** chunks from Payroll.pdf
   - **🗄️ Database Retrieval:** SQL query against `employee_salaries`, results in a table
3. The AI answer cites both sources

### Scenario 3: Dynamic Access Change

1. As **Bob** (Finance Team), ask about salaries — **⛔ Access Denied** on both Payroll.pdf and employee_salaries
2. Go to **Table Access Manager** → add `finance` role to `employee_salaries`
3. Ask again — Bob now gets database results (but still no Payroll.pdf access)

### Scenario 4: Live Document Editing

1. Open **Document Manager** → Edit Payroll.pdf → add new content
2. Click **🔄 Re-index Documents**
3. Ask a question about the new content — it appears in the answer

### Scenario 5: Group Management

1. Create a new group **"Auditor"** with roles `hr`, `finance`
2. Assign a user to the Auditor group
3. That user now has access to both HR documents and financial reports

---

## 🔧 Customization

### Adding New Documents

**Via UI:** Use the Document Manager in the sidebar to add documents with content, classification, owner, and allowed roles.

**Via file:** Edit `documents.json` directly:

```json
{
  "text": "Your document content here...",
  "source": "NewDocument.pdf",
  "allowed_roles": ["hr", "admin"],
  "description": "Description of the document",
  "classification": "Confidential",
  "owner": "Department Name"
}
```

### Adding New Database Tables

1. Add the table creation and seed data to `init.sql`
2. Add the table ACL to `table_access.json`:

```json
{
  "new_table_name": {
    "allowed_roles": ["finance", "admin"],
    "description": "Description of the table",
    "classification": "Internal",
    "owner": "Finance Department"
  }
}
```

3. Rebuild the PostgreSQL container:

```bash
docker compose down
docker volume rm data_gov_rag_pgdata
docker compose up -d
```

### Adding New Roles

Edit `ALL_ROLES` in `app.py`:

```python
ALL_ROLES = ["intern", "hr", "finance", "admin", "legal", "engineering"]
```

---

## 🐛 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|---------|
| ChromaDB `NotFoundError` | Click **Re-index Documents** to rebuild the vector store |
| PostgreSQL connection refused | Wait 10-15 seconds after startup for PostgreSQL to initialize |
| LLM not responding | Run `docker exec llama ollama pull llama3.2:3b` to ensure the model is downloaded |
| NVIDIA Embeddings failing | Verify your `NVIDIA_API_KEY` in `.env` is valid |
| Stale data after editing documents | Click **🔄 Re-index Documents** after making changes |

### Useful Commands

```bash
# View container logs
make logs

# Restart a specific container
docker restart data-gov-rag

# Check PostgreSQL data
docker exec -it gov-postgres psql -U govdemo -d governance -c "SELECT * FROM employee_salaries;"

# Check Ollama models
docker exec llama ollama list

# Full rebuild (clean slate)
make clean
docker volume rm data_gov_rag_pgdata
make build
make run
```

---

## 📄 License

This project is a demonstration application for educational and internal use.

---

> **Built for showcasing enterprise data governance patterns with AI-powered retrieval.**
