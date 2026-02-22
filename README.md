# 🚀 GB10 Demo Applications

Ready-to-run AI demo applications created by **Tamer AbdelFattah**, designed to showcase the capabilities of the **Dell GB10 (NVIDIA Grace Blackwell)** platform.

---

## 📦 Demos

| Demo | Description | Stack |
|------|-------------|-------|
| **[Data_Gov_RAG](Data_Gov_RAG/)** | Hybrid RAG with data governance — role-based access control over documents (ChromaDB) and database tables (PostgreSQL), with full governance logging | Streamlit, Ollama, ChromaDB, PostgreSQL, NVIDIA Embeddings |
| **[EXTRACT-elements](EXTRACT-elements/)** | Document extraction and knowledge graph pipeline — extract structured elements from Arabic and English documents | Streamlit, Ollama, ArangoDB, PostgreSQL |
| **[GB10-multiagent-chatbot](GB10-multiagent-chatbot/)** | Multi-agent chatbot system with tool-calling capabilities and multiple LLM model support | Streamlit, Ollama, LangChain |
| **[VLM](VLM/)** | Vision Language Model demos — visual question answering and image understanding | Ollama, Gradio |
| **[VSS](VSS/)** | Video Search & Surveillance — real-time video analytics with object detection and event recognition | NVIDIA DeepStream, Triton, Gradio |

---

## 🖥️ Platform

All demos are optimized for the **Dell PowerEdge with NVIDIA GB10 (Grace Blackwell)** featuring:
- NVIDIA Grace CPU + Blackwell GPU
- Unified memory architecture
- Local LLM inference via Ollama
- Fully containerized with Docker Compose

---

## 🚀 Quick Start

Each demo has its own `docker-compose.yml` and `Makefile`. To run any demo:

```bash
cd <demo-folder>
make build
make run
```

> **Note:** `.env` files with API keys are not included in this repo. Create them locally as needed — refer to each demo's README for details.

---

## 👤 Author

**Tamer AbdelFattah**  
NVIDIA AI Solutions Business Development Executive @ Dell Technologies  
📧 [tamer.abdelfattah@dell.com](mailto:tamer.abdelfattah@dell.com)
