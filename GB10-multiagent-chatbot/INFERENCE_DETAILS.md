# Inference Architecture & Memory Details

## 🏗️ Microservices Architecture

This application employs a **microservices architecture** where each model runs in its own dedicated container, exposing an OpenAI-compatible API. This approach allows for modularity and the use of specialized inference engines for different modalities.

### Component Breakdown

1.  **Inference Servers**:
    *   **Text & Code Models** (`gpt-oss-120b`, `deepseek-coder`):
        *   **Engine**: `llama.cpp` server (Python binding).
        *   **Role**: Handles general chat and code generation tasks.
        *   **Interface**: OpenAI-compatible API (`/v1/chat/completions`).
    *   **Embeddings** (`qwen3-embedding`):
        *   **Engine**: `llama.cpp` server.
        *   **Role**: specialised for generating vector embeddings for RAG (Retrieval-Augmented Generation).
    *   **Vision Model** (`qwen2.5-vl`):
        *   **Engine**: NVIDIA TensorRT-LLM (`trtllm-serve`).
        *   **Role**: High-performance image understanding and analysis.
        *   **Optimization**: Optimized specifically for NVIDIA GPUs.

2.  **Orchestration (The "Brain")**:
    *   **Backend (FastAPI)**: Serves as the central controller.
    *   **Model Context Protocol (MCP)**: Used to connect the backend to specialized "agents" (tools).
    *   **Flow**: Backend -> MCP Tool -> Inference Container API.

---

## 💾 Memory Management: Lazy Loading vs. vLLM

Users may notice that models in this setup do not immediately consume their full memory footprint upon container start, which differs from standard vLLM deployments.

### 1. `llama.cpp` vs. vLLM

*   **vLLM (Standard Datacenter Deployment)**:
    *   **Behavior**: Eagerly pre-allocates ~90% of available GPU memory at startup.
    *   **Reason**: Reserves space for model weights and a massive KV cache (PagedAttention) to prevent Out-Of-Memory (OOM) errors during high-throughput concurrent requests.
    *   **Observation**: "Hogs" memory immediately, even when idle.

*   **llama.cpp (Used in this App)**:
    *   **Behavior**: Uses **`mmap` (memory mapping)** to map the model file on disk to virtual memory.
    *   **Lazy Loading**: The Operating System only pages parts of the model (weights) into physical RAM/VRAM **when they are actually accessed** during a forward pass (inference).
    *   **Observation**: Low memory usage at idle; usage spikes only when a request is processed.

### 2. Unified Memory Architecture

The `llama.cpp` containers in this project are compiled with `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1`.

*   **Function**: Allows the GPU to access system RAM directly over the PCIe/NVLink bus.
*   **Benefit**: If the GPU runs out of VRAM, it can "overflow" into system RAM. This allows running larger models (like 120B parameters) that might otherwise barely fit or slightly exceed strict VRAM limits, at the cost of some performance speed.
*   **Startup**: Avoids a forced full-copy of weights to VRAM at boot time, contributing to the "lazy" memory characteristic.

## 🔌 External Integration

You can integrate other applications with this system at two different levels:

### 1. Direct Inference (OpenAI Compatible)
Since the inference containers expose standard OpenAI-compatible APIs, you can point **any application** (e.g., LangChain, AutoGen, or custom scripts) directly to them.

*   **Endpoint**: `http://localhost:8001/v1` (or `http://localhost:8000` inside docker network)
*   **API Key**: `api_key` (ignored by server, but usually required by clients)
*   **Example Usage**:
    ```python
    from openai import OpenAI
    
    client = OpenAI(base_url="http://localhost:8001/v1", api_key="sk-xxx")
    response = client.chat.completions.create(
        model="gpt-oss-120b", # or deepseek-coder
        messages=[{"role": "user", "content": "Hello!"}]
    )
    ```

### 2. Model Context Protocol (MCP)
The tools in `backend/tools/mcp_servers/` are standard **MCP Servers**. This means you can use them in **any MCP Client** (like Claude Desktop or an IDE extension).

*   **Protocol**: JSON-RPC over `stdio`
*   **Usage**: Configure your MCP client to run the python script:
    ```json
    {
      "mcpServers": {
        "code-tools": {
          "command": "python",
          "args": ["/path/to/backend/tools/mcp_servers/code_generation.py"]
        }
      }
    }
    ```

