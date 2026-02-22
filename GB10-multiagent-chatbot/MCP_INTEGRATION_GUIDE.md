# 🔌 MCP Integration Guide

This guide details how to use the Model Context Protocol (MCP) servers included in this project with external applications (e.g., Claude Desktop, IDEs, or other agents).

## 🌍 What are these Tools?

This project exposes three powerful capabilities as standardized MCP servers:

1.  **Code Generation** (`code_generation.py`): Generates code using `deepseek-coder` or `gpt-oss`.
2.  **RAG / Document Search** (`rag.py`): Searches your uploaded documents using `qwen3`; requires the Milvus database to be running.
3.  **Image Understanding** (`image_understanding.py`): Analyzes images using `qwen2.5-vl`.

## 🚀 How to Connect

The easiest way to use these tools is to run them via **Docker**. This ensures they have access to the internal network (to talk to the inference models) and all necessary Python dependencies without polluting your local environment.

### Prerequisites
*   The project Docker containers must be running (`backend`, `models`, etc.).
*   You must have `docker` installed on the machine running the client.

### Configuration Template

Add the following to your MCP Client configuration (e.g., `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "gb10-coder": {
      "command": "docker",
      "args": [
        "exec",
        "-i",
        "backend",
        "python",
        "/app/tools/mcp_servers/code_generation.py"
      ]
    },
    "gb10-rag": {
      "command": "docker",
      "args": [
        "exec",
        "-i",
        "backend",
        "python",
        "/app/tools/mcp_servers/rag.py"
      ]
    },
    "gb10-vision": {
      "command": "docker",
      "args": [
        "exec",
        "-i",
        "backend",
        "python",
        "/app/tools/mcp_servers/image_understanding.py"
      ]
    }
  }
}
```

> **Note**: The `-i` flag is crucial. It keeps `stdin` open, allowing the MCP JSON-RPC protocol to work over the docker tunnel.

## 🛠️ Specific Application Guides

### 1. Claude Desktop App
1.  Locate your config file:
    *   **Mac**: `~/Library/Application Support/Claude/claude_desktop_config.json`
    *   **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
    *   **Linux**: `~/.config/Claude/claude_desktop_config.json` (if supported)
2.  Paste the template above into the file.
3.  Restart Claude Desktop.
4.  You should now see the tools (e.g., 🔨 `write_code`, 🔨 `search_documents`) available in the chat interface.

### 2. Cursor / VS Code (via extensions)
If you are using an MCP-compatible extension:
1.  Look for "MCP Servers" or "Tools" configuration.
2.  Add a new server.
3.  **Command**: `docker`
4.  **Args**: `exec -i backend python /app/tools/mcp_servers/code_generation.py` (example).

## 🧪 Testing the Connection

You can verify the connection works by running the command manually in your terminal. You should see no immediate output (it's waiting for JSON-RPC input), but if it crashes or errors immediately, something is wrong.

```bash
# Test command
docker exec -i backend python /app/tools/mcp_servers/code_generation.py

# Type this (including newlines) to test handshake:
# { "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": { "protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": { "name": "test", "version": "1.0" } } }
```

## ⚠️ Important Considerations

*   **Network Access**: These tools rely on the backend container being able to reach the model containers (`deepseek-coder`, `qwen2.5-vl`, etc.) via the docker network. Running them via `docker exec backend ...` guarantees this works.
*   **Permissions**: The user running the MCP client (e.g., your desktop user) must have permission to run `docker` commands (i.e., be in the `docker` group).

## 🌐 Remote Access (Different Machine)

If your app (e.g., Claude Desktop) is on **Machine B** (e.g., your laptop) and the chatbot is on **Machine A** (GB10), you can use **SSH** to tunnel the request.

### How it works
Your local application runs an `ssh` command that executes `docker` on the remote server. The input/output (prompt/result) is streamed securely over the SSH connection.

### Configuration Template (Remote)

Update your config file on your local machine:

```json
{
  "mcpServers": {
    "gb10-coder": {
        "command": "ssh",
        "args": [
            "delluser26@<GB10-IP-ADDRESS>",
            "docker",
            "exec",
            "-i",
            "backend",
            "python",
            "/app/tools/mcp_servers/code_generation.py"
        ]
    },
    "gb10-rag": {
        "command": "ssh",
        "args": [
            "delluser26@<GB10-IP-ADDRESS>",
            "docker",
            "exec",
            "-i",
            "backend",
            "python",
            "/app/tools/mcp_servers/rag.py"
        ]
    },
    "gb10-vision": {
        "command": "ssh",
        "args": [
            "delluser26@<GB10-IP-ADDRESS>",
            "docker",
            "exec",
            "-i",
            "backend",
            "python",
            "/app/tools/mcp_servers/image_understanding.py"
        ]
    }
  }
}
```

### Requirements for Remote Access
1.  **SSH Key Auth**: You must set up **passwordless SSH login** (using SSH keys) from your laptop to the GB10 machine. If the command prompts for a password, the MCP connection will fail (it hangs waiting for input).
    *   Test it: Run `ssh user@gb10-ip echo test` from your laptop. If it prints "test" without asking for a password, you are ready.
2.  **Host Verification**: Run the ssh command manually once to accept the host key fingerprint if it's your first time connecting.

## 🤖 n8n Integration

If you want to use these tools within an **n8n workflow**, you have two main options:

### Option A: Use the "Execute Command" Node (Recommended)
You can directly call the MCP tools via SSH. This is useful for simple, one-off tool usage.

1.  **Node Type**: `Execute Command`
2.  **Execute Once**: Turned ON.
3.  **Command**:
    ```bash
    echo '{{ $json.payload }}' | ssh -o StrictHostKeyChecking=no delluser26@<GB10-IP-ADDRESS> docker exec -i backend python /app/tools/mcp_servers/rag.py
    ```
4.  **Input**: You need to construct the JSON-RPC payload in a previous node (Set Node) and pass it as `payload`.
    *   *Example Payload*: `{"jsonrpc": "2.0", "method": "callTool", "params": {"name": "search_documents", "arguments": {"query": "how to install"}}, "id": 1}`

### Option B: Use the OpenAI Chat Model Node
If you just want to "chat" with the model (without RAG/Tools), use the standard OpenAI node.

1.  **Credential Type**: OpenAI API
2.  **Base URL**: `http://<GB10-IP-ADDRESS>:8001/v1`
3.  **API Key**: `any_string`
4.  **Model**: `gpt-oss-120b` (or `deepseek-coder`)

> **Note**: Option B gives you the raw intelligence of the model but does NOT use the MCP tools (RAG, Vision) defined in the backend. For full agentic capabilities, you must use Option A or coordinate via the WebSocket API.

