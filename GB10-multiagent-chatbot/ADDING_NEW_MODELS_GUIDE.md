# Adding New Models Guide

This guide describes how to add new Large Language Models (LLMs) to the GB10 Multi-Agent Chatbot.

## Overview
The chatbot system uses `llama-server` (from `llama.cpp`) to run GGUF models. To add a new model, you need to:
1.  **Download** the model weights (GGUF format).
2.  **Define** a new service in `docker-compose-models.yml`.
3.  **Register** the model in `backend/config.json`.
4.  **Restart** the system.

---

## Step 1: Download Model Weights
You need to download the `.gguf` file for your model. We recommend using `huggingface-cli` or `wget`.

1.  Navigate to the models directory:
    ```bash
    cd /home/delluser26/APPS/GB10-multiagent-chatbot/models
    ```
2.  Download your model. For example, to download a hypothetical "NewModel-7B":
    ```bash
    # Example using wget
    wget https://huggingface.co/User/NewModel-7B-GGUF/resolve/main/newmodel-7b-q4_k_m.gguf
    ```

---

## Step 2: Add Service to Docker Compose
Open `/home/delluser26/APPS/GB10-multiagent-chatbot/docker-compose-models.yml` and add a new service block for your model.

Use the following template:

```yaml
  # Replace 'new-model-name' with a unique identifier (e.g., mistral-7b)
  new-model-name:
    <<: *build-base
    container_name: new-model-name
    volumes:
      - ./models:/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    command: 
      - "-m"
      - "/models/newmodel-7b-q4_k_m.gguf" # Exact filename of your downloaded model
      - "--port"
      - "8000"
      - "--host"
      - "0.0.0.0"
      - "-n"
      - "4096"        # Context window size (adjust as needed)
      - "--n-gpu-layers"
      - "999"         # Offload all layers to GPU
      - "--jinja"     # Enable Jinja2 template support
```

**Key Parameters:**
*   `container_name`: Must match the service name key.
*   `-m`: Path to the model file *inside the container* (mapped to `/models`).
*   `-n`: Context length (e.g., 2048, 4096, 32768).
*   `--n-gpu-layers`: Set to a high number (e.g., 999) to run entirely on GPU.

---

## Step 3: Register Model in Config
Open `/home/delluser26/APPS/GB10-multiagent-chatbot/backend/config.json`.
Add your model's **container name** to the `models` list.

```json
{
  "sources": [ ... ],
  "models": [
    "gpt-oss-120b",
    "deepseek-coder",
    "qwen2.5-vl",
    "new-model-name"  <-- ADD THIS LINE
  ],
  "selected_model": "gpt-oss-120b",
  ...
}
```
> **Important:** The string in `models` MUST match the `container_name` you defined in Step 2. The backend uses this name to resolve the hostname (e.g., `http://new-model-name:8000`).

---

## Step 4: Restart Services
To apply the changes, restart the backend and start the new model container.

```bash
cd /home/delluser26/APPS/GB10-multiagent-chatbot
docker compose up -d
```
This command will recreate the containers with the new configuration.

## Verification
1.  Check if the container is running:
    ```bash
    docker ps | grep new-model-name
    ```
2.  Refresh the web interface (`http://localhost:3000`). The new model should appear in the "Model" dropdown menu.
