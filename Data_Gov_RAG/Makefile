# Makefile for Data Governance RAG Demo (Dockerized + Local LLM)

IMAGE_NAME = data-gov-rag
CONTAINER_NAME = data-gov-rag
MODEL_DIR = /home/delluser26/APPS/GB10-multiagent-chatbot/models
MODEL_FILE = deepseek-coder-6.7b-instruct.Q8_0.gguf

.PHONY: help build run clean stop logs download-model

help:
	@echo "Available commands:"
	@echo "  make download-model - Download the Nemotron GGUF model (38GB)"
	@echo "  make build          - Build Docker images"
	@echo "  make run            - Run the application stack"
	@echo "  make stop           - Stop the application stack"
	@echo "  make clean          - Remove containers, images, and cache"
	@echo "  make logs           - View logs"

download-model:
	mkdir -p $(MODEL_DIR)
	pip install -U "huggingface_hub[cli]"
	hf download unsloth/Nemotron-3-Nano-30B-A3B-GGUF $(MODEL_FILE) --local-dir $(MODEL_DIR)

build:
	docker compose build

run:
	@echo "Starting application stack..."
	docker compose up -d
	@echo "App running at http://localhost:8501"
	@echo "Local LLM running at http://localhost:30000"

stop:
	docker compose down

clean:
	docker compose down --rmi all
	rm -rf __pycache__
	rm -rf .streamlit

logs:
	docker compose logs -f
