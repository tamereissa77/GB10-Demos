# Use NVIDIA PyTorch image (26.01) which supports GB10/Blackwell architecture
FROM nvcr.io/nvidia/pytorch:26.01-py3

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install base python dependencies (fastapi etc)
RUN pip3 install --no-cache-dir -r requirements.txt

# Install chandra-ocr dependencies MANUALLY to avoid triggering torch upgrade/overwrite
# 1. Install safe dependencies (transitive deps of transformers/accelerate)
RUN pip3 install --no-cache-dir \
    filelock \
    huggingface-hub \
    regex \
    packaging \
    tokenizers \
    safetensors \
    tqdm \
    psutil \
    "numpy<2.0.0" \
    pyyaml \
    requests \
    opencv-python-headless \
    scipy \
    tiktoken \
    attributes-doc \
    verovio \
    markdownify \
    python-dotenv \
    pydantic-settings \
    openai

# 2. Install risky packages with --no-deps to protect system torch
RUN pip3 install --no-cache-dir --no-deps \
    transformers>=4.37.0 \
    accelerate \
    chandra-ocr

# 3. Install remaining utils
RUN pip3 install --no-cache-dir \
    qwen-vl-utils \
    filetype \
    pypdfium2

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
