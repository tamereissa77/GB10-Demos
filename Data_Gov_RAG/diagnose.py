
import os
import sys
import traceback

try:
    print("Importing modules...")
    import streamlit
    import chromadb
    from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_core.documents import Document
    print("Imports successful.")
    
    # Mocking environment
    os.environ["NVIDIA_API_KEY"] = "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"
    
    print("Initializing Embeddings...")
    embeddings = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5")
    print("Embeddings initialized.")
    
    print("Initializing ChromaDB...")
    DOCUMENTS = [
        {"text": "Test", "metadata": {"source": "Test", "allowed_roles": ["admin"]}}
    ]
    docs = [Document(page_content=d["text"], metadata=d["metadata"]) for d in DOCUMENTS]
    
    # Force memory mode
    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="test_collection",
        persist_directory=None
    )
    print("ChromaDB initialized success.")

except Exception:
    traceback.print_exc()
