import logging
import os
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try importing the Chandra model
InferenceManager = None
load_file = None

try:
    from chandra.model import InferenceManager
    from chandra.input import load_file
    logger.info("Successfully imported InferenceManager from 'chandra.model'")
except ImportError as e:
    logger.warning(f"Could not import Chandra modules: {e}. Running in MOCK mode.")

# Global model instance
_model_instance = None

def get_model():
    global _model_instance
    if _model_instance is None and InferenceManager is not None:
        try:
            # Using 'hf' method as vllm might not be installed
            logger.info("Loading Chandra model (method='hf')... this may take a while")
            _model_instance = InferenceManager(method="hf")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    return _model_instance


def extract_triples_from_text(text, txt2kg_url):
    """Extract triples from a single chunk of text using Ollama via txt2kg.

    Timeouts can be tuned via env vars:
      - TRIPLE_EXTRACT_TIMEOUT_SECONDS (default: 300)
      - TRIPLE_EXTRACT_CONNECT_TIMEOUT_SECONDS (default: 30)
    """
    import requests
    import re
    
    if len(text.strip()) < 50:
        return []
    
    # Strip HTML tags (tables etc.) and truncate to avoid overwhelming Ollama
    clean_text = re.sub(r'<[^>]+>', ' ', text)  # Remove HTML tags
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()  # Collapse whitespace
    
    # Skip if after cleaning there's too little content
    if len(clean_text) < 50:
        logger.info("Page text was mostly HTML markup, skipping triple extraction")
        return []
    
    # Truncate to max 3000 chars to keep within Ollama's sweet spot
    MAX_TEXT_LEN = 3000
    if len(clean_text) > MAX_TEXT_LEN:
        logger.info(f"Truncating page text from {len(clean_text)} to {MAX_TEXT_LEN} chars")
        clean_text = clean_text[:MAX_TEXT_LEN]
    
    payload = {
        "text": clean_text,
        "useLangChain": False,
        "llmProvider": "ollama",
        "ollamaModel": "llama3.1:8b",
        "ollamaBaseUrl": "http://ollama:11434"
    }
    
    timeout_seconds = int(os.getenv("TRIPLE_EXTRACT_TIMEOUT_SECONDS", "300"))
    connect_timeout_seconds = int(os.getenv("TRIPLE_EXTRACT_CONNECT_TIMEOUT_SECONDS", "30"))

    try:
        resp = requests.post(
            f"{txt2kg_url}/api/extract-triples",
            json=payload,
            # (connect timeout, read timeout)
            timeout=(connect_timeout_seconds, timeout_seconds)
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("triples", [])
        else:
            logger.warning(f"Triple extraction failed with status {resp.status_code}")
            return []
    except requests.exceptions.Timeout:
        logger.warning(f"Triple extraction timed out after {timeout_seconds}s, skipping this page")
        return []
    except Exception as e:
        logger.warning(f"Triple extraction error: {e}")
        return []


def process_document_stream(file_path: str):
    """
    Process a document (PDF or Image) using Chandra OCR and yield results page-by-page.
    Triple extraction happens immediately after each page's OCR completes.
    """
    if InferenceManager is None:
        logger.info(f"Processing {file_path} in MOCK mode")
        yield json.dumps({
            "type": "progress",
            "page": 1,
            "total": 1,
            "text_snippet": "Mock result...",
            "page_data": {"markdown": "Mock text", "html": "<p>Mock</p>"}
        }) + "\n"
        
        yield json.dumps({
            "type": "complete",
            "status": "mock_success",
            "text": "This is a simulated result. Chandra OCR library was not found or could not be loaded.\n\nFile processed: " + os.path.basename(file_path),
            "html": "<div class='page'><p>This is a simulated result due to missing libraries.</p></div>",
            "raw_count": 1
        }) + "\n"
        return
    
    try:
        model = get_model()
        if model is None:
            raise RuntimeError("Model extraction failed to initialize")

        # Load file
        logger.info(f"Loading file: {file_path}")
        from pathlib import Path
        from chandra.model.schema import BatchInputItem
        
        # load_file returns a list of PIL Images
        images = load_file(str(file_path), {})
        
        # Run inference sequentially and extract triples after each page
        total_pages = len(images)
        logger.info(f"Starting sequential inference on {total_pages} pages")
        
        import requests
        TXT2KG_API_URL = "http://txt2kg-app:3000"
        
        results = []
        all_triples = []
        
        for i, img in enumerate(images):
            page_num = i + 1
            logger.info(f"Processing page {page_num}/{total_pages}...")
            
            # === OCR STEP: Run inference on this page ===
            batch_item = [BatchInputItem(image=img, prompt_type="ocr_layout")]
            page_result_list = model.generate(batch_item)
            
            if page_result_list:
                page_result = page_result_list[0]
                results.append(page_result)
                
                text_snippet = page_result.markdown[:100].replace('\n', ' ') + "..." if page_result.markdown else "No text found"
                logger.info(f"Finished page {page_num}/{total_pages}. Text snippet: {text_snippet}")
                
                # Yield OCR progress
                yield json.dumps({
                    "type": "progress",
                    "page": page_num,
                    "total": total_pages,
                    "text_snippet": text_snippet,
                    "page_data": {
                        "markdown": page_result.markdown,
                        "html": page_result.html
                    }
                }) + "\n"
                
                # === TRIPLE EXTRACTION: Extract triples from this page immediately ===
                if page_result.markdown and len(page_result.markdown.strip()) >= 50:
                    logger.info(f"Extracting KG triples from page {page_num}...")
                    
                    yield json.dumps({
                        "type": "progress",
                        "page": page_num,
                        "total": total_pages,
                        "text_snippet": f"Extracting KG triples from page {page_num}/{total_pages}...",
                        "page_data": {}
                    }) + "\n"
                    
                    page_triples = extract_triples_from_text(page_result.markdown, TXT2KG_API_URL)
                    
                    if page_triples:
                        all_triples.extend(page_triples)
                        logger.info(f"Page {page_num}: extracted {len(page_triples)} triples (total so far: {len(all_triples)})")
                    else:
                        logger.info(f"Page {page_num}: no triples extracted")
            else:
                logger.warning(f"No result returned for page {page_num}")

        # Combine all results
        full_text = "\n\n".join([r.markdown for r in results])
        full_html = "\n<hr>\n".join([r.html for r in results])
        
        # Store all accumulated triples in Graph DB
        logger.info(f"OCR and triple extraction complete. Total triples: {len(all_triples)}")
        
        if all_triples:
            try:
                yield json.dumps({
                    "type": "progress",
                    "page": total_pages,
                    "total": total_pages,
                    "text_snippet": f"Storing {len(all_triples)} triples in Knowledge Graph...",
                    "page_data": {}
                }) + "\n"

                # Process document with triples
                text_summary = full_text[:5000] if len(full_text) > 5000 else full_text
                payload_process = {
                    "text": text_summary,
                    "filename": os.path.basename(file_path),
                    "triples": all_triples,
                    "useLangChain": False,
                    "useGraphTransformer": False
                }
                logger.info("Calling /api/process-document with extracted triples...")
                resp_process = requests.post(f"{TXT2KG_API_URL}/api/process-document", json=payload_process, timeout=300)
                if resp_process.status_code == 200:
                    logger.info("Document processed successfully in txt2kg")
                else:
                    logger.error(f"Document processing failed: {resp_process.text}")

                # Store triples in Graph DB
                payload_store = {
                    "triples": all_triples,
                    "documentName": os.path.basename(file_path)
                }
                logger.info("Storing triples in Graph DB...")
                resp_store = requests.post(f"{TXT2KG_API_URL}/api/graph-db/triples", json=payload_store, timeout=120)
                if resp_store.status_code == 200:
                    logger.info(f"Successfully stored {len(all_triples)} triples in Graph DB")
                else:
                    logger.error(f"Failed to store triples: {resp_store.text}")

            except Exception as e:
                logger.error(f"txt2kg storage error: {e}")
        else:
            logger.warning("No triples were extracted from any page")

        # Step 4: Embed OCR text into Qdrant for vector search (GraphRAG)
        try:
            yield json.dumps({
                "type": "progress",
                "page": total_pages,
                "total": total_pages,
                "text_snippet": "Embedding text into vector database for search...",
                "page_data": {}
            }) + "\n"
            
            logger.info("Embedding OCR text into Qdrant for GraphRAG search...")
            doc_name = os.path.basename(file_path)
            
            payload_embed = {
                "documentId": doc_name,
                "content": full_text,
                "documentName": doc_name
            }
            resp_embed = requests.post(f"{TXT2KG_API_URL}/api/embeddings", json=payload_embed, timeout=120)
            if resp_embed.status_code == 200:
                embed_data = resp_embed.json()
                logger.info(f"Successfully embedded {embed_data.get('chunks', 0)} chunks into Qdrant")
            else:
                logger.warning(f"Embedding failed: {resp_embed.status_code} - {resp_embed.text[:200]}")
        except Exception as e:
            logger.warning(f"Embedding error (non-fatal): {e}")

        yield json.dumps({
            "type": "complete",
            "status": "success",
            "text": full_text,
            "html": full_html,
            "raw_count": len(results),
            "triples_count": len(all_triples)
        }) + "\n"
        
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        logger.error(f"Error during OCR processing: {error_msg}")
        yield json.dumps({
            "type": "error",
            "error": str(e)
        }) + "\n"
