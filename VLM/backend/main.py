from fastapi import FastAPI, UploadFile, File, HTTPException, Request
import logging
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from ocr_service import process_document_stream, get_model
import tempfile
import os
import shutil
import threading
import requests as http_requests

app = FastAPI(title="Arabic OCR API")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

TXT2KG_API_URL = os.environ.get("TXT2KG_API_URL", "http://txt2kg-app:3000")

# Track model loading state so the UI can poll readiness
_model_ready = False
_model_error: str | None = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_model_background():
    """Load the OCR model in a background thread so the server can start immediately."""
    global _model_ready, _model_error
    try:
        logger.info("Background: pre-loading OCR model...")
        get_model()
        _model_ready = True
        logger.info("Background: OCR model loaded successfully ✅")
    except Exception as e:
        _model_error = str(e)
        logger.error(f"Background: OCR model failed to load: {e}")


@app.on_event("startup")
async def startup_event():
    # Start model loading in a daemon thread – server is available immediately
    t = threading.Thread(target=_load_model_background, daemon=True)
    t.start()


@app.get("/")
async def root():
    return {"message": "Arabic OCR API is running", "model_ready": _model_ready}


@app.get("/api/health")
async def health():
    """Health / readiness endpoint polled by the UI."""
    if _model_error:
        return JSONResponse(
            status_code=503,
            content={"ready": False, "status": "error", "error": _model_error},
        )
    if not _model_ready:
        return JSONResponse(
            status_code=503,
            content={"ready": False, "status": "loading", "message": "OCR model is loading..."},
        )
    return {"ready": True, "status": "ok"}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    if not _model_ready:
        raise HTTPException(
            status_code=503,
            detail="OCR model is still loading. Please wait and try again.",
        )
    try:
        # Save uploaded file temporarily
        suffix = os.path.splitext(file.filename)[1]

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
        tmp.close()

        def iterfile():
            try:
                yield from process_document_stream(tmp_path)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        return StreamingResponse(iterfile(), media_type="application/x-ndjson")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/graphrag")
async def graphrag_query(request: Request):
    """Proxy GraphRAG queries to the txt2kg-app service"""
    try:
        body = await request.json()
        resp = http_requests.post(
            f"{TXT2KG_API_URL}/api/graphrag-query",
            json=body,
            timeout=180,
        )
        return JSONResponse(content=resp.json(), status_code=resp.status_code)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
