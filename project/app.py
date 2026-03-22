from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles
import os
import shutil
import tempfile
import uuid
import subprocess
import time
import requests
from pydantic import BaseModel
from workflow.pipeline import AdvisoryPipeline
from utils.pdf_watcher import start_pdf_watcher

app = FastAPI(
    title="Dual-Gated Retrieval-Constrained Legal Compliance Advisory System",
    description="API for retrieving legal compliance guidance with dual-gated inference safety.",
    version="1.0.0"
)

pipeline = AdvisoryPipeline(llm_model_name="llama3.2:1b", distance_threshold=0.35)

_pdf_observer = None

def is_ollama_running():
    try:
        response = requests.get("http://127.0.0.1:11434/", timeout=2)
        return response.status_code == 200
    except:
        return False

def auto_start_ollama():
    """Try to start Ollama automatically if not running"""
    print("⚡ Attempting to auto-start Ollama...")
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )
        # Wait up to 15 seconds for Ollama to be ready
        for i in range(15):
            time.sleep(1)
            if is_ollama_running():
                print(f"✅ Ollama auto-started successfully! (took {i+1}s)")
                return True
        print("❌ Ollama did not start in time. Please run 'ollama serve' manually.")
        return False
    except FileNotFoundError:
        print("❌ Ollama not found in PATH. Please install from https://ollama.com/download")
        return False
    except Exception as e:
        print(f"❌ Could not auto-start Ollama: {e}")
        return False

@app.on_event("startup")
async def startup_check():
    global _pdf_observer
    print("\n" + "="*55)
    print("🌾  AgriLegal FPO Assistant — Starting Up")
    print("="*55)

    # 1. Ollama check + auto-start
    if is_ollama_running():
        print("✅ Ollama Engine: ONLINE")
    else:
        print("⚠️  Ollama not detected. Trying to auto-start...")
        success = auto_start_ollama()
        if not success:
            print("💡 Manual fix: open a terminal and run: ollama serve")

    # 2. Start PDF watcher (also handles startup ingest)
    _pdf_observer = start_pdf_watcher()

    print("🚀 Server is ready! Visit http://127.0.0.1:8000")
    print("="*55 + "\n")

@app.on_event("shutdown")
async def shutdown():
    global _pdf_observer
    if _pdf_observer:
        _pdf_observer.stop()
        _pdf_observer.join()
        print("👀 PDF Watcher stopped.")



app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

class QueryRequest(BaseModel):
    query: str
    language: str = "en"

class QueryResponse(BaseModel):
    status: str
    message: str = None
    provision: str = None
    penalty: str = None
    action: str = None
    source: str = None
    raw: str = None
    reason: str = None
    final_reply: str = None

@app.post("/api/advise", response_model=QueryResponse)
def get_legal_advice(request: QueryRequest):
    if not is_ollama_running():
        return QueryResponse(
            status="error",
            message="Ollama is offline. Starting it up — please retry in 15 seconds."
        )
    result = pipeline.run(request.query)
    if result["status"] == "rejected":
        return QueryResponse(
            status="rejected",
            message=result["message"],
            reason=result.get("reason")
        )
    return QueryResponse(
        status="success",
        provision=result.get("provision"),
        penalty=result.get("penalty"),
        action=result.get("action"),
        source=result.get("source"),
        raw=result.get("raw"),
        final_reply=result.get("final_reply")
    )

@app.post("/api/voice")
async def handle_voice(audio: UploadFile = File(...), sender: str = Form("")):
    """Endpoint for the Node.js WhatsApp bridge"""
    ext = audio.filename.split('.')[-1] if '.' in audio.filename else 'ogg'
    temp_path = os.path.join(tempfile.gettempdir(), f"bridge_audio_{uuid.uuid4()}.{ext}")
    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)
        result = pipeline.run(input_data=temp_path, is_audio=True)
        return result
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/api/web_chat")
async def web_chat(query: str = Form(None), audio: UploadFile = File(None)):
    """Endpoint for the Web UI — handles text and voice"""
    if not is_ollama_running():
        return {
            "status": "error",
            "message": "Ollama is starting up — please wait 15 seconds and try again."
        }
    try:
        if audio:
            ext = audio.filename.split('.')[-1] if '.' in audio.filename else 'webm'
            temp_file = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.{ext}")
            try:
                with open(temp_file, "wb") as buffer:
                    shutil.copyfileobj(audio.file, buffer)
                pipeline_result = pipeline.run(input_data=temp_file, is_audio=True)
            finally:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        else:
            if not query:
                return {"status": "error", "message": "No input provided."}
            pipeline_result = pipeline.run(input_data=query, is_audio=False)

        if pipeline_result.get("status") == "error":
            reply_text = pipeline_result.get("message", "An error occurred.")
        elif pipeline_result.get("status") == "rejected":
            reply_text = pipeline_result.get("message", "I cannot find sufficient legal evidence.")
        else:
            reply_text = pipeline_result.get("final_reply", "Error formatting response.")

        return {
            "status": "success",
            "reply": reply_text,
            "transcribed_query": pipeline_result.get("transcribed_query", query)
        }
    except Exception as e:
        print(f"Web Chat Error: {e}")
        return {"status": "error", "message": "Internal server error."}

@app.get("/api/status")
def system_status():
    """Health check showing full system status"""
    from pathlib import Path
    pdf_count = len(list(Path("data/raw_pdfs").glob("*.pdf"))) if os.path.exists("data/raw_pdfs") else 0
    chroma_populated = os.path.exists("chroma_db") and len(list(Path("chroma_db").rglob("*"))) > 0
    return {
        "server": "online",
        "ollama": "online" if is_ollama_running() else "offline",
        "chromadb": "populated" if chroma_populated else "empty",
        "pdf_count": pdf_count,
        "watcher": "active" if _pdf_observer else "inactive"
    }

@app.get("/health")
def health_check():
    return {"status": "online"}
