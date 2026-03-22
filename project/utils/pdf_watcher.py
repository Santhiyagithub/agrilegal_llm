import os
import time
import threading
import subprocess
import sys
from pathlib import Path

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    class FileSystemEventHandler:
        pass

INGEST_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "ingestion", "ingest_documents.py")
RAW_PDFS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw_pdfs")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")

_ingest_lock = threading.Lock()
_pending_ingest = False

def run_ingestion(reason="manual"):
    global _pending_ingest
    with _ingest_lock:
        print(f"\n📥 [{reason}] Starting PDF ingestion...")
        try:
            result = subprocess.run(
                [sys.executable, INGEST_SCRIPT],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.dirname(__file__))
            )
            if result.returncode == 0:
                print(f"✅ Ingestion complete!\n{result.stdout[-500:] if result.stdout else ''}")
            else:
                print(f"❌ Ingestion failed:\n{result.stderr[-500:] if result.stderr else ''}")
        except Exception as e:
            print(f"❌ Ingestion error: {e}")
        _pending_ingest = False

def delayed_ingest(reason="file change", delay=3):
    """Wait a few seconds then ingest (debounce rapid file saves)"""
    global _pending_ingest
    if _pending_ingest:
        return
    _pending_ingest = True
    def _run():
        time.sleep(delay)
        run_ingestion(reason)
    threading.Thread(target=_run, daemon=True).start()

class PDFEventHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.pdf'):
            filename = os.path.basename(event.src_path)
            print(f"\n📄 New PDF detected: {filename}")
            delayed_ingest(reason=f"new file: {filename}")

    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('.pdf'):
            filename = os.path.basename(event.src_path)
            print(f"\n📄 PDF modified: {filename}")
            delayed_ingest(reason=f"modified: {filename}")

def is_chroma_empty():
    """Check if ChromaDB has any data"""
    if not os.path.exists(CHROMA_DIR):
        return True
    files = list(Path(CHROMA_DIR).rglob("*"))
    return len(files) == 0

def start_pdf_watcher():
    """Start the PDF watcher — call this from app.py startup"""
    os.makedirs(RAW_PDFS_DIR, exist_ok=True)

    # Auto-ingest on startup if ChromaDB is empty
    if is_chroma_empty():
        pdf_files = list(Path(RAW_PDFS_DIR).glob("*.pdf"))
        if pdf_files:
            print(f"\n📚 ChromaDB is empty. Found {len(pdf_files)} PDF(s). Auto-ingesting...")
            threading.Thread(target=run_ingestion, args=("startup",), daemon=True).start()
        else:
            print(f"\n⚠️  ChromaDB is empty and no PDFs found in data/raw_pdfs/")
            print(f"   Drop PDFs into: {RAW_PDFS_DIR}")
    else:
        print(f"\n✅ ChromaDB already populated. Skipping startup ingest.")

    # Start file watcher
    if WATCHDOG_AVAILABLE:
        observer = Observer()
        handler = PDFEventHandler()
        observer.schedule(handler, RAW_PDFS_DIR, recursive=False)
        observer.start()
        print(f"👀 PDF Watcher active — monitoring: data/raw_pdfs/")
        print(f"   Drop any PDF here and it will auto-ingest!\n")
        return observer
    else:
        print("⚠️  watchdog not installed. Run: pip install watchdog")
        print("   Auto-watch disabled, but startup ingest still works.\n")
        return None
