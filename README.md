# AgriLegal FPO Assistant 🌾⚖️

An advanced, locally hosted, dual-gated Retrieval-Augmented Generation (RAG) Artificial Intelligence designed explicitly to provide Farmer Producer Organizations (FPOs) with highly accurate, hallucination-free legal compliance advice. 

It natively supports **WhatsApp**, **Voice Notes**, and **Multilingual Translation** via local open-source models and the official Indian Government's Bhashini frameworks.

## 🌟 Key Features
* **Dual-Gated RAG Architecture:** Prevents AI hallucinations. Gate 1 verifies context against official legal PDFs. Gate 2 enforces strict markdown structure (Provision, Penalty, Checklist, Source).
* **Local Processing (Privacy First):** Core inference runs entirely offline utilizing Ollama (`llama3.2:1b`), ensuring sensitive FPO data is never transmitted to big tech APIs.
* **Auto-Ingesting Vector DB:** Built-in `watchdog` daemon. Simply drag-and-drop a PDF (like the *Companies Act 2013*) into the folder, and it automatically updates the AI's legal knowledge in 3 seconds.
* **Multilingual Voice Support:** Native support for Indian languages via Bhashini NMT and OpenAI Whisper integrations.
* **Dual Interfaces:**
  * **WhatsApp Web Simulator:** A pixel-perfect local UI running on `http://127.0.0.1:8000`.
  * **Real WhatsApp Bridge:** A Node.js wrapper that allows anyone to text the AI naturally via real WhatsApp numbers just by scanning a QR code.

---

## 🚀 Getting Started

To run the entire ecosystem, you only need 3 concurrent terminal commands. 

### 1. Start the Local AI Engine
Provide the raw intelligence to the system by booting Ollama.
```bash
ollama serve
```

### 2. Start the FastAPI Python Backend
Provides the brain, the vector database, the RAG logic, and the HTTP endpoints.
*(Ensure your virtual environment is activated)*
```bash
cd project
uvicorn app:app --reload --port 8000
```
> **Tip:** You can drop new legal PDFs into `project/data/raw_pdfs/` at any time while this server is running. It will auto-ingest them silently in the background!

### 3. Connect WhatsApp 
Boot the Node.js bridge to link your physical WhatsApp account to the AI.
```bash
cd whatsapp_bridge
node bridge.js
```
A massive QR code will print in the terminal. Open WhatsApp on your phone, go to **Linked Devices**, and scan it. Your bot is now live and will reply to anyone who texts you!

---

## 🛠️ Architecture
* **Language Model:** Ollama / LLaMA 3
* **Vector Database:** ChromaDB 
* **Embeddings:** SentenceTransformers (`all-mpnet-base-v2`)
* **Backend Framework:** FastAPI (Python)
* **WhatsApp API:** `whatsapp-web.js` (Node environment)
* **Voice / Translation:** OpenAI Whisper, Google Translate (DeepTranslator), Bhashini ASR/NMT.

## 🔒 Security
All API keys (like official Bhashini credentials) are securely abstracted into a local `.env` file that is ignored by git, ensuring your private keys are never leaked to the public repository.
