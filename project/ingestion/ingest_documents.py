import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw_pdfs")
CLEAN_DATA_DIR = os.path.join(BASE_DIR, "data", "clean_text")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")

def ingest_documents():
    print(f"Loading documents from {RAW_DATA_DIR}...")
    
    # Ensure data directories exist
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(CLEAN_DATA_DIR, exist_ok=True)
    
    loader = PyPDFDirectoryLoader(RAW_DATA_DIR)
    documents = loader.load()
    
    if not documents:
        print("No PDF documents found in the raw_pdfs directory. Add some PDFs and try again.")
        return

    print(f"Loaded {len(documents)} pages. Cleaning and saving text, then chunking (300-500 tokens)...")
    
    # Optional text cleaning step before chunking
    for i, doc in enumerate(documents):
        clean_content = doc.page_content.replace('\x00', '') # Remove null bytes
        doc.page_content = clean_content
        # Save to clean_text directory
        clean_path = os.path.join(CLEAN_DATA_DIR, f"clean_page_{i}.txt")
        with open(clean_path, "w", encoding="utf-8") as f:
            f.write(clean_content)
    
    # We use a chunk size of 400 characters, which roughly maps to less than 400 tokens
    # but since tokens != chars, let's bump it up slightly if we want 300-500 tokens.
    # Usually 1 token ≈ 4 chars in English, so let's do 1200 chars for ~300 tokens.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    print("Initializing embedding model (sentence-transformers/all-mpnet-base-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    print(f"Storing embeddings in ChromaDB at {CHROMA_DB_DIR}...")
    
    # Check if we should add to existing or create new
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    
    print("Ingestion complete. ChromaDB updated.")

if __name__ == "__main__":
    ingest_documents()
