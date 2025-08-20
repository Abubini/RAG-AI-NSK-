import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

class Config:
    # API Keys
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    
    # Model Configurations
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-2-v2")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama3-8b-8192")
    
    # Paths
    PERSIST_DIR: str = os.getenv("PERSIST_DIR", "./data/chroma")
    INBOX_DIR: str = os.getenv("INBOX_DIR", "./data/inbox")
    BM25_STORAGE_PATH: str = os.getenv("BM25_STORAGE_PATH", "./data/bm25/documents.json")
    
    # Retrieval Parameters
    TOP_K: int = int(os.getenv("TOP_K", "4"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))

config = Config()