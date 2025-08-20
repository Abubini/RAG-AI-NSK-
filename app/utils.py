import os
import hashlib
import json
from typing import List, Dict, Any
from langchain_core.documents import Document

def ensure_directories():
    """Ensure all required directories exist"""
    os.makedirs("./data/inbox", exist_ok=True)
    os.makedirs("./data/chroma", exist_ok=True)
    os.makedirs("./data/bm25", exist_ok=True)

def hash_content(content: str) -> str:
    """Generate hash for content deduplication"""
    return hashlib.md5(content.encode()).hexdigest()

def deduplicate_documents(docs: List[Document]) -> List[Document]:
    """Remove duplicate documents based on content"""
    seen = set()
    unique_docs = []
    
    for doc in docs:
        content_hash = hash_content(doc.page_content)
        if content_hash not in seen:
            seen.add(content_hash)
            unique_docs.append(doc)
    
    return unique_docs

def print_colored(text: str, color: str = "green"):
    """Print colored text in CLI"""
    colors = {
        "green": "\033[92m",
        "red": "\033[91m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "reset": "\033[0m"
    }
    print(f"{colors.get(color, colors['green'])}{text}{colors['reset']}")

def document_to_dict(doc: Document) -> Dict[str, Any]:
    """Convert Document to dictionary for JSON serialization"""
    return {
        "page_content": doc.page_content,
        "metadata": doc.metadata
    }

def dict_to_document(doc_dict: Dict[str, Any]) -> Document:
    """Convert dictionary back to Document"""
    return Document(
        page_content=doc_dict["page_content"],
        metadata=doc_dict.get("metadata", {})
    )

def save_documents_to_json(docs: List[Document], file_path: str):
    """Save documents to JSON file"""
    doc_dicts = [document_to_dict(doc) for doc in docs]
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(doc_dicts, f, ensure_ascii=False, indent=2)

def load_documents_from_json(file_path: str) -> List[Document]:
    """Load documents from JSON file"""
    if not os.path.exists(file_path):
        return []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            doc_dicts = json.load(f)
        return [dict_to_document(doc_dict) for doc_dict in doc_dicts]
    except:
        return []