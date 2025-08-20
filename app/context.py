from typing import List
from langchain_core.documents import Document

def format_context(docs: List[Document]) -> str:
    """Format context documents for display"""
    if not docs:
        return "No context available"
    
    context_str = ""
    for i, doc in enumerate(docs, 1):
        context_str += f"\n--- Document {i} ---\n"
        context_str += doc.page_content[:500]  # Show first 500 chars
        if len(doc.page_content) > 500:
            context_str += "..."
        context_str += "\n"
    
    return context_str

def get_citations(docs: List[Document]) -> List[str]:
    """Extract citation information from documents"""
    citations = []
    for i, doc in enumerate(docs, 1):
        source = getattr(doc.metadata, 'source', 'Unknown source')
        page = getattr(doc.metadata, 'page', 'Unknown page')
        citations.append(f"[{i}] Source: {source}, Page: {page}")
    return citations