import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from app.config import config

def load_pdf(file_path: str) -> List[Document]:
    """Load PDF document"""
    try:
        loader = PyPDFLoader(file_path)
        return loader.load()
    except Exception as e:
        raise Exception(f"Failed to load PDF {file_path}: {str(e)}")

def chunk_documents(documents: List[Document]) -> List[Document]:
    """Split documents into chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)

def process_file(file_path: str) -> List[Document]:
    """Process a single file and return chunks"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if file_path.lower().endswith('.pdf'):
        docs = load_pdf(file_path)
        return chunk_documents(docs)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def process_directory(directory_path: str) -> List[Document]:
    """Process all supported files in a directory"""
    all_chunks = []
    
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path) and filename.lower().endswith('.pdf'):
            try:
                chunks = process_file(file_path)
                all_chunks.extend(chunks)
                print(f"Processed {filename}: {len(chunks)} chunks")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    return all_chunks