import os
from typing import List, Optional
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredFileLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from app.config import config
import warnings
warnings.filterwarnings('ignore')

def load_document(file_path: str) -> List[Document]:
    """Load document with fallback mechanisms"""
    try:
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Try specific loaders first
        if file_ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_ext == '.txt':
            loader = TextLoader(file_path, encoding='utf-8')
        elif file_ext in ['.docx', '.doc']:
            # Try multiple DOCX loaders with fallbacks
            try:
                from langchain_community.document_loaders import Docx2txtLoader
                loader = Docx2txtLoader(file_path)
            except ImportError:
                # Fallback to unstructured
                from langchain_community.document_loaders import UnstructuredWordDocumentLoader
                loader = UnstructuredWordDocumentLoader(file_path)
        elif file_ext in ['.pptx', '.ppt']:
            # Try multiple PPTX loaders with fallbacks
            try:
                from langchain_community.document_loaders import UnstructuredPowerPointLoader
                loader = UnstructuredPowerPointLoader(file_path)
            except ImportError:
                # Basic fallback
                from langchain_community.document_loaders import UnstructuredFileLoader
                loader = UnstructuredFileLoader(file_path)
        elif file_ext in ['.xlsx', '.xls', '.csv']:
            from langchain_community.document_loaders import UnstructuredExcelLoader
            loader = UnstructuredExcelLoader(file_path)
        elif file_ext == '.md':
            from langchain_community.document_loaders import UnstructuredMarkdownLoader
            loader = UnstructuredMarkdownLoader(file_path)
        else:
            # Fallback for unknown formats
            from langchain_community.document_loaders import UnstructuredFileLoader
            loader = UnstructuredFileLoader(file_path)
        
        return loader.load()
        
    except Exception as e:
        # Ultimate fallback: try to read as text
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return [Document(page_content=content, metadata={"source": file_path})]
        except:
            raise Exception(f"Failed to load {file_path}: {str(e)}")

def chunk_documents(documents: List[Document]) -> List[Document]:
    """Split documents into chunks with metadata preservation"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        add_start_index=True
    )
    
    chunks = splitter.split_documents(documents)
    
    # Enhance metadata with chunk information
    for i, chunk in enumerate(chunks):
        if not chunk.metadata:
            chunk.metadata = {}
        chunk.metadata['chunk_id'] = i
        chunk.metadata['chunk_size'] = len(chunk.page_content)
    
    return chunks

def process_file(file_path: str) -> List[Document]:
    """Process a single file and return chunks"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    docs = load_document(file_path)
    return chunk_documents(docs)

def process_directory(directory_path: str) -> List[Document]:
    """Process all supported files in a directory"""
    all_chunks = []
    supported_extensions = ['.pdf', '.docx', '.doc', '.txt', '.pptx', '.ppt', '.xlsx', '.xls', '.csv', '.md']
    
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in supported_extensions:
                try:
                    chunks = process_file(file_path)
                    all_chunks.extend(chunks)
                    print(f"✅ Processed {filename}: {len(chunks)} chunks")
                except Exception as e:
                    print(f"❌ Error processing {filename}: {str(e)}")
                    # Try fallback processing
                    try:
                        print(f"🔄 Trying fallback processing for {filename}...")
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        fallback_doc = Document(page_content=content, metadata={"source": filename})
                        chunks = chunk_documents([fallback_doc])
                        all_chunks.extend(chunks)
                        print(f"✅ Fallback processed {filename}: {len(chunks)} chunks")
                    except:
                        print(f"❌ Failed fallback processing for {filename}")
    
    return all_chunks

def get_supported_formats() -> List[str]:
    """Return list of supported file formats"""
    return ['.pdf', '.txt', '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls', '.csv', '.md']