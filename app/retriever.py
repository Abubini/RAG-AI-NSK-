import hashlib
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable CUDA
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import Any, Dict, List, Optional
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from app.config import config
from app.utils import print_colored, save_documents_to_json, load_documents_from_json
from app.filters import MetadataFilter
from app.caching import CacheManager
import torch
import json

print(f"PyTorch using device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

class HybridRetriever:
    def __init__(self):
        # Force CPU usage for embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize reranker with CPU
        self.reranker = CrossEncoder(
            config.RERANKER_MODEL,
            device='cpu'
        )
        
        # Initialize storage
        self.all_documents = []
        self.vectorstore = None
        self.bm25_retriever = None
        
        # Load existing data
        self._load_existing_data()
        
        #metadata filter
        self.metadata_filter = MetadataFilter()

        #caching
        self.cache_manager = CacheManager()
    
    def _load_existing_data(self):
        """Load existing vectorstore and BM25 documents"""
        # Load BM25 documents from JSON
        self.all_documents = load_documents_from_json(config.BM25_STORAGE_PATH)
        print(f"📦 Loaded {len(self.all_documents)} documents from BM25 storage")
        
        # Load vectorstore if it exists
        if os.path.exists(config.PERSIST_DIR) and os.listdir(config.PERSIST_DIR):
            try:
                self.vectorstore = Chroma(
                    persist_directory=config.PERSIST_DIR,
                    embedding_function=self.embeddings
                )
                print("✅ Loaded existing vectorstore")
                
                # If we have documents but no BM25 storage, try to sync
                if self.all_documents and not self.bm25_retriever:
                    self._initialize_bm25_retriever()
                    
            except Exception as e:
                print(f"⚠️  Error loading vectorstore: {e}")
                self.vectorstore = None
    
    def _initialize_bm25_retriever(self):
        """Initialize BM25 retriever with current documents"""
        if self.all_documents:
            try:
                self.bm25_retriever = BM25Retriever.from_documents(self.all_documents)
                self.bm25_retriever.k = config.TOP_K
                print(f"✅ Initialized BM25 retriever with {len(self.all_documents)} documents")
            except Exception as e:
                print(f"❌ Error initializing BM25: {e}")
                self.bm25_retriever = None
    
    def _save_bm25_documents(self):
        """Save BM25 documents to JSON storage"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(config.BM25_STORAGE_PATH), exist_ok=True)
            save_documents_to_json(self.all_documents, config.BM25_STORAGE_PATH)
            print(f"💾 Saved {len(self.all_documents)} documents to BM25 storage")
        except Exception as e:
            print(f"❌ Error saving BM25 documents: {e}")
    
    def add_documents(self, chunks: List[Document]):
        """Add documents to both vectorstore and BM25 retriever"""
        try:
            # Add to all documents list for BM25
            self.all_documents.extend(chunks)
            
            # Initialize or update vectorstore
            if self.vectorstore is None:
                print("🆕 Creating new vectorstore...")
                self.vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=self.embeddings,
                    persist_directory=config.PERSIST_DIR
                )
            else:
                print("📝 Adding to existing vectorstore...")
                self.vectorstore.add_documents(chunks)
            
            # Initialize or update BM25 retriever
            print("🔄 Updating BM25 retriever...")
            self._initialize_bm25_retriever()
            
            # Save BM25 documents to persistent storage
            self._save_bm25_documents()
            
            print(f"✅ Added {len(chunks)} chunks. Total documents: {len(self.all_documents)}")
            
        except Exception as e:
            print(f"❌ Error adding documents: {str(e)}")
            raise
    
    def retrieve(self, query: str, top_k: Optional[int] = None, 
                filters: Optional[Dict[str, Any]] = None, 
                use_cache: bool = True) -> List[Document]:
        """Perform hybrid retrieval with optional metadata filtering"""
        if top_k is None:
            top_k = config.TOP_K


        if use_cache:
            cached_results = self.cache_manager.get_cached_results(query, filters)
            if cached_results is not None:
                return cached_results[:top_k]
        
        if self.vectorstore is None or self.bm25_retriever is None:
            print("⚠️  No documents indexed yet. Please ingest documents first.")
            return []
        
        
        try:
            # Apply metadata filtering if specified
            all_docs = self.all_documents
            if filters:
                all_docs = self.metadata_filter.filter_documents(all_docs, filters)
                print(f"   🔍 Applied filters: {filters}, {len(all_docs)} documents remain")
            print(f"🔍 Retrieving documents (top_k={top_k})...")
            
            # Semantic search
            print("   🤖 Semantic search...")
            semantic_docs = self.vectorstore.similarity_search(query, k=top_k)
            print(f"   ✅ Found {len(semantic_docs)} semantic results")
            
            # Keyword search
            print("   🔤 Keyword search...")
            keyword_docs = self.bm25_retriever.get_relevant_documents(query)
            print(f"   ✅ Found {len(keyword_docs)} keyword results")
            
            # Combine and deduplicate
            combined = {}
            for doc in semantic_docs + keyword_docs:
                # Use content hash for deduplication
                content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
                combined[content_hash] = doc
            
            unique_docs = list(combined.values())
            print(f"   🔄 Combined and deduplicated to {len(unique_docs)} unique documents")
            
            # Rerank if we have documents
            if unique_docs:
                print("   📊 Reranking documents...")
                reranked_docs = self._rerank(query, unique_docs, top_k)
                print(f"   ✅ Reranking complete. Returning top {len(reranked_docs)} documents")
                # Cache the results
                if use_cache and reranked_docs:
                    self.cache_manager.cache_results(query, reranked_docs, filters)
                return reranked_docs
            else:   
                return []
            
        except Exception as e:
            print(f"❌ Retrieval error: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def _rerank(self, query: str, docs: List[Document], top_k: int) -> List[Document]:
        """Rerank documents based on relevance to query"""
        if not docs:
            return []
        
        try:
            # For very large document sets, consider sampling for efficiency
            if len(docs) > 20:
                docs = docs[:20]  # Limit reranking to top 20 for performance
            
            pairs = [[query, doc.page_content] for doc in docs]
            scores = self.reranker.predict(pairs, show_progress_bar=False)
            
            reranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            return [doc for doc, score in reranked[:top_k]]
            
        except Exception as e:
            print(f"⚠️  Reranking error: {str(e)}, returning original order")
            return docs[:top_k]
    
    def get_stats(self):
        """Get statistics about the retriever"""
        return {
            "total_documents": len(self.all_documents),
            "vectorstore_ready": self.vectorstore is not None,
            "bm25_ready": self.bm25_retriever is not None,
            "bm25_storage_path": config.BM25_STORAGE_PATH
        }
    
    def clear_all_data(self):
        """Clear all stored data and reset in-memory state"""
        import shutil
        try:
            # Clear vectorstore directory
            if os.path.exists(config.PERSIST_DIR):
                shutil.rmtree(config.PERSIST_DIR)
                print("🗑️  Cleared vectorstore data")
            
            # Clear BM25 storage file
            bm25_dir = os.path.dirname(config.BM25_STORAGE_PATH)
            if os.path.exists(bm25_dir):
                shutil.rmtree(bm25_dir)
                print("🗑️  Cleared BM25 storage data")
            
            # Clear cache directory
            if os.path.exists(config.CACHE_DIR):
                shutil.rmtree(config.CACHE_DIR)
                print("🗑️  Cleared cache data")
            
            # Reset in-memory state
            self.all_documents = []
            self.vectorstore = None
            self.bm25_retriever = None
            
            # Recreate necessary directories
            os.makedirs(config.PERSIST_DIR, exist_ok=True)
            os.makedirs(os.path.dirname(config.BM25_STORAGE_PATH), exist_ok=True)
            os.makedirs(config.CACHE_DIR, exist_ok=True)
            
            print_colored("✅ All data cleared successfully!", "green")
            
        except Exception as e:
            print_colored(f"❌ Error clearing data: {e}", "red")
            raise