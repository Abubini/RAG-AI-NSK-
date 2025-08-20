import hashlib
import json
import os
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from langchain_core.documents import Document
from app.config import config

class CacheManager:
    def __init__(self, cache_dir: str = "./data/cache", ttl_hours: int = 24):
        self.cache_dir = cache_dir
        self.ttl = timedelta(hours=ttl_hours)
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, query: str, filters: Optional[Dict] = None) -> str:
        """Generate unique cache key for query and filters"""
        key_data = {'query': query, 'filters': filters or {}}
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        """Get file path for cache key"""
        return os.path.join(self.cache_dir, f"{cache_key}.json")
    
    def _is_cache_valid(self, cache_path: str) -> bool:
        """Check if cache is still valid based on TTL"""
        if not os.path.exists(cache_path):
            return False
        
        file_mtime = datetime.fromtimestamp(os.path.getmtime(cache_path))
        return datetime.now() - file_mtime < self.ttl
    
    def get_cached_results(self, query: str, filters: Optional[Dict] = None) -> Optional[List[Document]]:
        """Retrieve cached results if available and valid"""
        cache_key = self._get_cache_key(query, filters)
        cache_path = self._get_cache_path(cache_key)
        
        if self._is_cache_valid(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                
                # Convert back to Document objects
                documents = []
                for doc_data in cached_data.get('documents', []):
                    doc = Document(
                        page_content=doc_data['page_content'],
                        metadata=doc_data.get('metadata', {})
                    )
                    documents.append(doc)
                
                print(f"💾 Using cached results for: {query[:50]}...")
                return documents
                
            except Exception as e:
                print(f"❌ Cache read error: {e}")
        
        return None
    
    def cache_results(self, query: str, documents: List[Document], filters: Optional[Dict] = None):
        """Cache retrieval results"""
        cache_key = self._get_cache_key(query, filters)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            # Convert Documents to serializable format
            serializable_docs = []
            for doc in documents:
                serializable_docs.append({
                    'page_content': doc.page_content,
                    'metadata': doc.metadata
                })
            
            cache_data = {
                'query': query,
                'filters': filters,
                'documents': serializable_docs,
                'cached_at': datetime.now().isoformat(),
                'document_count': len(documents)
            }
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            print(f"💾 Cached {len(documents)} documents for query")
            
        except Exception as e:
            print(f"❌ Cache write error: {e}")
    
    def clear_expired_cache(self) -> int:
        """Clear expired cache files and return count cleared"""
        expired_files = []
        if os.path.exists(self.cache_dir):
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.json'):
                    cache_path = os.path.join(self.cache_dir, filename)
                    if not self._is_cache_valid(cache_path):
                        expired_files.append(cache_path)
        
        cleared_count = 0
        for file_path in expired_files:
            try:
                os.remove(file_path)
                cleared_count += 1
            except Exception as e:
                print(f"❌ Error clearing cache {file_path}: {e}")
        
        return cleared_count
    
    def clear_all_cache(self):
        """Clear all cache files"""
        if os.path.exists(self.cache_dir):
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.json'):
                    cache_path = os.path.join(self.cache_dir, filename)
                    try:
                        os.remove(cache_path)
                    except:
                        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {
            'total_files': 0,
            'expired_files': 0,
            'cache_dir': self.cache_dir,
            'files': []
        }
        
        if os.path.exists(self.cache_dir):
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.json'):
                    cache_path = os.path.join(self.cache_dir, filename)
                    stats['total_files'] += 1
                    
                    if not self._is_cache_valid(cache_path):
                        stats['expired_files'] += 1
                    
                    # Get file info
                    try:
                        file_stats = os.stat(cache_path)
                        stats['files'].append({
                            'name': filename,
                            'size': file_stats.st_size,
                            'modified': datetime.fromtimestamp(file_stats.st_mtime),
                            'is_valid': self._is_cache_valid(cache_path)
                        })
                    except:
                        pass
        
        return stats