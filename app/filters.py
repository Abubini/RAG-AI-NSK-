from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from datetime import datetime
import re

class MetadataFilter:
    def __init__(self):
        self.available_filters = {
            'source': self.filter_by_source,
            'date': self.filter_by_date,
            'page': self.filter_by_page,
            'chunk_size': self.filter_by_chunk_size,
            'content_length': self.filter_by_content_length
        }
    
    def filter_documents(self, docs: List[Document], filter_criteria: Dict[str, Any]) -> List[Document]:
        """Apply multiple filters to documents"""
        filtered_docs = docs
        
        for filter_type, filter_value in filter_criteria.items():
            if filter_type in self.available_filters:
                filtered_docs = self.available_filters[filter_type](filtered_docs, filter_value)
        
        return filtered_docs
    
    def filter_by_source(self, docs: List[Document], pattern: str) -> List[Document]:
        """Filter documents by source filename pattern"""
        pattern = pattern.lower()
        return [doc for doc in docs if 'source' in doc.metadata and pattern in doc.metadata['source'].lower()]
    
    def filter_by_date(self, docs: List[Document], date_range: str) -> List[Document]:
        """Filter documents by date range"""
        # Supports formats: "2024", "2024-01", "2024-01-01", "2024-01-01:2024-12-31"
        if ':' in date_range:
            start_str, end_str = date_range.split(':', 1)
            start_date = self._parse_date(start_str)
            end_date = self._parse_date(end_str)
            
            def in_date_range(doc):
                if 'created_date' in doc.metadata:
                    doc_date = self._parse_date(doc.metadata['created_date'])
                    return start_date <= doc_date <= end_date if doc_date else False
                return False
            
            return [doc for doc in docs if in_date_range(doc)]
        else:
            target_date = self._parse_date(date_range)
            return [doc for doc in docs if 'created_date' in doc.metadata and self._parse_date(doc.metadata['created_date']) == target_date]
    
    def filter_by_page(self, docs: List[Document], page_range: str) -> List[Document]:
        """Filter documents by page range"""
        if ':' in page_range:
            start_page, end_page = map(int, page_range.split(':', 1))
            return [doc for doc in docs if 'page' in doc.metadata and start_page <= doc.metadata['page'] <= end_page]
        else:
            target_page = int(page_range)
            return [doc for doc in docs if 'page' in doc.metadata and doc.metadata['page'] == target_page]
    
    def filter_by_chunk_size(self, docs: List[Document], size_range: str) -> List[Document]:
        """Filter documents by chunk size"""
        if ':' in size_range:
            min_size, max_size = map(int, size_range.split(':', 1))
            return [doc for doc in docs if 'chunk_size' in doc.metadata and min_size <= doc.metadata['chunk_size'] <= max_size]
        else:
            target_size = int(size_range)
            return [doc for doc in docs if 'chunk_size' in doc.metadata and doc.metadata['chunk_size'] == target_size]
    
    def filter_by_content_length(self, docs: List[Document], length_range: str) -> List[Document]:
        """Filter documents by content length"""
        if ':' in length_range:
            min_len, max_len = map(int, length_range.split(':', 1))
            return [doc for doc in docs if min_len <= len(doc.page_content) <= max_len]
        else:
            target_len = int(length_range)
            return [doc for doc in docs if len(doc.page_content) == target_len]
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse various date formats"""
        try:
            if len(date_str) == 4:  # Year only
                return datetime.strptime(date_str, '%Y')
            elif len(date_str) == 7:  # Year-Month
                return datetime.strptime(date_str, '%Y-%m')
            else:  # Full date
                return datetime.strptime(date_str, '%Y-%m-%d')
        except:
            return None

    def get_available_filters(self) -> List[str]:
        """Return list of available filter types"""
        return list(self.available_filters.keys())