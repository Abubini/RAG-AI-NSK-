#!/usr/bin/env python3
import argparse
import json
import os
import sys

# Set environment variables early to prevent CUDA issues
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable CUDA
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from app.ingestion import process_file, process_directory
from app.retriever import HybridRetriever
from app.generator import AnswerGenerator
from app.context import format_context, get_citations
from app.utils import ensure_directories, print_colored
from app.config import config
from app.evaluation import RAGEvaluator
from app.caching import CacheManager

class RAGCLI:
    def __init__(self):
        ensure_directories()
        self.retriever = HybridRetriever()
        self.generator = AnswerGenerator()
        self.evaluator = RAGEvaluator()
        self.cache_manager = CacheManager()
    
    def ingest(self, file_path: str = None, directory: str = None):
        """Ingest documents into the system"""
        try:
            if file_path:
                if not os.path.exists(file_path):
                    print_colored(f"❌ File not found: {file_path}", "red")
                    return
                chunks = process_file(file_path)
                print_colored(f"📄 Processed {file_path}: {len(chunks)} chunks", "green")
            elif directory:
                if not os.path.exists(directory):
                    print_colored(f"❌ Directory not found: {directory}", "red")
                    return
                chunks = process_directory(directory)
                print_colored(f"📁 Processed directory {directory}: {len(chunks)} total chunks", "green")
            else:
                print_colored("❌ Please specify either --file or --directory", "red")
                return
            
            if chunks:
                self.retriever.add_documents(chunks)
                stats = self.retriever.get_stats()
                print_colored(f"✅ Success! Total documents: {stats['total_documents']}", "green")
            else:
                print_colored("⚠️  No documents were processed", "yellow")
                
        except Exception as e:
            print_colored(f"❌ Error during ingestion: {str(e)}", "red")
            import traceback
            traceback.print_exc()
    
    def query(self, question: str, show_context: bool = False, 
            filters_str: str = None, use_cache: bool = True):
        """Query the RAG system with optional filters"""
        try:
            print_colored(f"\n🔍 Searching for: {question}", "blue")
            
            # Parse filters if provided
            filters = None
            if filters_str:
                try:
                    import json
                    filters = json.loads(filters_str)
                    print_colored(f"   🎯 Applying filters: {filters}", "yellow")
                except json.JSONDecodeError as e:
                    print_colored(f"❌ Invalid JSON filters: {e}", "red")
                    return
            
            # Check if retriever is ready
            stats = self.retriever.get_stats()
            if not stats['vectorstore_ready'] or not stats['bm25_ready']:
                print_colored("❌ No documents indexed yet. Please run 'ingest' first.", "red")
                return
            
            # Retrieve relevant documents with filters
            docs = self.retriever.retrieve(question, filters=filters, use_cache=use_cache)
            
            if show_context:
                print_colored("\n📄 Retrieved Context:", "yellow")
                context_str = format_context(docs)
                print(context_str if context_str else "No context available")
            
            # Generate answer
            print_colored("\n💡 Answer:", "green")
            answer = self.generator.generate_answer(question, docs)
            print(answer)
            
            # Show citations
            if docs:
                print_colored("\n📚 Citations:", "yellow")
                for citation in get_citations(docs):
                    print(f"  • {citation}")
            else:
                print_colored("\n⚠️  No documents found for citation", "yellow")
            
        except Exception as e:
            print_colored(f"❌ Error during query: {str(e)}", "red")
            import traceback
            traceback.print_exc()
    
    def status(self):
        """Show system status"""
        stats = self.retriever.get_stats()
        print_colored("\n📊 System Status:", "blue")
        print(f"   Total documents: {stats['total_documents']}")
        print(f"   Vectorstore ready: {'✅' if stats['vectorstore_ready'] else '❌'}")
        print(f"   BM25 ready: {'✅' if stats['bm25_ready'] else '❌'}")
        print(f"   Persist directory: {config.PERSIST_DIR}")
    
    def clear(self):
        """Clear all stored data"""
        confirm = input("❓ Are you sure you want to clear ALL data? This cannot be undone. (y/N): ")
        if confirm.lower() == 'y':
            try:
                self.retriever.clear_all_data()
                # Create a fresh retriever instance to ensure clean state
                self.retriever = HybridRetriever()
                print_colored("✅ All data cleared and system reset!", "green")
            except Exception as e:
                print_colored(f"❌ Error during clear: {str(e)}", "red")
        else:
            print_colored("❌ Clear operation cancelled", "yellow")
    
    def interactive_mode(self):
        """Start interactive query mode"""
        # Check status first
        self.status()
        
        print_colored("\n🤖 RAG CLI - Interactive Mode", "blue")
        print_colored("Type 'quit', 'exit', or 'q' to end the session", "yellow")
        print_colored("Type 'status' to see system status\n", "yellow")
        
        while True:
            try:
                question = input("❓ Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print_colored("👋 Goodbye!", "green")
                    break
                elif question.lower() == 'status':
                    self.status()
                    continue
                elif question.lower() == '':
                    continue
                
                if question:
                    self.query(question, show_context=True)
                    print()  # Empty line for readability
                    
            except KeyboardInterrupt:
                print_colored("\n👋 Goodbye!", "green")
                break
            except Exception as e:
                print_colored(f"❌ Error: {str(e)}", "red")
    def evaluate(self, num_samples: int = 5):
        """Run comprehensive evaluation"""
        try:
            results = self.evaluator.run_comprehensive_evaluation(
                self.retriever, self.generator, num_samples
            )
            
            print_colored("\n📊 Evaluation Results:", "blue")
            print(json.dumps(results, indent=2))
            
        except Exception as e:
            print_colored(f"❌ Evaluation error: {str(e)}", "red")
    
    def add_eval_pair(self, question: str, expected_docs: str, reference_answer: str = ""):
        """Add evaluation QA pair"""
        try:
            # Parse expected document IDs (comma-separated)
            expected_doc_ids = [doc_id.strip() for doc_id in expected_docs.split(',')]
            
            self.evaluator.add_evaluation_pair(question, expected_doc_ids, reference_answer)
            print_colored("✅ Evaluation pair added successfully!", "green")
            
        except Exception as e:
            print_colored(f"❌ Error adding evaluation pair: {str(e)}", "red")
    
    def clear_cache(self, expired_only: bool = False):
        """Clear cache files"""
        try:
            if expired_only:
                cleared_count = self.cache_manager.clear_expired_cache()
                print_colored(f"✅ Cleared {cleared_count} expired cache files", "green")
            else:
                self.cache_manager.clear_all_cache()
                print_colored("✅ Cleared all cache files", "green")
                
        except Exception as e:
            print_colored(f"❌ Error clearing cache: {str(e)}", "red")
    
    def cache_stats(self):
        """Show cache statistics"""
        cache_dir = self.cache_manager.cache_dir
        if os.path.exists(cache_dir):
            cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.json')]
            print_colored(f"\n📊 Cache Statistics:", "blue")
            print(f"   Total cache files: {len(cache_files)}")
            print(f"   Cache directory: {cache_dir}")
        else:
            print_colored("❌ Cache directory not found", "red")

def main():
    parser = argparse.ArgumentParser(description="RAG CLI System")
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest documents')
    ingest_group = ingest_parser.add_mutually_exclusive_group(required=True)
    ingest_group.add_argument('--file', help='Path to a single PDF file')
    ingest_group.add_argument('--directory', help='Path to directory containing PDFs')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query the system')
    query_parser.add_argument('question', help='Question to ask')
    query_parser.add_argument('--context', action='store_true', help='Show retrieved context')
    query_parser.add_argument('--filters', type=str, help='JSON string of filters to apply')
    query_parser.add_argument('--no-cache', action='store_true', help='Disable caching')

    # Add to main() function after other subparsers
    clear_parser = subparsers.add_parser('clear', help='Clear all stored data')
    
    # Status command
    subparsers.add_parser('status', help='Show system status')
    
    # Interactive command
    subparsers.add_parser('chat', help='Start interactive chat mode')

    # Evaluation commands
    eval_parser = subparsers.add_parser('evaluate', help='Run system evaluation')
    eval_parser.add_argument('--samples', type=int, default=5, help='Number of samples to evaluate')

    add_eval_parser = subparsers.add_parser('add-eval', help='Add evaluation QA pair')
    add_eval_parser.add_argument('--question', required=True, help='Evaluation question')
    add_eval_parser.add_argument('--expected-docs', required=True, help='Comma-separated expected document IDs')
    add_eval_parser.add_argument('--reference-answer', help='Reference answer (optional)')

    # Cache commands
    cache_parser = subparsers.add_parser('cache', help='Cache management')
    cache_parser.add_argument('--clear-expired', action='store_true', help='Clear only expired cache')
    cache_parser.add_argument('--clear-all', action='store_true', help='Clear all cache')
    cache_parser.add_argument('--stats', action='store_true', help='Show cache statistics')
    
    args = parser.parse_args()
    rag = RAGCLI()
    
    if args.command == 'ingest':
        rag.ingest(args.file, args.directory)
    elif args.command == 'query':
        rag.query(args.question, args.context, args.filters, not args.no_cache)
    elif args.command == 'status':
        rag.status()
    elif args.command == 'chat':
        rag.interactive_mode()
    elif args.command == 'evaluate':
        rag.evaluate(args.samples)
    elif args.command == 'add-eval':
        rag.add_eval_pair(args.question, args.expected_docs, args.reference_answer)
    elif args.command == 'clear':
        rag.clear()
    elif args.command == 'cache':
        if args.clear_expired:
            rag.clear_cache(expired_only=True)
        elif args.clear_all:
            rag.clear_cache(expired_only=False)
        elif args.stats:
            rag.cache_stats()
        else:
            cache_parser.print_help()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()