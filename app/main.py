#!/usr/bin/env python3
import argparse
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

class RAGCLI:
    def __init__(self):
        ensure_directories()
        self.retriever = HybridRetriever()
        self.generator = AnswerGenerator()
    
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
    
    def query(self, question: str, show_context: bool = False):
        """Query the RAG system"""
        try:
            print_colored(f"\n🔍 Searching for: {question}", "blue")
            
            # Check if retriever is ready
            stats = self.retriever.get_stats()
            if not stats['vectorstore_ready'] or not stats['bm25_ready']:
                print_colored("❌ No documents indexed yet. Please run 'ingest' first.", "red")
                return
            
            # Retrieve relevant documents
            docs = self.retriever.retrieve(question)
            
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
            self.retriever.clear_all_data()
            print_colored("✅ All data cleared!", "green")
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

    # Add to main() function after other subparsers
    clear_parser = subparsers.add_parser('clear', help='Clear all stored data')
    
    # Status command
    subparsers.add_parser('status', help='Show system status')
    
    # Interactive command
    subparsers.add_parser('chat', help='Start interactive chat mode')
    
    args = parser.parse_args()
    rag = RAGCLI()
    
    if args.command == 'ingest':
        rag.ingest(args.file, args.directory)
    elif args.command == 'query':
        rag.query(args.question, args.context)
    elif args.command == 'status':
        rag.status()
    elif args.command == 'chat':
        rag.interactive_mode()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()