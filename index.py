#!/usr/bin/env python3
import streamlit as st
import json
import os
import tempfile
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Set page config first
st.set_page_config(
    page_title="RAG AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply dark theme CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .sidebar .sidebar-content {
        background-color: #262730;
    }
    .stButton>button {
        background-color: #4F8BF9;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #3A7DE8;
    }
    .stTextInput>div>div>input {
        background-color: #262730;
        color: #FAFAFA;
        border: 1px solid #4F8BF9;
    }
    .stTextArea>div>div>textarea {
        background-color: #262730;
        color: #FAFAFA;
        border: 1px solid #4F8BF9;
    }
    .stSelectbox>div>div>select {
        background-color: #262730;
        color: #FAFAFA;
    }
    .success-box {
        background-color: #2E7D32;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #D32F2F;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #1976D2;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Import your RAG system components
try:
    from app.retriever import HybridRetriever
    from app.generator import AnswerGenerator
    from app.ingestion import process_file, process_directory, get_supported_formats
    from app.evaluation import RAGEvaluator
    from app.caching import CacheManager
    from app.config import config
    from app.context import format_context, get_citations
    from app.utils import ensure_directories
except ImportError as e:
    st.error(f"Failed to import RAG components: {e}")
    st.stop()

# Initialize session state
if 'rag_initialized' not in st.session_state:
    st.session_state.rag_initialized = False
    st.session_state.retriever = None
    st.session_state.generator = None
    st.session_state.evaluator = None
    st.session_state.cache_manager = None
    st.session_state.chat_history = []
    st.session_state.uploaded_files = []

def initialize_rag_system():
    """Initialize the RAG system components"""
    try:
        ensure_directories()
        st.session_state.retriever = HybridRetriever()
        st.session_state.generator = AnswerGenerator()
        st.session_state.evaluator = RAGEvaluator()
        st.session_state.cache_manager = CacheManager()
        st.session_state.rag_initialized = True
        return True
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {e}")
        return False

def main():
    st.title("🤖 RAG AI Assistant")
    st.markdown("### Intelligent Document Search and Q&A System")
    
    # Initialize RAG system if not already done
    if not st.session_state.rag_initialized:
        with st.spinner("Initializing RAG system..."):
            if initialize_rag_system():
                st.success("✅ RAG system initialized successfully!")
            else:
                st.error("❌ Failed to initialize RAG system")
                return
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "📁 Document Ingestion", 
        "💬 Chat & Query", 
        "📊 Evaluation", 
        "⚙️ System Management",
        "📈 Analytics"
    ])
    
    # Main content based on selected page
    if page == "📁 Document Ingestion":
        render_ingestion_page()
    elif page == "💬 Chat & Query":
        render_chat_page()
    elif page == "📊 Evaluation":
        render_evaluation_page()
    elif page == "⚙️ System Management":
        render_management_page()
    elif page == "📈 Analytics":
        render_analytics_page()

def render_ingestion_page():
    st.header("📁 Document Ingestion")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Files")
        uploaded_files = st.file_uploader(
            "Choose files to upload", 
            type=get_supported_formats(),
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files
            st.info(f"📄 {len(uploaded_files)} files ready for ingestion")
    
    with col2:
        st.subheader("Ingestion Options")
        process_method = st.radio(
            "Process method",
            ["Individual files", "Batch directory"]
        )
        
        if st.button("🚀 Process Documents", type="primary"):
            if process_method == "Individual files" and st.session_state.uploaded_files:
                process_uploaded_files()
            elif process_method == "Batch directory":
                process_directory_ui()
            else:
                st.warning("Please upload files first")

def process_uploaded_files():
    """Process uploaded files"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    successful = 0
    total = len(st.session_state.uploaded_files)
    
    for i, uploaded_file in enumerate(st.session_state.uploaded_files):
        try:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Process the file
            chunks = process_file(tmp_path)
            st.session_state.retriever.add_documents(chunks)
            
            # Clean up
            os.unlink(tmp_path)
            
            successful += 1
            status_text.text(f"✅ Processed {uploaded_file.name} ({len(chunks)} chunks)")
            
        except Exception as e:
            status_text.text(f"❌ Failed to process {uploaded_file.name}: {str(e)}")
        
        progress_bar.progress((i + 1) / total)
    
    if successful > 0:
        st.success(f"🎉 Successfully processed {successful}/{total} files!")
    else:
        st.error("❌ No files were processed successfully")

def process_directory_ui():
    """Process directory of files"""
    directory_path = st.text_input("Enter directory path", "./data/inbox/")
    
    if st.button("Process Directory", key="process_dir"):
        if os.path.exists(directory_path):
            try:
                with st.spinner("Processing directory..."):
                    chunks = process_directory(directory_path)
                    if chunks:
                        st.session_state.retriever.add_documents(chunks)
                        st.success(f"✅ Processed {len(chunks)} chunks from directory")
                    else:
                        st.warning("No documents found in directory")
            except Exception as e:
                st.error(f"❌ Error processing directory: {e}")
        else:
            st.error("❌ Directory does not exist")

# Replace the render_chat_page function with this fixed version:
def render_chat_page():
    st.header("💬 Chat & Query")
    
    # Chat history
    if st.session_state.chat_history:
        st.subheader("Chat History")
        for chat in st.session_state.chat_history[-10:]:  # Show last 10 messages
            with st.chat_message(chat["role"]):
                st.write(chat["content"])
                if chat.get("sources"):
                    with st.expander("📚 Sources"):
                        for source in chat["sources"]:
                            st.write(f"• {source}")
    
    # Query input
    query = st.chat_input("Ask a question about your documents...")
    
    if query:
        # Add user message to chat
        st.session_state.chat_history.append({"role": "user", "content": query})
        
        # Display user message immediately
        with st.chat_message("user"):
            st.write(query)
        
        # Process query
        with st.spinner("🔍 Searching documents..."):
            try:
                # Advanced filters section (moved outside to avoid duplication)
                filters = {}
                
                # Retrieve documents
                docs = st.session_state.retriever.retrieve(query, filters=filters)
                
                # Generate answer
                answer = st.session_state.generator.generate_answer(query, docs)
                
                # Get citations
                citations = get_citations(docs)
                
                # Add assistant response to chat
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": citations
                })
                
                # Display response
                with st.chat_message("assistant"):
                    st.write(answer)
                    if citations:
                        with st.expander("📚 Sources"):
                            for citation in citations:
                                st.write(f"• {citation}")
                
                # Show context if available
                if docs:
                    with st.expander("🔍 Retrieved Context"):
                        st.text(format_context(docs))
                        
            except Exception as e:
                error_msg = f"❌ Error processing query: {e}"
                st.error(error_msg)
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": error_msg
                })
    
    # Advanced filters (moved to separate section to avoid duplication)
    with st.expander("⚙️ Advanced Filters", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            source_filter = st.text_input("Source contains", "", key="source_filter")
            page_filter = st.text_input("Page range (e.g., 1:10)", "", key="page_filter")
        with col2:
            chunk_size_filter = st.text_input("Chunk size range", "", key="chunk_size_filter")
            content_length_filter = st.text_input("Content length range", "", key="content_length_filter")
    
    # Clear chat button - use a form to avoid rerun issues
    if st.button("🗑️ Clear Chat History", key="clear_chat"):
        st.session_state.chat_history = []
        st.success("Chat history cleared!")

def render_evaluation_page():
    st.header("📊 Evaluation")
    
    tab1, tab2, tab3 = st.tabs(["Add QA Pair", "Run Evaluation", "Evaluation Results"])
    
    with tab1:
        st.subheader("Add Evaluation QA Pair")
        
        question = st.text_area("Question")
        expected_docs = st.text_input("Expected Document IDs (comma-separated)")
        reference_answer = st.text_area("Reference Answer (optional)")
        
        if st.button("Add QA Pair"):
            if question and expected_docs:
                try:
                    expected_doc_ids = [doc_id.strip() for doc_id in expected_docs.split(",")]
                    st.session_state.evaluator.add_evaluation_pair(question, expected_doc_ids, reference_answer)
                    st.success("✅ QA pair added successfully!")
                except Exception as e:
                    st.error(f"❌ Error adding QA pair: {e}")
            else:
                st.warning("Please provide both question and expected documents")
    
    with tab2:
        st.subheader("Run Evaluation")
        
        num_samples = st.slider("Number of samples", 1, 20, 5)
        
        if st.button("🚀 Run Evaluation", type="primary"):
            with st.spinner("Running evaluation..."):
                try:
                    results = st.session_state.evaluator.run_comprehensive_evaluation(
                        st.session_state.retriever, st.session_state.generator, num_samples
                    )
                    st.session_state.eval_results = results
                    st.success("✅ Evaluation completed!")
                except Exception as e:
                    st.error(f"❌ Evaluation failed: {e}")
    
    with tab3:
        st.subheader("Evaluation Results")
        
        if 'eval_results' in st.session_state:
            results = st.session_state.eval_results
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Precision", f"{results['retrieval']['mean_precision']:.3f}")
            with col2:
                st.metric("Mean Recall", f"{results['retrieval']['mean_recall']:.3f}")
            with col3:
                st.metric("Mean F1 Score", f"{results['retrieval']['mean_f1']:.3f}")
            
            # Visualization
            fig = go.Figure(data=[
                go.Bar(name='Precision', x=['Retrieval'], y=[results['retrieval']['mean_precision']]),
                go.Bar(name='Recall', x=['Retrieval'], y=[results['retrieval']['mean_recall']]),
                go.Bar(name='F1 Score', x=['Retrieval'], y=[results['retrieval']['mean_f1']])
            ])
            fig.update_layout(barmode='group', title='Retrieval Metrics')
            st.plotly_chart(fig)
            
        else:
            st.info("Run an evaluation to see results here")

def render_management_page():
    st.header("⚙️ System Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("System Status")
        try:
            stats = st.session_state.retriever.get_stats()
            
            st.metric("Total Documents", stats['total_documents'])
            st.metric("Vectorstore Ready", "✅" if stats['vectorstore_ready'] else "❌")
            st.metric("BM25 Ready", "✅" if stats['bm25_ready'] else "❌")
            
            # Cache statistics
            cache_stats = st.session_state.cache_manager.get_stats()
            st.metric("Cache Files", cache_stats['total_files'])
            st.metric("Expired Cache Files", cache_stats['expired_files'])
            
        except Exception as e:
            st.error(f"Error getting system status: {e}")
    
    with col2:
        st.subheader("Management Actions")
        
        if st.button("🔄 Refresh Status", key="refresh_status"):
            st.rerun()
        
        # Clear data with confirmation
        if st.button("🗑️ Clear All Data", key="clear_data"):
            st.warning("This will delete ALL data including documents, cache, and chat history!")
            if st.checkbox("I understand this action cannot be undone", key="clear_confirm"):
                try:
                    st.session_state.retriever.clear_all_data()
                    st.session_state.chat_history = []
                    st.session_state.cache_manager.clear_all_cache()
                    st.success("✅ All data cleared successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error clearing data: {e}")
        
        # Cache management
        st.subheader("Cache Management")
        col_cache1, col_cache2 = st.columns(2)
        
        with col_cache1:
            if st.button("Clear Expired Cache", key="clear_expired"):
                try:
                    cleared = st.session_state.cache_manager.clear_expired_cache()
                    st.success(f"✅ Cleared {cleared} expired cache files")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error clearing cache: {e}")
        
        with col_cache2:
            if st.button("Clear All Cache", key="clear_all_cache"):
                try:
                    st.session_state.cache_manager.clear_all_cache()
                    st.success("✅ All cache cleared!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error clearing cache: {e}")

def render_analytics_page():
    st.header("📈 Analytics & Insights")
    
    # Document statistics
    stats = st.session_state.retriever.get_stats()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Documents", stats['total_documents'])
    with col2:
        st.metric("Vectorstore Status", "Active" if stats['vectorstore_ready'] else "Inactive")
    with col3:
        st.metric("BM25 Status", "Active" if stats['bm25_ready'] else "Inactive")
    
    # Chat statistics
    st.subheader("Chat Statistics")
    if st.session_state.chat_history:
        user_messages = len([m for m in st.session_state.chat_history if m['role'] == 'user'])
        assistant_messages = len([m for m in st.session_state.chat_history if m['role'] == 'assistant'])
        
        fig = px.pie(
            values=[user_messages, assistant_messages],
            names=['User Messages', 'Assistant Messages'],
            title='Message Distribution'
        )
        st.plotly_chart(fig)
    else:
        st.info("No chat history yet")
    
    # System information
    st.subheader("System Information")
    sys_info = {
        "Component": ["Embedding Model", "Reranker Model", "LLM Model", "Chunk Size", "Top K"],
        "Value": [
            config.EMBEDDING_MODEL,
            config.RERANKER_MODEL,
            config.LLM_MODEL,
            config.CHUNK_SIZE,
            config.TOP_K
        ]
    }
    st.table(pd.DataFrame(sys_info))

if __name__ == "__main__":
    main()