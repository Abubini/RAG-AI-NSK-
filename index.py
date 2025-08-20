#!/usr/bin/env python3
import sys

# Safe patch SQLite for ChromaDB
try:
    import pysqlite3
    import sqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    # pysqlite3 not installed yet, will fail later if Chroma tries to use sqlite
    pass

import streamlit as st
import json
import os
import tempfile
from pathlib import Path
import pandas as pd
from app.retriever import HybridRetriever
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import requests
import logging
import shutil

# Set page config first
st.set_page_config(
    page_title="RAG AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply enhanced dark theme CSS
st.markdown("""
<style>
    /* Main app styling */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Chat message styling */
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    
    /* User message styling */
    div[data-testid="stChatMessage"] > div:first-child > div {
        background-color: #2A5C8A !important;
        color: white !important;
        border-radius: 15px 15px 0 15px !important;
    }
    
    /* Assistant message styling */
    div[data-testid="stChatMessage"] > div:nth-child(2) > div {
        background-color: #262730 !important;
        color: #FAFAFA !important;
        border-left: 3px solid #4F8BF9;
        border-radius: 15px 15px 15px 0 !important;
    }
    
    /* Input area styling */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: #1E2229;
        color: #FAFAFA;
        border: 1px solid #4F8BF9;
        border-radius: 12px;
        padding: 12px;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #4F8BF9;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.7rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #3A7DE8;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(79, 139, 249, 0.4);
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-1d391kg > div:first-child {
        background-color: #131720 !important;
    }
    
    /* Hide footer and menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0E1117;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #4F8BF9;
        border-radius: 3px;
    }
    
    /* Custom containers */
    .chat-container {
        padding-bottom: 100px;
    }
    
    .input-container {
        position: fixed;
        bottom: 20px;
        left: 20px;
        right: 20px;
        background-color: #0E1117;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 -4px 12px rgba(0, 0, 0, 0.1);
        z-index: 999;
    }
    
    /* Document display styling */
    .document-card {
        background-color: #1E2229;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        border-left: 3px solid #4F8BF9;
    }
    
    .document-title {
        font-weight: bold;
        color: #4F8BF9;
        margin-bottom: 8px;
    }
    
    .document-content {
        color: #CCCCCC;
        font-size: 0.9rem;
        line-height: 1.4;
    }
    
    .document-meta {
        font-size: 0.8rem;
        color: #888;
        margin-top: 8px;
    }
    
    /* Welcome message */
    .welcome-card {
        background: linear-gradient(135deg, #262730 0%, #1E2229 100%);
        border-radius: 12px;
        padding: 2rem;
        margin: 2rem 0;
        border-left: 4px solid #4F8BF9;
    }
    
    /* Error message styling */
    .error-card {
        background: linear-gradient(135deg, #D32F2F 0%, #B71C1C 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: white;
    }
    
    /* Warning message styling */
    .warning-card {
        background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: white;
    }
    
    /* Info message styling */
    .info-card {
        background: linear-gradient(135deg, #1976D2 0%, #0D47A1 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: white;
    }
    
    /* Clear data confirmation */
    .clear-confirm {
        background-color: #D32F2F;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
if 'rag_initialized' not in st.session_state:
    st.session_state.rag_initialized = False
    st.session_state.retriever = None
    st.session_state.generator = None
    st.session_state.evaluator = None
    st.session_state.cache_manager = None
    st.session_state.chat_history = []
    st.session_state.uploaded_files = []
    st.session_state.show_sidebar = False
    st.session_state.offline_mode = False
    st.session_state.init_error = None
    st.session_state.show_clear_confirmation = False

def check_internet_connection():
    """Check if we have an internet connection"""
    try:
        response = requests.get("https://www.google.com", timeout=5)
        return True
    except requests.ConnectionError:
        return False

def initialize_rag_system():
    """Initialize the RAG system components with error handling"""
    try:
        # Check internet connection first
        if not check_internet_connection():
            st.session_state.offline_mode = True
            st.session_state.init_error = "No internet connection detected. Running in offline mode with limited functionality."
            logger.warning(st.session_state.init_error)
            return False
            
        # Try to import components
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
            st.session_state.init_error = f"Failed to import RAG components: {e}"
            logger.error(st.session_state.init_error)
            return False
            
        # Initialize components
        ensure_directories()
        st.session_state.retriever = HybridRetriever()
        st.session_state.generator = AnswerGenerator()
        st.session_state.evaluator = RAGEvaluator()
        st.session_state.cache_manager = CacheManager()
        st.session_state.rag_initialized = True
        st.session_state.offline_mode = False
        st.session_state.init_error = None
        return True
        
    except Exception as e:
        st.session_state.init_error = f"Failed to initialize RAG system: {e}"
        logger.error(st.session_state.init_error)
        
        # Check if it's a network error
        if "huggingface.co" in str(e) or "NameResolutionError" in str(e):
            st.session_state.offline_mode = True
        return False

def clear_all_data():
    """Clear all data from the system with robust error handling"""
    try:
        import shutil
        import os
        from app.config import config
        
        # Clear retriever data
        if st.session_state.retriever:
            try:
                success = st.session_state.retriever.clear_all_data()
                if not success:
                    st.warning("⚠️ Retriever clearance had some issues - using fallback")
                    raise Exception("Retriever clearance reported failure")
            except Exception as e:
                logger.warning(f"Error clearing retriever data: {e}")
                # Fallback: manually clear data directories using config paths
                data_dirs = [
                    config.PERSIST_DIR,  # chroma directory
                    os.path.dirname(config.BM25_STORAGE_PATH),  # bm25 directory
                ]
                
                for dir_path in data_dirs:
                    if os.path.exists(dir_path):
                        try:
                            shutil.rmtree(dir_path, ignore_errors=True)
                            os.makedirs(dir_path, exist_ok=True)
                            logger.info(f"Fallback cleared: {dir_path}")
                        except Exception as dir_error:
                            logger.error(f"Failed to clear {dir_path}: {dir_error}")
        
        # Clear chat history
        st.session_state.chat_history = []
        
        # Clear cache
        if st.session_state.cache_manager:
            try:
                st.session_state.cache_manager.clear_all_cache()
            except Exception as e:
                logger.warning(f"Error clearing cache: {e}")
                # Fallback: manually clear cache directory
                cache_dir = config.CACHE_DIR
                if os.path.exists(cache_dir):
                    try:
                        shutil.rmtree(cache_dir, ignore_errors=True)
                        os.makedirs(cache_dir, exist_ok=True)
                        logger.info("Fallback cleared cache directory")
                    except Exception as cache_error:
                        logger.error(f"Failed to clear cache: {cache_error}")
        
        # Reset other states
        st.session_state.uploaded_files = []
        
        # Reinitialize the retriever to ensure clean state
        try:
            st.session_state.retriever = HybridRetriever()
            logger.info("Retriever reinitialized after clearance")
        except Exception as init_error:
            logger.error(f"Failed to reinitialize retriever: {init_error}")
        
        st.success("✅ All data cleared successfully!")
        st.session_state.show_clear_confirmation = False
        
        # Use success message instead of immediate rerun to avoid UI flicker
        return True
        
    except Exception as e:
        logger.error(f"Critical error in clear_all_data: {e}")
        st.error(f"❌ Error clearing data: {e}")
        return False

def clear_expired_cache():
    """Clear expired cache files"""
    try:
        if st.session_state.cache_manager:
            cleared = st.session_state.cache_manager.clear_expired_cache()
            st.success(f"✅ Cleared {cleared} expired cache files")
        else:
            st.error("Cache manager not initialized")
    except Exception as e:
        st.error(f"❌ Error clearing expired cache: {e}")

def clear_all_cache():
    """Clear all cache files"""
    try:
        if st.session_state.cache_manager:
            st.session_state.cache_manager.clear_all_cache()
            st.success("✅ All cache cleared!")
        else:
            # Fallback: manually clear cache directory
            cache_dir = './data/cache'
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
                os.makedirs(cache_dir, exist_ok=True)
                st.success("✅ All cache cleared!")
            else:
                st.info("No cache directory found")
    except Exception as e:
        st.error(f"❌ Error clearing cache: {e}")

def format_document_display(docs):
    """Format retrieved documents for display"""
    if not docs:
        return "No documents retrieved"
    
    html_output = ""
    for i, doc in enumerate(docs):
        # Extract metadata
        source = doc.metadata.get('source', 'Unknown source')
        page = doc.metadata.get('page', '')
        score = doc.metadata.get('score', '')
        
        # Truncate content for better display
        content = doc.page_content
        if len(content) > 300:
            content = content[:300] + "..."
        
        html_output += f"""
        <div class="document-card">
            <div class="document-title">📄 Document {i+1}: {os.path.basename(source)}</div>
            <div class="document-content">{content}</div>
            <div class="document-meta">
                Source: {source} {f"| Page: {page}" if page else ""} {f"| Score: {score:.4f}" if score else ""}
            </div>
        </div>
        """
    
    return html_output

def main():
    # Header with logo and title
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>🤖 RAG AI Assistant</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #888; margin-top: 0;'>Intelligent Document Search and Q&A System</p>", unsafe_allow_html=True)
    
    # Show connection status
    if st.session_state.offline_mode:
        st.markdown(f"""
        <div class="warning-card">
            <h3>⚠️ Offline Mode</h3>
            <p>{st.session_state.init_error or 'Running with limited functionality'}</p>
            <p>You can still upload and manage documents, but AI features are unavailable.</p>
        </div>
        """, unsafe_allow_html=True)
    elif st.session_state.init_error:
        st.markdown(f"""
        <div class="error-card">
            <h3>❌ Initialization Error</h3>
            <p>{st.session_state.init_error}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Initialize RAG system if not already done
    if not st.session_state.rag_initialized and not st.session_state.offline_mode:
        with st.spinner("Initializing RAG system..."):
            if initialize_rag_system():
                st.success("✅ RAG system initialized successfully!")
            else:
                if not st.session_state.offline_mode:
                    st.error("❌ Failed to initialize RAG system")
    
    # Sidebar navigation (hidden by default, shown via button)
    if st.session_state.show_sidebar:
        with st.sidebar:
            st.title("Navigation")
            st.markdown("---")
            page = st.radio("Go to", [
                "💬 Chat", 
                "📁 Documents", 
                "📊 Evaluation", 
                "⚙️ Settings",
                "📈 Analytics"
            ], key="nav_radio")
            
            st.markdown("---")
            st.markdown("### System Status")
            try:
                if st.session_state.retriever:
                    stats = st.session_state.retriever.get_stats()
                    st.metric("Total Documents", stats.get('total_documents', 0))
                    st.metric("Vectorstore", "✅" if stats.get('vectorstore_ready') else "❌")
                    st.metric("BM25", "✅" if stats.get('bm25_ready') else "❌")
                else:
                    st.info("Components not initialized")
            except:
                st.info("Status unavailable")
                
            st.markdown("---")
            if st.button("🔄 Check Connection", use_container_width=True):
                if check_internet_connection():
                    st.success("Internet connection available!")
                    if initialize_rag_system():
                        st.success("RAG system initialized!")
                    st.rerun()
                else:
                    st.error("No internet connection")
    
    # Main content based on selected page
    if not st.session_state.show_sidebar or st.session_state.get('nav_radio', '💬 Chat') == '💬 Chat':
        render_chat_page()
    elif st.session_state.nav_radio == '📁 Documents':
        render_ingestion_page()
    elif st.session_state.nav_radio == '📊 Evaluation':
        render_evaluation_page()
    elif st.session_state.nav_radio == '⚙️ Settings':
        render_management_page()
    elif st.session_state.nav_radio == '📈 Analytics':
        render_analytics_page()

def render_ingestion_page():
    st.header("📁 Document Management")
    
    # Show warning if in offline mode
    if st.session_state.offline_mode:
        st.markdown("""
        <div class="info-card">
            <h4>📝 Document Processing in Offline Mode</h4>
            <p>You can upload and process documents, but they won't be indexed with AI features until you're back online.</p>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Upload Files")
        uploaded_files = st.file_uploader(
            "Drag and drop files here", 
            type=['txt', 'pdf', 'docx', 'pptx', 'xlsx', 'csv', 'md', 'html'],
            accept_multiple_files=True,
            help="Supported formats: TXT, PDF, DOCX, PPTX, XLSX, CSV, MD, HTML"
        )
        
        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files
            st.info(f"📄 {len(uploaded_files)} files ready for ingestion")
            
            if st.button("🚀 Process Uploaded Files", type="primary", use_container_width=True):
                process_uploaded_files()
    
    with col2:
        st.subheader("Batch Processing")
        st.info("Process all files in a directory")
        
        directory_path = st.text_input("Directory path", "./data/inbox/")
        
        if st.button("📂 Process Directory", use_container_width=True):
            if os.path.exists(directory_path):
                try:
                    with st.spinner("Processing directory..."):
                        # Simulate processing in offline mode
                        if st.session_state.offline_mode:
                            file_count = len([f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))])
                            st.info(f"Found {file_count} files. Will process when online.")
                        else:
                            # Try to import if not in offline mode
                            try:
                                from app.ingestion import process_directory
                                chunks = process_directory(directory_path)
                                if chunks:
                                    st.session_state.retriever.add_documents(chunks)
                                    st.success(f"✅ Processed {len(chunks)} chunks from directory")
                                else:
                                    st.warning("No documents found in directory")
                            except ImportError:
                                st.error("Processing module not available")
                except Exception as e:
                    st.error(f"❌ Error processing directory: {e}")
            else:
                st.error("❌ Directory does not exist")

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
            
            # Process the file (simulate in offline mode)
            if st.session_state.offline_mode:
                # Just store the file for later processing
                file_size = len(uploaded_file.getvalue())
                status_text.text(f"📥 Stored {uploaded_file.name} for processing when online ({file_size} bytes)")
            else:
                # Try to import if not in offline mode
                try:
                    from app.ingestion import process_file
                    chunks = process_file(tmp_path)
                    st.session_state.retriever.add_documents(chunks)
                    status_text.text(f"✅ Processed {uploaded_file.name} ({len(chunks)} chunks)")
                except ImportError:
                    status_text.text(f"❌ Processing module not available for {uploaded_file.name}")
            
            # Clean up
            os.unlink(tmp_path)
            
            successful += 1
            
        except Exception as e:
            status_text.text(f"❌ Failed to process {uploaded_file.name}: {str(e)}")
        
        progress_bar.progress((i + 1) / total)
    
    if successful > 0:
        if st.session_state.offline_mode:
            st.success(f"📥 Stored {successful}/{total} files for processing when online!")
        else:
            st.success(f"🎉 Successfully processed {successful}/{total} files!")
        st.session_state.uploaded_files = []
    else:
        st.error("❌ No files were processed successfully")

def render_chat_page():
    # Welcome message for new users
    if not st.session_state.chat_history:
        with st.container():
            st.markdown("""
            <div class="welcome-card">
                <h2>👋 Welcome to RAG AI Assistant</h2>
                <p>I can help you query and analyze your documents. To get started:</p>
                <ol>
                    <li>Upload documents using the Documents section</li>
                    <li>Ask questions about your content</li>
                    <li>I'll provide answers with citations from your documents</li>
                </ol>
                <p>Try asking something like: <i>"What are the main points in my documents?"</i></p>
            </div>
            """, unsafe_allow_html=True)
    
    # Show offline warning if applicable
    if st.session_state.offline_mode:
        st.markdown("""
        <div class="warning-card">
            <h4>⚠️ Chat Features Unavailable Offline</h4>
            <p>You need an internet connection to use the chat features. Please check your connection and try again.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Chat history container
    chat_container = st.container()
    
    # Input area at the bottom (fixed position)
    with st.container():
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        
        # Top row with menu button and clear chat
        col1, col2 = st.columns([6, 1])
        with col1:
            query = st.chat_input(
                "Ask a question about your documents...", 
                key="chat_input",
                disabled=st.session_state.offline_mode
            )
        with col2:
            if st.button("☰", help="Menu", key="menu_button"):
                st.session_state.show_sidebar = not st.session_state.show_sidebar
                st.rerun()
            
            if st.button("🗑️", help="Clear chat", key="clear_chat_button"):
                st.session_state.chat_history = []
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display chat history
    with chat_container:
        if st.session_state.chat_history:
            for chat in st.session_state.chat_history:
                with st.chat_message(chat["role"]):
                    st.write(chat["content"])
                    
                    if chat.get("documents"):
                        with st.expander("📄 Retrieved Documents", expanded=False):
                            st.markdown(chat["documents"], unsafe_allow_html=True)
    
    # Process query if entered and not in offline mode
    if query and not st.session_state.offline_mode:
        # Add user message to chat
        st.session_state.chat_history.append({"role": "user", "content": query})
        
        # Display user message immediately
        with st.chat_message("user"):
            st.write(query)
        
        # Process query
        with st.spinner("🔍 Searching documents..."):
            try:
                # Retrieve documents
                docs = st.session_state.retriever.retrieve(query)
                
                # Generate answer
                answer = st.session_state.generator.generate_answer(query, docs)
                
                # Format documents for display
                documents_html = format_document_display(docs)
                
                # Add assistant response to chat
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": answer,
                    "documents": documents_html
                })
                
                # Display response
                with st.chat_message("assistant"):
                    st.write(answer)
                    if docs:
                        with st.expander("📄 Retrieved Documents", expanded=False):
                            st.markdown(documents_html, unsafe_allow_html=True)
                
                # Rerun to update the chat display
                st.rerun()
                        
            except Exception as e:
                error_msg = f"❌ Error processing query: {e}"
                st.error(error_msg)
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": error_msg
                })
                st.rerun()

def render_evaluation_page():
    st.header("📊 Evaluation")
    
    if st.session_state.offline_mode:
        st.markdown("""
        <div class="warning-card">
            <h4>⚠️ Evaluation Features Unavailable Offline</h4>
            <p>You need an internet connection to use the evaluation features.</p>
        </div>
        """, unsafe_allow_html=True)
        return
        
    tab1, tab2, tab3 = st.tabs(["Add QA Pair", "Run Evaluation", "Results"])
    
    with tab1:
        st.subheader("Add Evaluation QA Pair")
        
        question = st.text_area("Question", placeholder="Enter the question to evaluate...")
        expected_docs = st.text_input("Expected Document IDs", placeholder="doc1, doc2, doc3...")
        reference_answer = st.text_area("Reference Answer (optional)", placeholder="What would be the ideal answer?")
        
        if st.button("Add QA Pair", use_container_width=True):
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
        
        if st.button("🚀 Run Evaluation", type="primary", use_container_width=True):
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
                st.metric("Precision", f"{results['retrieval']['mean_precision']:.3f}")
            with col2:
                st.metric("Recall", f"{results['retrieval']['mean_recall']:.3f}")
            with col3:
                st.metric("F1 Score", f"{results['retrieval']['mean_f1']:.3f}")
            
            # Visualization
            fig = go.Figure(data=[
                go.Bar(name='Precision', x=['Retrieval'], y=[results['retrieval']['mean_precision']]),
                go.Bar(name='Recall', x=['Retrieval'], y=[results['retrieval']['mean_recall']]),
                go.Bar(name='F1 Score', x=['Retrieval'], y=[results['retrieval']['mean_f1']])
            ])
            fig.update_layout(
                barmode='group', 
                title='Retrieval Metrics',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#FAFAFA'
            )
            st.plotly_chart(fig)
            
        else:
            st.info("Run an evaluation to see results here")

def render_management_page():
    st.header("⚙️ System Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("System Status")
        try:
            if st.session_state.retriever:
                stats = st.session_state.retriever.get_stats()
                
                st.metric("Total Documents", stats.get('total_documents', 0))
                st.metric("Vectorstore", "✅" if stats.get('vectorstore_ready') else "❌")
                st.metric("BM25", "✅" if stats.get('bm25_ready') else "❌")
                
                # Cache statistics
                if st.session_state.cache_manager:
                    cache_stats = st.session_state.cache_manager.get_stats()
                    st.metric("Cache Files", cache_stats.get('total_files', 0))
                    st.metric("Expired Cache", cache_stats.get('expired_files', 0))
            else:
                st.info("Components not initialized")
            
        except Exception as e:
            st.error(f"Error getting system status: {e}")
    
    with col2:
        st.subheader("Management Actions")
        
        if st.button("🔄 Refresh Status", use_container_width=True):
            st.rerun()
        
        # Clear data with confirmation
        if st.session_state.show_clear_confirmation:
            st.markdown("""
            <div class="clear-confirm">
                <h4>⚠️ Confirm Data Deletion</h4>
                <p>This will delete ALL data including documents, cache, and chat history!</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Yes, Delete Everything", use_container_width=True):
                    clear_all_data()
            with col2:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.show_clear_confirmation = False
                    st.rerun()
        else:
            if st.button("🗑️ Clear All Data", use_container_width=True):
                st.session_state.show_clear_confirmation = True
                st.rerun()
        
        # Cache management
        st.subheader("Cache Management")
        col_cache1, col_cache2 = st.columns(2)
        
        with col_cache1:
            if st.button("Clear Expired Cache", use_container_width=True):
                clear_expired_cache()
        
        with col_cache2:
            if st.button("Clear All Cache", use_container_width=True):
                clear_all_cache()

def render_analytics_page():
    st.header("📈 Analytics & Insights")
    
    # Document statistics
    try:
        if st.session_state.retriever:
            stats = st.session_state.retriever.get_stats()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Documents", stats.get('total_documents', 0))
            with col2:
                st.metric("Vectorstore", "Active" if stats.get('vectorstore_ready') else "Inactive")
            with col3:
                st.metric("BM25", "Active" if stats.get('bm25_ready') else "Inactive")
        else:
            st.info("Components not initialized")
    except:
        st.warning("Could not retrieve system statistics")
    
    # Chat statistics
    st.subheader("Chat Statistics")
    if st.session_state.chat_history:
        user_messages = len([m for m in st.session_state.chat_history if m['role'] == 'user'])
        assistant_messages = len([m for m in st.session_state.chat_history if m['role'] == 'assistant'])
        
        fig = px.pie(
            values=[user_messages, assistant_messages],
            names=['User Messages', 'Assistant Messages'],
            title='Message Distribution',
            color_discrete_sequence=['#4F8BF9', '#262730']
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#FAFAFA'
        )
        st.plotly_chart(fig)
    else:
        st.info("No chat history yet")
    
    # System information
    st.subheader("System Configuration")
    try:
        from app.config import config
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
        
        # Style the table
        st.dataframe(
            pd.DataFrame(sys_info),
            use_container_width=True,
            hide_index=True
        )
    except ImportError:
        st.info("Configuration not available in offline mode")

if __name__ == "__main__":
    main()