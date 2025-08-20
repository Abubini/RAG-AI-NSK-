# 🤖 RAG AI Assistant

A comprehensive Retrieval-Augmented Generation (RAG) system that enables intelligent document search and question-answering capabilities. This project features both a powerful CLI interface and a beautiful Streamlit web UI.

---

## 📋 Table of Contents

* [Features](#-features)
* [Architecture](#-architecture)
* [Installation](#-installation)
* [Configuration](#️-configuration)
* [CLI Usage](#-cli-usage)
* [Web UI Usage](#-web-ui-usage)
* [File Format Support](#-file-format-support)
* [API Integration](#-api-integration)
* [Deployment](#-deployment)
* [Troubleshooting](#-troubleshooting)
* [Contributing](#-contributing)
* [License](#-license)

---

## 🚀 Features

### Core RAG Capabilities

* **Hybrid Retrieval**: Combines semantic (ChromaDB) and keyword (BM25) search
* **Cross-Encoder Reranking**: Uses sentence transformers for relevance scoring
* **Multiple LLM Support**: Groq API with fallback to local models
* **Persistent Storage**: ChromaDB vector store with BM25 document persistence

### File Format Support

* Documents: **PDF, DOCX, PPTX, TXT**
* Data: **CSV, XLSX**
* Web: **URL content extraction**
* Markdown: **MD files**

### Advanced Features

* **Metadata Filtering**: Filter by source, page range, content length, and more
* **Smart Caching**: Automatic query caching with TTL expiration
* **Evaluation Framework**: Comprehensive QA evaluation with metrics
* **Modular Architecture**: Easy to extend and customize

---

## 🏗️ Architecture

```
text
├── app/
│   ├── main.py              # CLI entry point
│   ├── config.py            # Configuration management
│   ├── retriever.py         # Hybrid retrieval + reranking
│   ├── generator.py         # LLM response generation
│   ├── ingestion.py         # Document processing
│   ├── context.py           # Context formatting
│   ├── filters.py           # Metadata filtering
│   ├── evaluation.py        # QA evaluation metrics
│   ├── caching.py           # Query caching system
│   └── utils.py             # Utility functions
├── data/
│   ├── inbox/               # Raw document storage
│   ├── chroma/              # Vector database
│   ├── bm25/                # BM25 document storage
│   ├── cache/               # Query cache
│   └── evaluation/          # Evaluation data
├── index.py                 # Streamlit web UI
└── requirements.txt         # Python dependencies
```

---

## 📦 Installation

### Prerequisites

* Python **3.8+**
* `pip` package manager
* Groq API key (optional, for enhanced performance)

### Step-by-Step Setup

1. **Clone and setup environment:**

```bash
git clone <your-repo-url>
cd rag-ai
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Environment configuration:**

```bash
cp .env.example .env
# Edit .env with your API keys and preferences
```

4. **Directory setup:**

```bash
mkdir -p data/{inbox,chroma,bm25,cache,evaluation}
```

---

## ⚙️ Configuration

### Environment Variables (`.env`)

```env
# API Keys
GROQ_API_KEY=your_groq_api_key_here

# Model Configurations
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-2-v2
LLM_MODEL=llama3-8b-8192

# Paths
PERSIST_DIR=./data/chroma
INBOX_DIR=./data/inbox
BM25_STORAGE_PATH=./data/bm25/documents.json
CACHE_DIR=./data/cache
EVAL_DATA_PATH=./data/evaluation/qa_pairs.jsonl

# Retrieval Parameters
TOP_K=4
CHUNK_SIZE=800
CHUNK_OVERLAP=100
CACHE_TTL_HOURS=24
```

### Model Options

* **Embedding Models**: all-MiniLM-L6-v2, all-mpnet-base-v2, custom models
* **Reranker Models**: Various cross-encoder models from sentence-transformers
* **LLM Models**: Groq models or local Ollama models

---

## 💻 CLI Usage

### Document Management

```bash
# Ingest single file
python -m app.main ingest --file document.pdf

# Ingest directory
python -m app.main ingest --directory ./data/inbox/

# Ingest specific formats
python -m app.main ingest --file report.docx
python -m app.main ingest --file presentation.pptx
```

### Querying System

```bash
# Basic query
python -m app.main query "What are the main topics?"

# Query with context
python -m app.main query "Explain machine learning" --context

# Query with filters
python -m app.main query "data analysis" --filters '{"source": "report", "page": "1:10"}'

# Disable caching
python -m app.main query "latest information" --no-cache
```

### Evaluation System

```bash
# Add evaluation QA pair
python -m app.main add-eval --question "What is AI?" --expected-docs "1,2,3" --reference-answer "Artificial Intelligence is..."

# Run evaluation
python -m app.main evaluate --samples 5

# View evaluation results
python -c "from app.evaluation import RAGEvaluator; e = RAGEvaluator(); print(e.load_evaluation_data())"
```

### System Management

```bash
# Check system status
python -m app.main status

# Clear all data
python -m app.main clear

# Cache management
python -m app.main cache --stats
python -m app.main cache --clear-expired
python -m app.main cache --clear-all

# Interactive chat mode
python -m app.main chat
```

---

## 🌐 Web UI Usage

### Starting the Web Interface

```bash
streamlit run index.py
# or with custom port
streamlit run index.py --server.port 8501
```

### Web UI Features

**📁 Document Ingestion Page**

* Drag & Drop Upload
* Batch Processing
* Real-time Progress
* File Format Detection

**💬 Chat & Query Page**

* Real-time Chat Interface
* Advanced Filtering
* Context Display
* Citation Management
* Chat History

**📊 Evaluation Page**

* QA Pair Management
* Automated Testing
* Visual Metrics
* Results Export

**⚙️ System Management Page**

* Real-time Status
* Cache Management
* Data Management
* Configuration View

**📈 Analytics Page**

* Usage Statistics
* Performance Charts
* System Information
* Chat Analytics

### Web UI Keyboard Shortcuts

* **Enter**: Send message
* **Ctrl/Cmd + K**: Focus input
* **Esc**: Close modals

---

## 📄 File Format Support

| Format     | Extension   | Requirements   | Notes                         |
| ---------- | ----------- | -------------- | ----------------------------- |
| PDF        | .pdf        | PyPDF2         | Text extraction with metadata |
| Word       | .docx, .doc | python-docx    | Full formatting support       |
| PowerPoint | .pptx, .ppt | python-pptx    | Slide content extraction      |
| Text       | .txt        | Built-in       | UTF-8 encoding support        |
| CSV        | .csv        | pandas         | Tabular data processing       |
| Excel      | .xlsx, .xls | openpyxl       | Spreadsheet data extraction   |
| Markdown   | .md         | Built-in       | GitHub-flavored markdown      |
| Web Pages  | URLs        | beautifulsoup4 | Content extraction from URLs  |

### Metadata Extraction

* Document Properties: Title, author, creation date
* Structural Metadata: Page numbers, sections, headings
* Content Metadata: Word count, chunk size, content type
* Custom Metadata: User-defined fields

---

## 🔌 API Integration

### Supported LLM Providers

* **Groq API (Recommended)**
* **Ollama Local**
* **HuggingFace**
* **OpenAI**

### Example: Adding OpenAI Support

```python
from langchain_openai import ChatOpenAI

class CustomAnswerGenerator:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            api_key=os.getenv("OPENAI_API_KEY")
        )
    # ... rest of implementation
```

---

## 🚀 Deployment

### Local Deployment

```bash
# Production mode
streamlit run index.py --server.headless true --server.port 8501

# With logging
streamlit run index.py --logger.level=info
```

### Docker Deployment

**Dockerfile**

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "index.py", "--server.port=8501", "--server.headless=true"]
```

**Build & Run**

```bash
docker build -t rag-ai .
docker run -p 8501:8501 -v ./data:/app/data rag-ai
```

### Cloud Deployment Options

* Streamlit Cloud
* Heroku
* AWS EC2
* Google Cloud Run

---

## 🐛 Troubleshooting

### Common Issues

**Missing Dependencies:**

```bash
pip install -r requirements.txt
pip install python-docx python-pptx docx2txt
```

**CUDA Errors:**

```bash
export CUDA_VISIBLE_DEVICES=""
```

**API Key Issues:**

```bash
echo $GROQ_API_KEY
export USE_LOCAL_LLM=true
```

**File Permission Issues:**

```bash
chmod -R 755 data/
```

### Performance Optimization

* Use Lighter Models:

```env
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-2-v2
```

* Adjust Chunking:

```env
CHUNK_SIZE=500
CHUNK_OVERLAP=50
```

* Optimize Cache:

```env
CACHE_TTL_HOURS=6
TOP_K=3
```

---

## 🤝 Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

### Code Style

* Follow PEP 8
* Use type hints
* Add docstrings
* Include tests

### Testing

```bash
python -m pytest tests/
python -m pytest --cov=app tests/
python -m pytest tests/test_retriever.py -v
```

---

## 📊 Performance Benchmarks

* **Document Ingestion**: 100-500 docs/minute
* **Query Response**: 1-5 seconds
* **Accuracy**: 85-95%
* **Cache Hit Rate**: 60-80%

### Hardware Requirements

* Minimum: 4GB RAM, 2 CPU cores
* Recommended: 8GB RAM, 4 CPU cores
* Optimal: 16GB RAM, GPU acceleration

---

## 📝 License

This project is licensed under the **MIT License** - see the LICENSE file for details.

---

## 🙏 Acknowledgments

* **LangChain**: For the amazing framework
* **HuggingFace**: For transformer models
* **Streamlit**: For the excellent UI framework
* **Groq**: For high-performance inference

---

## 📞 Support

* Check the troubleshooting section
* Review existing GitHub issues
* Create a new issue
* Join our community Discord (if available)

---

🎉 **Happy Document Searching!**
