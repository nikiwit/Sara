# SARA - Smart Academic Research Assistant

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

SARA (Smart Academic Research Assistant) is an advanced Retrieval-Augmented Generation (RAG) chatbot system developed as part of the APU University project. The system provides intelligent question-answering capabilities by leveraging state-of-the-art natural language processing and vector-based retrieval techniques, specifically optimized for academic and institutional knowledge bases.

## 🎯 Project Overview

This project represents a comprehensive implementation of a production-ready RAG system designed to serve as an intelligent academic assistant. SARA combines modern machine learning techniques with robust system architecture to provide accurate, contextual responses to user queries across various academic domains.

**Key Achievements:**
- Advanced multi-format document processing pipeline
- Hybrid retrieval system with semantic and keyword matching
- Production-grade model lifecycle management
- Dual-environment configuration for development and deployment
- Comprehensive session management and conversation memory
- Real-time streaming responses with optimized user experience

## 🌟 Core Features

### 🤖 Intelligent Chatbot System
- **Conversational AI**: Natural language interaction with advanced context awareness
- **Follow-up Question Handling**: Intelligent query reformulation for contextual follow-ups
- **Session Management**: Isolated conversation histories with automatic cleanup (max 5 sessions)
- **Streaming Responses**: Real-time response generation for enhanced user experience
- **Memory Management**: Persistent conversation context across sessions

### 📚 Advanced Document Processing
- **Multi-Format Support**: 
  - PDF documents with advanced text extraction
  - Microsoft Word (DOCX) files
  - PowerPoint (PPTX) presentations
  - EPUB electronic books
  - Plain text and Markdown files
- **Intelligent Content Extraction**: Preserves document structure and hierarchy
- **Metadata Enhancement**: Automatic extraction of document metadata and context
- **APU-Specific Optimization**: Enhanced processing for university knowledge bases

### 🔍 Sophisticated Retrieval System
- **Hybrid Search Architecture**: Combines vector similarity and keyword matching
- **FAQ-Optimized Retrieval**: Specialized handling for frequently asked questions
- **Semantic Understanding**: Enhanced query comprehension using spaCy NLP models
- **Query Expansion**: Automatic query enhancement for improved search results
- **Context Compression**: Intelligent context optimization for better response generation

### ⚡ Production-Grade Infrastructure
- **Dual Environment Support**:
  - **Local Development**: Optimized for laptops and development machines
  - **Production Deployment**: Optimized for HGX H100 G593-SD2 servers
- **Model Lifecycle Management**: Automatic model updates and cache management
- **Resource Optimization**: Environment-specific resource allocation and management
- **Comprehensive Monitoring**: Health checks and performance monitoring

### 🎛️ Advanced Configuration Management
- **Environment-Specific Settings**: Separate configurations for development and production
- **Hardware Detection**: Automatic GPU detection and optimization (CUDA/Apple Silicon MPS)
- **Dynamic Resource Allocation**: Adaptive resource management based on available hardware
- **Extensive Customization**: Over 30 configurable parameters for fine-tuning

## 📋 System Requirements

### Minimum Requirements
- **Python**: Version 3.8 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 10GB free space for models and vector storage
- **Network**: Internet connection for initial setup and model downloads

### Recommended Hardware
- **Development**: Modern laptop with 16GB+ RAM
- **Production**: High-performance server (e.g., HGX H100 G593-SD2)
- **GPU**: CUDA-compatible GPU or Apple Silicon (optional but recommended)

### Software Dependencies
- **Ollama**: Required for local LLM inference
- **spaCy**: For advanced semantic processing
- **ChromaDB**: Vector database for document storage
- **PyTorch**: Machine learning framework
- **LangChain**: LLM application framework

## 🚀 Installation and Setup

### 1. Ollama Installation and Configuration

Ollama is required for local language model inference and must be installed first:

```bash
# macOS/Linux Installation
curl -fsSL https://ollama.ai/install.sh | sh

# Windows Installation
# Download installer from https://ollama.ai/download
```

**Pull Required Models:**
```bash
# For local development (lighter model)
ollama pull qwen2.5:3b-instruct

# For production deployment (more capable model)
ollama pull qwen2.5:7b-instruct
```

**Verify Ollama Installation:**
```bash
# Check available models
ollama list

# Start Ollama service (if not running)
ollama serve
```

### 2. Project Setup

**Clone Repository:**
```bash
git clone https://github.com/nikiwit/SARA.git
cd SARA
```

**Create Virtual Environment:**
```bash
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

**Install Dependencies:**
```bash
pip install -r requirements.txt
```

### 3. spaCy Model Installation

SARA requires spaCy language models for enhanced semantic processing:

```bash
# Install medium model (recommended for better semantic processing)
python -m spacy download en_core_web_md  # Medium model (~43MB) - includes word vectors

# Alternative: Install base model (smaller but limited semantic capabilities)
python -m spacy download en_core_web_sm  # Small model (~50MB) - no word vectors

# Optional: Install large model for production (best accuracy)
python -m spacy download en_core_web_lg  # Large model (~588MB) - full word vectors
```

### 4. Embedding Models

SARA automatically downloads and caches embedding models on first run:

**Local Development:**
- **Model**: `BAAI/bge-base-en-v1.5` (~438MB)
- **Purpose**: Document embeddings and semantic search
- **Location**: Cached in `model_cache/huggingface/`

**Production Environment:**
- **Model**: `BAAI/bge-large-en-v1.5` (~1.34GB)
- **Purpose**: Enhanced accuracy for document embeddings
- **Features**: Automatic cache management and integrity checking

**Note**: First startup will be slower as embedding models download automatically. Subsequent startups use cached models for faster initialization.

### 5. Environment Configuration

SARA supports dual-environment configuration with automatic detection:

**Environment Selection:**
```bash
# Run in local development mode (default)
python main.py

# Run in production mode
SARA_ENV=production python main.py
```

## 💻 Usage

### Starting the Application

**Basic Startup:**
```bash
# Ensure Ollama is running
ollama serve

# Start SARA (in separate terminal)
python main.py
```

**Advanced Usage:**
```bash
# Specify environment explicitly
SARA_ENV=production python main.py

# Enable debug logging
SARA_LOG_LEVEL=DEBUG python main.py

# Force document reindexing
SARA_FORCE_REINDEX=True python main.py
```

### Interactive Commands

SARA provides comprehensive CLI commands for system management:

**Document Management:**
- `reindex` - Rebuild document index from scratch
- `stats` - Display system and document statistics

**Session Management:**
- `new session [name]` - Create a new conversation session
- `list sessions` - Show all available sessions
- `switch session <id>` - Switch to a specific session
- `clear session` - Clear current session memory

**Model Management:**
- `model report` - Display model status and health information
- `model check` - Check for available model updates
- `model update` - Update models (requires confirmation)

**System Commands:**
- `clear` - Clear conversation memory
- `help` - Display available commands
- `exit` - Shutdown SARA gracefully

### Query Examples

**APU Knowledge Base Questions:**
```
"How do I submit ec?"
"What are the library operating hours?"
"Where do I get medical insurance from?"
"How do I pay my fees?"
"What is the process for the visa renewal?"
```

**Note**: SARA only answers questions based on the [APU Knowledge Base](https://apiit.atlassian.net/wiki/spaces/KB/overview?mode=global) content (last updated 13.08.2025). It cannot provide general knowledge or information outside of the university's knowledge base.

## 📁 Project Architecture

```
SARA/
├── 📋 Core Application
│   ├── main.py                     # Application entry point with error handling
│   ├── app.py                      # Main Sara class and CLI interface
│   ├── config.py                   # Base configuration with environment detection
│   ├── config_local.py             # Local development configuration
│   ├── config_production.py        # Production environment configuration
│   ├── resource_manager.py         # Hardware resource management
│   ├── sara_types.py               # Type definitions and data structures
│   └── input_processing.py         # User input processing and validation
│
├── 📄 Document Processing
│   ├── document_processing/
│   │   ├── loaders.py              # Multi-format document loaders
│   │   ├── parsers.py              # Content parsing and extraction
│   │   └── splitters.py            # Text chunking and segmentation
│
├── 🔍 Query and Retrieval
│   ├── query_handling/
│   │   ├── router.py               # Query routing and classification
│   │   ├── conversation.py         # Conversation flow management
│   │   └── commands.py             # CLI command processing
│   ├── retrieval/
│   │   ├── handler.py              # Main retrieval orchestrator
│   │   ├── context_processor.py    # Context optimization and compression
│   │   ├── faq_matcher.py          # FAQ-specific matching logic
│   │   └── reranker.py             # Result reranking and optimization
│
├── 🧠 AI and Response Generation
│   ├── response/
│   │   ├── generator.py            # LLM response generation with streaming
│   │   └── cache.py                # Response caching system
│   ├── spacy_semantic_processor.py # Advanced semantic processing
│
├── 💾 Data Management
│   ├── vector_management/
│   │   ├── manager.py              # Vector store operations and lifecycle
│   │   └── chromadb_manager.py     # ChromaDB client management
│   ├── session_management/
│   │   ├── session_manager.py      # Session lifecycle coordination
│   │   ├── session_storage.py      # JSON-based session persistence
│   │   └── session_types.py        # Session data structures
│
├── 📊 Data and Storage
│   ├── data/                       # Knowledge base documents
│   │   ├── apu_AA_kb.txt           # Academic Affairs knowledge base
│   │   ├── apu_BUR_kb.txt          # Bursar/Finance knowledge base
│   │   ├── apu_ITSM_kb.txt         # IT Service Management knowledge base
│   │   ├── apu_LIB_kb.txt          # Library services knowledge base
│   │   ├── apu_LNO_kb.txt          # Learning Network Office knowledge base
│   │   └── apu_VISA_kb.txt         # Visa and immigration knowledge base
│
└── 📚 Configuration and Documentation
    ├── requirements.txt            # Project dependencies
    ├── .gitignore                  # Version control exclusions
    ├── LICENSE                     # MIT License
    ├── CLAUDE.md                   # Development instructions for Claude Code
    └── README.md                   # This file
```

## ⚙️ Configuration

### Environment Variables

SARA uses environment variables for configuration, with support for environment-specific files:

**File Structure:**
- `config.py` - Base configuration with defaults
- `config_local.py` - Local development overrides
- `config_production.py` - Production deployment overrides

**Key Configuration Parameters:**

| Parameter | Local Default | Production Default | Description |
|-----------|---------------|-------------------|-------------|
| `SARA_ENV` | `local` | `production` | Environment selection |
| `SARA_EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | `BAAI/bge-large-en-v1.5` | Embedding model |
| `SARA_LLM_MODEL` | `qwen2.5:3b-instruct` | `qwen2.5:7b-instruct` | Language model |
| `SARA_CHUNK_SIZE` | `400` | `800` | Document chunk size |
| `SARA_MAX_CONTEXT_SIZE` | `7000` | `8000` | Maximum context tokens |
| `SARA_MAX_THREADS` | `2` | `32` | Processing threads |
| `SARA_MAX_MEMORY` | `2G` | `64G` | Memory allocation |

### Local Development Configuration

Optimized for development environments with resource constraints:

- **Resource Efficient**: Smaller models and reduced memory usage
- **Fast Iteration**: Quick startup and response times
- **Enhanced Logging**: Detailed debug information
- **Frequent Updates**: Weekly model update checks

### Production Configuration

Optimized for high-performance deployment:

- **Performance Focused**: Larger models and enhanced capabilities
- **Resource Intensive**: Full utilization of available hardware
- **Conservative Updates**: Monthly model checks with manual approval
- **Monitoring**: Comprehensive health checks and alerts

## 📚 Dependencies

### Core Machine Learning
- **torch**: PyTorch deep learning framework
- **sentence-transformers**: Sentence embedding models
- **transformers**: Hugging Face transformers library
- **langchain**: LLM application development framework
- **langchain-huggingface**: Hugging Face integration for LangChain
- **langchain-chroma**: ChromaDB integration for LangChain

### Document Processing
- **pypdf**: PDF document processing
- **python-docx**: Microsoft Word document handling
- **python-pptx**: PowerPoint presentation processing
- **ebooklib**: EPUB electronic book processing
- **unstructured**: Advanced document parsing

### Natural Language Processing
- **spacy**: Advanced NLP and semantic processing
- **nltk**: Natural Language Toolkit
- **huggingface-hub**: Model repository access

### Vector Database and Storage
- **chromadb**: Vector database for document embeddings
- **numpy**: Numerical computing support

### Web and API
- **requests**: HTTP requests for model updates
- **ollama**: Local LLM inference integration

### Utilities
- **python-dotenv**: Environment variable management
- **tqdm**: Progress bars for long-running operations
- **psutil**: System resource monitoring
- **typing-extensions**: Enhanced type annotations

For the complete dependency list with versions, see `requirements.txt`.

## 🔧 Advanced Configuration

### Model Management

SARA implements comprehensive model lifecycle management:

**Automatic Updates:**
- Periodic checks for model updates from Hugging Face Hub
- Age-based warnings for outdated models
- User prompts for update approval

**Cache Management:**
- Intelligent model caching with metadata tracking
- Automatic cleanup of incomplete downloads
- Backup and recovery capabilities

**Health Monitoring:**
- Model integrity verification
- Performance monitoring
- Error detection and recovery

### Performance Optimization

**Hardware Acceleration:**
- Automatic GPU detection (CUDA/MPS)
- Optimized tensor operations
- Memory-efficient processing

**Resource Management:**
- Environment-specific resource allocation
- Dynamic scaling based on available hardware
- Memory usage optimization

### Security and Reliability

**Data Safety:**
- No sensitive information in configuration
- Secure model caching
- Comprehensive error handling

**Reliability Features:**
- Graceful degradation on errors
- Automatic retry mechanisms
- Comprehensive logging

## 🤝 Contributing

Contributions are welcome and encouraged. Please follow these guidelines:

### Development Workflow
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with appropriate tests
4. Ensure code follows project standards
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Standards
- Follow Python PEP 8 style guidelines
- Include comprehensive docstrings
- Add type hints where appropriate
- Write unit tests for new functionality
- Update documentation as needed

### Areas for Contribution
- Additional document format support
- Enhanced semantic processing
- Performance optimizations
- UI/Web interface development
- Integration with external systems

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors and Acknowledgments

**Original Author:**
- **Nik** - [nikiwit.com](https://nikiwit.com/) - *Project Creator and Lead Developer*

**Acknowledgments:**
- APU University for project support and requirements
- The open-source community for excellent libraries and frameworks
- Contributors who have helped improve the project

## 📞 Support and Contact

For support, questions, or contributions:

- **Issues**: Open an issue in the GitHub repository
- **Documentation**: Refer to inline code documentation and CLAUDE.md
- **Community**: Engage with the APU community and contributors

## 🎓 Academic Context

This project was developed as part of academic research and development at APU University. It shows practical application of advanced AI technologies in educational settings and serves as a foundation for further research and development in intelligent academic assistance tools.

The system architecture and implementation choices reflect real-world production requirements while maintaining academic rigor and educational value.

---

**Made with ❤️ for the APU Community**

*SARA represents the intersection of cutting-edge AI technology and practical educational applications, designed to enhance the academic experience through intelligent, responsive, and reliable assistance.*