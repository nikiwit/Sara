# SARA - Smart Academic Retrieval Assistant

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

> **Advanced Retrieval-Augmented Generation (RAG) system for academic institutions**

SARA is a production-ready AI assistant designed specifically for academic environments. Built with cutting-edge NLP and vector retrieval technology, it provides intelligent, contextual responses to queries about university knowledge bases with enterprise-grade reliability.

## Table of Contents

- [Quick Start](#quick-start)
- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [API Documentation](#api-documentation)
- [Testing](#testing)
- [Performance](#performance)
- [Contributing](#contributing)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Quick Start

**Prerequisites:** Python 3.8+, 8GB RAM, 10GB free disk space

```bash
# 1. Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen2.5:3b-instruct

# 2. Clone and setup SARA
git clone https://github.com/nikiwit/SARA.git
cd SARA
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 3. Install language models
python -m spacy download en_core_web_md

# 4. Pre-download AI models (optional but recommended)
# This will download ~2GB of models for faster first startup
python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('BAAI/bge-large-en-v1.5'); CrossEncoder('BAAI/bge-reranker-large')"

# 5. Run SARA (models will auto-download if not cached)
python main.py
```

**First run takes ~2-3 minutes (based on Internet speed) to download AI models. Subsequent runs start in ~5 seconds.**

Ready to chat! Try asking: *"What can you do?"*

## Project Overview

This project represents a comprehensive implementation of a production-ready RAG system designed to serve as an intelligent academic assistant. SARA combines modern machine learning techniques with robust system architecture to provide accurate, contextual responses to user queries across various academic domains.

**Key Achievements:**
- Advanced multi-format document processing pipeline
- Hybrid retrieval system with semantic and keyword matching
- Production-grade model lifecycle management
- Dual-environment configuration for development and deployment
- Comprehensive session management and conversation memory
- Real-time streaming responses with optimized user experience

## Features

### Chatbot System
- **Conversational AI**: Natural language interaction with advanced context awareness
- **Language Detection**: Multi-signal hybrid approach with Unicode-based script detection and confidence scoring for non-English query handling
- **Query Disambiguation**: Pronoun resolution and typo correction using semantic analysis for improved query understanding
- **Follow-up Question Handling**: Intelligent query reformulation for contextual follow-ups
- **Session Management**: Isolated conversation histories with automatic cleanup (max 5 sessions)
- **Streaming Responses**: Real-time response generation for enhanced user experience
- **Memory Management**: Persistent conversation context across sessions

### Advanced Document Processing
- **Multi-Format Support**: 
  - PDF documents with advanced text extraction
  - Microsoft Word (DOCX) files
  - PowerPoint (PPTX) presentations
  - EPUB electronic books
  - Plain text and Markdown files
- **Intelligent Content Extraction**: Preserves document structure and hierarchy
- **Metadata Enhancement**: Automatic extraction of document metadata and context
- **APU-Specific Optimization**: Enhanced processing for university knowledge bases

### Retrieval System
- **Hybrid Search Architecture**: Combines vector similarity and keyword matching
- **Advanced Reranking**: State-of-the-art cross-encoder models for improved result relevance
- **FAQ-Optimized Retrieval**: Specialized handling for frequently asked questions
- **Semantic Understanding**: Enhanced query comprehension using spaCy NLP models
- **Query Expansion**: Automatic query enhancement for improved search results
- **Context Compression**: Intelligent context optimization for better response generation

### Production-Grade Infrastructure
- **Dual Environment Support**:
  - **Local Development**: Optimized for laptops and development machines
  - **Production Deployment**: Optimized for HGX H100 G593-SD2 servers
- **Model Lifecycle Management**: Automatic model updates and cache management
- **Resource Optimization**: Environment-specific resource allocation and management
- **Comprehensive Monitoring**: Health checks and performance monitoring

### Configuration Management
- **Environment-Specific Settings**: Separate configurations for development and production
- **Hardware Detection**: Automatic GPU detection and optimization (CUDA/Apple Silicon MPS)
- **Dynamic Resource Allocation**: Adaptive resource management based on available hardware
- **Extensive Customization**: Over 30 configurable parameters for fine-tuning

## Demo

### Chat Interface
```
🤖 SARA: Hello! I'm SARA, your Smart Academic Retrieval Assistant. 
How can I help you today?

👤 You: How do I submit EC?

🤖 SARA: To submit EC (Extracurricular activities), you need to:
1. Log into the Student Portal
2. Navigate to Academic Affairs → EC Submission
3. Upload your certificates and supporting documents
4. Fill out the EC form with activity details
5. Submit for approval

The deadline for EC submission is typically 2 weeks before semester end.
```

## System Requirements

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

## Installation

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

### 4. Embedding and Reranker Models

SARA automatically downloads and caches embedding and reranker models on first run:

**Embedding Models:**

**Local Development:**
- **Model**: `BAAI/bge-base-en-v1.5` (~438MB)
- **Purpose**: Document embeddings and semantic search
- **Location**: Cached in `model_cache/huggingface/`

**Production Environment:**
- **Model**: `BAAI/bge-large-en-v1.5` (~1.34GB)
- **Purpose**: Enhanced accuracy for document embeddings
- **Features**: Automatic cache management and integrity checking

**Reranker Models**

**Local Development:**
- **Model**: `BAAI/bge-reranker-base` (~560MB)
- **Purpose**: Improved result ranking for better relevance
- **Performance**: 90-95% of large model accuracy with faster inference

**Production Environment:**
- **Model**: `BAAI/bge-reranker-large` (~1.1GB)
- **Purpose**: State-of-the-art reranking for maximum accuracy
- **Performance**: Best-in-class English reranking performance

**Note**: First startup will be slower as models download automatically. Subsequent startups use cached models for faster initialization. The reranker significantly improves answer relevance by reordering search results before response generation.

#### Manual Model Download (Optional)

If you prefer to download models manually or have network restrictions, you can pre-download models using these commands:

**Install Hugging Face CLI:**
```bash
pip install huggingface_hub[cli]
```

**Download Embedding Models:**
```bash
# Local development model (~438MB)
huggingface-cli download BAAI/bge-base-en-v1.5 --local-dir ./model_cache/huggingface/sentence_transformers/models--BAAI--bge-base-en-v1.5

# Production model (~1.34GB)
huggingface-cli download BAAI/bge-large-en-v1.5 --local-dir ./model_cache/huggingface/sentence_transformers/models--BAAI--bge-large-en-v1.5
```

**Download Reranker Models:**
```bash
# Local development model (~560MB)
huggingface-cli download BAAI/bge-reranker-base --local-dir ./model_cache/huggingface/sentence_transformers/models--BAAI--bge-reranker-base

# Production model (~1.1GB)
huggingface-cli download BAAI/bge-reranker-large --local-dir ./model_cache/huggingface/sentence_transformers/models--BAAI--bge-reranker-large
```

**Alternative: Direct Python Download:**
```bash
# Set cache directory and download via Python
export HF_HOME=./model_cache/huggingface

# Download specific models as needed
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-base-en-v1.5')"
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-en-v1.5')"
python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-base')"
python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-large')"
```

**Verify Downloads:**
```bash
# Check downloaded models
ls -la model_cache/huggingface/sentence_transformers/
```

**Alternative Download Sources:**
- **Hugging Face Mirror**: https://hf-mirror.com (for restricted regions)
- **BAAI Official**: https://model.baai.ac.cn/models (alternative source)

### 5. Environment Configuration

SARA supports dual-environment configuration with automatic detection:

**Environment Selection:**
```bash
# Run in local development mode (default)
python main.py

# Run in production mode
SARA_ENV=production python main.py
```

## Usage

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

## Architecture

```
SARA/
├── Core Application
│   ├── main.py                     # Application entry point with error handling
│   ├── app.py                      # Main Sara class and CLI interface
│   ├── config.py                   # Base configuration with environment detection
│   ├── config_local.py             # Local development configuration
│   ├── config_production.py        # Production environment configuration
│   ├── resource_manager.py         # Hardware resource management
│   ├── sara_types.py               # Type definitions and data structures
│   └── input_processing.py         # User input processing and validation
│
├── Document Processing
│   ├── document_processing/
│   │   ├── loaders.py              # Multi-format document loaders
│   │   ├── parsers.py              # Content parsing and extraction
│   │   └── splitters.py            # Text chunking and segmentation
│
├── Query and Retrieval
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
├── AI and Response Generation
│   ├── response/
│   │   ├── generator.py            # LLM response generation with streaming
│   │   └── cache.py                # Response caching system
│   ├── spacy_semantic_processor.py # Advanced semantic processing
│
├── Data Management
│   ├── vector_management/
│   │   ├── manager.py              # Vector store operations and lifecycle
│   │   └── chromadb_manager.py     # ChromaDB client management
│   ├── session_management/
│   │   ├── session_manager.py      # Session lifecycle coordination
│   │   ├── session_storage.py      # JSON-based session persistence
│   │   └── session_types.py        # Session data structures
│
├── Data and Storage
│   ├── data/                       # Knowledge base documents
│   │   ├── apu_AA_kb.txt           # Academic Affairs knowledge base
│   │   ├── apu_BUR_kb.txt          # Bursar/Finance knowledge base
│   │   ├── apu_ITSM_kb.txt         # IT Service Management knowledge base
│   │   ├── apu_LIB_kb.txt          # Library services knowledge base
│   │   ├── apu_LNO_kb.txt          # Learning Network Office knowledge base
│   │   └── apu_VISA_kb.txt         # Visa and immigration knowledge base
│
└── Configuration and Documentation
    ├── requirements.txt            # Project dependencies
    ├── .gitignore                  # Version control exclusions
    ├── LICENSE                     # MIT License
    ├── CLAUDE.md                   # Development instructions for Claude Code
    └── README.md                   # This file
```

## Configuration

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
| `SARA_EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | `BAAI/bge-large-en-v1.5` | Embedding model |
| `SARA_LLM_MODEL` | `qwen2.5:3b-instruct` | `qwen2.5:7b-instruct` | Language model |
| `SARA_RERANKER_MODEL` | `BAAI/bge-reranker-large` | `BAAI/bge-reranker-large` | Reranker model |
| `SARA_CHUNK_SIZE` | `500` | `800` | Document chunk size |
| `SARA_CHUNK_OVERLAP` | `125` | `200` | Document chunk overlap |
| `SARA_RETRIEVER_K` | `6` | `8` | Number of retrieved documents |
| `SARA_MAX_CONTEXT_SIZE` | `8000` | `8000` | Maximum context tokens |
| `SARA_EXPANSION_FACTOR` | `2` | `3` | Query expansion multiplier |
| `SARA_MAX_THREADS` | `2` | `32` | Processing threads |
| `SARA_MAX_MEMORY` | `2G` | `64G` | Memory allocation |
| `SARA_LANGUAGE_DETECTION` | `True` | `True` | Enable language detection |
| `SARA_LANG_CONFIDENCE_THRESHOLD` | `0.65` | `0.70` | Language detection confidence |
| `SARA_AMBIGUITY_DETECTION` | `True` | `True` | Enable query disambiguation |
| `SARA_AMBIGUITY_THRESHOLD` | `0.7` | `0.7` | Query ambiguity threshold |

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

## Dependencies

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

## API Documentation

SARA provides a comprehensive CLI interface with the following command categories:

### Core Commands
| Command | Description | Example |
|---------|-------------|---------|
| `help` | Show available commands | `SARA > help` |
| `exit` | Shutdown gracefully | `SARA > exit` |
| `clear` | Clear conversation memory | `SARA > clear` |

### Document Management
| Command | Description | Example |
|---------|-------------|---------|
| `reindex` | Rebuild document index | `SARA > reindex` |
| `stats` | Display system statistics | `SARA > stats` |

### Session Management
| Command | Description | Example |
|---------|-------------|---------|
| `new session [name]` | Create new session | `SARA > new session Research` |
| `list sessions` | Show all sessions | `SARA > list sessions` |
| `switch session <id>` | Switch to session | `SARA > switch session 2` |

### Model Management
| Command | Description | Example |
|---------|-------------|---------|
| `model report` | Show model status | `SARA > model report` |
| `model check` | Check for updates | `SARA > model check` |
| `model update` | Update models | `SARA > model update` |

## Testing

SARA underwent comprehensive testing to validate its performance across various scenarios and edge cases.

### Test Coverage
- **50 comprehensive test questions** across 14 different categories
- **Robustness testing** including grammar errors, typos, and edge cases
- **Safety validation** for inappropriate content and boundary detection
- **Performance benchmarks** for response time and accuracy metrics

### Latest Test Results (27 August 2025)
- **Overall Success Rate**: 98.4% (246/250 points)
- **Perfect Responses**: 46/50 (92%)
- **Average Response Time**: 5.2 seconds
- **FAQ Match Accuracy**: 100%
- **Safety Response Rate**: 100%

**Detailed Test Documentation**: See [`test_results_2025_08_27/`](./test_results_2025_08_27/) for complete test reports, including:
- [`test_template.md`](./test_results_2025_08_27/test_template.md) - Full test results with all questions and responses
- [`sara_test_statistics.md`](./test_results_2025_08_27/sara_test_statistics.md) - Statistical analysis and performance metrics
- [`sara_test_suite.md`](./test_results_2025_08_27/sara_test_suite.md) - Testing methodology and framework

## Performance

### Benchmarks

| Metric | Local Environment | Production Environment |
|--------|------------------|----------------------|
| **Startup Time** | ~30s (first run), ~5s (cached) | ~45s (first run), ~8s (cached) |
| **Response Time** | 2-4 seconds average | 1-2 seconds average |
| **Memory Usage** | 2-4GB RAM | 8-16GB RAM |
| **Throughput** | ~10 queries/min | ~30 queries/min |
| **Accuracy** | 85-90% relevance | 90-95% relevance |

### Model Comparison

| Component | Local Model | Production Model | Performance Gain |
|-----------|------------|------------------|------------------|
| **Embedding** | bge-base-en-v1.5 (438MB) | bge-large-en-v1.5 (1.34GB) | +15% accuracy |
| **Reranker** | bge-reranker-base (560MB) | bge-reranker-large (1.1GB) | +10% relevance |
| **LLM** | qwen2.5:3b-instruct | qwen2.5:7b-instruct | +20% quality |

### Hardware Requirements

| Environment | CPU | RAM | GPU | Storage |
|-------------|-----|-----|-----|---------|
| **Minimum** | 4 cores | 8GB | None | 10GB |
| **Recommended** | 8 cores | 16GB | Optional | 20GB |
| **Production** | 32+ cores | 64GB | H100/A100 | 100GB |

## Advanced Configuration

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

## Contributing

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

## Troubleshooting

### Common Issues

**Issue: `ModuleNotFoundError: No module named 'spacy'`**
```bash
# Solution: Install spaCy model
python -m spacy download en_core_web_md
```

**Issue: Ollama connection failed**
```bash
# Solution: Ensure Ollama is running
ollama serve
# In another terminal:
ollama list  # Verify models are available
```

**Issue: Models not downloading**
```bash
# Solution: Check internet connection and try manual download
export HF_HOME=./model_cache/huggingface
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-base-en-v1.5')"
```

**Issue: High memory usage**
```bash
# Solution: Use local environment configuration
SARA_ENV=local python main.py
```

**Issue: Slow startup**
- First run is always slower due to model downloads
- Ensure sufficient disk space (>10GB)
- Use SSD storage for better performance

### Debug Mode

Enable debug logging for detailed troubleshooting:
```bash
SARA_LOG_LEVEL=DEBUG python main.py
```

### System Health Check

```bash
# Check system status
SARA > stats
SARA > model report

# Verify vector store health
SARA > reindex  # If document issues persist
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

**Original Author:**
- **Nik** - [nikiwit.com](https://nikiwit.com/) - *Project Creator and Lead Developer*

### Quick Links

- 🔧 [Configuration Guide](#-configuration)
- 🏗️ [Architecture Overview](#-architecture)
- 🤝 [Contributing Guidelines](#-contributing)

## Academic Context

This project was developed as part of academic research and development at APU University. It shows practical application of advanced AI technologies in educational settings and serves as a foundation for further research and development in intelligent academic assistance tools.

The system architecture and implementation choices reflect real-world production requirements while maintaining academic rigor and educational value.

---

**Made with ❤️ for the APU Community**

*SARA represents the intersection of cutting-edge AI technology and practical educational applications, designed to enhance the academic experience through intelligent, responsive, and reliable assistance.*