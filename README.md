# APURAG - Advanced Retrieval Augmented Generation System

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

APURAG is an advanced Retrieval Augmented Generation (RAG) system specifically designed for the APU Knowledge Base. It provides intelligent question-answering capabilities by leveraging natural language processing and vector-based retrieval techniques.

## 🌟 Features

- **Specialized APU Knowledge Base Integration**
  - Optimized for FAQ-style content
  - Preserves document structure and hierarchy
  - Enhanced metadata extraction

- **Advanced Document Processing**
  - Support for multiple document formats:
    - PDF
    - Word (DOCX)
    - PowerPoint (PPTX)
    - EPUB
  - Intelligent text extraction and processing
  - Structure preservation

- **Enhanced Query Processing**
  - Education-specific query classification
  - FAQ-optimized retrieval strategies
  - Improved direct question matching
  - Better context generation for Q&A content

- **Vector Management**
  - Efficient vector storage and retrieval
  - Optimized similarity search
  - Backup and restore capabilities

- **Dual Environment Configuration**
  - Local development environment optimized for laptops
  - Production environment optimized for HGX H100 G593-SD2 servers
  - Environment-specific resource management
  - Automatic hardware detection and optimization

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Sufficient disk space for vector storage
- Internet connection for initial package downloads
- [Ollama](https://ollama.ai/) installed and running locally
  - Required for local LLM inference
  - Must be running in the background while using APURAG
  - Minimum 8GB RAM recommended for optimal performance

## 🚀 Installation

1. Install and set up Ollama:
   ```bash
   # For macOS/Linux
   curl https://ollama.ai/install.sh | sh
   # or manual download and install from https://ollama.ai/download

   
   # For Windows
   # Download and install from https://ollama.ai/download
   ```

2. Pull the required Ollama model:
   ```bash
   # For local development
   ollama pull deepseek-r1:1.5b
   
   # For production
   ollama pull deepseek-r1:7b
   ```

3. Ensure Ollama is running:
   ```bash
   # Check Ollama status
   ollama list
   
   # If not running, start Ollama
   ollama serve
   ```

4. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/APURAG.git
   cd APURAG
   ```

5. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

6. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

7. Set up environment configuration:
   ```bash
   # For local development (default)
   cp .env.local .env
   
   # For production environment
   # cp .env.production .env
   ```

## 💻 Usage

1. Ensure Ollama is running in the background:
   ```bash
   ollama serve
   ```

2. Start the application:
   ```bash
   # Run with default environment (local)
   python main.py
   
   # Or specify environment explicitly
   APURAG_ENV=production python main.py
   ```

3. Follow the interactive CLI prompts to:
   - Process documents
   - Query the knowledge base
   - Manage vector storage
   - Configure system settings

## 📁 Project Structure

*Make sure to request environment files for up-to-date settings (.env files are in gitignore by default for security reasons).*

```
APURAG/
├── app.py                 # Main application logic
├── main.py                # Entry point
├── config.py              # Base configuration settings
├── config_local.py        # Local environment configuration
├── config_production.py   # Production environment configuration
├── resource_manager.py    # Hardware resource management
├── apurag_types.py        # Type definitions
├── input_processing.py    # Input handling
├── utils.py               # Utility functions
├── requirements.txt       # Project dependencies
├── .env.local             # Local environment variables template
├── .env.production        # Production environment variables template
├── .gitignore             # Git ignore rules
├── data/                  # Data directory
├── document_processing/   # Document processing modules
├── query_handling/        # Query processing modules
├── response/              # Response generation modules
├── retrieval/             # Retrieval system modules
└── vector_management/     # Vector storage management
```

## 🔧 Configuration

The system supports dual environment configuration:

### Environment Selection
- Set `APURAG_ENV` to either `local` (default) or `production`
- Or copy the appropriate `.env.local` or `.env.production` to `.env`

### Local Environment (Laptops)
- Optimized for lower resource usage
- Smaller models and chunk sizes
- Reduced memory footprint
- Debug-level logging

### Production Environment (HGX H100)
- Optimized for high-performance hardware
- Larger, more capable models
- Enhanced retrieval parameters
- Production-level logging
- Absolute file paths for reliability

### Configuration Files
- `config.py` - Base configuration with environment detection
- `config_local.py` - Local-specific configuration overrides
- `config_production.py` - Production-specific configuration overrides
- `.env.local` - Environment variables for local development
- `.env.production` - Environment variables for production deployment

## 📚 Dependencies

Major dependencies include:

- **Core:**
  - torch
  - langchain
  - sentence-transformers
  - chromadb

- **Document Processing:**
  - pypdf
  - python-docx
  - python-pptx
  - ebooklib
  - unstructured

- **Utilities:**
  - python-dotenv
  - numpy
  - tqdm
  - psutil

For a complete list, see `requirements.txt`.

## 📝 Version Control

The project includes a comprehensive `.gitignore` file to ensure that only necessary files are tracked in version control:

### Excluded from Version Control
- Environment-specific files (`.env`, `.env.local`, `.env.production`)
- Vector store data (`vector_store/*`)
- Log files (`*.log`)
- Cached Python files (`__pycache__/`, `*.pyc`)
- OS-specific files (`.DS_Store`, etc.)
- IDE configuration files (`.vscode/`, `.idea/`)

### Best Practices
- Always use the provided `.gitignore` to avoid committing sensitive or large files
- Keep data files in the `data/` directory (only `apu_kb.txt` is tracked)
- Store vector databases in the `vector_store/` directory (not tracked)
- Use environment-specific configuration files for local customization

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- **Nik** - [nikiwit.com](https://nikiwit.com/) - *Original Author* 

## 🙏 Acknowledgments

- Thanks to all contributors who have helped shape this project

## 📞 Support

For support, please open an issue in the GitHub repository or contact the maintainers.

---

Made with ❤️ for the APU community
