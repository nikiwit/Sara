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

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Sufficient disk space for vector storage
- Internet connection for initial package downloads

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/APURAG.git
   cd APURAG
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

1. Start the application:
   ```bash
   python main.py
   ```

2. Follow the interactive CLI prompts to:
   - Process documents
   - Query the knowledge base
   - Manage vector storage
   - Configure system settings

## 📁 Project Structure

```
APURAG/
├── app.py                 # Main application logic
├── main.py               # Entry point
├── config.py             # Configuration settings
├── apurag_types.py       # Type definitions
├── input_processing.py   # Input handling
├── utils.py             # Utility functions
├── requirements.txt      # Project dependencies
├── data/                # Data directory
├── document_processing/ # Document processing modules
├── query_handling/      # Query processing modules
├── response/           # Response generation modules
├── retrieval/          # Retrieval system modules
└── vector_management/  # Vector storage management
```

## 🔧 Configuration

The system can be configured through `config.py`. Key configuration options include:

- Vector store settings
- Document processing parameters
- Query handling preferences
- Logging configuration

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

For a complete list, see `requirements.txt`.

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

- **Nik** - *Original Author*

## 🙏 Acknowledgments

- Thanks to all contributors who have helped shape this project
- Special thanks to the APU community for their support and feedback

## 📞 Support

For support, please open an issue in the GitHub repository or contact the maintainers.

---

Made with ❤️ for the APU community 