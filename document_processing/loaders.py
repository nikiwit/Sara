"""
Document loaders and processors.
"""

import os
import logging
from typing import List
import html2text
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredPowerPointLoader, UnstructuredEPubLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import Config
from .parsers import APUKnowledgeBaseLoader
from .splitters import APUKnowledgeBaseTextSplitter

logger = logging.getLogger("CustomRAG")

class DocumentProcessor:
    """Handles loading, processing, and splitting documents."""
    
    @staticmethod
    def check_dependencies() -> bool:
        """Verify that required dependencies are installed."""
        missing_deps = []
        
        try:
            import docx2txt
        except ImportError:
            missing_deps.append("docx2txt (for DOCX files)")
        
        try:
            import pypdf
        except ImportError:
            missing_deps.append("pypdf (for PDF files)")
        
        try:
            import html2text
        except ImportError:
            missing_deps.append("html2text (for EPUB files)")
        
        try:
            import bs4
        except ImportError:
            missing_deps.append("beautifulsoup4 (for EPUB files)")
        
        if missing_deps:
            logger.warning(f"Missing dependencies: {', '.join(missing_deps)}")
            logger.warning("Some document types may not load correctly.")
            logger.warning("Install missing dependencies with: pip install " + " ".join([d.split(' ')[0] for d in missing_deps]))
            return False
            
        return True
    
    @staticmethod
    def get_file_loader(file_path: str):
        """Returns appropriate loader based on file extension with error handling."""
        ext = os.path.splitext(file_path)[1].lower()
        filename = os.path.basename(file_path)  # Define filename variable

        try:
            # Check if this is an APU knowledge base file
            if 'apu_kb' in filename.lower() and ext in ['.txt', '.md']:
                logger.info(f"Loading APU Knowledge Base file: {filename}")
                return APUKnowledgeBaseLoader(file_path)
            
            # Regular file types
            if ext == '.pdf':
                return PyPDFLoader(file_path)
            elif ext in ['.docx', '.doc']:
                return Docx2txtLoader(file_path)
            elif ext in ['.ppt', '.pptx']:
                return UnstructuredPowerPointLoader(file_path)
            elif ext == '.epub':
                logger.info(f"Loading EPUB file: {filename}")
                try:
                    return UnstructuredEPubLoader(file_path)
                except Exception as e:
                    logger.warning(f"UnstructuredEPubLoader failed: {e}, trying alternative EPUB loader")
                    # Use DocumentProcessor instead of cls
                    docs = DocumentProcessor.load_epub(file_path)
                    if docs:
                        class CustomEpubLoader(BaseLoader):
                            def __init__(self, documents):
                                self.documents = documents
                            def load(self):
                                return self.documents
                        return CustomEpubLoader(docs)
                    return None
            elif ext in ['.txt', '.md', '.csv']:
                return TextLoader(file_path)
            else:
                logger.warning(f"Unsupported file type: {ext} for file {filename}")
                return None
        except Exception as e:
            logger.error(f"Error creating loader for {file_path}: {str(e)}")
            return None
    
    @classmethod
    def load_documents(cls, path: str, extensions: List[str] = None) -> List:
        """
        Load documents from specified path with specified extensions.
        Returns a list of documents or empty list if none found.
        
        Filtering behavior:
        - When FILTER_APU_ONLY=true: Only loads files starting with "apu_"
        - Otherwise: Loads all files with supported extensions
        """
        if extensions is None:
            extensions = Config.SUPPORTED_EXTENSIONS
                
        # Use the configuration variable for filtering
        filter_apu_only = Config.FILTER_APU_ONLY
        
        if filter_apu_only:
            logger.info("APU-only filtering is ENABLED - loading only files starting with 'apu_'")
        else:
            logger.info("APU-only filtering is DISABLED - loading all compatible files")
        
        logger.info(f"Loading documents from: {path}")
        
        try:
            # Find all files with supported extensions
            all_files = []
            for root, _, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    ext = os.path.splitext(file_path)[1].lower()
                    
                    # Only process files with supported extensions
                    if ext in extensions:
                        # Apply APU-only filtering if enabled
                        if filter_apu_only and not file.lower().startswith("apu_"):
                            logger.info(f"Skipping non-APU document: {file}")
                            continue
                        
                        all_files.append(file_path)

            if not all_files:
                logger.warning(f"No compatible documents found in {path}")
                return []

            logger.info(f"Found {len(all_files)} compatible files")

            # Load each file with its appropriate loader
            all_documents = []
            for file_path in all_files:
                try:
                    logger.info(f"Loading: {os.path.basename(file_path)}")
                    loader = cls.get_file_loader(file_path)
                    if loader:
                        docs = loader.load()
                        if docs:
                            # Add source metadata to each document
                            for doc in docs:
                                if not hasattr(doc, 'metadata') or doc.metadata is None:
                                    doc.metadata = {}
                                doc.metadata['source'] = file_path
                                doc.metadata['filename'] = os.path.basename(file_path)
                                
                                # Add timestamp for sorting by recency if needed
                                try:
                                    doc.metadata['timestamp'] = os.path.getmtime(file_path)
                                except:
                                    doc.metadata['timestamp'] = 0

                            all_documents.extend(docs)
                            logger.info(f"Loaded {len(docs)} sections from {os.path.basename(file_path)}")
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {str(e)}")
                    continue

            # Filter out empty documents
            valid_documents = [doc for doc in all_documents if doc.page_content and doc.page_content.strip()]
            
            if not valid_documents:
                logger.warning("No document content could be extracted successfully")
                    
            logger.info(f"Successfully loaded {len(valid_documents)} total document sections")
            return valid_documents

        except Exception as e:
            logger.error(f"Document loading error: {e}")
            return []
    
    @staticmethod
    def load_epub(file_path: str):
        """
        Custom EPUB loader using ebooklib.
        Returns a list of LangChain Document objects.
        """
        try:
            from ebooklib import epub
            
            filename = os.path.basename(file_path)
            logger.info(f"Loading EPUB with custom loader: {filename}")
            
            # Load the EPUB file
            book = epub.read_epub(file_path)
            
            # Extract and process content
            documents = []
            h2t = html2text.HTML2Text()
            h2t.ignore_links = False
            
            # Get book title and metadata
            title = book.get_metadata('DC', 'title')[0][0] if book.get_metadata('DC', 'title') else "Unknown Title"
            
            # Process each chapter/item
            for item in book.get_items():
                if item.get_type() == epub.ITEM_DOCUMENT:
                    # Extract HTML content
                    html_content = item.get_content().decode('utf-8')
                    
                    # Parse with BeautifulSoup
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    # Get plain text content
                    text = h2t.handle(str(soup))
                    
                    if text.strip():
                        # Create a document with metadata
                        doc = Document(
                            page_content=text,
                            metadata={
                                'source': file_path,
                                'filename': filename,
                                'title': title,
                                'chapter': item.get_name(),
                            }
                        )
                        documents.append(doc)
            
            logger.info(f"Extracted {len(documents)} sections from EPUB")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading EPUB file {file_path}: {str(e)}")
            return None
    
    @staticmethod
    def split_documents(documents: List, chunk_size: int = None, chunk_overlap: int = None) -> List:
        """
        Split documents into smaller chunks for better retrieval.
        
        Args:
            documents: List of documents to split
            chunk_size: Size of each chunk (in characters)
            chunk_overlap: Overlap between chunks (in characters)
            
        Returns:
            List of document chunks
        """
        if not documents:
            logger.warning("No documents to split")
            return []
            
        if chunk_size is None:
            chunk_size = Config.CHUNK_SIZE
            
        if chunk_overlap is None:
            chunk_overlap = Config.CHUNK_OVERLAP
            
        logger.info(f"Splitting {len(documents)} documents into chunks (size={chunk_size}, overlap={chunk_overlap})")
        
        try:
            # Group documents by type
            apu_kb_docs = []
            standard_docs = []
            
            for doc in documents:
                if doc.metadata.get('content_type') == 'apu_kb_page':
                    apu_kb_docs.append(doc)
                else:
                    standard_docs.append(doc)
            
            chunked_documents = []
            
            # Use APU KB-specific splitter for knowledge base pages
            if apu_kb_docs:
                apu_kb_splitter = APUKnowledgeBaseTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=len,
                    is_separator_regex=False,
                )
                chunked_documents.extend(apu_kb_splitter.split_documents(apu_kb_docs))
            
            # Use standard splitter for other documents
            if standard_docs:
                standard_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=len,
                    is_separator_regex=False,
                )
                chunked_documents.extend(standard_splitter.split_documents(standard_docs))
            
            # Remove any empty chunks
            valid_chunks = [chunk for chunk in chunked_documents if chunk.page_content and chunk.page_content.strip()]
            
            # Log statistics
            logger.info(f"Created {len(valid_chunks)} chunks from {len(documents)} documents")
            
            return valid_chunks
            
        except Exception as e:
            logger.error(f"Error splitting documents: {e}")
            return []