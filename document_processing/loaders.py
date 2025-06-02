"""
Document loaders and processors for general document processing.
Handles loading, processing, and splitting documents from various file formats.
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

from config import config as Config

logger = logging.getLogger("CustomRAG")

class DocumentProcessor:
    """Handles loading, processing, and splitting documents from various file formats."""
    
    @staticmethod
    def check_dependencies() -> bool:
        """Verify that required dependencies are installed for document processing."""
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
    def should_process_file(filename: str) -> bool:
        """Determine if a file should be processed based on filtering settings."""
        # Check if content filtering is enabled
        if hasattr(Config, 'FILTER_CONTENT') and Config.FILTER_CONTENT:
            # Add your custom filtering logic here if needed
            # For now, process all files when no specific filter is set
            return True
        
        # Process all files by default
        return True
    
    @staticmethod
    def get_file_loader(file_path: str):
        """Returns appropriate loader based on file extension with comprehensive error handling."""
        ext = os.path.splitext(file_path)[1].lower()
        filename = os.path.basename(file_path)

        try:
            # Check for specialized knowledge base format (if you have any)
            if filename.lower().endswith(('.kb', '.knowledge')):
                logger.info(f"Detected knowledge base file: {filename} - Using specialized loader")
                
                class KnowledgeBaseLoader(BaseLoader):
                    def __init__(self, file_path):
                        self.file_path = file_path
                        self.filename = os.path.basename(file_path)
                    
                    def load(self):
                        logger.info(f"KnowledgeBaseLoader: Loading {self.filename}")
                        try:
                            with open(self.file_path, 'r', encoding='utf-8') as f:
                                text = f.read()
                            logger.info(f"KnowledgeBaseLoader: Read {len(text)} characters from {self.filename}")
                            
                            # Simple parsing - split by double newlines for sections
                            sections = text.split('\n\n')
                            docs = []
                            
                            for i, section in enumerate(sections):
                                if section.strip():
                                    doc = Document(
                                        page_content=section.strip(),
                                        metadata={
                                            'source': self.file_path,
                                            'filename': self.filename,
                                            'section': i + 1,
                                            'content_type': 'knowledge_base'
                                        }
                                    )
                                    docs.append(doc)
                            
                            logger.info(f"KnowledgeBaseLoader: Parsed {len(docs)} sections from {self.filename}")
                            return docs
                            
                        except Exception as e:
                            logger.error(f"KnowledgeBaseLoader error for {self.filename}: {e}")
                            return []
                
                return KnowledgeBaseLoader(file_path)
            
            # Standard file type handlers
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
                    logger.warning(f"UnstructuredEPubLoader failed for {filename}: {e}")
                    logger.info("Trying alternative EPUB loader")
                    
                    # Fallback to custom EPUB loader
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
                return TextLoader(file_path, encoding='utf-8')
            else:
                logger.warning(f"Unsupported file type: {ext} for file {filename}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating loader for {file_path}: {e}")
            return None
    
    @classmethod
    def load_documents(cls, path: str, extensions: List[str] = None) -> List:
        """
        Load documents from specified path with comprehensive file type support.
        """
        if extensions is None:
            extensions = getattr(Config, 'SUPPORTED_EXTENSIONS', ['.txt', '.pdf', '.docx', '.md', '.epub'])
        
        logger.info(f"Loading documents from: {path}")
        
        # Validate data directory
        if not os.path.exists(path):
            logger.error(f"Data directory does not exist: {path}")
            return []
        
        try:
            # List all files in directory for debugging
            all_files_in_dir = os.listdir(path)
            logger.info(f"Files in data directory: {all_files_in_dir}")
            
            # Find compatible files
            compatible_files = []
            skipped_files = []
            
            for root, _, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    ext = os.path.splitext(file)[1].lower()
                    
                    # Skip non-supported extensions
                    if ext not in extensions:
                        continue
                    
                    # Apply content filtering if enabled
                    if not cls.should_process_file(file):
                        skipped_files.append(file)
                        continue
                    
                    compatible_files.append(file_path)
                    logger.info(f"Added file to processing queue: {file}")
            
            # Log filtering results
            if skipped_files:
                logger.info(f"Skipped {len(skipped_files)} files due to filtering: {skipped_files}")
            
            if not compatible_files:
                logger.warning(f"No compatible documents found in {path}")
                return []

            logger.info(f"Found {len(compatible_files)} compatible files for processing")

            # Process each file
            all_documents = []
            
            for file_path in compatible_files:
                try:
                    filename = os.path.basename(file_path)
                    logger.info(f"Processing file: {filename}")
                    
                    # Get appropriate loader
                    loader = cls.get_file_loader(file_path)
                    
                    if not loader:
                        logger.warning(f"No loader available for {filename}")
                        continue
                    
                    # Load documents
                    logger.info(f"Loading documents from {filename} using {type(loader).__name__}")
                    docs = loader.load()
                    
                    if not docs:
                        logger.warning(f"No documents loaded from {filename}")
                        continue
                    
                    logger.info(f"Loaded {len(docs)} document sections from {filename}")
                    
                    # Add metadata to each document
                    for doc in docs:
                        if not hasattr(doc, 'metadata') or doc.metadata is None:
                            doc.metadata = {}
                        
                        doc.metadata.update({
                            'source': file_path,
                            'filename': filename,
                            'file_type': os.path.splitext(filename)[1].lower(),
                        })
                        
                        # Add timestamp
                        try:
                            doc.metadata['timestamp'] = os.path.getmtime(file_path)
                        except Exception as e:
                            logger.debug(f"Could not get timestamp for {filename}: {e}")
                            doc.metadata['timestamp'] = 0
                    
                    all_documents.extend(docs)
                    
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    continue

            # Filter out empty documents
            valid_documents = [doc for doc in all_documents if doc.page_content and doc.page_content.strip()]
            
            if not valid_documents:
                logger.warning("No valid document content could be extracted")
                return []
            
            # Log final statistics
            logger.info(f"Successfully loaded {len(valid_documents)} document sections from {len(compatible_files)} files")
            
            return valid_documents

        except Exception as e:
            logger.error(f"Error during document loading: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return []
    
    @staticmethod
    def load_epub(file_path: str):
        """
        Custom EPUB loader using ebooklib for comprehensive content extraction.
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
            h2t.ignore_images = True
            h2t.body_width = 0  # No line wrapping
            
            # Get book metadata
            title = "Unknown Title"
            try:
                title_meta = book.get_metadata('DC', 'title')
                if title_meta:
                    title = title_meta[0][0]
            except Exception as e:
                logger.debug(f"Could not extract EPUB title: {e}")
            
            # Process each chapter/item
            chapter_count = 0
            for item in book.get_items():
                if item.get_type() == epub.ITEM_DOCUMENT:
                    try:
                        # Extract HTML content
                        html_content = item.get_content().decode('utf-8')
                        
                        # Parse with BeautifulSoup for better cleaning
                        soup = BeautifulSoup(html_content, 'html.parser')
                        
                        # Remove script and style elements
                        for script in soup(["script", "style"]):
                            script.decompose()
                        
                        # Convert to text
                        text = h2t.handle(str(soup))
                        
                        if text.strip():
                            chapter_count += 1
                            
                            # Create document with comprehensive metadata
                            doc = Document(
                                page_content=text.strip(),
                                metadata={
                                    'source': file_path,
                                    'filename': filename,
                                    'title': title,
                                    'chapter': item.get_name() or f"Chapter_{chapter_count}",
                                    'chapter_number': chapter_count,
                                    'file_type': '.epub'
                                }
                            )
                            documents.append(doc)
                            
                    except Exception as e:
                        logger.warning(f"Error processing EPUB chapter {item.get_name()}: {e}")
                        continue
            
            logger.info(f"Extracted {len(documents)} chapters from EPUB: {filename}")
            return documents
            
        except ImportError:
            logger.error("ebooklib not available for EPUB processing")
            logger.error("Install with: pip install EbookLib")
            return None
        except Exception as e:
            logger.error(f"Error loading EPUB file {file_path}: {e}")
            return None
    
    @staticmethod
    def split_documents(documents: List, chunk_size: int = None, chunk_overlap: int = None) -> List:
        """
        Split documents into smaller chunks for optimal retrieval performance.
        """
        if not documents:
            logger.warning("No documents to split")
            return []
            
        if chunk_size is None:
            chunk_size = getattr(Config, 'CHUNK_SIZE', 1000)
            
        if chunk_overlap is None:
            chunk_overlap = getattr(Config, 'CHUNK_OVERLAP', 200)
            
        logger.info(f"Splitting {len(documents)} documents into chunks (size={chunk_size}, overlap={chunk_overlap})")
        
        try:
            # Group documents by type for specialized splitting if needed
            knowledge_base_docs = []
            standard_docs = []
            
            for doc in documents:
                if doc.metadata.get('content_type') == 'knowledge_base':
                    knowledge_base_docs.append(doc)
                else:
                    standard_docs.append(doc)
            
            logger.info(f"Document categorization: {len(knowledge_base_docs)} knowledge base docs, {len(standard_docs)} standard docs")
            
            chunked_documents = []
            
            # Use specialized splitter for knowledge base content if available
            if knowledge_base_docs:
                try:
                    # Try to use specialized splitter if available
                    try:
                        from .splitters import KnowledgeBaseTextSplitter
                        kb_splitter = KnowledgeBaseTextSplitter(
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            length_function=len,
                            is_separator_regex=False,
                        )
                        kb_chunks = kb_splitter.split_documents(knowledge_base_docs)
                        logger.info(f"Knowledge base splitter created {len(kb_chunks)} chunks from {len(knowledge_base_docs)} documents")
                        chunked_documents.extend(kb_chunks)
                    except ImportError:
                        logger.info("Specialized knowledge base splitter not available, using standard splitter")
                        standard_docs.extend(knowledge_base_docs)
                except Exception as e:
                    logger.error(f"Error in knowledge base splitting: {e}")
                    # Fallback to standard splitter
                    logger.info("Falling back to standard splitter for knowledge base docs")
                    standard_docs.extend(knowledge_base_docs)
            
            # Use standard splitter for other documents
            if standard_docs:
                try:
                    standard_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        length_function=len,
                        is_separator_regex=False,
                        separators=["\n\n", "\n", " ", ""]  # Standard hierarchical separators
                    )
                    std_chunks = standard_splitter.split_documents(standard_docs)
                    logger.info(f"Standard splitter created {len(std_chunks)} chunks from {len(standard_docs)} documents")
                    chunked_documents.extend(std_chunks)
                except Exception as e:
                    logger.error(f"Error in standard document splitting: {e}")
                    raise
            
            # Filter out empty or very small chunks
            min_chunk_size = max(10, chunk_size // 20)  # At least 10 chars, or 5% of chunk size
            valid_chunks = []
            
            for chunk in chunked_documents:
                if chunk.page_content and len(chunk.page_content.strip()) >= min_chunk_size:
                    valid_chunks.append(chunk)
                else:
                    logger.debug(f"Filtered out small chunk: {len(chunk.page_content) if chunk.page_content else 0} chars")
            
            # Final statistics
            logger.info(f"Document splitting complete:")
            logger.info(f"  Input: {len(documents)} documents")
            logger.info(f"  Output: {len(valid_chunks)} valid chunks")
            logger.info(f"  Filtered: {len(chunked_documents) - len(valid_chunks)} small/empty chunks")
            
            return valid_chunks
            
        except Exception as e:
            logger.error(f"Error splitting documents: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return []