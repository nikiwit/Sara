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
from .parsers import APUKnowledgeBaseLoader, APUKnowledgeBaseParser
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
            # Check if this is an APU knowledge base file - DIRECT HANDLING
            if 'apu_kb' in filename.lower():
                logger.info(f"Detected APU KB file: {filename} - Using direct KB loader")
                # Create a custom loader that directly calls the static method
                class DirectAPUKBLoader(BaseLoader):
                    def __init__(self, file_path):
                        self.file_path = file_path
                    def load(self):
                        logger.info(f"DirectAPUKBLoader: Loading {self.file_path}")
                        try:
                            with open(self.file_path, 'r', encoding='utf-8') as f:
                                text = f.read()
                                logger.info(f"DirectAPUKBLoader: Successfully read file with {len(text)} characters")
                            
                            # Parse directly with the parser
                            docs = APUKnowledgeBaseParser.parse_apu_kb(
                                text, 
                                source=self.file_path, 
                                filename=os.path.basename(self.file_path)
                            )
                            logger.info(f"DirectAPUKBLoader: Parsed {len(docs)} documents")
                            return docs
                        except Exception as e:
                            logger.error(f"DirectAPUKBLoader error: {e}")
                            return []
                
                return DirectAPUKBLoader(file_path)
            
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
                # Special case for apu_kb.txt (redundant check for safety)
                if 'apu_kb' in filename.lower():
                    logger.info(f"Detected APU KB file in .txt handler: {filename}")
                    # Create a custom loader that directly calls the static method
                    class DirectAPUKBLoader(BaseLoader):
                        def __init__(self, file_path):
                            self.file_path = file_path
                        def load(self):
                            logger.info(f"DirectAPUKBLoader (txt handler): Loading {self.file_path}")
                            try:
                                with open(self.file_path, 'r', encoding='utf-8') as f:
                                    text = f.read()
                                    logger.info(f"DirectAPUKBLoader (txt handler): Successfully read file with {len(text)} characters")
                                
                                # Parse directly with the parser
                                docs = APUKnowledgeBaseParser.parse_apu_kb(
                                    text, 
                                    source=self.file_path, 
                                    filename=os.path.basename(self.file_path)
                                )
                                logger.info(f"DirectAPUKBLoader (txt handler): Parsed {len(docs)} documents")
                                return docs
                            except Exception as e:
                                logger.error(f"DirectAPUKBLoader (txt handler) error: {e}")
                                return []
                    
                    return DirectAPUKBLoader(file_path)
                else:
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
        
        # Verify data directory exists and list all files
        try:
            if not os.path.exists(path):
                logger.error(f"Data directory does not exist: {path}")
                return []
                
            logger.info(f"Data directory exists: {path}")
            
            # List all files in the directory
            all_files_in_dir = os.listdir(path)
            logger.info(f"Files in data directory: {all_files_in_dir}")
            
            # Check specifically for apu_kb.txt
            apu_kb_path = os.path.join(path, "apu_kb.txt")
            if os.path.exists(apu_kb_path):
                logger.info(f"apu_kb.txt exists at: {apu_kb_path}")
                logger.info(f"apu_kb.txt size: {os.path.getsize(apu_kb_path)} bytes")
                logger.info(f"apu_kb.txt permissions: {oct(os.stat(apu_kb_path).st_mode)[-3:]}")
            else:
                logger.error(f"apu_kb.txt NOT FOUND at: {apu_kb_path}")
                
                # Try to find any file with apu_kb in the name
                apu_files = [f for f in all_files_in_dir if 'apu_kb' in f.lower()]
                if apu_files:
                    logger.info(f"Found potential APU KB files: {apu_files}")
                else:
                    logger.error("No files with 'apu_kb' in name found in data directory")
        except Exception as e:
            logger.error(f"Error checking data directory: {e}")
        
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
                        logger.info(f"Added file to processing list: {file}")

            if not all_files:
                logger.warning(f"No compatible documents found in {path}")
                return []

            logger.info(f"Found {len(all_files)} compatible files")

            # SPECIAL DIRECT HANDLING FOR APU_KB.TXT
            apu_kb_path = os.path.join(path, "apu_kb.txt")
            if os.path.exists(apu_kb_path) and os.path.isfile(apu_kb_path):
                logger.info(f"DIRECT HANDLING: Found apu_kb.txt at {apu_kb_path}")
                try:
                    with open(apu_kb_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                        logger.info(f"DIRECT HANDLING: Successfully read apu_kb.txt with {len(text)} characters")
                    
                    # Parse directly with the parser
                    apu_docs = APUKnowledgeBaseParser.parse_apu_kb(
                        text, 
                        source=apu_kb_path, 
                        filename="apu_kb.txt"
                    )
                    
                    if apu_docs:
                        logger.info(f"DIRECT HANDLING: Successfully parsed {len(apu_docs)} documents from apu_kb.txt")
                        
                        # Metadata
                        for doc in apu_docs:
                            doc.metadata['source'] = apu_kb_path
                            doc.metadata['filename'] = "apu_kb.txt"
                            try:
                                doc.metadata['timestamp'] = os.path.getmtime(apu_kb_path)
                            except:
                                doc.metadata['timestamp'] = 0
                        
                        # Load each file with its appropriate loader
                        all_documents = apu_docs
                        
                        # Also process other files
                        for file_path in all_files:
                            if 'apu_kb' not in os.path.basename(file_path).lower():  # Skip apu_kb.txt as we already processed it
                                try:
                                    filename = os.path.basename(file_path)
                                    logger.info(f"Processing file: {filename}")
                                    
                                    # Regular loader path
                                    logger.info(f"Getting loader for: {filename}")
                                    loader = cls.get_file_loader(file_path)
                                    
                                    if loader:
                                        logger.info(f"Loader created for {filename}, type: {type(loader).__name__}")
                                        docs = loader.load()
                                        
                                        if docs:
                                            logger.info(f"Loader returned {len(docs)} documents")
                                            # Source metadata for each document
                                            for doc in docs:
                                                if not hasattr(doc, 'metadata') or doc.metadata is None:
                                                    doc.metadata = {}
                                                doc.metadata['source'] = file_path
                                                doc.metadata['filename'] = os.path.basename(file_path)
                                                
                                                # Timestamp for sorting by recency if needed
                                                try:
                                                    doc.metadata['timestamp'] = os.path.getmtime(file_path)
                                                except:
                                                    doc.metadata['timestamp'] = 0

                                            all_documents.extend(docs)
                                            logger.info(f"Loaded {len(docs)} sections from {os.path.basename(file_path)}")
                                        else:
                                            logger.warning(f"Loader returned no documents for {filename}")
                                    else:
                                        logger.warning(f"No loader created for {filename}")
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
                    logger.error(f"DIRECT HANDLING: Error processing apu_kb.txt: {e}")
                    # Continue with regular processing as fallback
            else:
                logger.warning(f"DIRECT HANDLING: apu_kb.txt not found at {apu_kb_path}")

            # Load each file with its appropriate loader
            all_documents = []
            for file_path in all_files:
                try:
                    filename = os.path.basename(file_path)
                    logger.info(f"Processing file: {filename}")
                    
                    # Special direct handling for apu_kb.txt
                    if 'apu_kb' in filename.lower():
                        logger.info(f"DIRECT PROCESSING for APU KB file: {filename}")
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                text = f.read()
                                logger.info(f"Successfully read APU KB file with {len(text)} characters")
                            
                            # Parse directly with the parser
                            docs = APUKnowledgeBaseParser.parse_apu_kb(
                                text, 
                                source=file_path, 
                                filename=filename
                            )
                            
                            if docs:
                                logger.info(f"Successfully parsed {len(docs)} documents from APU KB file")
                                # Timestamp for sorting by recency if needed
                                for doc in docs:
                                    try:
                                        doc.metadata['timestamp'] = os.path.getmtime(file_path)
                                    except:
                                        doc.metadata['timestamp'] = 0
                                
                                all_documents.extend(docs)
                                logger.info(f"Added {len(docs)} APU KB documents to collection")
                            else:
                                logger.warning(f"No documents parsed from APU KB file: {filename}")
                            
                            continue  # Skip the regular loader path
                        except Exception as e:
                            logger.error(f"Error in direct APU KB processing: {e}")
                            # Fall through to regular loader as backup
                    
                    # Regular loader path
                    logger.info(f"Getting loader for: {filename}")
                    loader = cls.get_file_loader(file_path)
                    
                    if loader:
                        logger.info(f"Loader created for {filename}, type: {type(loader).__name__}")
                        docs = loader.load()
                        
                        if docs:
                            logger.info(f"Loader returned {len(docs)} documents")
                            # Source metadata to each document
                            for doc in docs:
                                if not hasattr(doc, 'metadata') or doc.metadata is None:
                                    doc.metadata = {}
                                doc.metadata['source'] = file_path
                                doc.metadata['filename'] = os.path.basename(file_path)
                                
                                # Timestamp for sorting by recency if needed
                                try:
                                    doc.metadata['timestamp'] = os.path.getmtime(file_path)
                                except:
                                    doc.metadata['timestamp'] = 0

                            all_documents.extend(docs)
                            logger.info(f"Loaded {len(docs)} sections from {os.path.basename(file_path)}")
                        else:
                            logger.warning(f"Loader returned no documents for {filename}")
                    else:
                        logger.warning(f"No loader created for {filename}")
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
            
            logger.info(f"Document split: {len(apu_kb_docs)} APU KB docs, {len(standard_docs)} standard docs")
            
            chunked_documents = []
            
            # Use APU KB specific splitter for knowledge base pages
            if apu_kb_docs:
                apu_kb_splitter = APUKnowledgeBaseTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=len,
                    is_separator_regex=False,
                )
                kb_chunks = apu_kb_splitter.split_documents(apu_kb_docs)
                logger.info(f"APU KB splitter created {len(kb_chunks)} chunks from {len(apu_kb_docs)} documents")
                chunked_documents.extend(kb_chunks)
            
            # Use standard splitter for other documents
            if standard_docs:
                standard_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=len,
                    is_separator_regex=False,
                )
                std_chunks = standard_splitter.split_documents(standard_docs)
                logger.info(f"Standard splitter created {len(std_chunks)} chunks from {len(standard_docs)} documents")
                chunked_documents.extend(std_chunks)
            
            # Remove any empty chunks
            valid_chunks = [chunk for chunk in chunked_documents if chunk.page_content and chunk.page_content.strip()]
            
            # Log statistics
            logger.info(f"Created {len(valid_chunks)} chunks from {len(documents)} documents")
            
            return valid_chunks
            
        except Exception as e:
            logger.error(f"Error splitting documents: {e}")
            return []
