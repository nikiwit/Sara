"""
Enhanced CustomRAG - An Advanced Retrieval Augmented Generation System for APU Knowledge Base
----------------------------------------------------------------------------------
This application allows users to query the APU knowledge base using natural language.
It has been specially optimized for FAQ-style content organized in pages.

Features:
- Specialized APU knowledge base parsing and structure preservation
- Enhanced metadata extraction from structured content
- Education-specific query classification
- FAQ-optimized retrieval strategies
- Better context generation for Q&A content
- Improved direct question matching

Original Author: Nik
Enhanced By: Claude
License: MIT
"""

import os
import shutil
import sqlite3
import time
import logging
import torch
import sys
import json
import re
import random
import math
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple, Set, Iterator
from pathlib import Path
from collections import Counter, defaultdict
from enum import Enum
from datetime import datetime
from chromadb.config import Settings
import types

# NLP imports
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.util import ngrams

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)

# EPUB processing imports
import html2text
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader

# Document processing imports
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredPowerPointLoader, UnstructuredEPubLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter

# Vector store and embedding imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# LLM and prompt imports
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
import requests

# Configure logging - production settings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("customrag.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CustomRAG")

import chromadb
logger.info(f"Using ChromaDB version: {chromadb.__version__}")

#############################################################################
# CONFIGURATION
#############################################################################

class Config:
    """Configuration settings for the RAG application."""
    
    # Paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.environ.get("CUSTOMRAG_DATA_PATH", os.path.join(SCRIPT_DIR, "data"))
    PERSIST_PATH = os.environ.get("CUSTOMRAG_VECTOR_PATH", os.path.join(SCRIPT_DIR, "vector_store"))
    
    # Embedding and retrieval settings
    EMBEDDING_MODEL_NAME = os.environ.get("CUSTOMRAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    LLM_MODEL_NAME = os.environ.get("CUSTOMRAG_LLM_MODEL", "deepseek-r1:1.5b")
    
    # Chunking settings
    CHUNK_SIZE = int(os.environ.get("CUSTOMRAG_CHUNK_SIZE", "500"))
    CHUNK_OVERLAP = int(os.environ.get("CUSTOMRAG_CHUNK_OVERLAP", "150"))
    
    # Retrieval settings
    RETRIEVER_K = int(os.environ.get("CUSTOMRAG_RETRIEVER_K", "6"))
    RETRIEVER_SEARCH_TYPE = os.environ.get("CUSTOMRAG_SEARCH_TYPE", "hybrid")  # Changed to hybrid as default
    KEYWORD_RATIO = float(os.environ.get("CUSTOMRAG_KEYWORD_RATIO", "0.4"))  # 40% weight to keywords by default for FAQ
    FAQ_MATCH_WEIGHT = float(os.environ.get("CUSTOMRAG_FAQ_MATCH_WEIGHT", "0.5"))  # Weight for direct FAQ matches
    
    # Query processing settings
    USE_QUERY_EXPANSION = os.environ.get("CUSTOMRAG_QUERY_EXPANSION", "True").lower() in ("true", "1", "t")
    EXPANSION_FACTOR = int(os.environ.get("CUSTOMRAG_EXPANSION_FACTOR", "3"))
    
    # Context processing settings
    MAX_CONTEXT_SIZE = int(os.environ.get("CUSTOMRAG_MAX_CONTEXT_SIZE", "4000"))
    USE_CONTEXT_COMPRESSION = os.environ.get("CUSTOMRAG_CONTEXT_COMPRESSION", "True").lower() in ("true", "1", "t")
    
    # Ollama API
    OLLAMA_BASE_URL = os.environ.get("CUSTOMRAG_OLLAMA_URL", "http://localhost:11434")
    
    # Miscellaneous
    FORCE_REINDEX = os.environ.get("CUSTOMRAG_FORCE_REINDEX", "False").lower() in ("true", "1", "t")
    LOG_LEVEL = os.environ.get("CUSTOMRAG_LOG_LEVEL", "INFO")
    
    # APU filtering setting - set default to True to only process APU files
    FILTER_APU_ONLY = os.environ.get("FILTER_APU_ONLY", "False").lower() in ("true", "1", "t")
    
    # Supported file types
    SUPPORTED_EXTENSIONS = ['.pdf', '.txt', '.docx', '.doc', '.md', '.ppt', '.pptx', '.epub']
    
    # APU KB specific settings
    APU_KB_ANSWER_CONTEXT_SIZE = int(os.environ.get("CUSTOMRAG_APU_KB_ANSWER_SIZE", "3"))
    APU_KB_EXACT_MATCH_BOOST = float(os.environ.get("CUSTOMRAG_APU_KB_EXACT_MATCH_BOOST", "2.0"))

    @classmethod
    def setup(cls):
        """Set up the configuration and ensure directories exist."""
        # Set logging level based on configuration
        log_level = getattr(logging, cls.LOG_LEVEL.upper(), logging.INFO)
        logger.setLevel(log_level)
        
        # Ensure data directory exists
        os.makedirs(cls.DATA_PATH, exist_ok=True)
        
        logger.info(f"Data directory: {cls.DATA_PATH}")
        logger.info(f"Vector store directory: {cls.PERSIST_PATH}")
        logger.info(f"Embedding model: {cls.EMBEDDING_MODEL_NAME}")
        logger.info(f"LLM model: {cls.LLM_MODEL_NAME}")
        logger.info(f"Search type: {cls.RETRIEVER_SEARCH_TYPE}")
        
        if cls.RETRIEVER_SEARCH_TYPE == "hybrid":
            logger.info(f"Keyword ratio: {cls.KEYWORD_RATIO}")
            logger.info(f"FAQ match weight: {cls.FAQ_MATCH_WEIGHT}")
        
        if cls.USE_QUERY_EXPANSION:
            logger.info(f"Query expansion enabled with factor: {cls.EXPANSION_FACTOR}")
            
        if cls.USE_CONTEXT_COMPRESSION:
            logger.info(f"Context compression enabled")
            
        # Log APU filtering status
        if cls.FILTER_APU_ONLY:
            logger.info("APU document filtering is ENABLED - only files starting with 'apu_' will be processed")
        else:
            logger.info("APU document filtering is DISABLED - all compatible files will be processed")

#############################################################################
# ENUMS AND TYPES
#############################################################################

class QueryType(Enum):
    """Enum for different types of queries."""
    FACTUAL = "factual"           # Asking for specific facts
    PROCEDURAL = "procedural"     # How to do something
    CONCEPTUAL = "conceptual"     # Understanding concepts
    EXPLORATORY = "exploratory"   # Open-ended exploration
    COMPARATIVE = "comparative"   # Comparing things
    CONVERSATIONAL = "conversational"  # Social conversation
    COMMAND = "command"           # System commands
    IDENTITY = "identity"         # Questions about the assistant itself
    # APU specific query types
    ACADEMIC = "academic"         # Academic-related queries (courses, exams)
    ADMINISTRATIVE = "administrative"  # Administrative processes
    FINANCIAL = "financial"       # Fees, payments
    UNKNOWN = "unknown"           # Unclassified

class DocumentRelevance(Enum):
    """Enum for document relevance levels."""
    HIGH = "high"       # Directly relevant
    MEDIUM = "medium"   # Somewhat relevant
    LOW = "low"         # Tangentially relevant
    NONE = "none"       # Not relevant

class RetrievalStrategy(Enum):
    """Enum for retrieval strategies."""
    SEMANTIC = "semantic"         # Semantic similarity search
    KEYWORD = "keyword"           # Keyword-based search
    HYBRID = "hybrid"             # Combined semantic and keyword
    MMR = "mmr"                   # Maximum Marginal Relevance for diversity
    FAQ_MATCH = "faq_match"       # Direct matching for FAQ content

#############################################################################
# APU KB SPECIFIC DOCUMENT PROCESSING
#############################################################################

class APUKnowledgeBaseParser:
    """Parser for APU Knowledge Base format."""
    
    PAGE_PATTERN = r"--- PAGE: (.*?) ---\s*(.*?)(?=--- PAGE:|$)"
    RELATED_PAGES_PATTERN = r"Related Pages –\s*(.*?)$"
    
    @classmethod
    def parse_knowledge_base(cls, content: str) -> List[Dict[str, Any]]:
        """
        Parse the APU Knowledge Base content into structured pages.
        
        Args:
            content: Full content of the knowledge base file
            
        Returns:
            List of dictionaries representing each page with metadata
        """
        # Find all pages using regex pattern
        pages = []
        for match in re.finditer(cls.PAGE_PATTERN, content, re.DOTALL):
            title = match.group(1).strip()
            content = match.group(2).strip()
            
            # Skip empty pages
            if not content:
                continue
            
            # Extract related pages if present
            related_pages = []
            related_match = re.search(cls.RELATED_PAGES_PATTERN, content, re.MULTILINE | re.DOTALL)
            if related_match:
                related_text = related_match.group(1).strip()
                # Process "label in ( ... )" format or regular links
                if "label in" in related_text:
                    # Just store as is, can be processed later if needed
                    related_pages = [related_text]
                else:
                    # Split by newlines and strip each line
                    related_pages = [line.strip() for line in related_text.split('\n') if line.strip()]
                    
                # Clean content by removing the related pages section
                content = content.replace(related_match.group(0), "").strip()
            
            # Create a structured page object
            page = {
                "title": title,
                "content": content,
                "related_pages": related_pages,
                "is_faq": cls._is_faq_page(title, content)
            }
            
            pages.append(page)
        
        logger.info(f"Parsed {len(pages)} pages from APU Knowledge Base")
        return pages
    
    @staticmethod
    def _is_faq_page(title: str, content: str) -> bool:
        """
        Determine if a page is an FAQ type page.
        
        Args:
            title: Page title
            content: Page content
            
        Returns:
            Boolean indicating if the page is an FAQ
        """
        # Check if the title is a question
        has_question_mark = "?" in title
        
        # Check for question words in title
        question_words = ["how", "what", "where", "when", "why", "who", "can", "do", "is", "are", "will"]
        starts_with_question_word = any(title.lower().startswith(word) for word in question_words)
        
        # Either has a question mark or starts with a question word
        return has_question_mark or starts_with_question_word

class APUKnowledgeBaseLoader(BaseLoader):
    """Specialized loader for APU Knowledge Base files."""
    
    def __init__(self, file_path: str):
        """Initialize with file path."""
        self.file_path = file_path
    
    def load(self) -> List[Document]:
        """
        Load the APU Knowledge Base file into Document objects.
        
        Returns:
            List of Document objects
        """
        try:
            # Read the file content
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the knowledge base
            pages = APUKnowledgeBaseParser.parse_knowledge_base(content)
            
            # Convert pages to LangChain Document objects
            documents = []
            for page in pages:
                # Create metadata dictionary
                metadata = {
                    "source": self.file_path,
                    "filename": os.path.basename(self.file_path),
                    "page_title": page["title"],
                    "is_faq": page["is_faq"],
                    "related_pages": page["related_pages"],
                    "content_type": "apu_kb_page"
                }
                
                # Add tags for improved searchability
                metadata["tags"] = self._extract_tags(page["title"], page["content"])
                
                # Create Document object
                doc = Document(
                    page_content=page["content"],
                    metadata=metadata
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error loading APU Knowledge Base file {self.file_path}: {str(e)}")
            return []
    
    @staticmethod
    def _extract_tags(title: str, content: str) -> List[str]:
        """
        Extract relevant tags from the page title and content.
        
        Args:
            title: Page title
            content: Page content
            
        Returns:
            List of tag strings
        """
        tags = []
        
        # Add the title as a tag (lowercase)
        tags.append(title.lower())
        
        # Extract possible key terms from title
        title_words = title.lower().split()
        for word in title_words:
            if len(word) > 3 and word not in ["what", "when", "where", "how", "does", "will"]:
                tags.append(word)
        
        # Extract key acronyms from content
        acronyms = re.findall(r'\b[A-Z]{2,}\b', content)
        for acronym in acronyms:
            tags.append(acronym)
        
        # Extract course codes (e.g., APU1F2103CS)
        course_codes = re.findall(r'\b[A-Z]{2,}[0-9]{1,}[A-Z0-9]{2,}\b', content)
        for code in course_codes:
            tags.append(code)
        
        return list(set(tags))  # Remove duplicates

class APUKnowledgeBaseTextSplitter:
    """Custom text splitter for APU Knowledge Base content."""
    
    def __init__(self, chunk_size=500, chunk_overlap=150, **kwargs):
        # Accept and ignore any additional kwargs
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split APU KB documents into chunks."""
        result_docs = []
        total_docs = len(documents)
        
        logger.info(f"APUKnowledgeBaseTextSplitter processing {total_docs} documents")
        
        for i, doc in enumerate(documents):
            # Log progress periodically 
            if i % 20 == 0:
                logger.info(f"Processing APU KB document {i+1}/{total_docs}")
                
            # Most APU KB documents are small FAQs - if they're under chunk size, keep them intact
            if len(doc.page_content) <= self.chunk_size:
                result_docs.append(doc)
                continue
                
            # For longer documents, add the document as is without chunking to avoid processing issues
            # This preserves the FAQ structure better for the APU KB
            result_docs.append(doc)
        
        logger.info(f"APUKnowledgeBaseTextSplitter finished processing {len(result_docs)} documents")
        return result_docs
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        # If the text is short enough, don't split
        if len(text) <= self.chunk_size:
            return [text]
        
        # Try to split on newlines first to preserve question/answer structure
        chunks = []
        sections = text.split("\n\n")
        
        current_chunk = ""
        for section in sections:
            # If adding this section would exceed chunk size and current chunk is not empty,
            # save current chunk and start a new one
            if len(current_chunk) + len(section) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = section
            # Otherwise, add section to current chunk
            else:
                if current_chunk:
                    current_chunk += "\n\n" + section
                else:
                    current_chunk = section
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # If any chunks are still too large, split them further
        result = []
        for chunk in chunks:
            if len(chunk) > self.chunk_size:
                # Use a simple character-based split as fallback
                result.extend(self._basic_split(chunk))
            else:
                result.append(chunk)
        
        return result
    
    def _basic_split(self, text: str) -> List[str]:
        """Basic character-based split for oversized chunks."""
        result = []
        # Split with overlap
        start = 0
        while start < len(text):
            # Find a good breakpoint (end of sentence if possible)
            end = min(start + self.chunk_size, len(text))
            if end < len(text):
                # Try to find a sentence boundary
                last_period = text.rfind('. ', start, end)
                if last_period > start:
                    end = last_period + 1  # Include the period
            
            # Extract the chunk
            chunk = text[start:end].strip()
            if chunk:
                result.append(chunk)
            
            # Move start position with overlap
            start = end - self.chunk_overlap if end < len(text) else end
        
        return result

#############################################################################
# 1. INPUT PROCESSING
#############################################################################

class InputProcessor:
    """Processes user input to enhance retrieval quality."""
    
    def __init__(self):
        """Initialize the input processor with NLP components."""
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Enhanced synonym dictionary with APU-specific terms
        self.synonyms = {
            "document": ["file", "text", "paper", "doc", "content"],
            "find": ["locate", "search", "discover", "retrieve", "get"],
            "explain": ["describe", "clarify", "elaborate", "detail", "elucidate"],
            "show": ["display", "present", "exhibit", "demonstrate"],
            "information": ["info", "data", "details", "facts", "knowledge"],
            "create": ["make", "generate", "produce", "build", "develop"],
            "modify": ["change", "alter", "adjust", "edit", "update"],
            "remove": ["delete", "eliminate", "erase", "take out"],
            "important": ["significant", "essential", "critical", "key", "vital"],
            "problem": ["issue", "challenge", "difficulty", "trouble", "obstacle"],
            "solution": ["answer", "resolution", "fix", "remedy", "workaround"],
            # APU specific synonyms
            "exam": ["examination", "test", "assessment", "final", "docket"],
            "course": ["module", "class", "subject", "unit"],
            "assignment": ["coursework", "homework", "project", "paper", "submission"],
            "fee": ["payment", "charge", "cost", "expense", "tuition"],
            "lecturer": ["professor", "teacher", "instructor", "tutor", "faculty"],
            "result": ["grade", "mark", "score", "outcome", "performance"],
            "deferment": ["postponement", "delay", "extension", "adjournment"],
            "referral": ["resit", "retake", "redo", "resubmission"],
            "attendance": ["presence", "participation", "turn up", "show up"],
            "certificate": ["diploma", "degree", "qualification", "credential"],
            "transcript": ["record", "academic record", "results"],
            "docket": ["exam slip", "hall ticket", "exam ticket"],
            "EC": ["extenuating circumstances", "special circumstances", "exception", "exemption"],
            "registration": ["enrollment", "signup", "joining", "application"],
            "scholarship": ["funding", "financial aid", "grant", "bursary"],
            "internship": ["industrial training", "placement", "practical training", "industry experience"],
            "visa": ["student pass", "immigration document", "permit"],
            "APU": ["Asia Pacific University", "university", "school", "campus"],
            "timetable": ["schedule", "class schedule", "calendar"],
            "intake": ["batch", "cohort", "entry", "enrollment period"],
        }
        
        # APU-specific abbreviations
        self.apu_abbreviations = {
            "EC": ["extenuating circumstances"],
            "APU": ["Asia Pacific University"],
            "APIIT": ["Asia Pacific Institute of Information Technology"],
            "VC": ["Vice Chancellor"],
            "MPU": ["Mata Pelajaran Umum"],
            "PL": ["Programme Leader"],
            "SIS": ["Student Information System"],
            "EB": ["Examination Board"],
            "UAC": ["University Appeals Committee"],
            "SAC": ["School Appeals Committee"],
            "FYP": ["Final Year Project"],
            "LRT": ["Light Rail Transit"],
            "MCO": ["Movement Control Order"],
            "APKey": ["APU Key", "student ID"],
            "HOS": ["Head of School"],
            "CGPA": ["Cumulative Grade Point Average"],
            "GPA": ["Grade Point Average"],
        }
    
    def normalize_query(self, query: str) -> str:
        """
        Normalize query by converting to lowercase, removing punctuation,
        and standardizing whitespace.
        """
        # Convert to lowercase
        normalized = query.lower()
        
        # Remove punctuation except apostrophes in contractions
        normalized = re.sub(r'[^\w\s\']', ' ', normalized)
        
        # Replace multiple spaces with a single space
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized.strip()
    
    def expand_abbreviations(self, query: str) -> str:
        """
        Expand abbreviations in the query with their full forms.
        
        Args:
            query: Original query string
            
        Returns:
            Query with expanded abbreviations
        """
        words = query.split()
        expanded_query = query
        
        # Check for abbreviations and expand them
        for abbr, expansions in self.apu_abbreviations.items():
            if abbr in words:
                # Add the expansion to the query but don't replace original
                for expansion in expansions:
                    if expansion not in expanded_query:
                        expanded_query += f" {expansion}"
        
        return expanded_query
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze the query to extract features for better retrieval.
        
        Returns:
            A dictionary with query analysis including:
            - tokens: List of tokens
            - normalized_query: Normalized query text
            - lemmatized_tokens: Lemmatized tokens
            - keywords: Key terms (non-stopwords)
            - query_type: Type of query (enum)
            - expanded_queries: List of expanded query variants
            - extracted_terms: Education/APU-specific terms extracted
        """
        # Normalize the query
        normalized_query = self.normalize_query(query)
        
        # Expand abbreviations
        expanded_query = self.expand_abbreviations(normalized_query)
        
        # Tokenize
        tokens = word_tokenize(expanded_query)
        
        # Remove stopwords and get keywords
        keywords = [token for token in tokens if token.lower() not in self.stop_words and len(token) > 1]
        
        # Lemmatize tokens
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Generate n-grams for phrases (bigrams and trigrams)
        bigrams = list(ngrams(tokens, 2))
        trigrams = list(ngrams(tokens, 3))
        
        # Extract bigram and trigram phrases
        bigram_phrases = [' '.join(bg) for bg in bigrams]
        trigram_phrases = [' '.join(tg) for tg in trigrams]
        
        # Extract education/APU-specific terms
        edu_terms = self._extract_education_terms(normalized_query, tokens)
        
        # Identify query type (enhanced with APU-specific types)
        query_type = self.classify_query_type(normalized_query, tokens, edu_terms)
        
        # Generate expanded queries if enabled
        expanded_queries = []
        if Config.USE_QUERY_EXPANSION:
            expanded_queries = self.expand_query(expanded_query, keywords)
        
        return {
            "original_query": query,
            "normalized_query": normalized_query,
            "expanded_query": expanded_query,
            "tokens": tokens,
            "keywords": keywords,
            "lemmatized_tokens": lemmatized_tokens,
            "bigram_phrases": bigram_phrases,
            "trigram_phrases": trigram_phrases,
            "query_type": query_type,
            "expanded_queries": expanded_queries,
            "edu_terms": edu_terms
        }
    
    def _extract_education_terms(self, query: str, tokens: List[str]) -> List[str]:
        """
        Extract education and APU-specific terms from the query.
        
        Args:
            query: Normalized query string
            tokens: List of tokens
            
        Returns:
            List of education/APU-specific terms
        """
        edu_terms = []
        
        # Common education terms
        edu_keywords = [
            "exam", "docket", "test", "assessment", "module", "course", "class",
            "lecturer", "referral", "resit", "retake", "fee", "payment", "transcript",
            "certificate", "registration", "visa", "EC", "extenuating", "circumstances",
            "attendance", "appeal", "degree", "diploma", "graduation", "intake", "semester",
            "programme", "program", "scholarship", "internship", "placement", "deferment",
            "withdrawal", "compensation", "supervisor", "submission", "deadline",
            "assignment", "project", "dissertation", "library", "campus", "accommodation",
            "timetable", "schedule", "result", "grade", "mark", "CGPA", "GPA"
        ]
        
        # Check for education terms in tokens
        for token in tokens:
            if token.lower() in edu_keywords:
                edu_terms.append(token.lower())
        
        # Check for course codes (e.g., APU1F2103CS)
        course_codes = re.findall(r'\b[A-Z]{2,}[0-9]{1,}[A-Z0-9]{2,}\b', query)
        edu_terms.extend(course_codes)
        
        # Check for common abbreviations
        for abbr in self.apu_abbreviations.keys():
            if abbr in query:
                edu_terms.append(abbr.lower())
        
        return list(set(edu_terms))  # Remove duplicates
    
    def classify_query_type(self, query: str, tokens: List[str], edu_terms: List[str]) -> QueryType:
        """
        Classify the query into different types, including APU-specific types.
        """
        # Check for command queries first
        command_patterns = [
            r'\b(?:exit|quit|bye|goodbye)\b',
            r'\bclear\b',
            r'\breindex\b',
            r'\bstats\b',
            r'\bhelp\b'
        ]
        
        for pattern in command_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return QueryType.COMMAND
        
        # Check for identity queries (NEW)
        identity_patterns = [
            r'\bwho\s+(?:are|r)\s+(?:you|u)\b',
            r'\bwhat\s+(?:are|r)\s+(?:you|u)\b',
            r'\byour\s+name\b',
            r'\bur\s+name\b',
            r'\bwho\s+(?:created|made|built|developed)\s+(?:you|u|this)\b',
            r'\bhow\s+(?:were|r)\s+(?:you|u)\s+(?:made|created|built|developed)\b',
            r'\bintroduce\s+yourself\b',
            r'\babout\s+yourself\b',
            r'\bwhat\s+(?:can|do)\s+you\s+do\b',
            r'\byour\s+purpose\b',
            r'\bhow\s+do\s+you\s+work\b'
        ]
        
        for pattern in identity_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return QueryType.IDENTITY
        
        # Check for conversational queries
        conversational_patterns = [
            r'\b(?:hi|hello|hey|greetings|howdy|good\s*(?:morning|afternoon|evening)|what\'s\s*up)\b',
            r'\bhow\s+are\s+you\b',
            r'\b(?:thanks|thank\s*you)\b',
            r'\bappreciate\s*(?:it|that)\b',
        ]
        
        for pattern in conversational_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return QueryType.CONVERSATIONAL
        
        # Check for APU-specific academic queries
        academic_patterns = [
            r'\b(?:exam|docket|test|assessment|module|course|class|lecture|assignment|submission)\b',
            r'\b(?:referral|resit|retake|grade|mark|result|CGPA|GPA)\b',
            r'\b(?:dissertation|project|FYP|lecturer|supervisor|class|timetable|schedule)\b',
            r'\b(?:registration|compensation|curriculum)\b'
        ]
        
        for pattern in academic_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return QueryType.ACADEMIC
        
        # Check for APU-specific administrative queries
        administrative_patterns = [
            r'\b(?:EC|extenuating circumstances|appeal|application|form|document|ID card)\b',
            r'\b(?:accommodation|housing|library|deadline|attendance|participation)\b',
            r'\b(?:visa|student pass|internship|placement|deferment|withdrawal|letter|certificate)\b',
            r'\b(?:transcript|admission|orientation|enrollment|transfer|VC\'s list)\b'
        ]
        
        for pattern in administrative_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return QueryType.ADMINISTRATIVE
        
        # Check for APU-specific financial queries
        financial_patterns = [
            r'\b(?:fee|payment|charge|cost|expense|tuition|scholarship|funding|financial aid)\b',
            r'\b(?:bursary|loan|refund|installment|billing|invoice|receipt|credit|debit)\b',
            r'\b(?:sponsorship|discount|waiver|rebate|compensation|penalty)\b'
        ]
        
        for pattern in financial_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return QueryType.FINANCIAL
        
        # Fall back to standard query classification if no APU-specific match
        
        # Check for factual queries
        factual_patterns = [
            r'\bwhat\s+is\b',
            r'\bwhat\s+are\b',
            r'\bwho\s+is\b',
            r'\bwhen\s+\w+\b',
            r'\bwhere\s+\w+\b',
            r'\blist\b',
            r'\bdefine\b',
            r'\btell\s+me\s+about\b'
        ]
        
        for pattern in factual_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return QueryType.FACTUAL
        
        # Check for procedural queries
        procedural_patterns = [
            r'\bhow\s+to\b',
            r'\bhow\s+do\s+I\b',
            r'\bhow\s+can\s+I\b',
            r'\bsteps\s+to\b',
            r'\bguide\b',
            r'\btutorial\b',
            r'\bprocedure\b',
            r'\bprocess\b'
        ]
        
        for pattern in procedural_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return QueryType.PROCEDURAL
        
        # Check for conceptual queries
        conceptual_patterns = [
            r'\bexplain\b',
            r'\bconcept\b',
            r'\btheory\b',
            r'\bwhy\s+\w+\b',
            r'\breasons?\s+for\b',
            r'\bmean\b',
            r'\bunderstand\b'
        ]
        
        for pattern in conceptual_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return QueryType.CONCEPTUAL
        
        # Check for comparative queries
        comparative_patterns = [
            r'\bcompare\b',
            r'\bcontrast\b',
            r'\bdifference\b',
            r'\bsimilar\b',
            r'\bversus\b',
            r'\bvs\b',
            r'\bbetter\b',
            r'\badvantages?\b',
            r'\bdisadvantages?\b'
        ]
        
        for pattern in comparative_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return QueryType.COMPARATIVE
        
        # Check for exploratory queries
        exploratory_patterns = [
            r'\btell\s+me\s+more\b',
            r'\blearn\s+about\b',
            r'\bdiscover\b',
            r'\bexplore\b',
            r'\boverview\b',
            r'\bintroduction\b'
        ]
        
        for pattern in exploratory_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return QueryType.EXPLORATORY
        
        # Default to unknown if no patterns match
        return QueryType.UNKNOWN
    
    def expand_query(self, query: str, keywords: List[str]) -> List[str]:
        """
        Expand the query with synonyms and variations, including APU-specific terms.
        
        Returns:
            List of expanded query strings
        """
        expanded_queries = [query]  # Start with the original query
        
        # Skip expansion for very short queries or conversational queries
        if len(keywords) < 2:
            return expanded_queries
        
        # Expand using synonyms
        for i, keyword in enumerate(keywords):
            if keyword in self.synonyms:
                for synonym in self.synonyms[keyword][:Config.EXPANSION_FACTOR]:
                    # Replace the keyword with its synonym
                    new_keywords = keywords.copy()
                    new_keywords[i] = synonym
                    expanded_query = ' '.join(new_keywords)
                    if expanded_query not in expanded_queries:
                        expanded_queries.append(expanded_query)
        
        # Expand abbreviations
        for i, keyword in enumerate(keywords):
            if keyword.upper() in self.apu_abbreviations:
                for expansion in self.apu_abbreviations[keyword.upper()]:
                    # Replace the abbreviation with its expanded form
                    new_keywords = keywords.copy()
                    new_keywords[i] = expansion
                    expanded_query = ' '.join(new_keywords)
                    if expanded_query not in expanded_queries:
                        expanded_queries.append(expanded_query)
        
        # If we haven't generated enough variations, try some combinations
        if len(expanded_queries) < Config.EXPANSION_FACTOR and len(keywords) >= 3:
            # Generate variations by removing one non-essential keyword at a time
            for i in range(len(keywords)):
                variation = ' '.join(keywords[:i] + keywords[i+1:])
                if variation not in expanded_queries:
                    expanded_queries.append(variation)
                
                # Stop if we have enough variations
                if len(expanded_queries) >= Config.EXPANSION_FACTOR:
                    break
        
        return expanded_queries[:Config.EXPANSION_FACTOR]  # Limit to avoid too many variations

#############################################################################
# 2. QUERY ROUTING & CLASSIFICATION
#############################################################################

class QueryRouter:
    """Routes queries to appropriate handlers based on query type."""
    
    def __init__(self, conversation_handler, retrieval_handler, command_handler):
        """Initialize with handlers for different query types."""
        self.conversation_handler = conversation_handler
        self.retrieval_handler = retrieval_handler
        self.command_handler = command_handler
    
    def route_query(self, query_analysis: Dict[str, Any], stream=False) -> Tuple[Any, bool]:
        """
        Route the query to the appropriate handler.
        
        Args:
            query_analysis: Analysis output from InputProcessor
            stream: Whether to stream the response
            
        Returns:
            Tuple of (handler_result, should_continue)
        """
        query_type = query_analysis["query_type"]
        original_query = query_analysis["original_query"]
        
        # Route based on query type
        if query_type == QueryType.COMMAND:
            # Handle system commands (not streaming these)
            return self.command_handler.handle_command(original_query)
        
        elif query_type == QueryType.CONVERSATIONAL:
            # Handle conversational queries (now with streaming support)
            response = self.conversation_handler.handle_conversation(original_query, stream=stream)
            return response, True
        
        else:
            # Handle all other query types with retrieval system
            response = self.retrieval_handler.process_query(query_analysis, stream=stream)
            return response, True

class ConversationHandler:
    """Handles conversational queries that don't require document retrieval."""
    
    def __init__(self, memory):
        """Initialize with a memory for conversation history."""
        self.memory = memory
        
        # Greeting patterns and responses
        self.greeting_patterns = [
            r'\b(?:hi|hello|hey|greetings|howdy|good\s*(?:morning|afternoon|evening)|what\'s\s*up)\b',
            r'\bhow\s+are\s+you\b',
        ]

        self.greeting_responses = [
            "Hello! I'm your APU knowledge base assistant. How can I help you with your questions about APU today?",
            "Hi there! I'm ready to help answer questions about APU. What would you like to know?",
            "Greetings! I'm here to assist with information about APU. What are you looking for?",
            "Hello! I'm your APU RAG assistant. I can help you find information about academics, administrative processes, and more.",
            "Hi! I'm ready to help you navigate APU-related questions. What would you like to learn about?"
        ]

        # Acknowledgement patterns and responses
        self.acknowledgement_patterns = [
            r'\b(?:thanks|thank\s*you)\b',
            r'\bappreciate\s*(?:it|that)\b',
            r'\b(?:awesome|great|cool|nice)\b',
            r'\bthat\s*(?:helps|helped)\b',
            r'\bgot\s*it\b',
        ]

        self.acknowledgement_responses = [
            "You're welcome! Is there anything else you'd like to know about APU?",
            "Happy to help! Let me know if you have any other questions about APU.",
            "My pleasure! Feel free to ask if you need anything else.",
            "Glad I could assist. Any other questions about APU?",
            "You're welcome! I'm here if you need more information about APU procedures, policies, or services."
        ]
    
    def handle_conversation(self, query: str, stream=False) -> str:
        """
        Handles conversational queries.
        
        Args:
            query: The user's query
            stream: Whether to stream the response
            
        Returns:
            Either a string response or an iterator for streaming
        """
        query_lower = query.lower().strip()
        response = None
        
        # Check for greetings
        for pattern in self.greeting_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                response = random.choice(self.greeting_responses)
                break
        
        # Check for acknowledgements
        if not response:
            for pattern in self.acknowledgement_patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    response = random.choice(self.acknowledgement_responses)
                    break
        
        # If no specific pattern matched, give a generic response
        if not response:
            response = "I'm here to help you with information about APU. What would you like to know?"
        
        # Update conversation memory - do this for non-streaming case
        if not stream:
            self.memory.chat_memory.add_user_message(query)
            self.memory.chat_memory.add_ai_message(response)
            return response
        
        # For streaming, use the Ollama streaming approach so UI shows smooth typing
        if stream:
            # Store response for updating memory after streaming completes
            self._last_response = response
            self._last_query = query
            
            # Return iterator that simulates streaming for simple responses
            return self._stream_response(response)

    def _stream_response(self, response: str):
        """
        Simulates streaming for conversational responses.
        
        Args:
            response: The full response to stream
            
        Returns:
            An iterator yielding tokens
        """
        # Update memory before starting to stream
        self.memory.chat_memory.add_user_message(self._last_query)
        self.memory.chat_memory.add_ai_message(response)
        
        # Simple streaming simulation - character by character
        # You could adjust chunk sizes or speed for more natural flow
        for char in response:
            yield char
            time.sleep(0.01)  # Slight delay between characters

class CommandHandler:
    """Handles system commands."""
    
    def __init__(self, rag_system):
        """Initialize with a reference to the RAG system."""
        self.rag_system = rag_system
    
    def handle_command(self, command: str) -> Tuple[str, bool]:
        """
        Handles system commands.
        
        Args:
            command: The command string
            
        Returns:
            Tuple of (response, should_continue)
        """
        command_lower = command.lower().strip()
        
        # Help command
        if command_lower in ["help", "menu", "commands"]:
            help_text = """
            Available Commands:
            - help: Display this help menu
            - exit, quit: Stop the application
            - clear: Reset the conversation memory
            - reindex: Reindex all documents
            - stats: See document statistics
            """
            return help_text, True
        
        # Exit commands
        elif command_lower in ["exit", "quit", "bye", "goodbye"]:
            return "Goodbye! Have a great day!", False
            
        # Clear memory command
        elif command_lower == "clear":
            # Reset memory but keep system message
            system_message = self.rag_system.memory.chat_memory.messages[0] if self.rag_system.memory.chat_memory.messages else None
            self.rag_system.memory.clear()
            if system_message:
                self.rag_system.memory.chat_memory.messages.append(system_message)
            return "Conversation memory has been reset.", True
            
        # Stats command
        elif command_lower == "stats":
            # Capture printed output
            import io
            from contextlib import redirect_stdout
            
            f = io.StringIO()
            with redirect_stdout(f):
                VectorStoreManager.print_document_statistics(self.rag_system.vector_store)
            
            output = f.getvalue()
            if not output.strip():
                output = "No document statistics available."
                
            return output, True
            
        # Reindex command
        elif command_lower == "reindex":
            result = self.rag_system.reindex_documents()
            if result:
                return "Documents have been successfully reindexed.", True
            else:
                return "Failed to reindex documents. Check the log for details.", True
        
        # Unknown command
        else:
            return f"Unknown command: {command}. Type 'help' to see available commands.", True
        
#############################################################################
# 3. CHROMA DB SETTINGS
#############################################################################
class ChromaDBManager:
    """Singleton manager for ChromaDB client lifecycle."""
    _instance = None
    _client = None
    
    @classmethod
    def get_client(cls, persist_directory=None, reset=False):
        """Get or create a ChromaDB client with proper configuration.
        
        Args:
            persist_directory: Path to the persistence directory
            reset: Force creation of a new client instance
            
        Returns:
            Configured ChromaDB client
        """
        if persist_directory is None:
            persist_directory = Config.PERSIST_PATH
            
        # Create a new client if requested or if none exists
        if cls._client is None or reset:
            try:
                # Close existing client if applicable
                if cls._client is not None and hasattr(cls._client, 'close'):
                    try:
                        cls._client.close()
                    except Exception as e:
                        logger.warning(f"Error closing existing ChromaDB client: {e}")
                
                # Create new client with proper settings
                from chromadb.config import Settings
                cls._client = chromadb.PersistentClient(
                    path=persist_directory,
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True,
                        is_persistent=True
                    )
                )
                logger.info(f"Initialized ChromaDB client for {persist_directory}")
            except Exception as e:
                logger.error(f"Error initializing ChromaDB client: {e}")
                raise
        
        return cls._client
    
    @classmethod
    def get_or_create_collection(cls, client, name, metadata=None, embedding_function=None):
        """Get or create a collection with metadata.
        
        Args:
            client: ChromaDB client
            name: Collection name
            metadata: Collection metadata
            embedding_function: Function for embeddings
            
        Returns:
            Tuple of (collection, langchain_chroma_instance)
        """
        if metadata is None:
            metadata = {}
            
        # Add version tracking to metadata
        full_metadata = {
            "hnsw:space": "cosine",
            "embedding_model": Config.EMBEDDING_MODEL_NAME,
            "embedding_version": "1.0",
            "app_version": "1.0",
            "created_at": datetime.now().isoformat()
        }
        
        # Merge with provided metadata
        full_metadata.update(metadata)
        
        try:
            # Check if collection exists
            all_collections = client.list_collections()
            collection_exists = any(c.name == name for c in all_collections)
            
            if collection_exists:
                logger.info(f"Using existing collection: {name}")
                collection = client.get_collection(name)
                
                # Update metadata for existing collection
                if hasattr(collection, 'modify') and full_metadata:
                    try:
                        collection.modify(metadata=full_metadata)
                    except Exception as e:
                        logger.warning(f"Could not update collection metadata: {e}")
            else:
                logger.info(f"Creating new collection: {name}")
                collection = client.create_collection(name=name, metadata=full_metadata)
            
            # Create LangChain wrapper
            from langchain_chroma import Chroma
            langchain_store = Chroma(
                client=client,
                collection_name=name,
                embedding_function=embedding_function
            )
            
            return collection, langchain_store
            
        except Exception as e:
            logger.error(f"Error getting/creating collection {name}: {e}")
            raise
    
    @classmethod
    def close(cls):
        """Close the ChromaDB client connection."""
        if cls._client is not None:
            try:
                # Different ChromaDB versions may have different cleanup methods
                if hasattr(cls._client, 'close'):
                    cls._client.close()
                elif hasattr(cls._client, 'persist'):
                    # Some versions use persist to ensure data is saved
                    cls._client.persist()
                # Simply clear the reference if no cleanup method is available
                cls._client = None
                logger.info("Released ChromaDB client resources")
            except Exception as e:
                logger.error(f"Error during ChromaDB client cleanup: {e}")

#############################################################################
# 3. RETRIEVAL SYSTEM
#############################################################################

class RetrievalHandler:
    """Handles document retrieval using multiple strategies."""
    
    def __init__(self, vector_store, embeddings, memory, context_processor):
        """Initialize the retrieval handler."""
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.memory = memory
        self.context_processor = context_processor
        
        # Create retrievers for different strategies
        self.retrievers = self._create_retrievers()
        
        # Initialize FAQ matcher for direct FAQ matching
        self.faq_matcher = FAQMatcher(vector_store)
    
    def _create_retrievers(self) -> Dict[RetrievalStrategy, Any]:
        """Create retrievers for different strategies."""
        retrievers = {}
        
        # Semantic search retriever
        retrievers[RetrievalStrategy.SEMANTIC] = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": Config.RETRIEVER_K}
        )
        
        # MMR retriever for diversity
        retrievers[RetrievalStrategy.MMR] = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": Config.RETRIEVER_K,
                "fetch_k": Config.RETRIEVER_K * 3,
                "lambda_mult": 0.5
            }
        )
        
        # For keyword and hybrid search, we'll implement custom methods
        
        return retrievers
    
    def process_query(self, query_analysis: Dict[str, Any], stream=False) -> Union[str, Iterator[str]]:
        """
        Process a query using the appropriate retrieval strategy.
        
        Args:
            query_analysis: The analysis from InputProcessor
            stream: Whether to stream the response
            
        Returns:
            Response string or iterator of response tokens if stream=True
        """
        query_type = query_analysis["query_type"]
        original_query = query_analysis["original_query"]
        expanded_queries = query_analysis.get("expanded_queries", [original_query])
        
        logger.info(f"Processing {query_type.value} query: {original_query}")
        
        # Handle identity questions with predefined responses
        if query_type == QueryType.IDENTITY:
            # Map of identity questions to predefined answers
            identity_responses = {
                "who are you": "I'm the APU Knowledge Base Assistant, designed to help you find information about Asia Pacific University's academic procedures, administrative processes, and university services.",
                "what are you": "I'm an AI-powered retrieval system specifically built to help APU students and staff access information from the APU knowledge base quickly and accurately.",
                "your name": "You can call me the APU Knowledge Base Assistant. I'm here to help you with any questions about APU.",
                "how do you work": "I use a technique called Retrieval Augmented Generation to find relevant information in the APU knowledge base and create helpful responses to your questions.",
                "who made you": "I was developed by the APU technology team to provide quick and accurate answers about university procedures and policies.",
                "what can you do": "I can answer questions about APU's academic procedures, administrative processes, fees, exams, and other university services. Just ask me anything related to APU!",
                "your purpose": "My purpose is to help APU students and staff quickly find accurate information about university procedures, policies, and services.",
            }
            
            # Default response if no specific match
            response = "I'm the APU Knowledge Base Assistant. I'm here to help you find information about APU's academic procedures, administrative processes, and university services."
            
            # Look for specific patterns in the query
            query_lower = original_query.lower()
            for key, value in identity_responses.items():
                if key in query_lower:
                    response = value
                    break
                    
            # Update memory
            self.memory.chat_memory.add_user_message(original_query)
            self.memory.chat_memory.add_ai_message(response)
            
            # If streaming, return as iterator
            if stream:
                return iter([response])
            else:
                return response
        
        # First try direct FAQ matching for a quick answer
        faq_match_result = self.faq_matcher.match_faq(query_analysis)
        
        if faq_match_result and faq_match_result.get("match_score", 0) > 0.7:
            # High confidence direct FAQ match
            logger.info(f"Found direct FAQ match with score {faq_match_result['match_score']}")
            
            # Create a simplified context with just the matched FAQ
            context = self._format_faq_match(faq_match_result)
            
            # Generate response using the matched FAQ
            input_dict = {
                "question": original_query,
                "context": context,
                "chat_history": self.memory.chat_memory.messages,
                "is_faq_match": True,
                "match_score": faq_match_result["match_score"]
            }
            
            # Generate response through LLM with streaming if requested
            response = RAGSystem.stream_ollama_response(
                self._create_prompt(input_dict), 
                Config.LLM_MODEL_NAME,
                stream_output=stream
            )
            
            # If we're not streaming, update memory
            if not stream:
                self.memory.chat_memory.add_user_message(original_query)
                self.memory.chat_memory.add_ai_message(response)
            
            return response
        
        # If no good FAQ match, proceed with standard retrieval
        # Select retrieval strategy based on query type
        retrieval_strategy = self._select_retrieval_strategy(query_type)
        logger.info(f"Selected retrieval strategy: {retrieval_strategy.value}")
        
        # Retrieve relevant documents
        context_docs = self._retrieve_documents(expanded_queries, retrieval_strategy)
        
        # If no documents found, try a fallback strategy
        if not context_docs and retrieval_strategy != RetrievalStrategy.HYBRID:
            logger.info("No documents found, trying hybrid fallback strategy")
            context_docs = self._retrieve_documents(expanded_queries, RetrievalStrategy.HYBRID)
        
        # Process the retrieved documents
        context = self.context_processor.process_context(context_docs, query_analysis)
        
        # Generate response using RAG system
        input_dict = {
            "question": original_query,
            "context": context,
            "chat_history": self.memory.chat_memory.messages,
            "is_faq_match": False
        }
        
        # Generate response through LLM with streaming if requested
        response = RAGSystem.stream_ollama_response(
            self._create_prompt(input_dict), 
            Config.LLM_MODEL_NAME,
            stream_output=stream
        )
        
        # If we're not streaming, update memory
        if not stream:
            self.memory.chat_memory.add_user_message(original_query)
            self.memory.chat_memory.add_ai_message(response)
        
        return response
    
    def _format_faq_match(self, match_result: Dict[str, Any]) -> str:
        """Format a direct FAQ match into a context for the LLM."""
        doc = match_result["document"]
        score = match_result["match_score"]
        
        # Get metadata
        title = doc.metadata.get("page_title", "Unknown Title")
        source = doc.metadata.get("source", "Unknown Source")
        filename = doc.metadata.get("filename", os.path.basename(source) if source != "Unknown Source" else "Unknown File")
        
        # Format the context with extremely clear instructions
        context = f"--- DIRECT FAQ MATCH (Confidence: {score:.2f}) ---\n\n"
        context += f"Question: {title}\n\n"
        context += f"SOURCE TEXT (REPRODUCE THIS EXACTLY WITHOUT ADDING ANY EMAIL ADDRESSES THAT ARE NOT SHOWN HERE):\n\"\"\"\n{doc.page_content}\n\"\"\"\n\n"
        
        # Add the ONLY valid email addresses that exist
        emails_in_text = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', doc.page_content)
        if emails_in_text:
            context += "VALID EMAIL ADDRESSES IN TEXT (ONLY USE THESE EXACT ADDRESSES):\n"
            for email in emails_in_text:
                context += f"- {email}\n"
        else:
            context += "NOTE: THERE ARE NO EMAIL ADDRESSES IN THE SOURCE TEXT. DO NOT ADD ANY.\n"
        
        # Add related pages if available
        related_pages = doc.metadata.get("related_pages", [])
        if related_pages:
            context += "\nRelated information can be found in:\n"
            for page in related_pages[:3]:
                context += f"- {page}\n"
        
        return context
    
    def _select_retrieval_strategy(self, query_type: QueryType) -> RetrievalStrategy:
        """
        Select the appropriate retrieval strategy based on query type.
        """
        if Config.RETRIEVER_SEARCH_TYPE != "auto":
            # Use the configured strategy if not set to auto
            return RetrievalStrategy(Config.RETRIEVER_SEARCH_TYPE)
        
        # Select strategy based on query type
        if query_type in [QueryType.FACTUAL, QueryType.ACADEMIC]:
            return RetrievalStrategy.HYBRID  # Precise for factual/academic questions
        
        elif query_type in [QueryType.PROCEDURAL, QueryType.ADMINISTRATIVE]:
            return RetrievalStrategy.HYBRID  # Procedural and admin processes need semantic and keyword
        
        elif query_type == QueryType.FINANCIAL:
            return RetrievalStrategy.HYBRID  # Financial questions need precision
        
        elif query_type == QueryType.COMPARATIVE:
            return RetrievalStrategy.MMR  # Diverse results for comparison
        
        elif query_type == QueryType.EXPLORATORY:
            return RetrievalStrategy.MMR  # Diverse results for exploration
        
        else:
            return RetrievalStrategy.HYBRID  # Default to hybrid
    
    def _retrieve_documents(self, queries: List[str], strategy: RetrievalStrategy) -> List[Document]:
        """
        Retrieve documents using the specified strategy.
        
        Args:
            queries: List of query strings (original and expanded)
            strategy: Retrieval strategy to use
            
        Returns:
            List of retrieved documents
        """
        all_docs = []
        seen_ids = set()  # Track document IDs to avoid duplicates
        
        # Process each query (original + expanded)
        for query in queries:
            try:
                if strategy == RetrievalStrategy.SEMANTIC or strategy == RetrievalStrategy.MMR:
                    # Use the appropriate retriever
                    docs = self.retrievers[strategy].invoke(query)
                    
                elif strategy == RetrievalStrategy.KEYWORD:
                    # Use keyword-based retrieval
                    docs = self._keyword_retrieval(query)
                    
                elif strategy == RetrievalStrategy.HYBRID:
                    # Use hybrid retrieval (combination of semantic and keyword)
                    docs = self._hybrid_retrieval(query)
                
                # Add unique documents to the result list
                for doc in docs:
                    # Create a unique ID based on content and source
                    doc_id = f"{doc.metadata.get('source', '')}-{hash(doc.page_content[:100])}"
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        all_docs.append(doc)
            
            except Exception as e:
                logger.error(f"Error retrieving documents for query '{query}': {e}")
        
        # Sort docs by relevance score if available
        all_docs = self._rerank_documents(all_docs, queries[0])
        
        logger.info(f"Retrieved {len(all_docs)} unique documents")
        return all_docs
    
    def _keyword_retrieval(self, query: str) -> List[Document]:
        """
        Perform keyword-based retrieval, enhanced for APU knowledge base.
        
        Args:
            query: Query string
            
        Returns:
            List of documents
        """
        # Get all documents from the vector store
        all_docs = self.vector_store.get()
        
        if not all_docs or not all_docs.get('documents'):
            return []
            
        documents = all_docs.get('documents', [])
        metadatas = all_docs.get('metadatas', [])
        ids = all_docs.get('ids', [])
        
        # Extract keywords from the query
        tokens = word_tokenize(query.lower())
        stop_words = set(stopwords.words('english'))
        keywords = [token for token in tokens if token not in stop_words and len(token) > 2]
        
        # Score documents based on keyword matches
        scored_docs = []
        for i, doc_text in enumerate(documents):
            if not doc_text:
                continue
                
            score = 0
            doc_lower = doc_text.lower()
            metadata = metadatas[i] if i < len(metadatas) else {}
            
            # Check if this is an APU KB page
            is_apu_kb = metadata.get('content_type') == 'apu_kb_page'
            page_title = metadata.get('page_title', '')
            
            # Count exact keyword matches
            for keyword in keywords:
                # Count occurrences of the keyword
                count = doc_lower.count(keyword)
                score += count
                
                # Bonus for exact phrase match
                if query.lower() in doc_lower:
                    score += 5
                    
                # For APU KB pages, check title match
                if is_apu_kb and keyword in page_title.lower():
                    score += 3  # Boost score for title matches
            
            # Bonus for exact question match in FAQ pages
            if is_apu_kb and metadata.get('is_faq', False):
                question_similarity = self._calculate_question_similarity(query, page_title)
                score += question_similarity * 10  # High bonus for question similarity
            
            # Check for match in tags
            if is_apu_kb and 'tags' in metadata:
                for tag in metadata['tags']:
                    if any(kw in tag for kw in keywords):
                        score += 2  # Bonus for tag matches
            
            # Give a boost to direct question matches for FAQ pages
            if is_apu_kb and metadata.get('is_faq', False) and query.strip().endswith('?'):
                # Calculate similarity between query and page title
                title_similarity = self._calculate_question_similarity(query, page_title)
                if title_similarity > 0.7:  # High similarity
                    score += 15
                elif title_similarity > 0.5:  # Moderate similarity
                    score += 8
                
            if score > 0:
                doc_id = ids[i] if i < len(ids) else str(i)
                
                # Create Document object
                doc = Document(
                    page_content=doc_text,
                    metadata={
                        **metadata,
                        'score': score,
                        'id': doc_id
                    }
                )
                scored_docs.append((score, doc))
        
        # Sort by score (descending) and return top k
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        
        # Extract just the documents from the scored list
        result_docs = [doc for _, doc in scored_docs[:Config.RETRIEVER_K]]
        
        return result_docs
    
    def _calculate_question_similarity(self, query: str, title: str) -> float:
        """
        Calculate similarity between a query and a question title.
        
        Args:
            query: The user query
            title: The FAQ title
            
        Returns:
            Similarity score between 0 and 1
        """
        # Normalize both strings
        query = query.lower().strip()
        title = title.lower().strip()
        
        # Remove question marks
        query = query.rstrip('?')
        title = title.rstrip('?')
        
        # Tokenize
        query_tokens = set(word_tokenize(query))
        title_tokens = set(word_tokenize(title))
        
        # Remove stop words
        stop_words = set(stopwords.words('english'))
        query_tokens = {token for token in query_tokens if token not in stop_words}
        title_tokens = {token for token in title_tokens if token not in stop_words}
        
        # Return 0 if either set is empty after removing stop words
        if not query_tokens or not title_tokens:
            return 0
        
        # Calculate Jaccard similarity
        intersection = len(query_tokens.intersection(title_tokens))
        union = len(query_tokens.union(title_tokens))
        
        return intersection / union if union > 0 else 0
    
    def _hybrid_retrieval(self, query: str) -> List[Document]:
        """
        Perform hybrid retrieval (semantic + keyword), optimized for APU KB.
        
        Args:
            query: Query string
            
        Returns:
            List of documents
        """
        # Get semantic search results
        semantic_docs = self.retrievers[RetrievalStrategy.SEMANTIC].invoke(query)
        
        # Get keyword search results
        keyword_docs = self._keyword_retrieval(query)
        
        # Try direct FAQ matching
        faq_match_result = self.faq_matcher.match_faq({"original_query": query})
        faq_docs = []
        if faq_match_result and faq_match_result.get("match_score", 0) > 0.5:
            faq_docs = [faq_match_result["document"]]
        
        # Combine results with weighting
        semantic_weight = 1 - Config.KEYWORD_RATIO - Config.FAQ_MATCH_WEIGHT
        keyword_weight = Config.KEYWORD_RATIO
        faq_weight = Config.FAQ_MATCH_WEIGHT
        
        # Track documents by ID to avoid duplicates
        combined_docs = {}
        
        # Add FAQ docs with highest priority
        for i, doc in enumerate(faq_docs):
            doc_id = f"{doc.metadata.get('source', '')}-{hash(doc.page_content[:100])}"
            
            # Use match score if available
            faq_score = faq_match_result.get("match_score", 0.8) * faq_weight
            
            # Store with score
            combined_docs[doc_id] = {
                "doc": doc,
                "score": faq_score
            }
        
        # Add semantic docs with their weights
        for i, doc in enumerate(semantic_docs):
            doc_id = f"{doc.metadata.get('source', '')}-{hash(doc.page_content[:100])}"
            
            # Inverse rank scoring (higher rank = lower score)
            semantic_score = semantic_weight * (1.0 - (i / max(1, len(semantic_docs))))
            
            # Check if this is an APU KB page and apply appropriate boosts
            if doc.metadata.get('content_type') == 'apu_kb_page':
                # Boost for FAQ pages
                if doc.metadata.get('is_faq', False):
                    semantic_score *= 1.2
                
                # Bonus for title match
                title = doc.metadata.get('page_title', '').lower()
                if any(word in title for word in query.lower().split()):
                    semantic_score *= 1.3
            
            # Store with score
            if doc_id in combined_docs:
                combined_docs[doc_id]["score"] += semantic_score
            else:
                combined_docs[doc_id] = {
                    "doc": doc,
                    "score": semantic_score
                }
        
        # Add keyword docs with their weights
        for i, doc in enumerate(keyword_docs):
            doc_id = f"{doc.metadata.get('source', '')}-{hash(doc.page_content[:100])}"
            
            # Get original keyword score if available, otherwise use inverse rank
            keyword_score = (doc.metadata.get('score', 0) / max(1, max([d.metadata.get('score', 1) for d in keyword_docs])))
            
            # Adjust by weight
            keyword_score = keyword_weight * keyword_score
            
            # Update score if document already exists, otherwise add it
            if doc_id in combined_docs:
                combined_docs[doc_id]["score"] += keyword_score
            else:
                combined_docs[doc_id] = {
                    "doc": doc,
                    "score": keyword_score
                }
        
        # Sort by combined score
        sorted_docs = sorted(combined_docs.values(), key=lambda x: x["score"], reverse=True)
        
        # Return top K documents
        result_docs = [item["doc"] for item in sorted_docs[:Config.RETRIEVER_K]]
        
        return result_docs
    
    def _rerank_documents(self, documents: List[Document], query: str) -> List[Document]:
        """
        Rerank documents based on relevance to the query, optimized for APU KB.
        
        Args:
            documents: List of documents to rerank
            query: Original query string
            
        Returns:
            Reranked document list
        """
        if not documents:
            return []
        
        # Embed the query
        try:
            query_embedding = self.embeddings.embed_query(query)
            
            # Score each document
            scored_docs = []
            for doc in documents:
                # Try to get precomputed embedding if available
                doc_embedding = None
                if hasattr(doc, 'embedding') and doc.embedding is not None:
                    doc_embedding = doc.embedding
                else:
                    # Compute embedding
                    doc_embedding = self.embeddings.embed_query(doc.page_content)
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, doc_embedding)
                
                # Get keyword score if available
                keyword_score = doc.metadata.get('score', 0)
                
                # Get base score
                base_score = (similarity * 0.6) + (keyword_score * 0.4)
                
                # Apply APU KB specific boosts
                if doc.metadata.get('content_type') == 'apu_kb_page':
                    # Boost FAQ pages for question-like queries
                    if doc.metadata.get('is_faq', False) and query.strip().endswith('?'):
                        base_score *= 1.3
                    
                    # Boost for title match
                    title = doc.metadata.get('page_title', '').lower()
                    if any(word in title for word in query.lower().split()):
                        base_score *= 1.2
                    
                    # Calculate question similarity for FAQ pages
                    if doc.metadata.get('is_faq', False):
                        question_similarity = self._calculate_question_similarity(query, title)
                        base_score += question_similarity * 0.5
                
                scored_docs.append((base_score, doc))
            
            # Sort by score (descending)
            scored_docs.sort(reverse=True, key=lambda x: x[0])
            
            # Return reranked documents
            return [doc for _, doc in scored_docs]
            
        except Exception as e:
            logger.error(f"Error reranking documents: {e}")
            return documents  # Return original order if reranking fails
    
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm_a = math.sqrt(sum(a * a for a in vec1))
        norm_b = math.sqrt(sum(b * b for b in vec2))
        
        if norm_a == 0 or norm_b == 0:
            return 0
            
        return dot_product / (norm_a * norm_b)
    
    def _create_prompt(self, input_dict):
        """Create a prompt for the LLM based on query type and context."""
        # Check if this is an FAQ match
        is_faq_match = input_dict.get("is_faq_match", False)
        match_score = input_dict.get("match_score", 0)
        
        # Detect financial questions
        question = input_dict.get("question", "").lower()
        is_financial = any(term in question for term in [
            "fee", "payment", "pay", "cash", "credit", "debit", "invoice", 
            "receipt", "outstanding", "due", "overdue", "installment",
            "scholarship", "loan", "charge", "refund", "deposit"
        ])
        
        if is_faq_match and match_score > 0.85:
            template = """
            You are an AI assistant answering questions about APU (Asia Pacific University).
            
            CRITICAL RULES YOU MUST FOLLOW:
            1. NEVER invent email addresses - only use the exact email addresses listed in "VALID EMAIL ADDRESSES IN TEXT" section
            2. If a person or department is mentioned without an email (e.g., "Programme Leader"), do NOT create an email for them
            3. NEVER put email addresses in square brackets like [email@example.com]
            4. Do NOT use markdown, bold text, or "Answer:" labels in your response
            5. Your answer must be based ONLY on the information in "SOURCE TEXT" - do not add any details not present there
            
            Context from APU knowledge base:
            {context}
            
            Question: {question}
            
            Your response:
            """
        elif is_financial:
            # Enhanced template for financial questions
            template = """
            You are a helpful AI assistant answering questions about financial matters at APU (Asia Pacific University) in Malaysia.
            
            CRITICAL INSTRUCTIONS:
            1. Focus specifically on providing detailed payment information including methods, deadlines, and procedures
            2. Include EXACT fee amounts, account numbers, and payment details if available in the context
            3. Specify payment locations, online systems, or bank accounts to use
            4. Mention what documentation students need when making payments
            5. Copy ALL email addresses and contact information EXACTLY as they appear in the context
            6. Do NOT invent or modify any payment methods or procedures not mentioned in the context
            
            If specific payment information is not available in the context, clearly state:
            "The specific payment procedure for this situation is not detailed in my knowledge base. Please contact the Finance Office at finance@apu.edu.my or visit the Finance Counter during operating hours (Monday-Friday, 9am-5pm)."
            
            Context from APU knowledge base:
            {context}
            
            Chat History:
            {chat_history}
            
            Question: {question}
            """
        else:
            # Standard RAG prompt with similar instructions
            template = """
            You are a helpful AI assistant answering questions about APU (Asia Pacific University) in Malaysia. 
            
            CRITICAL INSTRUCTIONS:
            1. Copy ALL email addresses EXACTLY as they appear in the context (e.g., admin@apu.edu.my)
            2. Do NOT invent or modify any email addresses or contact information
            3. Do NOT add any markdown formatting, bold text, or "Answer:" labels
            4. Provide only a single, straightforward response
            
            Additional guidelines:
            - If you find the answer in the APU knowledge base, respond directly and precisely
            - Focus on providing clear, action-oriented information when answering procedural questions
            - Use the specific terminology from APU (e.g., "EC" for "Extenuating Circumstances")
            - If information about specific fees is available, include the exact amount
            - If you cannot find specific information in the knowledge base, say "I don't have specific information about that in the APU knowledge base"
            
            Context from APU knowledge base:
            {context}
            
            Chat History:
            {chat_history}
            
            Question: {question}
            """
        
        # Format chat history
        chat_history = ""
        for message in input_dict["chat_history"]:
            if isinstance(message, HumanMessage):
                chat_history += f"Human: {message.content}\n"
            elif isinstance(message, AIMessage):
                chat_history += f"AI: {message.content}\n"
        
        # Fill in the template
        prompt = template.format(
            context=input_dict["context"],
            chat_history=chat_history,
            question=input_dict["question"]
        )
        
        return prompt

class FAQMatcher:
    """Specialized matcher for FAQ content in the APU knowledge base."""
    
    def __init__(self, vector_store):
        """Initialize the FAQ matcher."""
        self.vector_store = vector_store
    
    def match_faq(self, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Try to find a direct FAQ match for the query.
        
        Args:
            query_analysis: Query analysis or dictionary with original_query
            
        Returns:
            Dictionary with match result or None if no good match found
        """
        # Extract query from analysis or get original_query
        if isinstance(query_analysis, dict):
            if "original_query" in query_analysis:
                query = query_analysis["original_query"]
            else:
                return None
        else:
            return None
        
        # Check if query is a question (ends with ?)
        is_question = query.strip().endswith('?')
        
        try:
            # Get all documents
            all_docs = self.vector_store.get()
            
            if not all_docs or not all_docs.get('documents'):
                return None
                
            documents = all_docs.get('documents', [])
            metadatas = all_docs.get('metadatas', [])
            
            # Track best match
            best_match = None
            best_score = 0
            
            # Process each document
            for i, doc_text in enumerate(documents):
                if i >= len(metadatas):
                    continue
                    
                metadata = metadatas[i]
                
                # Only consider APU KB pages that are FAQs
                if metadata.get('content_type') != 'apu_kb_page' or not metadata.get('is_faq', False):
                    continue
                
                # Get the page title (which should be the question)
                title = metadata.get('page_title', '')
                
                # Skip if no title
                if not title:
                    continue
                
                # Calculate similarity between query and title
                similarity = self._calculate_faq_similarity(query, title)
                
                # Boost for question-to-question matching
                if is_question and title.strip().endswith('?'):
                    similarity *= 1.2
                
                # Also check content for exact matches to the query
                content_match = 0
                if query.lower() in doc_text.lower():
                    content_match = 0.3
                
                # Combined score
                score = similarity + content_match
                
                # Update best match if better
                if score > best_score and score > 0.5:  # Threshold for considering a match
                    best_score = score
                    best_match = {
                        "document": Document(
                            page_content=doc_text,
                            metadata=metadata
                        ),
                        "match_score": score
                    }
            
            return best_match
            
        except Exception as e:
            logger.error(f"Error in FAQ matching: {e}")
            return None
    
    def _calculate_faq_similarity(self, query: str, title: str) -> float:
        """
        Calculate the similarity between a query and an FAQ title with enhanced financial question matching.
        
        Args:
            query: User query
            title: FAQ title
            
        Returns:
            Similarity score between 0 and 1
        """
        # Normalize
        query = query.lower().strip().rstrip('?')
        title = title.lower().strip().rstrip('?')
        
        # Check for exact match
        if query == title:
            return 1.0
        
        # Check for financial keywords match
        financial_keywords = [
            "fee", "payment", "pay", "cash", "credit", "debit", "invoice", 
            "receipt", "outstanding", "due", "overdue", "installment",
            "scholarship", "loan", "charge", "refund", "deposit"
        ]
        
        query_has_financial = any(kw in query for kw in financial_keywords)
        title_has_financial = any(kw in title for kw in financial_keywords)
        
        # Apply finance-specific boosting
        finance_boost = 0
        if query_has_financial and title_has_financial:
            finance_boost = 0.3
        
        # Check for substring match
        if query in title or title in query:
            # Calculate relative length ratio for boosting based on how close in length they are
            length_ratio = min(len(query), len(title)) / max(len(query), len(title))
            substring_score = 0.8 * length_ratio
            return substring_score + finance_boost
        
        # Tokenize
        query_tokens = set(word_tokenize(query))
        title_tokens = set(word_tokenize(title))
        
        # Remove stop words
        stop_words = set(stopwords.words('english'))
        query_tokens = {token for token in query_tokens if token not in stop_words}
        title_tokens = {token for token in title_tokens if token not in stop_words}
        
        # Calculate Jaccard similarity
        intersection = len(query_tokens.intersection(title_tokens))
        union = len(query_tokens.union(title_tokens))
        
        jaccard = intersection / union if union > 0 else 0
        
        # Check for overlapping n-grams (phrases)
        ngram_match = self._check_ngram_overlap(query, title)
        
        # Combine scores (weighted)
        return (jaccard * 0.6) + (ngram_match * 0.2) + finance_boost
    
    def _check_ngram_overlap(self, text1: str, text2: str) -> float:
        """
        Check for overlapping n-grams between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Overlap score between 0 and 1
        """
        # Tokenize
        tokens1 = word_tokenize(text1)
        tokens2 = word_tokenize(text2)
        
        # Generate n-grams (bigrams and trigrams)
        bigrams1 = set(ngrams(tokens1, 2)) if len(tokens1) >= 2 else set()
        bigrams2 = set(ngrams(tokens2, 2)) if len(tokens2) >= 2 else set()
        
        trigrams1 = set(ngrams(tokens1, 3)) if len(tokens1) >= 3 else set()
        trigrams2 = set(ngrams(tokens2, 3)) if len(tokens2) >= 3 else set()
        
        # Calculate overlap
        bigram_overlap = len(bigrams1.intersection(bigrams2)) / max(1, min(len(bigrams1), len(bigrams2)))
        trigram_overlap = len(trigrams1.intersection(trigrams2)) / max(1, min(len(trigrams1), len(trigrams2)))
        
        # Combine (trigrams are stronger indicators of similarity)
        combined_overlap = (bigram_overlap * 0.4) + (trigram_overlap * 0.6)
        
        return combined_overlap

#############################################################################
# 4. CONTEXT PROCESSING
#############################################################################

class ContextProcessor:
    """Processes retrieved documents into a coherent context for the LLM."""
    
    def __init__(self):
        """Initialize the context processor."""
        self.max_context_size = Config.MAX_CONTEXT_SIZE
        self.use_compression = Config.USE_CONTEXT_COMPRESSION
    
    def process_context(self, documents: List[Document], query_analysis: Dict[str, Any]) -> str:
        """Process documents with monitoring for context size."""
        if not documents:
            return "No relevant information found in the APU knowledge base."
        
        # Get query elements
        query_type = query_analysis["query_type"]
        keywords = query_analysis["keywords"]
        
        # Score and prioritize documents
        scored_docs = self._score_documents(documents, keywords, query_type)
        
        # Select documents to include based on priority and size constraints
        selected_docs = self._select_documents(scored_docs)
        
        # Format the selected documents
        formatted_context = self._format_documents(selected_docs, query_analysis)
        
        # Add monitoring
        context_size = len(formatted_context)
        context_ratio = context_size / self.max_context_size
        
        if context_ratio > 0.95:
            logger.warning(f"Context size critical: {context_size}/{self.max_context_size} ({context_ratio:.1%})")
        elif context_ratio > 0.85:
            logger.info(f"Context size approaching limit: {context_size}/{self.max_context_size} ({context_ratio:.1%})")
        
        return formatted_context
    
    def _score_documents(self, documents: List[Document], keywords: List[str], query_type: QueryType) -> List[Tuple[Document, float, DocumentRelevance]]:
        """
        Score documents based on relevance to the query, with APU KB optimizations.
        
        Args:
            documents: List of documents
            keywords: List of query keywords
            query_type: Type of query
            
        Returns:
            List of tuples (document, score, relevance_level)
        """
        scored_docs = []
        
        for doc in documents:
            # Start with base score (if available in metadata)
            base_score = doc.metadata.get('score', 0.5)
            
            # Check if this is an APU KB page
            is_apu_kb = doc.metadata.get('content_type') == 'apu_kb_page'
            is_faq = doc.metadata.get('is_faq', False)
            
            # Adjust score based on keyword matches
            keyword_score = 0
            content_lower = doc.page_content.lower()
            
            for keyword in keywords:
                if keyword.lower() in content_lower:
                    # Count occurrences
                    count = content_lower.count(keyword.lower())
                    keyword_score += min(count / 10.0, 0.5)  # Cap at 0.5
            
            # Boost for APU KB pages
            if is_apu_kb:
                # Higher score for FAQ pages
                if is_faq:
                    base_score += 0.2
                
                # Check title match for keywords
                title = doc.metadata.get('page_title', '').lower()
                if any(keyword in title for keyword in keywords):
                    base_score += 0.3
                
                # Check tags for keyword matches
                if 'tags' in doc.metadata:
                    for tag in doc.metadata.get('tags', []):
                        if any(keyword in tag for keyword in keywords):
                            base_score += 0.1
            
            # Consider document length (prefer medium-sized chunks)
            length = len(doc.page_content)
            length_score = 0
            if 200 <= length <= 1000:
                length_score = 0.2  # Prefer medium chunks
            elif length > 1000:
                length_score = 0.1  # Long chunks are okay
            else:
                length_score = 0  # Short chunks less preferred
            
            # Adjust score based on query type and document content
            type_score = 0
            
            if query_type == QueryType.ACADEMIC:
                # For academic queries, prefer documents with relevant terms
                academic_terms = ["course", "module", "exam", "grade", "assessment", "lecture", "assignment"]
                if any(term in content_lower for term in academic_terms):
                    type_score += 0.3
                    
            elif query_type == QueryType.ADMINISTRATIVE:
                # For administrative queries, prefer documents with process terms
                admin_terms = ["form", "application", "procedure", "process", "submit", "request", "office"]
                if any(term in content_lower for term in admin_terms):
                    type_score += 0.3
                    
            elif query_type == QueryType.FINANCIAL:
                # For financial queries, prefer documents with financial terms
                financial_terms = ["fee", "payment", "scholarship", "loan", "refund", "invoice", "charge"]
                if any(term in content_lower for term in financial_terms):
                    type_score += 0.3
                    
            elif query_type == QueryType.FACTUAL:
                # For factual queries, prefer documents with data, numbers, definitions
                if re.search(r'\b(?:defined?|mean|refer|is a|are a|definition)\b', content_lower):
                    type_score += 0.3
                if re.search(r'\d+', content_lower):
                    type_score += 0.2
                    
            elif query_type == QueryType.PROCEDURAL:
                # For procedural queries, prefer step-by-step content
                if re.search(r'\b(?:step|procedure|process|how to|guide|instruction)\b', content_lower):
                    type_score += 0.3
                if re.search(r'\b(?:first|second|third|next|then|finally)\b', content_lower):
                    type_score += 0.3
                    
            elif query_type == QueryType.CONCEPTUAL:
                # For conceptual queries, prefer explanations
                if re.search(r'\b(?:concept|theory|explanation|principle|understand|because)\b', content_lower):
                    type_score += 0.3
                    
            elif query_type == QueryType.COMPARATIVE:
                # For comparative queries, prefer content with comparisons
                if re.search(r'\b(?:compare|contrast|versus|vs|difference|similarity|advantage|disadvantage)\b', content_lower):
                    type_score += 0.4
            
            # Combine scores
            combined_score = (base_score * 0.4) + (keyword_score * 0.3) + (length_score * 0.1) + (type_score * 0.2)
            
            # Determine relevance level
            relevance = DocumentRelevance.MEDIUM  # Default
            if combined_score > 0.7:
                relevance = DocumentRelevance.HIGH
            elif combined_score < 0.3:
                relevance = DocumentRelevance.LOW
            
            scored_docs.append((doc, combined_score, relevance))
        
        # Sort by score (descending)
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return scored_docs
    
    def _select_documents(self, scored_docs: List[Tuple[Document, float, DocumentRelevance]]) -> List[Tuple[Document, DocumentRelevance]]:
        """Select documents with stricter prioritization and size management."""
        selected_docs = []
        current_size = 0
        max_size = self.max_context_size
        
        # Group documents by relevance and sort by score within each group
        high_docs = [(doc, score, rel) for doc, score, rel in scored_docs if rel == DocumentRelevance.HIGH]
        high_docs.sort(key=lambda x: x[1], reverse=True)  # Sort by score
        
        medium_docs = [(doc, score, rel) for doc, score, rel in scored_docs if rel == DocumentRelevance.MEDIUM]
        medium_docs.sort(key=lambda x: x[1], reverse=True)
        
        low_docs = [(doc, score, rel) for doc, score, rel in scored_docs if rel == DocumentRelevance.LOW]
        
        # Process high relevance documents first with strict size management
        for doc, _, relevance in high_docs:
            doc_size = len(doc.page_content)
            
            # For very large documents, always compress
            if doc_size > max_size * 0.5:  # If document takes more than 50% of context
                if self.use_compression:
                    compressed_content = self._compress_document(doc.page_content)
                    compressed_size = len(compressed_content)
                    
                    # Only add if it fits in remaining space
                    if current_size + compressed_size <= max_size:
                        compressed_doc = Document(
                            page_content=compressed_content,
                            metadata=doc.metadata
                        )
                        selected_docs.append((compressed_doc, relevance))
                        current_size += compressed_size
                continue
            
            # Add normal sized documents if they fit
            if current_size + doc_size <= max_size:
                selected_docs.append((doc, relevance))
                current_size += doc_size
        
        # Add medium and low relevance only if significant space remains
        remaining_docs = medium_docs + low_docs
        for doc, _, relevance in remaining_docs:
            # Stop if we've reached 90% capacity
            if current_size >= max_size * 0.9:
                break
                
            doc_size = len(doc.page_content)
            if current_size + doc_size <= max_size:
                selected_docs.append((doc, relevance))
                current_size += doc_size
        
        logger.info(f"Selected {len(selected_docs)} documents for context (size: {current_size}/{max_size}) - {current_size/max_size:.1%} capacity")
        return selected_docs
    
    def _compress_document(self, content: str) -> str:
        """Enhanced document compression that preserves key information."""
        # For short content, don't compress
        if len(content) <= self.max_context_size / 2:
            return content
        
        # First pass: Remove excessive whitespace
        compressed = re.sub(r'\s+', ' ', content).strip()
        
        # Second pass: Identify and extract key information sections
        # For FAQ content, preserve question and direct answer
        if re.search(r'\?', compressed[:100]):  # Likely a question
            # Try to keep question and first paragraph of answer
            question_end = compressed.find('?', 0, 100) + 1
            first_para_end = compressed.find('\n\n', question_end)
            
            if question_end > 0 and first_para_end > question_end:
                key_content = compressed[:first_para_end].strip()
                if len(key_content) <= self.max_context_size / 2:
                    return key_content
        
        # For other content, apply progressive compression
        compressed = self._remove_filler_phrases(compressed)
        compressed = self._deduplicate_sentences(compressed)
        
        # If still too long, extract key sentences
        if len(compressed) > self.max_context_size / 2:
            compressed = self._extract_key_sentences(compressed, self.max_context_size / 2)
        
        return compressed

    def _remove_filler_phrases(self, text):
        """Remove common filler phrases."""
        filler_patterns = [
            (r'in order to', 'to'),
            (r'due to the fact that', 'because'),
            (r'in spite of the fact that', 'although'),
            (r'as a matter of fact', ''),
            (r'for the most part', 'mostly'),
            (r'for all intents and purposes', ''),
            (r'with regard to', 'regarding'),
            (r'in the event that', 'if'),
            (r'in the process of', 'while'),
        ]
        
        for pattern, replacement in filler_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text

    def _deduplicate_sentences(self, text):
        """Remove duplicate or near-duplicate sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        unique_sentences = []
        fingerprints = set()
        
        for sentence in sentences:
            words = re.findall(r'\b\w+\b', sentence.lower())
            if len(words) > 3:
                fingerprint = ' '.join(sorted(words[:5]))
                if fingerprint not in fingerprints:
                    fingerprints.add(fingerprint)
                    unique_sentences.append(sentence)
            else:
                unique_sentences.append(sentence)
        
        return ' '.join(unique_sentences)

    def _extract_key_sentences(self, text, max_length):
        """Extract key sentences based on importance markers."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Score sentences based on importance markers
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            score = 0
            
            # Sentences at beginning and end are often important
            if i < 3:
                score += 3
            elif i >= len(sentences) - 3:
                score += 2
                
            # Sentences with numbers often contain key facts
            if re.search(r'\d', sentence):
                score += 2
                
            # Sentences with key transitional phrases
            if re.search(r'\b(?:however|therefore|thus|in conclusion|importantly|note that|remember)\b', 
                        sentence, re.IGNORECASE):
                score += 2
            
            # Add for reference to FAQ-like constructs
            if re.search(r'\b(?:question|answer|ask|query|refer|check|contact)\b', sentence, re.IGNORECASE):
                score += 1
                
            scored_sentences.append((score, sentence))
        
        # Sort by score and select until we reach max length
        scored_sentences.sort(reverse=True, key=lambda x: x[0])
        
        result = []
        current_length = 0
        
        for _, sentence in scored_sentences:
            if current_length + len(sentence) + 1 <= max_length:
                result.append(sentence)
                current_length += len(sentence) + 1
            else:
                break
        
        # Reorder sentences to maintain original flow
        original_order = sorted([(sentences.index(s), s) for s in result])
        return ' '.join(s for _, s in original_order)
    
    def _format_documents(self, selected_docs: List[Tuple[Document, DocumentRelevance]], query_analysis: Dict[str, Any]) -> str:
        """Format selected documents with overhead estimation to stay within limits."""
        if not selected_docs:
            return "No relevant information found in the APU knowledge base."
                
        # Estimate formatting overhead per document
        estimated_overhead_per_doc = 60  # Characters for headers, labels, etc.
        estimated_doc_separator = 20     # Extra characters between docs
        
        # Calculate total overhead
        total_overhead = len(selected_docs) * estimated_overhead_per_doc + (len(selected_docs) - 1) * estimated_doc_separator
        
        # Calculate available space for actual content
        available_content_space = self.max_context_size - total_overhead
        
        # Get query details for contextual formatting
        query_type = query_analysis["query_type"]
        
        # Group documents by relevance
        high_docs = []
        medium_docs = []
        low_docs = []
        unique_titles = set()
        
        for doc, relevance in selected_docs:
            if relevance == DocumentRelevance.HIGH:
                high_docs.append(doc)
            elif relevance == DocumentRelevance.MEDIUM:
                medium_docs.append(doc)
            else:
                low_docs.append(doc)
            
            # Track unique page titles for APU KB pages
            if doc.metadata.get('content_type') == 'apu_kb_page':
                title = doc.metadata.get('page_title', 'Untitled')
                unique_titles.add(title)

        # Calculate current content size (without formatting)
        current_content_size = sum(len(doc.page_content) for doc in high_docs + medium_docs + low_docs)
        
        # If content will exceed available space, reduce documents
        if current_content_size > available_content_space:
            # Prioritize keeping high relevance docs
            content_to_remove = current_content_size - available_content_space
            
            # Remove low relevance docs first
            while low_docs and content_to_remove > 0:
                doc = low_docs.pop()
                content_to_remove -= len(doc.page_content)
            
            # Then medium relevance if still needed
            while medium_docs and content_to_remove > 0:
                doc = medium_docs.pop()
                content_to_remove -= len(doc.page_content)
                
            # In extreme cases, remove lower-scored high relevance docs
            if content_to_remove > 0:
                # Sort high docs by length (remove longest first to minimize info loss)
                high_docs.sort(key=lambda doc: len(doc.page_content), reverse=True)
                
                while high_docs and content_to_remove > 0:
                    doc = high_docs.pop()
                    content_to_remove -= len(doc.page_content)

        # Begin formatting
        formatted_docs = []
        
        # Add a summary of documents
        doc_count = len(high_docs) + len(medium_docs) + len(low_docs)
        title_count = len(unique_titles)
        
        summary_header = f"Found {doc_count} relevant sections"
        if title_count > 0:
            summary_header += f" from {title_count} topics in the APU knowledge base"
        
        # Add a more specific header based on query type
        if query_type == QueryType.ACADEMIC:
            summary_header = f"Found {doc_count} relevant sections about academic procedures at APU"
        elif query_type == QueryType.ADMINISTRATIVE:
            summary_header = f"Found {doc_count} relevant sections about administrative processes at APU"
        elif query_type == QueryType.FINANCIAL:
            summary_header = f"Found {doc_count} relevant sections about financial matters at APU"
        
        formatted_docs.append(f"{summary_header}\n")
        
        # Format the documents
        formatted_content = self._format_doc_group(high_docs, "HIGHLY RELEVANT INFORMATION", formatted_docs, 0)
        formatted_content = self._format_doc_group(medium_docs, "ADDITIONAL RELEVANT INFORMATION", formatted_docs, len(high_docs))
        formatted_content = self._format_doc_group(low_docs, "SUPPLEMENTARY INFORMATION", formatted_docs, len(high_docs) + len(medium_docs))
        
        final_context = "\n".join(formatted_docs)
        
        # Log the actual final size
        context_size = len(final_context)
        context_ratio = context_size / self.max_context_size
        
        if context_ratio > 0.95:
            logger.warning(f"Context size critical: {context_size}/{self.max_context_size} ({context_ratio:.1%})")
        elif context_ratio > 0.85:
            logger.info(f"Context size approaching limit: {context_size}/{self.max_context_size} ({context_ratio:.1%})")
        
        return final_context

    def _format_doc_group(self, docs: List[Document], section_title: str, formatted_docs: List[str], start_index: int) -> List[str]:
        """Helper to format a group of documents with consistent style."""
        if not docs:
            return formatted_docs
        
        formatted_docs.append(f"--- {section_title} ---")
        
        for i, doc in enumerate(docs):
            is_apu_kb = doc.metadata.get('content_type') == 'apu_kb_page'
            is_faq = doc.metadata.get('is_faq', False)
            
            if is_apu_kb:
                title = doc.metadata.get('page_title', 'Untitled')
                if is_faq:
                    # Format as question-answer (more compact)
                    formatted_text = f"Q: {title}\nA: {doc.page_content}\n"
                else:
                    # Format as topic (more compact)
                    formatted_text = f"Topic: {title}\n{doc.page_content}\n"
                
                # Add related pages if available (more compact)
                related_pages = doc.metadata.get('related_pages', [])
                if related_pages and not any("label in" in page for page in related_pages):
                    if len(related_pages) > 2:
                        formatted_text += f"Related: {', '.join(related_pages[:2])}\n"
                    else:
                        formatted_text += f"Related: {', '.join(related_pages)}\n"
            else:
                # Standard document format (more compact)
                source = doc.metadata.get('source', 'Unknown source')
                filename = doc.metadata.get('filename', os.path.basename(source) if source != 'Unknown source' else 'Unknown file')
                
                formatted_text = f"Doc {i+1+start_index} ({filename}): {doc.page_content}\n"
            
            formatted_docs.append(formatted_text)
        
        return formatted_docs

#############################################################################
# DOCUMENT PROCESSING
#############################################################################

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

#############################################################################
# VECTOR STORE MANAGEMENT
#############################################################################

class VectorStoreManager:
    """Manages the vector database operations."""
    
    @staticmethod
    def check_vector_store_health(vector_store):
        """Perform comprehensive health check on vector store."""
        if not vector_store:
            logger.warning("No vector store provided for health check")
            return False
            
        try:
            # Check 1: Can we get collection info?
            collection_info = False
            if hasattr(vector_store, '_collection') and vector_store._collection is not None:
                collection = vector_store._collection
                try:
                    count = collection.count()
                    logger.info(f"Collection reports {count} documents")
                    if count > 0:
                        collection_info = True
                except Exception as e:
                    logger.warning(f"Failed to get collection count: {e}")
            
            # Check 2: Can we perform a test query?
            query_success = False
            try:
                test_results = vector_store.similarity_search("test query APU university", k=1)
                if test_results:
                    logger.info(f"Query test successful: retrieved {len(test_results)} documents")
                    query_success = True
            except Exception as e:
                logger.warning(f"Query test failed: {e}")
            
            # Check 3: Can we get all documents?
            get_success = False
            try:
                all_docs = vector_store.get()
                doc_count = len(all_docs.get('documents', []))
                logger.info(f"get() method reports {doc_count} documents")
                if doc_count > 0:
                    get_success = True
            except Exception as e:
                logger.warning(f"get() method failed: {e}")
            
            # Overall health assessment
            health_status = collection_info or (query_success and get_success)
            
            if health_status:
                logger.info("Vector store health check: PASSED")
            else:
                logger.warning("Vector store health check: FAILED")
                
            return health_status
            
        except Exception as e:
            logger.error(f"Error during vector store health check: {e}")
            return False
    
    @staticmethod
    def get_embedding_device():
        """Determine the best available device for embeddings."""
        if torch.cuda.is_available():
            logger.info("Using CUDA GPU for embeddings")
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("Using Apple Silicon MPS for embeddings")
            return 'mps'
        else:
            logger.info("Using CPU for embeddings")
            return 'cpu'
        
    @staticmethod
    def save_embeddings_backup(vector_store, filepath=None):
        """Save a backup of embeddings and metadata."""
        if not vector_store:
            return False
            
        if filepath is None:
            filepath = os.path.join(os.path.dirname(Config.PERSIST_PATH), "embeddings_backup.pkl")
            
        try:
            data = vector_store.get()
            if not data or not data.get('ids') or len(data.get('ids', [])) == 0:
                logger.warning("No data to backup - vector store appears empty")
                return False
                
            # Add timestamp for versioning
            data['backup_time'] = datetime.now().isoformat()
            data['doc_count'] = len(data.get('ids', []))
            
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved embeddings backup with {data['doc_count']} documents to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save embeddings backup: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    @staticmethod
    def load_embeddings_backup(embeddings, filepath=None, collection_name="apu_kb_collection"):
        """Load embeddings from backup if main store is empty."""
        if filepath is None:
            filepath = os.path.join(os.path.dirname(Config.PERSIST_PATH), "embeddings_backup.pkl")
            
        if not os.path.exists(filepath):
            logger.warning(f"No embeddings backup found at {filepath}")
            return None
            
        try:
            # Load backup
            import pickle
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                
            # Check if data appears valid
            if not data or 'ids' not in data or len(data['ids']) == 0:
                logger.warning("Backup file exists but contains no valid data")
                return None
                
            logger.info(f"Loaded embeddings backup with {len(data['ids'])} documents from {filepath}")
            
            # Create a new in-memory Chroma instance with this data
            from langchain_chroma import Chroma
            import chromadb
            from chromadb.config import Settings
            
            # Create a memory client
            client = chromadb.Client(Settings(anonymized_telemetry=False))
            
            # Create a new collection
            collection = client.create_collection(name=collection_name)
            
            # Add the data in batches to avoid memory issues
            batch_size = 100
            total_items = len(data['ids'])
            
            for i in range(0, total_items, batch_size):
                end_idx = min(i + batch_size, total_items)
                
                # Prepare batch
                batch_ids = data['ids'][i:end_idx]
                batch_embeddings = data['embeddings'][i:end_idx] if data.get('embeddings') else None
                batch_metadatas = data['metadatas'][i:end_idx] if data.get('metadatas') else None
                batch_documents = data['documents'][i:end_idx] if data.get('documents') else None
                
                # Add to collection
                collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    documents=batch_documents
                )
                
                logger.info(f"Restored batch {i//batch_size + 1}/{(total_items+batch_size-1)//batch_size}: {len(batch_ids)} documents")
            
            # Create a LangChain wrapper around this collection
            vector_store = Chroma(
                client=client,
                collection_name=collection_name,
                embedding_function=embeddings
            )
            
            logger.info(f"Successfully restored vector store from backup with {total_items} documents")
            return vector_store
                
        except Exception as e:
            logger.error(f"Failed to load embeddings backup: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
        
    @staticmethod
    def fix_chromadb_collection(vector_store):
        """
        Fix for ChromaDB collections that appear empty despite existing in the database.
        This is a workaround for a known issue with ChromaDB persistence.
        """
        if not vector_store:
            return False
            
        try:
            # Check if collection exists but reports 0 documents
            if hasattr(vector_store, '_collection') and vector_store._collection is not None:
                count = vector_store._collection.count()
                if count == 0:
                    logger.warning("Collection exists but reports 0 documents - attempting fix")
                    
                    # Try to force a collection reload through direct access
                    if hasattr(vector_store._collection, '_client'):
                        client = vector_store._collection._client
                        collection_name = vector_store._collection.name
                        
                        # Try a direct query to wake up the collection
                        try:
                            logger.info("Attempting direct ChromaDB query to fix collection")
                            from chromadb.api.types import QueryResult
                            results = client.query(
                                collection_name=collection_name,
                                query_texts=["test query for collection fix"],
                                n_results=1,
                            )
                            logger.info(f"Direct query results: {results}")
                            
                            # Check collection again
                            count_after = vector_store._collection.count()
                            logger.info(f"Collection count after fix attempt: {count_after}")
                            
                            return count_after > 0
                        except Exception as e:
                            logger.error(f"Error during collection fix attempt: {e}")
            
            return False
        except Exception as e:
            logger.error(f"Error in fix_chromadb_collection: {e}")
            return False
        
    @staticmethod
    def sanitize_metadata(documents: List[Document]) -> List[Document]:
        """
        Sanitize document metadata to ensure compatibility with ChromaDB.
        
        Args:
            documents: List of documents to process
            
        Returns:
            List of documents with sanitized metadata
        """
        sanitized_docs = []
        
        for doc in documents:
            # Create a copy of the metadata
            metadata = doc.metadata.copy() if doc.metadata else {}
            
            # Process each metadata field
            for key, value in list(metadata.items()):
                # Convert lists to strings
                if isinstance(value, list):
                    if value:  # If list is not empty
                        metadata[key] = json.dumps(value)
                    else:
                        # Remove empty lists
                        metadata.pop(key)
                # Remove None values
                elif value is None:
                    metadata.pop(key)
                # Keep other primitive types as is
            
            # Create a new document with sanitized metadata
            sanitized_doc = Document(
                page_content=doc.page_content,
                metadata=metadata
            )
            sanitized_docs.append(sanitized_doc)
        
        return sanitized_docs
    
    @staticmethod
    def create_embeddings(model_name=None):
        """Create the embedding model."""
        if model_name is None:
            model_name = Config.EMBEDDING_MODEL_NAME
            
        device = VectorStoreManager.get_embedding_device()
        
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': True}
            )
            return embeddings
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise
    
    @staticmethod
    def reset_chroma_db(persist_directory):
        """Reset the ChromaDB environment - handles file system operations safely."""
        logger.info(f"Resetting vector store at {persist_directory}")
        
        # Release resources via garbage collection
        import gc
        gc.collect()
        time.sleep(0.5)
        
        # Remove existing directory if it exists
        if os.path.exists(persist_directory):
            try:
                # Make directory writable first (for Windows compatibility)
                if sys.platform == 'win32':
                    for root, dirs, files in os.walk(persist_directory):
                        for dir in dirs:
                            os.chmod(os.path.join(root, dir), 0o777)
                        for file in files:
                            os.chmod(os.path.join(root, file), 0o777)
                
                # Try Python's built-in directory removal first
                shutil.rmtree(persist_directory)
            except Exception as e:
                logger.warning(f"Error removing directory with shutil: {e}")
                
                # Fallback to system commands
                try:
                    if sys.platform == 'win32':
                        os.system(f"rd /s /q \"{persist_directory}\"")
                    else:
                        os.system(f"rm -rf \"{persist_directory}\"")
                except Exception as e2:
                    logger.error(f"Failed to remove directory: {e2}")
                    return False
        
        # Create fresh directory structure
        try:
            os.makedirs(persist_directory, exist_ok=True)
            
            # Set appropriate permissions
            if sys.platform != 'win32':
                os.chmod(persist_directory, 0o755)
            
            # Create .chroma subdirectory for ChromaDB to recognize
            os.makedirs(os.path.join(persist_directory, ".chroma"), exist_ok=True)
            
            return True
        except Exception as e:
            logger.error(f"Failed to create vector store directory: {e}")
            return False
    
    @classmethod
    def get_or_create_vector_store(cls, chunks=None, embeddings=None, persist_directory=None):
        """Get existing vector store or create a new one with enhanced debugging."""
        if persist_directory is None:
            persist_directory = Config.PERSIST_PATH
            
        # Use a consistent collection name
        collection_name = "apu_kb_collection"
        logger.info(f"Using collection name: {collection_name}")
        
        # Check vector store directory
        cls._check_vector_store_directory(persist_directory)
        
        # Reset directory if needed
        if Config.FORCE_REINDEX or not os.path.exists(persist_directory) or not os.listdir(persist_directory):
            cls.reset_chroma_db(persist_directory)
        
        # Get ChromaDB client
        client = ChromaDBManager.get_client(persist_directory)
        
        # Load existing vector store or create new one
        if chunks is None and os.path.exists(persist_directory) and os.listdir(persist_directory):
            return cls._load_existing_vector_store(client, collection_name, embeddings)
        elif chunks:
            return cls._create_new_vector_store(client, collection_name, chunks, embeddings)
        else:
            logger.error("Cannot create or load vector store - no chunks provided and no existing store")
            return None

    @classmethod
    def _check_vector_store_directory(cls, directory):
        """Check vector store directory and log information."""
        if os.path.exists(directory):
            logger.info(f"Vector store directory exists with contents: {os.listdir(directory)}")
            sqlite_path = os.path.join(directory, "chroma.sqlite3")
            if os.path.exists(sqlite_path):
                file_size = os.path.getsize(sqlite_path)
                logger.info(f"chroma.sqlite3 exists with size: {file_size} bytes")

    @classmethod
    def _load_existing_vector_store(cls, client, collection_name, embeddings):
        """Load existing vector store."""
        try:
            logger.info(f"Loading existing vector store")
            
            # Get collection and langchain wrapper
            collection, vector_store = ChromaDBManager.get_or_create_collection(
                client, collection_name, embedding_function=embeddings)
            
            # Verify collection has documents
            try:
                count = collection.count()
                logger.info(f"Vector store reports {count} documents after loading")
                
                if count > 0:
                    logger.info(f"Successfully loaded vector store with {count} documents")
                    return vector_store
                else:
                    logger.warning("Vector store exists but is empty - will try backup")
                    return None
            except Exception as e:
                logger.error(f"Error verifying collection: {e}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    @classmethod
    def _create_new_vector_store(cls, client, collection_name, chunks, embeddings):
        """Create new vector store with chunks."""
        try:
            logger.info(f"Creating new vector store with {len(chunks)} chunks")
            
            # Sanitize metadata before creating vector store
            sanitized_chunks = cls.sanitize_metadata(chunks)
            
            # Get or create collection
            metadata = {
                "document_count": len(sanitized_chunks),
                "chunk_size": Config.CHUNK_SIZE,
                "chunk_overlap": Config.CHUNK_OVERLAP
            }
            
            _, vector_store = ChromaDBManager.get_or_create_collection(
                client, collection_name, metadata, embeddings)
            
            # Add documents to the vector store
            vector_store.add_documents(documents=sanitized_chunks)
            
            # Explicitly persist
            if hasattr(vector_store, 'persist'):
                vector_store.persist()
                logger.info(f"Vector store persisted successfully with {len(sanitized_chunks)} chunks")
            
            return vector_store
                
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    @staticmethod
    def print_document_statistics(vector_store):
        """Print statistics about indexed documents."""
        if not vector_store:
            logger.warning("No vector store available for statistics")
            return
            
        try:
            # Initialize counters
            doc_counts = {}
            apu_kb_count = 0
            faq_count = 0
            
            # Access collection directly first
            collection = None
            if hasattr(vector_store, '_collection') and vector_store._collection is not None:
                collection = vector_store._collection
                logger.info("Accessing vector store statistics via direct collection")
                count = collection.count()
                logger.info(f"Collection reports {count} documents")
            
            # Get all documents
            all_docs = vector_store.get()
            documents = all_docs.get('documents', [])
            all_metadata = all_docs.get('metadatas', [])
            
            doc_count = len(documents)
            logger.info(f"Vector store get() method reports {doc_count} documents")
            
            if doc_count == 0:
                # Try a test query to see if documents can be retrieved
                try:
                    test_results = vector_store.similarity_search("test query", k=1)
                    if test_results:
                        logger.info(f"Found {len(test_results)} documents via search - data exists but get() not working")
                except Exception as e:
                    logger.error(f"Error during test search: {e}")
                    
            if doc_count == 0 and (collection is None or collection.count() == 0):
                logger.warning("Vector store appears to be empty")
                return
            
            # Count documents by filename
            for metadata in all_metadata:
                if metadata and 'filename' in metadata:
                    filename = metadata['filename']
                    doc_counts[filename] = doc_counts.get(filename, 0) + 1
                
                # Count APU KB specific pages
                if metadata.get('content_type') == 'apu_kb_page':
                    apu_kb_count += 1
                    if metadata.get('is_faq', False):
                        faq_count += 1
            
            total_chunks = len(documents)
            unique_files = len(doc_counts)
            
            logger.info(f"Vector store contains {total_chunks} chunks from {unique_files} files")
            
            # Print file statistics
            print(f"\nKnowledge base contains {unique_files} documents ({total_chunks} total chunks):")
            for filename, count in sorted(doc_counts.items()):
                print(f"  - {filename}: {count} chunks")
            
            # Print APU KB specific statistics
            if apu_kb_count > 0:
                print(f"\nAPU Knowledge Base: {apu_kb_count} pages, including {faq_count} FAQs")
                
            # Print only the most recently added document if timestamp is available
            recent_docs = []
            for i, metadata in enumerate(all_metadata):
                if metadata and 'timestamp' in metadata:
                    recent_docs.append((metadata['timestamp'], metadata.get('filename', 'Unknown')))
            
            if recent_docs:
                recent_docs.sort(reverse=True)
                # Get only the most recent document
                timestamp, filename = recent_docs[0]
                date_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                print(f"\nMost recently added document: {filename} (added: {date_str}).")
                    
        except Exception as e:
            logger.error(f"Error getting document statistics: {e}")
            import traceback
            logger.error(traceback.format_exc())  # Add detailed error info
            print("Error retrieving document statistics.")
    
    @staticmethod
    def verify_document_indexed(vector_store, doc_name):
        """Verify if a specific document is properly indexed."""
        if not vector_store:
            return False
            
        try:
            # Search for the document name in the vector store
            results = vector_store.similarity_search(f"information from {doc_name}", k=3)
            
            # Check if any results match this filename
            for doc in results:
                filename = doc.metadata.get('filename', '')
                if doc_name.lower() in filename.lower():
                    logger.info(f"Document '{doc_name}' is indexed in the vector store")
                    return True
                    
            logger.warning(f"Document '{doc_name}' was not found in the vector store")
            return False
                
        except Exception as e:
            logger.error(f"Error verifying document indexing: {e}")
            return False

#############################################################################
# 5. RESPONSE GENERATION
#############################################################################

class RAGSystem:
    """Manages the RAG processing pipeline."""
    
    @staticmethod
    def format_docs(docs):
        """Format retrieved documents for inclusion in the prompt."""
        if not docs:
            return "No relevant documents found."
            
        formatted_docs = []
        unique_filenames = set()
        
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'Unknown source')
            filename = doc.metadata.get('filename', os.path.basename(source) if source != 'Unknown source' else 'Unknown file')
            unique_filenames.add(filename)
            
            # Format differently for APU KB pages
            if doc.metadata.get('content_type') == 'apu_kb_page':
                title = doc.metadata.get('page_title', 'Untitled')
                formatted_text = f"Document {i+1} (from {filename}, Topic: {title}):\n{doc.page_content}\n\n"
            else:
                # Include metadata if available
                page_info = f"page {doc.metadata.get('page', '')}" if doc.metadata.get('page', '') else ""
                chunk_info = f"chunk {i+1}/{len(docs)}"
                
                metadata_line = f"Document {i+1} (from {filename} {page_info} {chunk_info}):\n"
                formatted_text = f"{metadata_line}{doc.page_content}\n\n"
                
            formatted_docs.append(formatted_text)

        # Add a summary of documents
        summary = f"Retrieved {len(docs)} chunks from {len(unique_filenames)} files: {', '.join(unique_filenames)}\n\n"
        
        return summary + "\n".join(formatted_docs)
    
    @staticmethod
    def stream_ollama_response(prompt, model_name=None, base_url=None, stream_output=False):
        """Stream response from Ollama API with token-by-token output.
        
        Args:
            prompt: The prompt to send to Ollama
            model_name: The name of the model to use
            base_url: The base URL for the Ollama API
            stream_output: Whether to stream output in real-time (yield tokens) or return full response
            
        Returns:
            If stream_output is True, yields tokens as they are generated
            If stream_output is False, returns the full response as a string
        """
        if model_name is None:
            model_name = Config.LLM_MODEL_NAME
            
        if base_url is None:
            base_url = Config.OLLAMA_BASE_URL
            
        url = f"{base_url}/api/generate"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": model_name,
            "prompt": prompt,
            "stream": True
        }

        full_response = ""

        try:
            # Test connection to Ollama API
            test_url = f"{base_url}/api/tags"
            try:
                test_response = requests.get(test_url, timeout=5)
                if test_response.status_code != 200:
                    error_msg = f"Error: Could not connect to Ollama API. Make sure Ollama is running."
                    logger.error(f"Ollama API unavailable: HTTP {test_response.status_code}")
                    return error_msg if not stream_output else iter([error_msg])
            except requests.RequestException as e:
                error_msg = f"Error: Could not connect to Ollama API. Make sure Ollama is running and accessible."
                logger.error(f"Failed to connect to Ollama: {e}")
                return error_msg if not stream_output else iter([error_msg])

            # Process the streaming response
            with requests.post(url, headers=headers, json=data, stream=True, timeout=30) as response:
                if response.status_code != 200:
                    error_msg = f"Error: Failed to generate response (HTTP {response.status_code})"
                    logger.error(f"Ollama API error: {response.status_code}")
                    return error_msg if not stream_output else iter([error_msg])

                for line in response.iter_lines():
                    if line:
                        try:
                            json_line = json.loads(line.decode('utf-8'))
                            if 'response' in json_line:
                                token = json_line['response']
                                full_response += token
                                
                                # If streaming, yield each token
                                if stream_output:
                                    yield token

                            if json_line.get('done', False):
                                break
                        except json.JSONDecodeError:
                            logger.error(f"Invalid JSON from Ollama API: {line}")
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logger.error(f"Error during Ollama request: {e}")
            return error_msg if not stream_output else iter([error_msg])
        
        # Return the full response if not streaming
        if not stream_output:
            return full_response

#############################################################################
# MAIN APPLICATION
#############################################################################

class CustomRAG:
    """Main RAG application class for command line interface."""
    
    def __init__(self):
        """Initialize the RAG application."""
        self.vector_store = None
        self.embeddings = None
        self.memory = None
        self.input_processor = None
        self.context_processor = None
        self.retrieval_handler = None
        self.conversation_handler = None
        self.command_handler = None
        self.query_router = None
    
    def initialize(self):
        """Set up all components of the RAG system."""
        # Setup configuration
        Config.setup()
        
        # Check dependencies
        DocumentProcessor.check_dependencies()
        
        # Initialize components
        self.input_processor = InputProcessor()
        self.context_processor = ContextProcessor()
        
        # Create memory
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history"
        )
        
        # Add a system message to memory
        system_message = SystemMessage(content="I am an AI assistant that helps with answering questions about APU. I can provide information about academic procedures, administrative processes, and university services.")
        self.memory.chat_memory.messages.append(system_message)
        
        # Create embeddings
        try:
            self.embeddings = VectorStoreManager.create_embeddings()
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            return False
        
        # Initialize vector store with health checks
        vector_store_valid = self.initialize_vector_store()
        
        if not vector_store_valid:
            logger.error("Failed to initialize a valid vector store")
            return False
        
        # Print statistics
        VectorStoreManager.print_document_statistics(self.vector_store)
        
        # Initialize handlers
        self.conversation_handler = ConversationHandler(self.memory)
        self.retrieval_handler = RetrievalHandler(self.vector_store, self.embeddings, self.memory, self.context_processor)
        self.command_handler = CommandHandler(self)
        
        # Initialize query router
        self.query_router = QueryRouter(
            self.conversation_handler,
            self.retrieval_handler,
            self.command_handler
        )
        
        return True

    def initialize_vector_store(self):
        """Initialize vector store with health checks and fallbacks."""
        # Path for backup
        backup_path = os.path.join(os.path.dirname(Config.PERSIST_PATH), "embeddings_backup.pkl")
        backup_exists = os.path.exists(backup_path)
        
        # First try to load from ChromaDB (normal flow)
        vector_store_valid = False
        if not Config.FORCE_REINDEX and os.path.exists(Config.PERSIST_PATH):
            try:
                logger.info("Attempting to load existing vector store")
                self.vector_store = VectorStoreManager.get_or_create_vector_store(None, self.embeddings)
                
                # Check if it has documents and is healthy
                if self.vector_store:
                    vector_store_valid = VectorStoreManager.check_vector_store_health(self.vector_store)
            except Exception as e:
                logger.error(f"Error loading vector store: {e}")
        
        # If ChromaDB load failed but we have a backup, try to restore from backup
        if not vector_store_valid and backup_exists:
            logger.info("Attempting to restore from embeddings backup")
            self.vector_store = VectorStoreManager.load_embeddings_backup(self.embeddings)
            if self.vector_store:
                vector_store_valid = VectorStoreManager.check_vector_store_health(self.vector_store)
        
        # If both ChromaDB and backup failed, or we're forcing reindex, create from scratch
        if not vector_store_valid or Config.FORCE_REINDEX:
            logger.info("Creating new vector store from documents")
            
            # Load and process documents
            documents = DocumentProcessor.load_documents(Config.DATA_PATH)
            if not documents:
                logger.error("No documents found to index")
                return False
                
            chunks = DocumentProcessor.split_documents(documents)
            if not chunks:
                logger.error("Failed to create document chunks")
                return False
                
            # Create vector store with chunks
            self.vector_store = VectorStoreManager.get_or_create_vector_store(chunks, self.embeddings)
            
            # If successful, create a backup
            if self.vector_store:
                vector_store_valid = VectorStoreManager.check_vector_store_health(self.vector_store)
                if vector_store_valid:
                    # Create backup
                    VectorStoreManager.save_embeddings_backup(self.vector_store)
        
        return vector_store_valid
    
    def reindex_documents(self):
        """Reindex all documents in the data directory."""
        logger.info("Reindexing documents")
        print("Reindexing documents. This may take a while...")
        
        # Close existing resources
        if self.vector_store is not None:
            try:
                # Force reset of ChromaDB client
                ChromaDBManager.close()
                self.vector_store = None
                
                # Force garbage collection
                import gc
                gc.collect()
                time.sleep(0.5)
            except Exception as e:
                logger.warning(f"Error closing vector store: {e}")
        
        # Reset the vector store directory
        if not VectorStoreManager.reset_chroma_db(Config.PERSIST_PATH):
            logger.error("Failed to reset vector store")
            return False
        
        # Load and process documents
        try:
            documents = DocumentProcessor.load_documents(Config.DATA_PATH)
            if not documents:
                logger.error("No documents found to index")
                return False
                
            chunks = DocumentProcessor.split_documents(documents)
            if not chunks:
                logger.error("Failed to create document chunks")
                return False
                
            # Get a fresh client
            client = ChromaDBManager.get_client(reset=True)
            
            # Create fresh embeddings
            self.embeddings = VectorStoreManager.create_embeddings()
                
            # Create vector store with chunks
            self.vector_store = VectorStoreManager._create_new_vector_store(
                client, "apu_kb_collection", chunks, self.embeddings)
            
            if not self.vector_store:
                logger.error("Failed to create vector store")
                return False
                
            # Create a backup of the newly created vector store
            VectorStoreManager.save_embeddings_backup(self.vector_store)
                
            # Print statistics
            VectorStoreManager.print_document_statistics(self.vector_store)
            
            # Reinitialize retrieval handler with new vector store
            self.retrieval_handler = RetrievalHandler(self.vector_store, self.embeddings, self.memory, self.context_processor)
            
            # Update query router with new retrieval handler
            self.query_router = QueryRouter(
                self.conversation_handler,
                self.retrieval_handler,
                self.command_handler
            )
                
            print(f"Reindexing completed successfully! Added {len(chunks)} chunks to the vector store.")
            return True
            
        except Exception as e:
            logger.error(f"Error during reindexing: {e}")
            return False
    
    def run_cli(self):
        """Run the interactive command line interface."""
        if not self.initialize():
            logger.error("Failed to initialize RAG system")
            return
            
        # Print banner and instructions
        print("\n" + "="*60)
        print("📚 Enhanced CustomRAG - APU Knowledge Base Assistant 📚")
        print("="*60)
        print("Ask questions about APU using natural language.")
        print("Commands:")
        print("  - Type 'help' to see available commands")
        print("  - Type 'exit' or 'quit' to stop")
        print("  - Type 'clear' to reset the conversation memory")
        print("  - Type 'reindex' to reindex all documents")
        print("  - Type 'stats' to see document statistics")
        print("="*60 + "\n")
        
        # Main interaction loop
        while True:
            try:
                query = input("\nYour Question: ").strip()
                
                if not query:
                    print("Please enter a question.")
                    continue
                
                # Process query
                start_time = time.time()
                
                print("\nThinking...\n")
                
                try:
                    # Process and route query
                    query_analysis = self.input_processor.analyze_query(query)
                    
                    # Route the query to appropriate handler with streaming enabled
                    response, should_continue = self.query_router.route_query(query_analysis, stream=True)
                    
                    if response:
                        # Handle streamed responses
                        if isinstance(response, types.GeneratorType) or hasattr(response, '__iter__'):
                            print("Response: ", end="", flush=True)
                            full_response = ""
                            for token in response:
                                print(token, end="", flush=True)
                                full_response += token
                            print()  # Add newline after streaming completes
                            
                            # Now update memory with the full response
                            self.memory.chat_memory.add_user_message(query)
                            self.memory.chat_memory.add_ai_message(full_response)
                        else:
                            # Handle non-streamed responses
                            print(f"Response: {response}")
                    
                    if not should_continue:
                        break
                        
                except Exception as e:
                    logger.error(f"Error processing query: {e}")
                    print("\n[Error occurred during response generation]")
                
                end_time = time.time()
                logger.info(f"Query processed in {end_time - start_time:.2f} seconds")
                
            except KeyboardInterrupt:
                print("\nGoodbye! Have a great day!")
                break
            except Exception as e:
                logger.error(f"Error in CLI: {e}")
                print("\n[An unexpected error occurred. Please try again.]")
                
    def cleanup(self):
        """Clean up resources before shutdown."""
        logger.info("Cleaning up resources")
        
        # Close ChromaDB client
        ChromaDBManager.close()
        
        # Clear vector store reference
        self.vector_store = None
        
        # Force garbage collection
        import gc
        gc.collect()
        
        logger.info("Cleanup completed")

def main():
    """Entry point for the CustomRAG application."""
    app = None
    try:
        app = CustomRAG()
        app.run_cli()
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"Error: {e}")
        return 1
    finally:
        # Clean up resources
        if app is not None:
            try:
                app.cleanup()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
    
