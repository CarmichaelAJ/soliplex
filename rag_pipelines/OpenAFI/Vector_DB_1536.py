import os
import glob
import json
import time
import re
from datetime import datetime
from typing import Dict, Any, List, Tuple
from difflib import SequenceMatcher
import pdfplumber  # Better PDF text extraction than PyPDF2

# chromadb imports
import chromadb
from chromadb.config import Settings

import openai  # Import OpenAI library
from pathlib import Path

# ============ CONFIG ============

PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_DIR.parent.parent

# Set OpenAI API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set. Please set it before running this script.")
openai.api_key = openai_api_key

FOLDER_PATH = str((REPO_ROOT / "docs").resolve())  # Soliplex docs as dataset
METADATA_JSON = str((PIPELINE_DIR / "document_tracker.json").resolve())      # Where we store document statuses
CHROMA_DIR = str((PIPELINE_DIR / "chroma_storage_1536").resolve())
COLLECTION_NAME = "pdf_chunks_collection"
TXT_FOLDER_PATH = str((PIPELINE_DIR / "text_cache").resolve())  # Cache extracted text
DATASET_METADATA = str((PIPELINE_DIR / "dataset_metadata.json").resolve())
DOC_PATTERNS = ("*.md", "*.markdown", "*.txt", "*.pdf")
TEXT_EXTENSIONS = {".md", ".markdown", ".txt"}

# Size thresholds for paragraph-based chunking
MAX_DOCUMENT_SIZE_MB = 10  # Size in MB above which we use hierarchical chunking
MAX_PARAGRAPHS_PER_CHUNK = 3  # Number of paragraphs per chunk
PARAGRAPH_OVERLAP = 1  # Number of paragraphs to overlap between chunks (1 up, 1 down)

# Define the embedding model to use consistently throughout the application
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI model
EMBEDDING_DIM = 1536  # Expected dimension for OpenAI embedding

# Define a custom OpenAI embedding function.
class OpenAIEmbeddingFunction:
    def __init__(self, model_name=EMBEDDING_MODEL):
        self.model_name = model_name
        self.dimension = EMBEDDING_DIM

    def name(self):
        """Return the name of this embedding function."""
        return f"openai_{self.model_name}"

    def __call__(self, input):
        # Ensure input is a list
        if isinstance(input, str):
            input = [input]
        embeddings = []
        for text in input:
            response = openai.embeddings.create(
                model=self.model_name,
                input=text,
                dimensions=EMBEDDING_DIM
            )
            # Extract embedding from response
            vector = response.data[0].embedding
            
            # Print vector dimensions
            print(f"Vector length: {len(vector)}")
            embeddings.append(vector)
        return embeddings
    
    def embed_query(self, input):
        """Embed a query text (used by ChromaDB for queries)."""
        return self.__call__(input)

# Initialize ChromaDB with OpenAI embeddings
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=OpenAIEmbeddingFunction(model_name=EMBEDDING_MODEL)
)

# Ensure text folder exists
os.makedirs(TXT_FOLDER_PATH, exist_ok=True)

# ============ HELPER FUNCTIONS ============

def load_or_initialize_metadata(json_path: str) -> Dict[str, Any]:
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return {}

def save_metadata(metadata: Dict[str, Any], json_path: str):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

def list_source_documents() -> List[str]:
    """Return all dataset files matching configured extensions."""
    files = []
    for pattern in DOC_PATTERNS:
        glob_pattern = os.path.join(FOLDER_PATH, "**", pattern)
        files.extend(glob.glob(glob_pattern, recursive=True))
    # Deduplicate if multiple patterns match same file
    return sorted(set(files))

def get_file_modification_time(filepath: str) -> float:
    return os.path.getmtime(filepath)

def convert_pdf_to_text(pdf_path, save_txt=True):
    """Extract text from source documents and optionally cache the content."""
    source_path = Path(pdf_path)
    
    # Markdown / text documents can be read directly
    if source_path.suffix.lower() in TEXT_EXTENSIONS:
        try:
            return source_path.read_text(encoding='utf-8')
        except Exception as exc:
            print(f"Error reading text document {source_path}: {exc}")
            return ""
    
    txt_path = Path(TXT_FOLDER_PATH) / f"{source_path.stem}.txt"
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    
    # If txt file already exists and is newer than the PDF, just read it
    if txt_path.exists() and get_file_modification_time(str(txt_path)) > get_file_modification_time(str(source_path)):
        print(f"Using existing text file: {txt_path}")
        with txt_path.open('r', encoding='utf-8') as f:
            return f.read()
    
    print(f"Converting PDF to text: {source_path}")
    text_content = []
    
    try:
        with pdfplumber.open(str(source_path)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_content.append(page_text)
        
        full_text = "\n\n".join(text_content)
        
        # Save to txt file if requested
        if save_txt:
            with txt_path.open('w', encoding='utf-8') as f:
                f.write(full_text)
            print(f"Saved text to {txt_path}")
            
        return full_text
    except Exception as e:
        print(f"Error extracting text from {source_path}: {str(e)}")
        return ""

def simple_token_count(text: str) -> int:
    """
    Simple token counter - approximates tokens as words divided by 0.75
    (OpenAI's rule of thumb: 1 token ≈ 0.75 words)
    """
    words = len(text.split())
    return int(words / 0.75)

def extract_paragraphs_from_text(text):
    """Split text into paragraphs based on double newlines."""
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    if not paragraphs and text.strip():
        # If no paragraphs were found but there's text, treat the whole text as one paragraph
        paragraphs = [text.strip()]
    return paragraphs

def chunk_by_paragraphs(text: str, paragraphs_per_chunk: int = MAX_PARAGRAPHS_PER_CHUNK, overlap: int = PARAGRAPH_OVERLAP) -> List[Tuple[str, int, int, int]]:
    """
    Chunk text by paragraphs with overlap.
    Returns a list of tuples: (chunk_text, start_para_idx, end_para_idx, total_paragraphs)
    This allows us to track which paragraphs are in each chunk and retrieve adjacent ones.
    """
    paragraphs = extract_paragraphs_from_text(text)
    
    if not paragraphs:
        return []
    
    chunks = []
    total_paragraphs = len(paragraphs)
    i = 0
    
    while i < total_paragraphs:
        # Determine the range of paragraphs for this chunk
        start_idx = i
        end_idx = min(i + paragraphs_per_chunk, total_paragraphs)
        
        # Get the paragraphs for this chunk
        chunk_paragraphs = paragraphs[start_idx:end_idx]
        chunk_text = '\n\n'.join(chunk_paragraphs)
        
        # Store chunk with its paragraph indices
        chunks.append((chunk_text, start_idx, end_idx - 1, total_paragraphs))
        
        # Move to next chunk, accounting for overlap
        # We move forward by (paragraphs_per_chunk - overlap) paragraphs
        step = max(1, paragraphs_per_chunk - overlap)
        i += step
    
    return chunks

def get_paragraph_context(text, chunk_text):
    """
    Find the paragraph containing the chunk and get surrounding paragraphs.
    Returns a tuple of (prev_paragraph, parent_paragraph, next_paragraph)
    """
    paragraphs = extract_paragraphs_from_text(text)
    
    # If we couldn't split into paragraphs, return empty context
    if not paragraphs:
        return (None, chunk_text, None)
    
    # If there's only one paragraph, it must be the parent
    if len(paragraphs) == 1:
        return (None, paragraphs[0], None)
    
    # Find which paragraph contains the chunk
    parent_idx = -1
    chunk_text_clean = chunk_text.strip()
    
    # First try exact containment
    for i, para in enumerate(paragraphs):
        if chunk_text_clean in para:
            parent_idx = i
            break
    
    # If we couldn't find the exact chunk in any paragraph,
    # use fuzzy matching to find the closest paragraph
    if parent_idx == -1:
        best_ratio = 0
        for i, para in enumerate(paragraphs):
            # Skip very short paragraphs for matching
            if len(para) < 20:
                continue
                
            ratio = SequenceMatcher(None, chunk_text_clean, para).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                parent_idx = i
        
        # If best match is still weak, use the first substantial paragraph
        if best_ratio < 0.3:
            for i, para in enumerate(paragraphs):
                if len(para) > 100:  # A reasonable paragraph length
                    parent_idx = i
                    break
            
            # If still no good match, use the first paragraph
            if parent_idx == -1 and paragraphs:
                parent_idx = 0
    
    # Get surrounding paragraphs
    prev_para = paragraphs[parent_idx-1] if parent_idx > 0 else None
    parent_para = paragraphs[parent_idx] if parent_idx >= 0 and parent_idx < len(paragraphs) else chunk_text_clean
    next_para = paragraphs[parent_idx+1] if parent_idx >= 0 and parent_idx < len(paragraphs)-1 else None
    
    return (prev_para, parent_para, next_para)

def extract_pdf_pages(pdf_path: str) -> list:
    """Extract text from a document by pseudo-pages, returning (page_num, text)."""
    source_path = Path(pdf_path)
    
    def _split_text_to_pages(text: str) -> list:
        lines = text.split('\n')
        avg_lines_per_page = 50  # Approximation
        pages = []
        for i in range(0, len(lines), avg_lines_per_page):
            page_num = i // avg_lines_per_page + 1
            page_text = '\n'.join(lines[i:i+avg_lines_per_page])
            if page_text.strip():
                pages.append((page_num, page_text))
        return pages
    
    if source_path.suffix.lower() in TEXT_EXTENSIONS:
        try:
            return _split_text_to_pages(source_path.read_text(encoding='utf-8'))
        except Exception as exc:
            print(f"Error reading pages from {source_path}: {exc}")
            return []
    
    # First try to use the cached text version if available
    txt_path = Path(TXT_FOLDER_PATH) / f"{source_path.stem}.txt"
    if txt_path.exists():
        with txt_path.open('r', encoding='utf-8') as f:
            text = f.read()
        return _split_text_to_pages(text)
    
    # Fall back to pdfplumber if text file not available
    pages = []
    try:
        with pdfplumber.open(str(source_path)) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    pages.append((i+1, page_text))
        return pages
    except Exception as e:
        print(f"Error extracting pages from {source_path}: {str(e)}")
        return []

def detect_chapters(text: str) -> List[Tuple[str, str]]:
    """
    Detect chapter or section boundaries in text.
    Returns a list of (chapter_title, chapter_content) tuples.
    """
    # Common chapter/section patterns
    chapter_patterns = [
        r'(?:\n|\r\n|\r)(?:CHAPTER|Chapter)\s+\d+[\.\:]?\s+([^\n\r]+)(?:\n|\r\n|\r)',
        r'(?:\n|\r\n|\r)(?:SECTION|Section)\s+\d+[\.\:]?\s+([^\n\r]+)(?:\n|\r\n|\r)',
        r'(?:\n|\r\n|\r)\d+[\.\d]*\s+([A-Z][^\n\r]{2,60})(?:\n|\r\n|\r)',  # Numbered sections with UPPERCASE or Title Case headers
        r'(?:\n|\r\n|\r)(?:[A-Z][^\n\r]{2,60})(?:\n|\r\n|\r)'  # Just UPPERCASE or Title Case headers on their own line
    ]
    
    # Try to find chapters
    chapters = []
    remaining_text = text
    
    # First, see if the document has a table of contents we can use to identify chapter patterns
    toc_pattern = r'(?:TABLE\s+OF\s+CONTENTS|CONTENTS|Table\s+of\s+Contents)'
    toc_match = re.search(toc_pattern, text[:5000])  # Look in first 5000 chars
    
    if toc_match:
        # Extract chapter titles from TOC to look for in main text
        toc_section = text[toc_match.start():toc_match.start() + 5000]  # Get up to 5000 chars of TOC
        toc_lines = [line.strip() for line in toc_section.split('\n') if line.strip()]
        
        # Look for chapter patterns in TOC
        custom_patterns = []
        for line in toc_lines:
            # Look for "Chapter X" or "Section X" patterns
            if re.search(r'(?:Chapter|CHAPTER|Section|SECTION)\s+\d+', line):
                # Extract the name pattern
                title_match = re.search(r'(?:Chapter|CHAPTER|Section|SECTION)\s+\d+\.?\s+(.*?)\s*(?:\d+)?$', line)
                if title_match and title_match.group(1):
                    title_pattern = re.escape(title_match.group(1).strip())
                    custom_patterns.append(fr'(?:\n|\r\n|\r)(?:Chapter|CHAPTER|Section|SECTION)\s+\d+\.?\s+{title_pattern}')
        
        # Add custom patterns from TOC
        if custom_patterns:
            chapter_patterns = custom_patterns + chapter_patterns
    
    # Look for potential dividers like long sequences of dashes or similar
    divider_pattern = r'(?:\n|\r\n|\r)[-=_]{3,}(?:\n|\r\n|\r)'
    dividers = list(re.finditer(divider_pattern, text))
    
    if len(dividers) > 2:  # If we have several dividers, they might be section breaks
        last_pos = 0
        for i, match in enumerate(dividers):
            section_text = text[last_pos:match.start()]
            if section_text.strip():
                # Try to extract a title from the first few lines
                first_lines = section_text.strip().split('\n')[:3]
                potential_title = ' - '.join([l.strip() for l in first_lines if l.strip()])[:100]
                chapters.append((f"Section {i+1}: {potential_title}", section_text))
            last_pos = match.end()
        
        # Add the final section
        final_section = text[last_pos:]
        if final_section.strip():
            first_lines = final_section.strip().split('\n')[:3]
            potential_title = ' - '.join([l.strip() for l in first_lines if l.strip()])[:100]
            chapters.append((f"Section {len(dividers)+1}: {potential_title}", final_section))
    
    # If we didn't find chapters via dividers, try regex patterns
    if not chapters:
        # Try each pattern
        for pattern in chapter_patterns:
            matches = list(re.finditer(pattern, text))
            if len(matches) > 1:  # We need at least 2 matches to split into chapters
                chapter_starts = [match.start() for match in matches]
                chapter_titles = [match.group(1) if len(match.groups()) > 0 else f"Chapter {i+1}" 
                                for i, match in enumerate(matches)]
                
                # Extract chapters
                for i in range(len(chapter_starts)):
                    start = chapter_starts[i]
                    end = chapter_starts[i+1] if i < len(chapter_starts) - 1 else len(text)
                    chapter_text = text[start:end]
                    chapter_title = chapter_titles[i]
                    chapters.append((f"Chapter: {chapter_title}", chapter_text))
                
                break  # Stop after finding chapters with the first successful pattern
    
    # If still no chapters found, look for page number markers and try to use those
    if not chapters:
        page_markers = list(re.finditer(r'(?:\n|\r\n|\r)\s*[\-\–\—]{0,3}\s*(\d+)\s*[\-\–\—]{0,3}\s*(?:\n|\r\n|\r)', text))
        
        if len(page_markers) > 2:  # Need at least a few page numbers
            # Group text by pages with clear page number markers
            last_pos = 0
            current_section = []
            
            for match in page_markers:
                page_text = text[last_pos:match.start()]
                if page_text.strip():
                    current_section.append(page_text)
                
                # Every 10 pages or so, create a new chapter
                if len(current_section) >= 10:
                    section_text = "\n".join(current_section)
                    page_num = match.group(1)
                    chapters.append((f"Pages up to {page_num}", section_text))
                    current_section = []
                
                last_pos = match.end()
            
            # Add the final section
            if current_section:
                final_section = "\n".join(current_section) + text[last_pos:]
                if final_section.strip():
                    chapters.append((f"Final pages", final_section))
    
    # If we still don't have chapters, split by size
    if not chapters:
        # Split into roughly equal chunks based on character count
        total_length = len(text)
        ideal_chunks = 5  # Try to get about 5 chunks
        chunk_size = total_length // ideal_chunks
        
        for i in range(ideal_chunks):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < ideal_chunks - 1 else total_length
            
            # Try to find a paragraph break near the intended end
            break_pos = text.rfind('\n\n', start, end)
            if break_pos > start + (chunk_size // 2):
                end = break_pos
            
            chunk_text = text[start:end]
            if chunk_text.strip():
                chapters.append((f"Part {i+1}", chunk_text))
    
    # If we somehow still don't have chapters (unlikely at this point), return the whole text
    if not chapters:
        chapters = [("Complete Document", text)]
    
    return chapters

def fallback_chunker(text: str, max_chunk_size=8000, overlap=200) -> list:
    """Split text into chunks of approximately max_chunk_size characters with overlap."""
    if not text or not text.strip():
        return []
        
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + max_chunk_size, text_length)
        
        # Try to find a paragraph or sentence break for a cleaner split
        if end < text_length:
            # Look for paragraph breaks
            paragraph_end = text.rfind('\n\n', start, end)
            if paragraph_end > start + max_chunk_size // 2:
                end = paragraph_end + 2
            else:
                # Look for sentence breaks
                sentence_end = text.rfind('. ', start, end)
                if sentence_end > start + max_chunk_size // 2:
                    end = sentence_end + 2
        
        # Get the chunk
        chunk = text[start:end]
        chunks.append(chunk)
        
        # Move start with overlap
        start = max(start + 1000, end - overlap)  # Ensure we make progress
        
        # Avoid tiny chunks at the end
        if text_length - start < max_chunk_size // 4:
            if start < text_length:  # Make sure we haven't reached the end
                chunks.append(text[start:])
            break
    
    return chunks

def add_chunk_to_chroma(chunk_text, source_pdf, document_metadata, chunk_index, total_chunks, start_para_idx=None, end_para_idx=None, total_paragraphs=None):
    base_filename = os.path.basename(source_pdf)
    timestamp = datetime.now().isoformat()
    doc_id = f"{base_filename}_{chunk_index}_{int(time.time())}"
    token_count = simple_token_count(chunk_text)
    
    chunk_metadata = {
        "source_pdf": base_filename,
        "chunk_index": chunk_index,
        "chunk_of_total": f"{chunk_index+1}/{total_chunks}",
        "chunked_at": timestamp,
        "token_count": token_count,
        "doc_number": document_metadata.get("number", ""),
        "doc_title": document_metadata.get("title", ""),
        "doc_version": document_metadata.get("version", ""),
        "published_date": document_metadata.get("published_date", ""),
        "certified_current": document_metadata.get("certified_current", ""),
        "last_action": document_metadata.get("last_action", ""),
        "doc_url": document_metadata.get("url", ""),
    }
    
    # Add paragraph tracking information if provided
    if start_para_idx is not None and end_para_idx is not None and total_paragraphs is not None:
        chunk_metadata["start_paragraph_index"] = start_para_idx
        chunk_metadata["end_paragraph_index"] = end_para_idx
        chunk_metadata["total_paragraphs"] = total_paragraphs
        chunk_metadata["paragraph_range"] = f"{start_para_idx}-{end_para_idx} of {total_paragraphs}"
    
    try:
        collection.add(
            documents=[chunk_text],
            metadatas=[chunk_metadata],
            ids=[doc_id]
        )
        
        return {
            "chunk_id": doc_id,
            "token_count": token_count,
            "timestamp": timestamp
        }
    except Exception as e:
        print(f"Error in add_chunk_to_chroma: {str(e)}")
        print(f"Chunk text preview (first 100 chars): {chunk_text[:100]}")
        raise  # Re-raise the exception for the caller to handle

def chunk_document(text: str, pdf_path: str = None) -> List[Tuple[str, int, int, int]]:
    """
    Chunk document using simple paragraph-based chunking.
    Returns a list of tuples: (chunk_text, start_para_idx, end_para_idx, total_paragraphs)
    """
    if not text or not text.strip():
        return []
    
    print(f"Chunking document with paragraph-based approach...")
    print(f"Document length: {len(text)} characters")
    
    # Use paragraph-based chunking
    chunks = chunk_by_paragraphs(text, MAX_PARAGRAPHS_PER_CHUNK, PARAGRAPH_OVERLAP)
    
    if chunks:
        print(f"Created {len(chunks)} chunks with {MAX_PARAGRAPHS_PER_CHUNK} paragraphs per chunk and {PARAGRAPH_OVERLAP} paragraph overlap")
    else:
        print("Warning: No chunks were created from the document")
    
    return chunks

def remove_deleted_documents_from_database():
    print("Checking for documents removed from source...")
    # Get list of PDFs in destination folder
    current_pdfs = set(os.path.basename(f) for f in list_source_documents())
    
    # Get list of all unique source PDFs in the database
    try:
        all_db_results = collection.get(include=["metadatas"])
        if all_db_results and "metadatas" in all_db_results:
            db_source_pdfs = set(meta["source_pdf"] for meta in all_db_results["metadatas"] if "source_pdf" in meta)
            
            # Find PDFs in database that no longer exist in folder
            removed_pdfs = db_source_pdfs - current_pdfs
            
            if removed_pdfs:
                print(f"Found {len(removed_pdfs)} removed documents to clean from database:")
                for pdf in removed_pdfs:
                    print(f" - {pdf}")
                    # Get all chunks for this PDF
                    pdf_chunks = collection.get(
                        where={"source_pdf": pdf},
                        include=["metadatas", "ids"]
                    )
                    if pdf_chunks and "ids" in pdf_chunks and pdf_chunks["ids"]:
                        # Delete all chunks for this PDF
                        print(f"   Removing {len(pdf_chunks['ids'])} chunks from database")
                        collection.delete(ids=pdf_chunks["ids"])
            else:
                print("No removed documents detected in database")
    except Exception as e:
        print(f"Error checking for removed documents: {str(e)}")

# ============ MAIN LOGIC ============

def manage_and_chunk_pdfs():
    chunking_metadata = load_or_initialize_metadata(METADATA_JSON)
    print(f"Loaded chunking metadata for {len(chunking_metadata)} documents")
    
    pdf_metadata = {}
    if os.path.exists(DATASET_METADATA):
        with open(DATASET_METADATA, "r", encoding="utf-8") as f:
            pdf_metadata = json.load(f)
        print(f"Loaded dataset metadata for {len(pdf_metadata)} documents")
    
    pdf_files = list_source_documents()
    print(f"Found {len(pdf_files)} source documents in {FOLDER_PATH}")
    
    chunking_stats = []
    
    # Create a list of problematic files from previous runs
    # Filter out any corrupted entries (where value is not a dict)
    problematic_files = [k for k, v in chunking_metadata.items() 
                        if isinstance(v, dict) and v.get("error") is not None]
    if problematic_files:
        print(f"Found {len(problematic_files)} problematic files from previous runs")
    
    # Clean up any corrupted metadata entries
    valid_metadata = {k: v for k, v in chunking_metadata.items() if isinstance(v, dict)}
    if len(valid_metadata) != len(chunking_metadata):
        print(f"Warning: Removed {len(chunking_metadata) - len(valid_metadata)} corrupted metadata entries")
        chunking_metadata = valid_metadata
    
    for pdf_file in pdf_files:
        file_name = os.path.basename(pdf_file)
        file_mod_time = get_file_modification_time(pdf_file)
        doc_metadata = pdf_metadata.get(file_name, {})
        
        if file_name not in chunking_metadata:
            chunking_metadata[file_name] = {
                "found_date": str(datetime.now()),
                "last_modified_timestamp": file_mod_time,
                "last_url": doc_metadata.get("url", ""),
                "chunked_date": None,
                "last_certified_current": doc_metadata.get("certified_current", ""),
                "last_action_date": doc_metadata.get("last_action", ""),
                "error": None
            }
        
        record = chunking_metadata[file_name]
        needs_chunking = False
        chunking_reason = []
        
        # Check if this was a problematic file that we should retry
        if record.get("error") is not None:
            needs_chunking = True
            chunking_reason.append("retrying previously failed file")
        elif record.get("chunked_date") is None:
            needs_chunking = True
            chunking_reason.append("never chunked before")
        elif file_mod_time > record.get("last_modified_timestamp", 0):
            needs_chunking = True
            chunking_reason.append("file modified since last chunking")
        elif doc_metadata.get("url") != record.get("last_url", ""):
            needs_chunking = True
            chunking_reason.append("document URL changed")
        elif doc_metadata.get("certified_current") != record.get("last_certified_current", ""):
            needs_chunking = True
            chunking_reason.append("certified current date changed")
        elif doc_metadata.get("last_action") != record.get("last_action_date", ""):
            needs_chunking = True
            chunking_reason.append("last action date changed")
        
        if needs_chunking:
            print(f"Chunking {file_name} ({', '.join(chunking_reason)}) ...")
            try:
                try:
                    base_filename = os.path.basename(file_name)
                    existing_results = collection.get(
                        where={"source_pdf": base_filename},
                        include=["metadatas"]
                    )
                    if existing_results and len(existing_results["ids"]) > 0:
                        print(f"Found {len(existing_results['ids'])} existing chunks for {base_filename}. Removing them...")
                        collection.delete(ids=existing_results["ids"])
                except Exception as e:
                    print(f"No existing chunks found or error when checking: {str(e)}")
                
                # First convert PDF to text
                text_content = convert_pdf_to_text(pdf_file, save_txt=True)
                if not text_content or not text_content.strip():
                    raise ValueError("Failed to extract text from PDF")
                
                # Use paragraph-based chunking
                chunks_data = chunk_document(text_content, pdf_file)
                
                total_chunks = len(chunks_data)
                if total_chunks == 0:
                    raise ValueError("No chunks were generated from the document")
                    
                total_tokens = 0
                chunk_stats = []
                
                for i, (chunk_text, start_para_idx, end_para_idx, total_paragraphs) in enumerate(chunks_data):
                    try:
                        chunk_result = add_chunk_to_chroma(
                            chunk_text, 
                            file_name, 
                            doc_metadata, 
                            i, 
                            total_chunks,
                            start_para_idx=start_para_idx,
                            end_para_idx=end_para_idx,
                            total_paragraphs=total_paragraphs
                        )
                        total_tokens += chunk_result["token_count"]
                        chunk_stats.append(chunk_result)
                    except Exception as e:
                        print(f"Error adding chunk {i+1}/{total_chunks} from {file_name}: {str(e)}")
                
                doc_stats = {
                    "document": file_name,
                    "chunks_added": len(chunk_stats),
                    "total_tokens": total_tokens,
                    "avg_chunk_size": total_tokens / len(chunk_stats) if chunk_stats else 0,
                    "timestamp": datetime.now().isoformat()
                }
                chunking_stats.append(doc_stats)
                
                record["chunked_date"] = str(datetime.now())
                record["last_modified_timestamp"] = file_mod_time
                record["last_url"] = doc_metadata.get("url", "")
                record["last_certified_current"] = doc_metadata.get("certified_current", "")
                record["last_action_date"] = doc_metadata.get("last_action", "")
                record["chunks_count"] = total_chunks
                record["total_tokens"] = total_tokens
                record["avg_chunk_size"] = total_tokens / total_chunks if total_chunks else 0
                record["error"] = None  # Clear any previous errors
                
                save_metadata(chunking_metadata, METADATA_JSON)
                
            except Exception as e:
                error_message = str(e)
                print(f"Error chunking {file_name}: {error_message}")
                record["error"] = error_message
                save_metadata(chunking_metadata, METADATA_JSON)
        else:
            print(f"{file_name} is unchanged and already chunked; skipping.")
    
    save_metadata(chunking_metadata, METADATA_JSON)
    
    if chunking_stats:
        stats_file = str(Path(PIPELINE_DIR) / "chunking_stats.json")
        existing_stats = []
        if os.path.exists(stats_file):
            try:
                with open(stats_file, "r") as f:
                    existing_stats = json.load(f)
            except:
                existing_stats = []
        
        with open(stats_file, "w") as f:
            json.dump(existing_stats + chunking_stats, f, indent=2)
    
    chunked_count = sum(1 for record in chunking_metadata.values() if record.get("chunked_date") is not None)
    error_count = sum(1 for record in chunking_metadata.values() if record.get("error") is not None)
    print(f"Chunking complete. {chunked_count}/{len(chunking_metadata)} documents are chunked.")
    print(f"There are {error_count} documents with errors.")
    print(f"Metadata saved to {METADATA_JSON}")
    if chunking_stats:
        print(f"Chunking statistics appended to chunking_stats.json")
    
    return chunking_metadata

if __name__ == "__main__":
    # Test the embedding model before proceeding
    try:
        print(f"Testing OpenAI embedding model {EMBEDDING_MODEL}...")
        test_text = "Testing embedding model dimensions"
        embed_fn = OpenAIEmbeddingFunction(model_name=EMBEDDING_MODEL)
        test_embedding = embed_fn(test_text)
        print(f"Successfully generated test embedding with dimension: {len(test_embedding[0])}")
        # Continue only if no exceptions
    except Exception as e:
        print(f"ERROR: Failed to generate test embedding: {str(e)}")
        print("Please make sure you have set the OPENAI_API_KEY environment variable")
        import sys
        sys.exit(1)
    
    # First process new/updated documents
    manage_and_chunk_pdfs()
    
    # Then remove chunks for deleted documents
    remove_deleted_documents_from_database()
