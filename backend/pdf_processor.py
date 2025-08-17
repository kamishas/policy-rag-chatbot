"""
PDF Processing Module

This module handles downloading and processing Air Force PDF documents.
It extracts "Roles and Responsibilities" sections and breaks them into chunks
for vector storage and search.

Key Concepts:
- PDF extraction: Converting PDF files to readable text
- Section extraction: Finding specific parts of documents
- Text chunking: Breaking large text into smaller, searchable pieces
- Overlap: Ensuring context isn't lost between chunks
"""

import requests
import io
from pdfminer.high_level import extract_text
from typing import List, Dict, Optional
import re
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AirForcePDFProcessor:
    """
    Processor for Air Force PDF documents.
    
    This class downloads PDFs from URLs, extracts the "Roles and Responsibilities"
    sections, and breaks them into chunks suitable for vector search.
    """
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        """
        Initialize the PDF processor.
        
        Args:
            chunk_size: Number of words per chunk (500 is a good balance)
            overlap: Number of words to overlap between chunks (preserves context)
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # Request headers to avoid being blocked by servers
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        logger.info(f"📄 PDF Processor initialized")
        logger.info(f"Chunk size: {chunk_size} words")
        logger.info(f"Overlap: {overlap} words")
    
    async def process_pdf_from_url(self, pdf_url: str) -> List[Dict]:
        """
        Download and process a PDF from a URL.
        
        This is the main method that:
        1. Downloads the PDF
        2. Extracts text
        3. Finds the roles & responsibilities section
        4. Breaks it into chunks
        5. Returns structured document data
        
        Args:
            pdf_url: URL of the PDF to process
            
        Returns:
            List of document chunks, each containing:
            - text: The actual text content
            - source: Original PDF URL
            - chunk_id: Number identifying this chunk
            - doc_type: Type of document (AFI, AFMAN, etc.)
            - section: Always "roles_responsibilities"
        """
        try:
            logger.info(f"🔄 Processing PDF: {pdf_url}")
            
            # Step 1: Download the PDF
            pdf_content = await self._download_pdf(pdf_url)
            if not pdf_content:
                return []
            
            # Step 2: Extract text from PDF
            full_text = self._extract_text_from_pdf(pdf_content)
            if not full_text:
                logger.warning(f"❌ No text extracted from {pdf_url}")
                return []
            
            # Step 3: Find the roles & responsibilities section
            roles_text = self._extract_roles_section(full_text)
            if not roles_text:
                logger.warning(f"❌ No roles section found in {pdf_url}")
                return []
            
            # Step 4: Break into chunks
            chunks = self._chunk_text(roles_text)
            if not chunks:
                logger.warning(f"❌ No valid chunks created from {pdf_url}")
                return []
            
            # Step 5: Create structured document objects
            documents = []
            doc_type = self._extract_doc_type(pdf_url)
            
            for i, chunk in enumerate(chunks):
                documents.append({
                    "text": chunk,
                    "source": pdf_url,
                    "chunk_id": i,
                    "doc_type": doc_type,
                    "section": "roles_responsibilities",
                    "total_chunks": len(chunks)
                })
            
            logger.info(f"✅ Successfully processed {len(documents)} chunks from {pdf_url}")
            return documents
            
        except Exception as e:
            logger.error(f"❌ Error processing PDF {pdf_url}: {str(e)}")
            return []
    
    async def _download_pdf(self, pdf_url: str) -> Optional[bytes]:
        """
        Download PDF content from URL.
        
        Args:
            pdf_url: URL to download from
            
        Returns:
            PDF content as bytes, or None if failed
        """
        try:
            logger.info(f"⬇️ Downloading: {pdf_url}")
            
            # Make HTTP request with timeout and retries
            response = requests.get(
                pdf_url, 
                headers=self.headers,
                timeout=30,  # 30 second timeout
                stream=True  # Download in chunks for large files
            )
            response.raise_for_status()  # Raise exception for HTTP errors
            
            # Check if it's actually a PDF
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' not in content_type:
                logger.warning(f"⚠️ File may not be PDF. Content-Type: {content_type}")
            
            pdf_content = response.content
            logger.info(f"✅ Downloaded {len(pdf_content)} bytes")
            return pdf_content
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to download {pdf_url}: {str(e)}")
            return None
    
    def _extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """
        Extract text from PDF bytes using pdfminer.
        
        pdfminer is chosen because it's very reliable for government documents
        and handles complex formatting better than alternatives.
        
        Args:
            pdf_content: PDF file as bytes
            
        Returns:
            Extracted text as string
        """
        try:
            # Create a file-like object from bytes
            pdf_file = io.BytesIO(pdf_content)
            
            # Extract text using pdfminer
            # This handles complex PDF layouts well
            text = extract_text(pdf_file)
            
            # Clean up the text
            if text:
                # Remove excessive whitespace but preserve structure
                text = re.sub(r'\n\s*\n', '\n\n', text)  # Multiple newlines to double
                text = re.sub(r' +', ' ', text)  # Multiple spaces to single
                text = text.strip()
            
            logger.info(f"📝 Extracted {len(text)} characters of text")
            return text
            
        except Exception as e:
            logger.error(f"❌ Failed to extract text from PDF: {str(e)}")
            return ""
    
    def _extract_roles_section(self, text: str) -> str:
        """
        Extract the "Roles and Responsibilities" section from Air Force documents.
        
        Air Force documents follow specific formatting patterns:
        - Usually Chapter 2 contains roles and responsibilities
        - Section starts with "ROLES AND RESPONSIBILITIES"
        - Ends with next chapter or reference section
        
        Args:
            text: Full document text
            
        Returns:
            Just the roles and responsibilities section
        """
        if not text:
            return ""
        
        # Define patterns to find the start of the section
        start_patterns = [
            r"ROLES AND RESPONSIBILITIES",
            r"Chapter 2.*ROLES AND RESPONSIBILITIES", 
            r"2\.\s*ROLES AND RESPONSIBILITIES",
            r"CHAPTER 2.*ROLES AND RESPONSIBILITIES"
        ]
        
        start_index = -1
        
        # Try each pattern to find the section start
        for pattern in start_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                start_index = match.start()
                logger.info(f"📍 Found roles section using pattern: {pattern}")
                break
        
        if start_index == -1:
            logger.warning("❌ Could not find 'ROLES AND RESPONSIBILITIES' section")
            return ""
        
        # Sometimes the first match is in the table of contents
        # Look for a second occurrence which is usually the actual section
        second_match = text.find("ROLES AND RESPONSIBILITIES", start_index + 1)
        if second_match != -1 and second_match - start_index < 50000:  # Increased range for longer documents
            start_index = second_match
            logger.info("📍 Using second occurrence (skipping table of contents)")
        
        # Also check for third occurrence in case there are multiple TOC references
        third_match = text.find("ROLES AND RESPONSIBILITIES", second_match + 1) if second_match != -1 else -1
        if third_match != -1 and third_match - start_index < 50000:
            # Check if this third occurrence seems more substantial (more content after it)
            remaining_after_third = len(text) - third_match
            remaining_after_second = len(text) - second_match if second_match != -1 else 0
            if remaining_after_third > 1000 and remaining_after_third > remaining_after_second * 0.5:
                start_index = third_match
                logger.info("📍 Using third occurrence (more substantial content found)")
        
        # Define patterns that typically end the roles section
        # Based on user feedback: Chapter 2 (roles) ends at Chapter 3
        end_patterns = [
    r"\n\s*Chapter 3\b",  # Roles section ends at Chapter 3 (start of line)
    r"\n\s*CHAPTER 3\b", 
    r"\n\s*3\.\s+[A-Z][A-Z\s]+",  # Chapter 3 with capitalized title
    r"\n\s*Chapter 4\b",  # Fallback if no Chapter 3 (start of line)
    r"\n\s*CHAPTER 4\b", 
    r"\n\s*4\.\s+[A-Z][A-Z\s]+",  # Chapter 4 with capitalized title
    r"\n\s*REFERENCES\s*\n",
    r"\n\s*BIBLIOGRAPHY\s*\n",
    r"\n\s*GLOSSARY\s*\n", 
    r"\n\s*ATTACHMENTS\s*\n",
    r"\n\s*APPENDIX\s*\n",
    r"\n\s*INDEX\s*\n"
]
        
        # Find the earliest end marker
        end_index = len(text)
        end_pattern_found = None
        
        for pattern in end_patterns:
            match = re.search(pattern, text[start_index:], re.IGNORECASE)
            if match:
                potential_end = start_index + match.start()
                if potential_end < end_index:
                    end_index = potential_end
                    end_pattern_found = pattern
        
        if end_pattern_found:
            logger.info(f"📍 Section ends at pattern: {end_pattern_found}")
        
        # Extract the section
        roles_text = text[start_index:end_index].strip()
        
        # Validate we got something meaningful
        if len(roles_text) < 100:
            logger.warning(f"⚠️ Roles section very short ({len(roles_text)} chars), might be incorrect")
            return ""
        
        logger.info(f"✅ Extracted roles section: {len(roles_text)} characters")
        return roles_text
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Break text into personnel-based chunks for better search.
        
        Each chunk represents one personnel role (e.g., 2.1, 2.2, 2.12)
        This ensures each role is individually searchable and maintains
        complete context for that specific position.
        
        Args:
            text: Text to chunk (roles and responsibilities section)
            
        Returns:
            List of personnel role chunks
        """
        if not text:
            return []
        
        chunks = []
        
        # Strategy: Split text on main personnel roles (2.X.) but NOT sub-responsibilities (2.X.Y.)
        # Only match roles like 2.1, 2.12, etc. (NOT 2.1.1, 2.12.1, etc.)
        
        # Find all main personnel roles (2.X where X has no dots)
        main_role_pattern = r'\n\s*2\.(\d+)\.\s*'
        
        # Find all main role positions
        main_roles = []
        for match in re.finditer(main_role_pattern, text):
            role_number = match.group(1)
            start_pos = match.start()
            main_roles.append((role_number, start_pos))
        
        logger.info(f"📋 Found {len(main_roles)} main personnel roles: {[f'2.{r[0]}' for r in main_roles[:5]]}")
        
        # Create chunks by extracting from each main role to the next main role
        for i, (role_number, start_pos) in enumerate(main_roles):
            # Determine end position (start of next main role or end of text)
            if i + 1 < len(main_roles):
                end_pos = main_roles[i + 1][1]  # Start of next main role
            else:
                end_pos = len(text)  # End of text
            
            # Extract complete section for this personnel
            section_text = text[start_pos:end_pos].strip()
            
            if len(section_text) > 100:  # Only keep substantial chunks
                chunks.append(section_text)
                
                # Count sub-responsibilities in this chunk
                sub_responsibilities = re.findall(rf'2\.{role_number}\.(\d+)\.', section_text)
                logger.info(f"📋 Created chunk for role 2.{role_number}: {len(section_text)} chars, {len(sub_responsibilities)} sub-responsibilities")
                
                # Log first few sub-responsibilities for verification
                if sub_responsibilities:
                    sub_list = [f"2.{role_number}.{sub}" for sub in sub_responsibilities[:3]]
                    more_text = f" (+{len(sub_responsibilities)-3} more)" if len(sub_responsibilities) > 3 else ""
                    logger.info(f"    📝 Sub-responsibilities: {', '.join(sub_list)}{more_text}")
            else:
                logger.warning(f"⚠️ Skipping small chunk for role 2.{role_number}: {len(section_text)} chars")
        
        # If no matches found with patterns, fall back to simple approach
        if not chunks:
            logger.warning("⚠️ No personnel roles found, using fallback chunking")
            # Try to split on any 2.X pattern
            parts = re.split(r'\n\s*2\.(\d+)', text)
            for i in range(1, len(parts), 2):
                if i + 1 < len(parts):
                    role_num = parts[i]
                    content = parts[i + 1]
                    chunk = f"2.{role_num}{content}".strip()
                    if len(chunk) > 50:
                        chunks.append(chunk)
        
        logger.info(f"📄 Created {len(chunks)} personnel-based chunks")
        
        # Log examples of created chunks for debugging
        if chunks:
            for i, chunk in enumerate(chunks[:3]):  # Show first 3 examples
                first_line = chunk.split('\n')[0][:100]
                logger.info(f"📋 Chunk {i+1}: {first_line}...")
        
        return chunks
    
    def _extract_doc_type(self, url: str) -> str:
        """
        Determine document type from URL.
        
        Air Force documents have different types:
        - AFI: Air Force Instruction (detailed procedures)
        - AFMAN: Air Force Manual (how-to guides)
        - AFPD: Air Force Policy Directive (high-level policy)
        - DAFI: Department of Air Force Instruction
        
        Args:
            url: Document URL
            
        Returns:
            Document type abbreviation
        """
        url_lower = url.lower()
        
        if 'afi' in url_lower and 'dafi' not in url_lower:
            return "AFI"
        elif 'afman' in url_lower:
            return "AFMAN"
        elif 'afpd' in url_lower:
            return "AFPD"
        elif 'dafi' in url_lower:
            return "DAFI"
        elif 'afpam' in url_lower:
            return "AFPAM"
        else:
            return "UNKNOWN"
    
    def get_processor_stats(self) -> dict:
        """
        Get information about processor configuration.
        
        Returns:
            Dictionary with processor settings
        """
        return {
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "description": "Air Force PDF processor for roles and responsibilities extraction"
        }


# Global processor instance
pdf_processor = AirForcePDFProcessor()

"""
Usage Example:

# Process a single PDF
documents = await pdf_processor.process_pdf_from_url(
    "https://static.e-publishing.af.mil/production/1/af_a3/publication/afi10-2402/afi10-2402.pdf"
)

# Each document in the list looks like:
{
    "text": "2.3.1. Assistant Secretary of the Air Force for Acquisition...",
    "source": "https://static.e-publishing.af.mil/.../afi10-2402.pdf",
    "chunk_id": 0,
    "doc_type": "AFI",
    "section": "roles_responsibilities",
    "total_chunks": 5
}
"""