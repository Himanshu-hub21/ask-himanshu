import os
import re
from typing import List, Dict
import PyPDF2
import pdfplumber
from pathlib import Path

class DocumentProcessor:
    """
    Process PDFs and create chunks for RAG
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, str]:
        """
        Extract text from PDF using pdfplumber (better quality)
        """
        text_by_page = {}
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text:
                        text_by_page[f"page_{page_num}"] = text
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            return {}
        
        return text_by_page
    
    def clean_text(self, text: str) -> str:
        """
        Clean extracted text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-]', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        return text.strip()
    
    def chunk_text(self, text: str, source: str, page: str = None) -> List[Dict]:
        """
        Split text into chunks with metadata
        """
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            # Create chunk with metadata
            chunk = {
                'text': chunk_text,
                'source': source,
                'page': page,
                'chunk_id': f"{source}_{page}_{i}",
                'word_count': len(chunk_words)
            }
            
            chunks.append(chunk)
        
        return chunks
    
    def detect_pii(self, text: str) -> bool:
        """
        Detect if text contains PII (phone, email, address)
        """
        # Phone patterns (Indian)
        phone_pattern = r'(\+91|0)?[6-9]\d{9}'
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        
        # Check for patterns
        has_phone = bool(re.search(phone_pattern, text))
        has_email = bool(re.search(email_pattern, text))
        
        return has_phone or has_email
    
    def process_document(self, pdf_path: str) -> List[Dict]:
        """
        Complete pipeline: extract → clean → chunk → detect PII
        """
        print(f"Processing: {pdf_path}")
        
        # Extract text
        text_by_page = self.extract_text_from_pdf(pdf_path)
        
        # Process each page
        all_chunks = []
        for page, text in text_by_page.items():
            # Clean text
            cleaned_text = self.clean_text(text)
            
            # Create chunks
            chunks = self.chunk_text(
                text=cleaned_text,
                source=os.path.basename(pdf_path),
                page=page
            )
            
            # Tag PII
            for chunk in chunks:
                chunk['has_pii'] = self.detect_pii(chunk['text'])
            
            all_chunks.extend(chunks)
        
        print(f"  → Created {len(all_chunks)} chunks")
        return all_chunks
