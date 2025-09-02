import os
import re
from typing import List, Dict
from pathlib import Path
import PyPDF2
from docx import Document as DocxDocument
from django.conf import settings
from .models import Document, DocumentChunk
from .rag_utils import EmbeddingManager


class DocumentProcessor:
    """Handles document text extraction and chunking"""
    
    def __init__(self):
        self.embedding_manager = EmbeddingManager()
    
    def extract_text(self, document: Document) -> str:
        """Extract text from document based on file type"""
        file_path = document.file.path
        
        try:
            if document.file_type == 'pdf':
                return self._extract_pdf_text(file_path)
            elif document.file_type == 'docx':
                return self._extract_docx_text(file_path)
            elif document.file_type == 'txt':
                return self._extract_txt_text(file_path)
            else:
                raise ValueError(f"Unsupported file type: {document.file_type}")
        except Exception as e:
            raise Exception(f"Error extracting text from {document.title}: {str(e)}")
    
    def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            raise Exception(f"Error reading PDF file: {str(e)}")
        
        return text.strip()
    
    def _extract_docx_text(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = DocxDocument(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
        except Exception as e:
            raise Exception(f"Error reading DOCX file: {str(e)}")
        
        return text.strip()
    
    def _extract_txt_text(self, file_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        text = file.read()
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise Exception("Could not decode text file with any supported encoding")
        except Exception as e:
            raise Exception(f"Error reading TXT file: {str(e)}")
        
        return text.strip()
    
    def chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """Split text into chunks with overlap"""
        if chunk_size is None:
            chunk_size = settings.CHUNK_SIZE
        if overlap is None:
            overlap = settings.CHUNK_OVERLAP
        
        if not text:
            return []
        
        # Clean the text
        text = self._clean_text(text)
        
        # Split into sentences for better chunking
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = ""
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed chunk size, finalize current chunk
            if current_length + sentence_length > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk, overlap)
                current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                current_length = len(current_chunk)
            else:
                # Add sentence to current chunk
                current_chunk += " " + sentence if current_chunk else sentence
                current_length += sentence_length
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if len(chunk.strip()) > 50]  # Filter out very short chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\']+', ' ', text)
        
        # Remove multiple consecutive periods
        text = re.sub(r'\.{3,}', '...', text)
        
        return text.strip()
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting - can be improved with NLTK or spaCy
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap_text(self, text: str, overlap_length: int) -> str:
        """Get overlap text from the end of a chunk"""
        if len(text) <= overlap_length:
            return text
        
        # Try to get overlap at sentence boundaries
        sentences = self._split_into_sentences(text)
        overlap_text = ""
        
        for sentence in reversed(sentences):
            if len(overlap_text + sentence) <= overlap_length:
                overlap_text = sentence + " " + overlap_text
            else:
                break
        
        return overlap_text.strip()
    
    def process_document(self, document: Document) -> Dict:
        """Process a document: extract text, create chunks, and generate embeddings"""
        try:
            # Extract text
            text = self.extract_text(document)
            
            if not text:
                raise Exception("No text could be extracted from the document")
            
            # Create chunks
            chunks = self.chunk_text(text)
            
            if not chunks:
                raise Exception("No valid chunks could be created from the document")
            
            # Generate embeddings and save chunks
            chunk_count = 0
            for i, chunk_text in enumerate(chunks):
                try:
                    # Generate embedding
                    embedding = self.embedding_manager.generate_embedding(chunk_text)
                    
                    # Create chunk record
                    chunk = DocumentChunk.objects.create(
                        document=document,
                        chunk_index=i,
                        text=chunk_text,
                        embedding_vector=embedding.tolist()  # Store as list
                    )
                    chunk_count += 1
                    
                except Exception as e:
                    print(f"Error processing chunk {i} for document {document.title}: {str(e)}")
                    continue
            
            # Update document
            document.total_chunks = chunk_count
            document.processed = True
            document.save()
            
            return {
                'success': True,
                'chunks_created': chunk_count,
                'text_length': len(text),
                'message': f'Successfully processed {document.title}'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f'Failed to process {document.title}'
            }


class DocumentManager:
    """Manages document operations"""
    
    def __init__(self):
        self.processor = DocumentProcessor()
    
    def upload_document(self, file, title: str, file_type: str, user) -> Document:
        """Upload and create a new document"""
        # Determine file type if not provided
        if not file_type:
            file_extension = file.name.lower().split('.')[-1]
            file_type = file_extension if file_extension in ['pdf', 'docx', 'txt'] else 'txt'
        
        # Create document
        document = Document.objects.create(
            title=title or file.name,
            file=file,
            file_type=file_type,
            uploaded_by=user
        )
        
        return document
    
    def delete_document(self, document: Document):
        """Delete a document and its associated chunks"""
        try:
            # Remove from vector store
            from .rag_utils import rag_pipeline
            rag_pipeline.remove_document(str(document.id))
            
            # Delete document (will cascade to chunks)
            document.delete()
            
            return {'success': True, 'message': 'Document deleted successfully'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_document_stats(self, document: Document) -> Dict:
        """Get statistics for a document"""
        stats = {
            'total_chunks': document.total_chunks,
            'processed': document.processed,
            'file_size': document.file.size if document.file else 0,
        }
        
        if document.processed:
            # Calculate total text length
            total_text_length = sum(
                len(chunk.text) for chunk in document.chunks.all()
            )
            stats['total_text_length'] = total_text_length
            stats['avg_chunk_length'] = total_text_length / document.total_chunks if document.total_chunks > 0 else 0
        
        return stats
