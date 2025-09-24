import os
import json
import faiss
import numpy as np
from typing import List, Dict, Tuple, Optional
import google.generativeai as genai
from django.conf import settings
from .models import Document, DocumentChunk

# Conditional import for Pinecone
if getattr(settings, 'USE_PINECONE', True):
    try:
        from pinecone import Pinecone
        PINECONE_AVAILABLE = True
    except ImportError:
        PINECONE_AVAILABLE = False
        print("Warning: Pinecone not available, falling back to FAISS")
else:
    PINECONE_AVAILABLE = False


class EmbeddingManager:
    """Manages text embeddings using Google's Embedding API"""
    
    def __init__(self):
        if settings.GEMINI_API_KEY:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self._dimension = 768  # Google's text-embedding-004 model dimension
        else:
            raise ValueError("GEMINI_API_KEY is required for embeddings")
    
    @property
    def dimension(self):
        """Get embedding dimension"""
        return self._dimension
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text using Google's API"""
        try:
            # Clean and validate text
            if not text or not isinstance(text, str):
                print("Warning: Empty or invalid text for embedding")
                return np.zeros(self._dimension, dtype=np.float32)
            
            # Clean text for Google API
            cleaned_text = self._clean_text_for_api(text)
            
            # Additional validation after cleaning
            if len(cleaned_text) < 3:
                print(f"Warning: Text too short after cleaning: '{cleaned_text}'")
                return np.zeros(self._dimension, dtype=np.float32)
            
            # Add retry logic with exponential backoff
            import time
            max_retries = 3
            base_delay = 1.0
            
            for attempt in range(max_retries):
                try:
                    # Add rate limiting for production
                    if 'RENDER' in os.environ and attempt > 0:
                        time.sleep(0.5)  # Rate limit in production
                    
                    response = genai.embed_content(
                        model="models/text-embedding-004",
                        content=cleaned_text,
                        task_type="retrieval_document"  # Specify task type
                    )
                    
                    # Validate response structure
                    if not response or not isinstance(response, dict):
                        print(f"Warning: Invalid response type from Google API: {type(response)}")
                        if attempt < max_retries - 1:
                            time.sleep(base_delay * (2 ** attempt))
                            continue
                        return np.zeros(self._dimension, dtype=np.float32)
                    
                    if "embedding" not in response:
                        print(f"Warning: No 'embedding' key in response: {list(response.keys()) if isinstance(response, dict) else response}")
                        if attempt < max_retries - 1:
                            time.sleep(base_delay * (2 ** attempt))
                            continue
                        return np.zeros(self._dimension, dtype=np.float32)
                    
                    embedding = response["embedding"]
                    
                    # Validate embedding structure
                    if not isinstance(embedding, list) or len(embedding) != self._dimension:
                        print(f"Warning: Invalid embedding format or dimension: type={type(embedding)}, len={len(embedding) if hasattr(embedding, '__len__') else 'N/A'}")
                        if attempt < max_retries - 1:
                            time.sleep(base_delay * (2 ** attempt))
                            continue
                        return np.zeros(self._dimension, dtype=np.float32)
                    
                    # Convert to numpy array and validate values
                    embedding_array = np.array(embedding, dtype=np.float32)
                    if np.any(np.isnan(embedding_array)) or np.any(np.isinf(embedding_array)):
                        print("Warning: Invalid embedding values (NaN or Inf)")
                        if attempt < max_retries - 1:
                            time.sleep(base_delay * (2 ** attempt))
                            continue
                        return np.zeros(self._dimension, dtype=np.float32)
                    
                    return embedding_array
                    
                except Exception as api_error:
                    error_str = str(api_error).lower()
                    print(f"API attempt {attempt + 1}/{max_retries} failed: {api_error}")
                    
                    # Check if this is a retryable error
                    retryable_errors = [
                        'timeout', 'rate limit', 'quota', 'temporary', 
                        'service unavailable', 'internal error', 'network'
                    ]
                    
                    is_retryable = any(err in error_str for err in retryable_errors)
                    
                    if attempt < max_retries - 1 and is_retryable:
                        retry_delay = base_delay * (2 ** attempt)
                        print(f"Retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        print(f"Not retrying error (final attempt or non-retryable): {api_error}")
                        break
            
            # If we get here, all attempts failed
            return np.zeros(self._dimension, dtype=np.float32)
            
        except Exception as e:
            print(f"Error generating embedding: {e}")
            print(f"Text length: {len(text) if text else 0}")
            print(f"Text preview: {text[:100] if text else 'None'}...")
            # Return zero vector as fallback
            return np.zeros(self._dimension, dtype=np.float32)
    
    def _clean_text_for_api(self, text: str) -> str:
        """Clean text for Google API compatibility"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Limit length (Google API has stricter limits in production)
        max_length = 8000  # More conservative limit for production environments
        if len(text) > max_length:
            text = text[:max_length]
        
        # Remove problematic characters that might cause API issues
        # Only keep basic ASCII printable characters and common whitespace
        import string
        import re
        
        # First pass: only keep safe characters
        allowed_chars = set(string.ascii_letters + string.digits + string.punctuation + ' \n\t')
        cleaned = ""
        for char in text:
            if char in allowed_chars:
                cleaned += char
            elif char.isspace():
                cleaned += " "  # Replace any other whitespace with regular space
        
        # Second pass: remove control characters and normalize
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', ' ', cleaned)  # Remove control chars
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Normalize whitespace
        cleaned = re.sub(r'[^\x20-\x7E\s]', ' ', cleaned)  # Remove non-ASCII printable
        
        # Final cleanup
        cleaned = cleaned.strip()
        
        # Ensure minimum length
        if len(cleaned) < 3:
            return "placeholder text"  # Fallback for very short text
        
        return cleaned
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts"""
        embeddings = []
        for text in texts:
            embedding = self.generate_embedding(text)
            embeddings.append(embedding)
        return np.array(embeddings)


class FAISSVectorStore:
    """Manages FAISS vector database operations"""
    
    def __init__(self):
        self.embedding_manager = EmbeddingManager()
        self.dimension = self.embedding_manager.dimension
        self.index = None
        self.chunk_metadata = []
        self.index_path = settings.FAISS_INDEX_PATH
        self.metadata_path = self.index_path / 'metadata.json'
        
        # Load existing index if available
        self.load_index()
    
    def load_index(self):
        """Load FAISS index and metadata from disk"""
        index_file = self.index_path / 'index.faiss'
        if index_file.exists() and self.metadata_path.exists():
            try:
                self.index = faiss.read_index(str(index_file))
                with open(self.metadata_path, 'r') as f:
                    self.chunk_metadata = json.load(f)
                print(f"Loaded FAISS index with {self.index.ntotal} vectors")
            except Exception as e:
                print(f"Error loading FAISS index: {e}")
                self._initialize_empty_index()
        else:
            self._initialize_empty_index()
    
    def _initialize_empty_index(self):
        """Initialize empty FAISS index"""
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product similarity
        self.chunk_metadata = []
    
    def save_index(self):
        """Save FAISS index and metadata to disk"""
        try:
            os.makedirs(self.index_path, exist_ok=True)
            index_file = self.index_path / 'index.faiss'
            faiss.write_index(self.index, str(index_file))
            
            with open(self.metadata_path, 'w') as f:
                json.dump(self.chunk_metadata, f)
            print(f"Saved FAISS index with {self.index.ntotal} vectors")
        except Exception as e:
            print(f"Error saving FAISS index: {e}")
    
    def add_chunk(self, chunk: DocumentChunk):
        """Add a document chunk to the vector store"""
        if not chunk.embedding_vector:
            return False
        
        try:
            # Convert embedding to numpy array
            embedding = np.array(chunk.embedding_vector, dtype=np.float32).reshape(1, -1)
            
            # Normalize for cosine similarity (with inner product)
            faiss.normalize_L2(embedding)
            
            # Add to FAISS index
            self.index.add(embedding)
            
            # Add metadata
            metadata = {
                'chunk_id': str(chunk.id),
                'document_id': str(chunk.document.id),
                'document_title': chunk.document.title,
                'chunk_index': chunk.chunk_index,
                'text': chunk.text[:500]  # Store first 500 chars for quick reference
            }
            self.chunk_metadata.append(metadata)
            
            return True
        except Exception as e:
            print(f"Error adding chunk to vector store: {e}")
            return False
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar chunks"""
        if self.index.ntotal == 0:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_manager.generate_embedding(query)
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(query_embedding)
            
            # Search FAISS index
            scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
            
            # Prepare results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.chunk_metadata):
                    result = self.chunk_metadata[idx].copy()
                    result['similarity_score'] = float(score)
                    results.append(result)
            
            return results
        except Exception as e:
            print(f"Error searching vector store: {e}")
            return []
    
    def remove_document_chunks(self, document_id: str):
        """Remove all chunks belonging to a document"""
        # Note: FAISS doesn't support individual vector deletion easily
        # For now, we'll rebuild the index without the document's chunks
        try:
            # Filter out chunks from the document
            new_metadata = [
                meta for meta in self.chunk_metadata 
                if meta['document_id'] != document_id
            ]
            
            if len(new_metadata) == len(self.chunk_metadata):
                return  # No chunks to remove
            
            # Rebuild index
            self._initialize_empty_index()
            
            # Re-add remaining chunks
            for meta in new_metadata:
                try:
                    chunk = DocumentChunk.objects.get(id=meta['chunk_id'])
                    if chunk.embedding_vector:
                        self.add_chunk(chunk)
                except DocumentChunk.DoesNotExist:
                    continue
            
            self.save_index()
            print(f"Removed chunks for document {document_id}")
        except Exception as e:
            print(f"Error removing document chunks: {e}")
    
    def rebuild_from_database(self):
        """Rebuild the vector store from all processed documents in the database"""
        try:
            print("Rebuilding vector store from database...")
            
            # Initialize empty index
            self._initialize_empty_index()
            
            # Get all processed documents with chunks
            processed_documents = Document.objects.filter(processed=True, total_chunks__gt=0)
            total_chunks_added = 0
            
            for document in processed_documents:
                chunks = DocumentChunk.objects.filter(document=document).order_by('chunk_index')
                for chunk in chunks:
                    if chunk.embedding_vector and self.add_chunk(chunk):
                        total_chunks_added += 1
            
            # Save the rebuilt index
            if total_chunks_added > 0:
                self.save_index()
                print(f"Rebuilt vector store with {total_chunks_added} chunks from {processed_documents.count()} documents")
            else:
                print("No chunks found to rebuild vector store")
                
            return total_chunks_added
            
        except Exception as e:
            print(f"Error rebuilding vector store from database: {e}")
            return 0
    
    def clear_all_data(self):
        """Clear all vectors and metadata from the vector store"""
        try:
            print("Clearing all vector store data...")
            
            # Initialize empty index
            self._initialize_empty_index()
            
            # Save the empty index
            self.save_index()
            
            print("Successfully cleared all vector store data")
            return True
            
        except Exception as e:
            print(f"Error clearing vector store data: {e}")
            return False


class PineconeVectorStore:
    """Manages Pinecone vector database operations"""
    
    def __init__(self):
        self.embedding_manager = EmbeddingManager()
        self.dimension = self.embedding_manager.dimension
        
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone client not available. Install with: pip install pinecone-client")
        
        # Initialize Pinecone
        try:
            self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            self.index = self.pc.Index(settings.PINECONE_INDEX_NAME)
            print(f"Connected to Pinecone index: {settings.PINECONE_INDEX_NAME}")
        except Exception as e:
            print(f"Error connecting to Pinecone: {e}")
            raise
    
    def add_chunk(self, chunk: DocumentChunk):
        """Add a document chunk to the Pinecone vector store"""
        if not chunk.embedding_vector:
            return False
        
        try:
            # Prepare metadata
            metadata = {
                'chunk_id': str(chunk.id),
                'document_id': str(chunk.document.id),
                'document_title': chunk.document.title,
                'chunk_index': chunk.chunk_index,
                'text': chunk.text[:1000]  # Store first 1000 chars
            }
            
            # Upsert to Pinecone
            self.index.upsert(
                vectors=[{
                    'id': str(chunk.id),
                    'values': chunk.embedding_vector,
                    'metadata': metadata
                }]
            )
            
            return True
        except Exception as e:
            print(f"Error adding chunk to Pinecone: {e}")
            return False
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar chunks in Pinecone"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_manager.generate_embedding(query).tolist()
            
            # Search Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            # Format results
            formatted_results = []
            for match in results['matches']:
                result = match['metadata'].copy()
                result['similarity_score'] = float(match['score'])
                formatted_results.append(result)
            
            return formatted_results
        except Exception as e:
            print(f"Error searching Pinecone: {e}")
            return []
    
    def remove_document_chunks(self, document_id: str):
        """Remove all chunks belonging to a document from Pinecone"""
        try:
            # Query for all chunks of this document
            chunks = DocumentChunk.objects.filter(document_id=document_id)
            
            # Delete from Pinecone
            ids_to_delete = [str(chunk.id) for chunk in chunks]
            if ids_to_delete:
                self.index.delete(ids=ids_to_delete)
                print(f"Removed {len(ids_to_delete)} chunks for document {document_id}")
        except Exception as e:
            print(f"Error removing document chunks from Pinecone: {e}")
    
    def rebuild_from_database(self):
        """Rebuild the Pinecone index from all processed documents in the database"""
        try:
            print("Rebuilding Pinecone index from database...")
            
            # Clear existing data (with proper error handling)
            self.clear_all_data()
            
            # Get all processed documents with chunks
            processed_documents = Document.objects.filter(processed=True, total_chunks__gt=0)
            total_chunks_added = 0
            
            if not processed_documents.exists():
                print("No processed documents found to rebuild index")
                return 0
            
            # Batch upsert for efficiency
            batch_size = 50  # Reduced for API rate limits
            vectors_to_upsert = []
            
            for document in processed_documents:
                chunks = DocumentChunk.objects.filter(document=document).order_by('chunk_index')
                for chunk in chunks:
                    # Regenerate embedding with Google API (768 dimensions)
                    try:
                        new_embedding = self.embedding_manager.generate_embedding(chunk.text)
                        
                        # Skip if embedding generation failed
                        if np.all(new_embedding == 0):
                            print(f"Skipping chunk {chunk.id}: embedding generation failed")
                            continue
                        
                        # Update the chunk with new embedding
                        chunk.embedding_vector = new_embedding.tolist()
                        chunk.save()
                        
                        metadata = {
                            'chunk_id': str(chunk.id),
                            'document_id': str(chunk.document.id),
                            'document_title': chunk.document.title,
                            'chunk_index': chunk.chunk_index,
                            'text': chunk.text[:1000]
                        }
                        
                        vectors_to_upsert.append({
                            'id': str(chunk.id),
                            'values': new_embedding.tolist(),
                            'metadata': metadata
                        })
                        
                        # Batch upsert when batch is full
                        if len(vectors_to_upsert) >= batch_size:
                            try:
                                self.index.upsert(vectors=vectors_to_upsert)
                                total_chunks_added += len(vectors_to_upsert)
                                vectors_to_upsert = []
                                print(f"Processed {total_chunks_added} chunks...")
                            except Exception as upsert_error:
                                print(f"Error upserting batch: {upsert_error}")
                                # Continue with next batch
                                vectors_to_upsert = []
                    
                    except Exception as e:
                        print(f"Error processing chunk {chunk.id}: {e}")
                        continue
            
            # Upsert remaining vectors
            if vectors_to_upsert:
                try:
                    self.index.upsert(vectors=vectors_to_upsert)
                    total_chunks_added += len(vectors_to_upsert)
                except Exception as final_upsert_error:
                    print(f"Error upserting final batch: {final_upsert_error}")
            
            print(f"Rebuilt Pinecone index with {total_chunks_added} chunks from {processed_documents.count()} documents")
            return total_chunks_added
            
        except Exception as e:
            print(f"Error rebuilding Pinecone index from database: {e}")
            return 0
    
    def clear_all_data(self):
        """Clear all vectors from the Pinecone index"""
        try:
            print("Clearing all Pinecone index data...")
            
            # Try to get index stats first to check if it exists and has data
            try:
                stats = self.index.describe_index_stats()
                total_vectors = stats.get('total_vector_count', 0)
                print(f"Index currently has {total_vectors} vectors")
                
                if total_vectors > 0:
                    # Delete all vectors in the index
                    self.index.delete(delete_all=True)
                    print("Successfully cleared all Pinecone index data")
                else:
                    print("Index is already empty")
                    
            except Exception as stats_error:
                print(f"Could not get index stats: {stats_error}")
                # Try to delete anyway, but catch the error
                try:
                    self.index.delete(delete_all=True)
                    print("Successfully cleared all Pinecone index data")
                except Exception as delete_error:
                    if "not found" in str(delete_error).lower() or "404" in str(delete_error):
                        print("Index/namespace is already empty or doesn't exist")
                    else:
                        raise delete_error
            
            return True
            
        except Exception as e:
            print(f"Error clearing Pinecone index data: {e}")
            return False
    
    def save_index(self):
        """No-op for Pinecone (auto-saved)"""
        pass


class GeminiLLM:
    """Manages Gemini LLM interactions"""
    
    def __init__(self):
        if settings.GEMINI_API_KEY:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            # Use the newer model name
            self.model = genai.GenerativeModel('gemini-2.5-flash')
        else:
            self.model = None
            print("Warning: GEMINI_API_KEY not set in settings")
    
    def generate_response(self, query: str, context_chunks: List[Dict]) -> Tuple[str, List[Dict]]:
        """Generate response using Gemini with RAG context"""
        if not self.model:
            return "Error: Gemini API not configured", []
        
        try:
            # Prepare context from chunks
            context_texts = []
            sources = []
            
            for i, chunk in enumerate(context_chunks):
                context_texts.append(f"[Document {i+1}: {chunk['document_title']}]\n{chunk['text']}")
                sources.append({
                    'document_id': chunk['document_id'],
                    'document_title': chunk['document_title'],
                    'chunk_index': chunk['chunk_index'],
                    'similarity_score': chunk.get('similarity_score', 0)
                })
            
            context = "\n\n".join(context_texts)
            
            # Create the prompt
            prompt = f"""You are a helpful AI assistant that answers questions based on the provided documents. 
Use the following context to answer the user's question. If the answer cannot be found in the context, 
say that you don't have enough information to answer the question.

Context:
{context}

Question: {query}

Please provide a comprehensive answer based on the context provided. If you reference specific information, 
mention which document it came from."""

            # Generate response
            response = self.model.generate_content(prompt)
            
            return response.text, sources
            
        except Exception as e:
            print(f"Error generating Gemini response: {e}")
            return f"Error generating response: {str(e)}", []


class RAGPipeline:
    """Main RAG pipeline that orchestrates the entire process"""
    
    def __init__(self):
        # Choose vector store based on settings
        if getattr(settings, 'USE_PINECONE', True) and PINECONE_AVAILABLE:
            self.vector_store = PineconeVectorStore()
            self.use_pinecone = True
        else:
            self.vector_store = FAISSVectorStore()
            self.use_pinecone = False
        
        self.llm = GeminiLLM()
    
    def process_query(self, query: str, top_k: int = 5) -> Tuple[str, List[Dict]]:
        """Process a user query through the RAG pipeline"""
        # For Pinecone, we don't need to check if index is empty (it's cloud-based)
        # For FAISS, check if we need to rebuild
        if not self.use_pinecone:
            if self.vector_store.index.ntotal == 0:
                print("Vector store is empty, checking for processed documents...")
                processed_count = Document.objects.filter(processed=True, total_chunks__gt=0).count()
                if processed_count > 0:
                    print(f"Found {processed_count} processed documents, rebuilding vector store...")
                    self.vector_store.rebuild_from_database()
        
        # Search for relevant chunks
        relevant_chunks = self.vector_store.search(query, top_k)
        
        if not relevant_chunks:
            return "I don't have any relevant documents to answer your question. Please upload some documents first.", []
        
        # Generate response using LLM
        response, sources = self.llm.generate_response(query, relevant_chunks)
        
        return response, sources
    
    def add_document_chunks(self, document: Document):
        """Add all chunks from a document to the vector store"""
        chunks_added = 0
        for chunk in document.chunks.all():
            if self.vector_store.add_chunk(chunk):
                chunks_added += 1
        
        if chunks_added > 0:
            self.vector_store.save_index()
        
        return chunks_added
    
    def remove_document(self, document_id: str):
        """Remove a document and its chunks from the vector store"""
        self.vector_store.remove_document_chunks(document_id)
    
    def clear_all_vector_data(self):
        """Clear all vector data and optionally delete all document chunks"""
        return self.vector_store.clear_all_data()


# Global RAG pipeline instance
rag_pipeline = RAGPipeline()
