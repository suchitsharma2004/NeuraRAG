from rest_framework import viewsets, status, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.views import APIView
from django.shortcuts import get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.db.models import Q
from .models import Document, DocumentChunk, ChatSession, ChatMessage, ProcessingTask
from .serializers import (
    DocumentSerializer, DocumentUploadSerializer, DocumentChunkSerializer,
    ChatSessionSerializer, ChatMessageSerializer, ChatQuerySerializer,
    ChatResponseSerializer, ProcessingTaskSerializer
)
from .document_utils import DocumentManager
from .rag_utils import rag_pipeline
from .tasks import process_document_task
import uuid
import json


class DocumentViewSet(viewsets.ModelViewSet):
    """ViewSet for managing documents"""
    queryset = Document.objects.all()
    serializer_class = DocumentSerializer
    permission_classes = [permissions.AllowAny]  # Allow anonymous access for development
    parser_classes = [MultiPartParser, FormParser]
    
    def get_queryset(self):
        """Filter documents by user if authenticated"""
        if self.request.user.is_authenticated:
            return Document.objects.filter(uploaded_by=self.request.user)
        return Document.objects.all()  # For development - show all documents
    
    def create(self, request, *args, **kwargs):
        """Upload a new document"""
        print(f"Document upload called with data: {request.data}")
        print(f"Content type: {request.content_type}")
        print(f"FILES: {request.FILES}")
        
        serializer = DocumentUploadSerializer(data=request.data)
        if serializer.is_valid():
            try:
                # Create document
                document_manager = DocumentManager()
                user = self.request.user if self.request.user.is_authenticated else None
                document = document_manager.upload_document(
                    file=serializer.validated_data['file'],
                    title=serializer.validated_data['title'],
                    file_type=serializer.validated_data['file_type'],
                    user=user
                )
                
                # Try to start background processing, fallback to sync processing
                try:
                    task = process_document_task.delay(str(document.id))
                    
                    # Create processing task record
                    ProcessingTask.objects.create(
                        document=document,
                        celery_task_id=task.id,
                        status='pending'
                    )
                    
                    processing_started = True
                    task_id = task.id
                except Exception as celery_error:
                    print(f"Celery not available, processing synchronously: {celery_error}")
                    # Process document synchronously
                    try:
                        from .tasks import process_document_sync
                        sync_result = process_document_sync(str(document.id))
                        print(f"Sync processing result: {sync_result}")
                        processing_started = True  # Mark as processed since we did it sync
                        task_id = None
                    except Exception as sync_error:
                        print(f"Synchronous processing failed: {sync_error}")
                        processing_started = False
                        task_id = None
                
                response_serializer = DocumentSerializer(document)
                return Response(
                    {
                        'document': response_serializer.data,
                        'processing_started': processing_started,
                        'task_id': task_id
                    },
                    status=status.HTTP_201_CREATED
                )
            except Exception as e:
                return Response(
                    {'error': str(e)},
                    status=status.HTTP_400_BAD_REQUEST
                )
        print(f"Document serializer errors: {serializer.errors}")
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    def destroy(self, request, *args, **kwargs):
        """Delete a document"""
        document = self.get_object()
        document_manager = DocumentManager()
        result = document_manager.delete_document(document)
        
        if result['success']:
            return Response({'message': result['message']}, status=status.HTTP_204_NO_CONTENT)
        else:
            return Response({'error': result['error']}, status=status.HTTP_400_BAD_REQUEST)
    
    @action(detail=True, methods=['get'])
    def chunks(self, request, pk=None):
        """Get chunks for a specific document"""
        document = self.get_object()
        chunks = DocumentChunk.objects.filter(document=document).order_by('chunk_index')
        serializer = DocumentChunkSerializer(chunks, many=True)
        return Response(serializer.data)
    
    @action(detail=True, methods=['get'])
    def processing_status(self, request, pk=None):
        """Get processing status for a document"""
        document = self.get_object()
        try:
            processing_task = ProcessingTask.objects.get(document=document)
            serializer = ProcessingTaskSerializer(processing_task)
            return Response(serializer.data)
        except ProcessingTask.DoesNotExist:
            return Response({'status': 'no_task'}, status=status.HTTP_404_NOT_FOUND)
    
    @action(detail=True, methods=['post'])
    def reprocess(self, request, pk=None):
        """Reprocess a document"""
        document = self.get_object()
        
        # Clear existing chunks
        DocumentChunk.objects.filter(document=document).delete()
        document.processed = False
        document.total_chunks = 0
        document.save()
        
        # Start new processing task
        task = process_document_task.delay(str(document.id))
        
        # Update or create processing task
        processing_task, created = ProcessingTask.objects.get_or_create(
            document=document,
            defaults={'celery_task_id': task.id, 'status': 'pending'}
        )
        if not created:
            processing_task.celery_task_id = task.id
            processing_task.status = 'pending'
            processing_task.error_message = ''
            processing_task.save()
        
        return Response({
            'message': 'Document reprocessing started',
            'task_id': task.id
        })


class ChatSessionViewSet(viewsets.ModelViewSet):
    """ViewSet for managing chat sessions"""
    queryset = ChatSession.objects.all()
    serializer_class = ChatSessionSerializer
    permission_classes = [permissions.AllowAny]  # Allow anonymous access for development
    
    def get_queryset(self):
        """Filter sessions by user if authenticated"""
        if self.request.user.is_authenticated:
            return ChatSession.objects.filter(user=self.request.user)
        return ChatSession.objects.all()  # For development
    
    def create(self, request, *args, **kwargs):
        """Create a new chat session"""
        title = request.data.get('title', '')
        session = ChatSession.objects.create(
            user=request.user,
            title=title
        )
        serializer = self.get_serializer(session)
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    
    @action(detail=True, methods=['get'])
    def messages(self, request, pk=None):
        """Get messages for a chat session"""
        session = self.get_object()
        messages = ChatMessage.objects.filter(session=session).order_by('created_at')
        serializer = ChatMessageSerializer(messages, many=True)
        return Response(serializer.data)
    
    @action(detail=True, methods=['post'])
    def send_message(self, request, pk=None):
        """Send a message in a chat session"""
        session = self.get_object()
        serializer = ChatQuerySerializer(data=request.data)
        
        if serializer.is_valid():
            message_text = serializer.validated_data['message']
            
            # Create user message
            user_message = ChatMessage.objects.create(
                session=session,
                message_type='user',
                content=message_text
            )
            
            # Process query through RAG pipeline
            try:
                response_text, sources = rag_pipeline.process_query(message_text)
                
                # Create assistant message
                assistant_message = ChatMessage.objects.create(
                    session=session,
                    message_type='assistant',
                    content=response_text,
                    sources=sources
                )
                
                # Update session title if it's the first message
                if not session.title and session.messages.count() == 2:
                    # Use first few words of user message as title
                    words = message_text.split()[:5]
                    session.title = ' '.join(words) + ('...' if len(words) == 5 else '')
                    session.save()
                
                # Prepare response
                response_data = {
                    'session_id': session.id,
                    'user_message': ChatMessageSerializer(user_message).data,
                    'assistant_message': ChatMessageSerializer(assistant_message).data
                }
                
                return Response(response_data)
                
            except Exception as e:
                # Create error message
                error_message = ChatMessage.objects.create(
                    session=session,
                    message_type='assistant',
                    content=f"I'm sorry, I encountered an error: {str(e)}"
                )
                
                response_data = {
                    'session_id': session.id,
                    'user_message': ChatMessageSerializer(user_message).data,
                    'assistant_message': ChatMessageSerializer(error_message).data,
                    'error': True
                }
                
                return Response(response_data, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@method_decorator(csrf_exempt, name='dispatch')
class ChatAPIView(APIView):
    """Simple chat API for direct queries"""
    permission_classes = [permissions.AllowAny]  # Allow anonymous access for development
    
    def post(self, request):
        """Process a chat query"""
        try:
            # Debug logging
            print(f"Chat API called with data: {request.data}")
            print(f"Content type: {request.content_type}")
            
            serializer = ChatQuerySerializer(data=request.data)
            if not serializer.is_valid():
                print(f"Serializer errors: {serializer.errors}")
                return Response({'errors': serializer.errors}, status=status.HTTP_400_BAD_REQUEST)
        
            message = serializer.validated_data['message']
            session_id = serializer.validated_data.get('session_id')
            query_mode = serializer.validated_data.get('query_mode', 'general')
            
            # Debug logging
            print(f"Processing query with mode: {query_mode}")
            print(f"Query: {message}")
            
            # Get or create session
            if session_id:
                try:
                    if self.request.user.is_authenticated:
                        session = ChatSession.objects.get(id=session_id, user=self.request.user)
                    else:
                        session = ChatSession.objects.get(id=session_id)
                except ChatSession.DoesNotExist:
                    session = ChatSession.objects.create(user=self.request.user if self.request.user.is_authenticated else None)
            else:
                session = ChatSession.objects.create(user=self.request.user if self.request.user.is_authenticated else None)
            
            # Create user message
            user_message = ChatMessage.objects.create(
                session=session,
                message_type='user',
                content=message
            )
            
            # Process query
            try:
                response_text, sources = rag_pipeline.process_query(message)
                
                # Create assistant message
                assistant_message = ChatMessage.objects.create(
                    session=session,
                    message_type='assistant',
                    content=response_text,
                    sources=sources
                )
                
                return Response({
                    'session_id': str(session.id),
                    'response': response_text,
                    'sources': sources,
                    'user_message_id': str(user_message.id),
                    'assistant_message_id': str(assistant_message.id)
                })
                
            except Exception as e:
                print(f"RAG pipeline error: {str(e)}")
                return Response({
                    'error': f"Error processing query: {str(e)}"
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                
        except Exception as e:
            print(f"Chat API error: {str(e)}")
            return Response({
                'error': f"API error: {str(e)}"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ProcessingTaskViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for viewing processing tasks"""
    queryset = ProcessingTask.objects.all()
    serializer_class = ProcessingTaskSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        """Filter tasks by user's documents"""
        return ProcessingTask.objects.filter(document__uploaded_by=self.request.user)


# Simple function-based views for additional endpoints
@csrf_exempt
def search_documents(request):
    """Search across all documents"""
    if not request.user.is_authenticated:
        return JsonResponse({'error': 'Authentication required'}, status=401)
    
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=405)
    
    query = request.POST.get('query', '').strip()
    if not query:
        return JsonResponse({'error': 'Query parameter required'}, status=400)
    
    try:
        # Search using RAG pipeline
        relevant_chunks = rag_pipeline.vector_store.search(query, top_k=10)
        
        return JsonResponse({
            'query': query,
            'results': relevant_chunks,
            'total_results': len(relevant_chunks)
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt  
def get_statistics(request):
    """Get overall statistics"""
    if not request.user.is_authenticated:
        return JsonResponse({'error': 'Authentication required'}, status=401)
    
    # Get user's documents
    user_documents = Document.objects.filter(uploaded_by=request.user)
    
    stats = {
        'total_documents': user_documents.count(),
        'processed_documents': user_documents.filter(processed=True).count(),
        'total_chunks': sum(doc.total_chunks for doc in user_documents),
        'total_sessions': ChatSession.objects.filter(user=request.user).count(),
        'total_messages': ChatMessage.objects.filter(session__user=request.user).count(),
        'vector_store_size': rag_pipeline.vector_store.index.ntotal if rag_pipeline.vector_store.index else 0
    }
    
    return JsonResponse(stats)

def chat_interface(request):
    """Serve the chat interface HTML"""
    from django.shortcuts import render
    return render(request, 'chat.html')

def index(request):
    """Serve the home/index page"""
    from django.shortcuts import render
    return render(request, 'index.html')

@csrf_exempt  
def clear_vector_database(request):
    """Clear all vector database data"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=405)
    
    try:
        # Clear vector store
        success = rag_pipeline.clear_all_vector_data()
        
        if success:
            return JsonResponse({
                'success': True,
                'message': 'Vector database cleared successfully'
            })
        else:
            return JsonResponse({
                'success': False,
                'message': 'Failed to clear vector database'
            }, status=500)
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt  
def debug_documents(request):
    """Debug endpoint to show document processing status and vector store info"""
    from .models import Document, DocumentChunk
    
    documents = Document.objects.all().order_by('-uploaded_at')
    
    debug_info = {
        'total_documents': documents.count(),
        'processed_documents': documents.filter(processed=True).count(),
        'vector_store_size': rag_pipeline.vector_store.index.ntotal if rag_pipeline.vector_store.index else 0,
        'documents': []
    }
    
    for doc in documents:
        chunks = DocumentChunk.objects.filter(document=doc)
        doc_info = {
            'id': str(doc.id),
            'title': doc.title,
            'file_type': doc.file_type,
            'processed': doc.processed,
            'total_chunks': doc.total_chunks,
            'actual_chunks_in_db': chunks.count(),
            'uploaded_at': doc.uploaded_at.isoformat() if doc.uploaded_at else None,
            'processed_at': doc.processed_at.isoformat() if doc.processed_at else None,
        }
        debug_info['documents'].append(doc_info)
    
    # Check vector store metadata
    if hasattr(rag_pipeline.vector_store, 'chunk_metadata'):
        debug_info['vector_store_metadata_count'] = len(rag_pipeline.vector_store.chunk_metadata)
        debug_info['documents_in_vector_store'] = list(set([
            meta.get('document_id') for meta in rag_pipeline.vector_store.chunk_metadata
        ]))
    else:
        debug_info['vector_store_metadata_count'] = 0
        debug_info['documents_in_vector_store'] = []
    
    return JsonResponse(debug_info, indent=2)

@csrf_exempt  
def rebuild_vector_store(request):
    """Force rebuild the vector store from all processed documents"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=405)
    
    try:
        from .models import Document
        
        # Clear the vector store
        rag_pipeline.vector_store._initialize_empty_index()
        
        # Rebuild from all processed documents
        processed_docs = Document.objects.filter(processed=True, total_chunks__gt=0)
        total_chunks_added = 0
        documents_processed = 0
        
        for document in processed_docs:
            chunks_added = rag_pipeline.add_document_chunks(document)
            if chunks_added > 0:
                total_chunks_added += chunks_added
                documents_processed += 1
                print(f"Added {chunks_added} chunks from document: {document.title}")
        
        # Save the rebuilt index
        rag_pipeline.vector_store.save_index()
        
        return JsonResponse({
            'success': True,
            'message': f'Vector store rebuilt successfully',
            'documents_processed': documents_processed,
            'total_chunks_added': total_chunks_added,
            'final_vector_store_size': rag_pipeline.vector_store.index.ntotal if rag_pipeline.vector_store.index else 0
        })
        
    except Exception as e:
        print(f"Error rebuilding vector store: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)
