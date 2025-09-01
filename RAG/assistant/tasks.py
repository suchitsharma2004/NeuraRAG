from celery import shared_task
from django.utils import timezone
from .models import Document, ProcessingTask
from .document_utils import DocumentProcessor, DocumentManager
from .rag_utils import rag_pipeline, FAISSVectorStore


@shared_task(bind=True)
def process_document_task(self, document_id):
    """Celery task to process a document in the background"""
    try:
        # Get the document
        document = Document.objects.get(id=document_id)
        
        # Get or create processing task
        processing_task, created = ProcessingTask.objects.get_or_create(
            document=document,
            defaults={'celery_task_id': self.request.id}
        )
        
        if not created:
            processing_task.celery_task_id = self.request.id
            processing_task.save()
        
        # Update task status
        processing_task.status = 'processing'
        processing_task.started_at = timezone.now()
        processing_task.save()
        
        # Process the document
        processor = DocumentProcessor()
        result = processor.process_document(document)
        
        if result['success']:
            # Add chunks to vector store
            chunks_added = rag_pipeline.add_document_chunks(document)
            
            # Mark as completed
            processing_task.status = 'completed'
            processing_task.completed_at = timezone.now()
            processing_task.save()
            
            return {
                'success': True,
                'document_id': str(document.id),
                'chunks_created': result['chunks_created'],
                'chunks_added_to_vector_store': chunks_added,
                'message': result['message']
            }
        else:
            # Mark as failed
            processing_task.status = 'failed'
            processing_task.error_message = result['error']
            processing_task.completed_at = timezone.now()
            processing_task.save()
            
            return {
                'success': False,
                'document_id': str(document.id),
                'error': result['error'],
                'message': result['message']
            }
            
    except Document.DoesNotExist:
        return {
            'success': False,
            'error': f'Document with id {document_id} not found'
        }
    except Exception as e:
        # Mark task as failed if it exists
        try:
            processing_task = ProcessingTask.objects.get(document_id=document_id)
            processing_task.status = 'failed'
            processing_task.error_message = str(e)
            processing_task.completed_at = timezone.now()
            processing_task.save()
        except ProcessingTask.DoesNotExist:
            pass
        
        return {
            'success': False,
            'error': str(e)
        }


@shared_task
def cleanup_old_processing_tasks():
    """Clean up old processing tasks (run periodically)"""
    from datetime import timedelta
    
    cutoff_date = timezone.now() - timedelta(days=7)
    
    # Delete old completed and failed tasks
    deleted_count = ProcessingTask.objects.filter(
        completed_at__lt=cutoff_date,
        status__in=['completed', 'failed']
    ).delete()
    
    return f"Cleaned up {deleted_count[0]} old processing tasks"


@shared_task
def rebuild_vector_store():
    """Rebuild the entire vector store from scratch"""
    try:
        # Clear existing vector store
        rag_pipeline.vector_store._initialize_empty_index()
        
        # Process all documents
        processed_docs = 0
        total_chunks = 0
        
        for document in Document.objects.filter(processed=True):
            chunks_added = rag_pipeline.add_document_chunks(document)
            if chunks_added > 0:
                processed_docs += 1
                total_chunks += chunks_added
        
        # Save the updated index
        rag_pipeline.vector_store.save_index()
        
        return {
            'success': True,
            'processed_documents': processed_docs,
            'total_chunks': total_chunks,
            'message': 'Vector store rebuilt successfully'
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'message': 'Failed to rebuild vector store'
        }


def process_document_sync(document_id):
    """
    Synchronous version of document processing for when Celery is not available
    """
    try:
        document = Document.objects.get(id=document_id)
        document_processor = DocumentProcessor()
        
        # Process the document
        result = document_processor.process_document(document)
        
        if result['success']:
            # The DocumentProcessor already creates chunks and marks document as processed
            # Now we need to add chunks to the GLOBAL vector store (same one used for queries)
            chunks_added = rag_pipeline.add_document_chunks(document)
            
            return {
                'success': True,
                'document_id': str(document.id),
                'chunks_created': result['chunks_created'],
                'chunks_added_to_vector_store': chunks_added,
                'message': result['message']
            }
        else:
            return {
                'success': False,
                'document_id': str(document.id),
                'error': result['error'],
                'message': result['message']
            }
            
    except Document.DoesNotExist:
        return {
            'success': False,
            'error': f'Document with id {document_id} not found'
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }
