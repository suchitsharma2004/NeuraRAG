from django.core.management.base import BaseCommand
from assistant.models import Document
from assistant.document_utils import DocumentProcessor
from assistant.rag_utils import rag_pipeline


class Command(BaseCommand):
    help = 'Process unprocessed documents'

    def add_arguments(self, parser):
        parser.add_argument(
            '--document-id',
            type=str,
            help='Process a specific document by ID',
        )
        parser.add_argument(
            '--all',
            action='store_true',
            help='Process all unprocessed documents',
        )

    def handle(self, *args, **options):
        if options['document_id']:
            # Process specific document
            try:
                document = Document.objects.get(id=options['document_id'])
                self.process_single_document(document)
            except Document.DoesNotExist:
                self.stdout.write(
                    self.style.ERROR(f'Document with ID {options["document_id"]} not found')
                )
        elif options['all']:
            # Process all unprocessed documents
            unprocessed_docs = Document.objects.filter(processed=False)
            if not unprocessed_docs.exists():
                self.stdout.write(self.style.WARNING('No unprocessed documents found'))
                return
            
            for document in unprocessed_docs:
                self.process_single_document(document)
        else:
            # Show available options
            self.stdout.write(self.style.WARNING('Please specify --document-id <id> or --all'))
            
            # List unprocessed documents
            unprocessed_docs = Document.objects.filter(processed=False)
            if unprocessed_docs.exists():
                self.stdout.write('\nUnprocessed documents:')
                for doc in unprocessed_docs:
                    self.stdout.write(f'  ID: {doc.id} - {doc.title}')

    def process_single_document(self, document):
        self.stdout.write(f'Processing document: {document.title}')
        
        try:
            # Process the document
            processor = DocumentProcessor()
            result = processor.process_document(document)
            
            if result['success']:
                self.stdout.write(
                    self.style.SUCCESS(
                        f'✓ Successfully processed {document.title} - '
                        f'{result["chunks_created"]} chunks created'
                    )
                )
                
                # Add to vector store
                chunks_added = rag_pipeline.add_document_chunks(document)
                self.stdout.write(
                    self.style.SUCCESS(f'✓ Added {chunks_added} chunks to vector store')
                )
                
            else:
                self.stdout.write(
                    self.style.ERROR(f'✗ Failed to process {document.title}: {result["error"]}')
                )
                
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'✗ Error processing {document.title}: {str(e)}')
            )
