# RAG Assistant - Retrieval Augmented Generation Chatbot

A Django-based RAG (Retrieval-Augmented Generation) chatbot that allows users to upload documents and ask questions about their content using AI.

## Features

- 📄 **Document Upload**: Support for PDF, DOCX, and TXT files
- 🔍 **Smart Chunking**: Intelligent text splitting with overlap for better context
- 🤖 **AI-Powered Responses**: Uses Google Gemini for natural language responses
- 🔍 **Vector Search**: FAISS-based semantic search for relevant document chunks
- 💬 **Chat Interface**: Web-based chat interface for natural conversations
- 🔄 **Background Processing**: Async document processing with Celery
- 📊 **Admin Interface**: Django admin for managing documents and chats
- 🌐 **REST API**: Complete API for integration with other applications

## Architecture

```
User uploads document → Text extraction → Chunking → Embedding generation → FAISS vector store
                                                                               ↓
User asks question → Query embedding → Vector search → Retrieved chunks → LLM (Gemini) → Response
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd RAG_prototype

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment file
cp .env.example .env

# Edit .env and set your Gemini API key
# Get API key from: https://makersuite.google.com/app/apikey
nano .env
```

### 3. Database Setup

```bash
cd RAG
python manage.py migrate
python manage.py createsuperuser
```

### 4. Start the Application

```bash
# Start Django development server
python manage.py runserver

# In another terminal, start Celery worker (optional, for background processing)
cd RAG
celery -A RAG worker --loglevel=info

# In another terminal, start Redis (required for Celery)
redis-server
```

### 5. Access the Application

- **Chat Interface**: http://localhost:8000/
- **Django Admin**: http://localhost:8000/admin/
- **API Documentation**: http://localhost:8000/api/

## Usage

### Web Interface

1. Open http://localhost:8000/
2. Upload documents using the upload area
3. Wait for processing to complete
4. Start asking questions about your documents

### API Usage

#### Upload a Document

```bash
curl -X POST http://localhost:8000/api/documents/ \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf" \
  -F "title=My Document" \
  -F "file_type=pdf"
```

#### Ask a Question

```bash
curl -X POST http://localhost:8000/api/chat/ \
  -H "Content-Type: application/json" \
  -d '{"message": "What is this document about?"}'
```

#### Get Documents

```bash
curl http://localhost:8000/api/documents/
```

## API Endpoints

### Documents
- `GET /api/documents/` - List all documents
- `POST /api/documents/` - Upload a new document
- `GET /api/documents/{id}/` - Get document details
- `DELETE /api/documents/{id}/` - Delete a document
- `GET /api/documents/{id}/chunks/` - Get document chunks
- `GET /api/documents/{id}/processing_status/` - Check processing status

### Chat
- `POST /api/chat/` - Send a chat message
- `GET /api/chat-sessions/` - List chat sessions
- `POST /api/chat-sessions/` - Create new chat session
- `GET /api/chat-sessions/{id}/messages/` - Get session messages

### Other
- `POST /api/search/` - Search across documents
- `GET /api/stats/` - Get usage statistics

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key | Required |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379/0` |
| `DEBUG` | Django debug mode | `True` |
| `CHUNK_SIZE` | Text chunk size | `1000` |
| `CHUNK_OVERLAP` | Overlap between chunks | `200` |
| `EMBEDDING_MODEL` | Sentence transformer model | `all-MiniLM-L6-v2` |

### Django Settings

Key settings in `RAG/settings.py`:

```python
# RAG Configuration
GEMINI_API_KEY = ""  # Set this with your API key
FAISS_INDEX_PATH = BASE_DIR / 'data' / 'faiss_index'
DOCUMENTS_STORAGE_PATH = BASE_DIR / 'data' / 'documents'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
```

## Components

### Models
- **Document**: Stores uploaded documents
- **DocumentChunk**: Text chunks with embeddings
- **ChatSession**: User chat sessions
- **ChatMessage**: Individual messages
- **ProcessingTask**: Background processing status

### Core Classes
- **DocumentProcessor**: Handles text extraction and chunking
- **EmbeddingManager**: Generates embeddings using sentence-transformers
- **FAISSVectorStore**: Manages vector database operations
- **GeminiLLM**: Interfaces with Google Gemini API
- **RAGPipeline**: Orchestrates the entire RAG process

## File Processing

### Supported Formats
- **PDF**: Extracted using PyPDF2
- **DOCX**: Extracted using python-docx
- **TXT**: Direct text reading with encoding detection

### Text Processing Pipeline
1. Extract text from uploaded file
2. Clean and normalize text
3. Split into chunks with overlap
4. Generate embeddings for each chunk
5. Store in FAISS vector database
6. Index for fast retrieval

## Development

### Project Structure

```
RAG_prototype/
├── RAG/                    # Django project
│   ├── RAG/               # Project settings
│   ├── assistant/         # Main app
│   │   ├── models.py      # Database models
│   │   ├── views.py       # API views
│   │   ├── serializers.py # DRF serializers
│   │   ├── rag_utils.py   # RAG core logic
│   │   ├── document_utils.py # Document processing
│   │   ├── tasks.py       # Celery tasks
│   │   └── templates/     # HTML templates
│   └── manage.py
├── requirements.txt
├── .env.example
└── README.md
```

### Adding New Features

1. **New Document Types**: Extend `DocumentProcessor._extract_*_text()` methods
2. **Custom Embeddings**: Modify `EmbeddingManager` class
3. **Different LLMs**: Implement new LLM classes similar to `GeminiLLM`
4. **UI Improvements**: Modify templates in `assistant/templates/`

### Testing

```bash
# Run Django tests
python manage.py test

# Test API endpoints
curl -X GET http://localhost:8000/api/documents/
```

## Deployment

### Production Considerations

1. **Environment Variables**: Set all required environment variables
2. **Database**: Use PostgreSQL instead of SQLite
3. **Static Files**: Configure static file serving
4. **Security**: Set `DEBUG=False` and configure ALLOWED_HOSTS
5. **Celery**: Use proper message broker (Redis/RabbitMQ)
6. **File Storage**: Use cloud storage for uploaded files

### Docker Deployment (Optional)

Create a `Dockerfile`:

```dockerfile
FROM python:3.11

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
WORKDIR /app/RAG

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
```

## Troubleshooting

### Common Issues

1. **GEMINI_API_KEY not set**: Set your API key in the `.env` file
2. **Redis connection error**: Make sure Redis is running
3. **File upload fails**: Check file size limits and permissions
4. **Embeddings not working**: Ensure sentence-transformers model is downloaded

### Logs

- Django logs: Check console output
- Celery logs: Check Celery worker output
- Processing errors: Check Django admin → Processing Tasks

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Future Enhancements

- [ ] Support for more document formats (Excel, PowerPoint)
- [ ] Multiple language support
- [ ] Integration with Slack/Teams
- [ ] Advanced search filters
- [ ] Document version control
- [ ] User permissions and sharing
- [ ] Analytics and usage tracking
- [ ] Custom embedding models
- [ ] Conversation memory and context
- [ ] Export chat history
