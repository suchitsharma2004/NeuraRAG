# Environment Variables for FREE Render Deployment (SQLite + Pinecone)

## Required Environment Variables

Set these environment variables in your Render dashboard:

### 1. Secret Key (Required)
```
SECRET_KEY=your-super-secret-django-key-here
```
**Note**: Generate a new secret key for production using:
```bash
python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
```

### 2. Gemini API Key (Required for RAG functionality)
```
GEMINI_API_KEY=your-gemini-api-key-here
```
**Note**: Get this from Google AI Studio (https://makersuite.google.com/app/apikey)

### 3. Allowed Hosts (Required for production)
```
ALLOWED_HOSTS=your-app-name.onrender.com
```
**Note**: Replace `your-app-name` with your actual Render service name

### 4. Pinecone Configuration (Required for persistent vectors)
```
USE_PINECONE=True
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_NAME=neurarag-vectors
PINECONE_ENVIRONMENT=us-east-1
```
**Note**: 
- Get Pinecone API key from [Pinecone Console](https://app.pinecone.io/)
- Free tier includes 1M vectors, 1 pod, 1 index
- Using AWS `us-east-1` region with NVIDIA hosting
- Pod type: `p1.x1` (free tier)

### 5. Debug Mode (Optional, defaults to False)
```
DEBUG=False
```

## How to Set Up Pinecone (Free)

1. **Sign up**: Go to [Pinecone.io](https://www.pinecone.io/) and create account
2. **Create Index**: 
   - Name: `neurarag-vectors`
   - Dimensions: `384` (for all-MiniLM-L6-v2 model)
   - Metric: `cosine`
   - Environment: `gcp-starter` (free)
3. **Get API Key**: Copy from Pinecone console
4. **Add to Render**: Set environment variables above

## Architecture

- **Database**: SQLite (Django models, users, sessions) - FREE
- **Vector Storage**: Pinecone (document embeddings) - FREE (1M vectors)
- **File Storage**: Local Render disk (uploaded documents) - FREE
- **Web Service**: Render free tier - FREE

## Benefits

✅ **Persistent Vectors**: Pinecone vectors survive Render restarts
✅ **Scalable**: Pinecone handles 1M vectors on free tier  
✅ **Fast Search**: Optimized vector similarity search
✅ **Zero Cost**: Everything stays on free tiers

### 1. Secret Key (Required)
```
SECRET_KEY=your-super-secret-django-key-here
```
**Note**: Generate a new secret key for production using:
```bash
python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
```

### 2. Gemini API Key (Required for RAG functionality)
```
GEMINI_API_KEY=your-gemini-api-key-here
```
**Note**: Get this from Google AI Studio (https://makersuite.google.com/app/apikey)

### 3. Allowed Hosts (Required for production)
```
ALLOWED_HOSTS=your-app-name.onrender.com,www.your-domain.com
```
**Note**: Replace with your actual Render app URL and any custom domains

### 4. CORS Origins (Optional, if you have a frontend)
```
CORS_ALLOWED_ORIGINS=https://your-frontend-url.com,https://your-app-name.onrender.com
```

### 5. Debug Mode (Optional, defaults to False)
```
DEBUG=False
```

### 6. Redis URL (Optional, for Celery background tasks)
```
REDIS_URL=redis://your-redis-instance-url:6379/0
```
**Note**: You can add a Redis service in render.yaml if needed for background tasks

## Automatic Environment Variables

These are automatically set by Render:

- `DATABASE_URL` - Automatically configured from the PostgreSQL service
- `RENDER` - Set to `true` on Render platform (used in settings.py)

## How to Set Environment Variables in Render

1. Go to your Render dashboard
2. Select your web service
3. Go to "Environment" tab
4. Click "Add Environment Variable"
5. Add each variable one by one

## Security Notes

- Never commit real environment variables to git
- Use strong, unique values for SECRET_KEY
- Keep your API keys secure and rotate them regularly
- Set DEBUG=False in production
