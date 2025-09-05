# Environment Variables for Render Deployment

## Required Environment Variables

Set these environment variables in your Render dashboard:

### 1. Secret Key (Required)
```
SECRET_KEY=your-super-secret-django-key-here
```
**Note**: Generate a new secret key for production. You can use Django's `get_random_secret_key()` function or an online generator.

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
