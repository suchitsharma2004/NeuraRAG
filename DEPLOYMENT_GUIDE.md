# NeuraRAG Deployment Guide

## 🚀 Deploying to Render

This guide will walk you through deploying your Django RAG-powered chatbot to Render for free.

### Prerequisites

1. ✅ GitHub repository with your code
2. ✅ Render account (free tier)
3. ✅ Pinecone account with API key
4. ✅ OpenAI API key

### Step 1: Prepare Your Repository

Ensure all files are committed and pushed to GitHub:

```bash
# Check git status
git status

# Add all files
git add .

# Commit changes
git commit -m "Prepare for Render deployment"

# Push to GitHub
git push origin main
```

### Step 2: Create Render Web Service

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository
4. Configure the service:
   - **Name**: `neurarag` (or your preferred name)
   - **Environment**: `Python`
   - **Build Command**: `./build.sh`
   - **Start Command**: `cd RAG && gunicorn RAG.wsgi:application`
   - **Instance Type**: `Free`

### Step 3: Set Environment Variables

In your Render service settings, add these environment variables:

#### Required Variables:
```
DEBUG=False
DJANGO_SETTINGS_MODULE=RAG.settings
GEMINI_API_KEY=your_gemini_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX_NAME=neurarag-vectors
USE_PINECONE=True
DJANGO_SECRET_KEY=django-insecure-1x2i*f==tzwgb911!l*zuo%ou_3_hjr1%!p-cb$$$$4e(j1kt^8dc
```

#### Generate Django Secret Key:
```python
from django.core.management.utils import get_random_secret_key
print(get_random_secret_key())
```

### Step 4: Deploy

1. Click **"Create Web Service"**
2. Render will automatically build and deploy your app
3. Monitor the build logs for any issues
4. Once deployed, you'll get a URL like: `https://neurarag.onrender.com`

### Step 5: Test Your Deployment

1. Visit your deployed URL
2. Upload a test document
3. Try asking questions to verify RAG functionality
4. Check that Pinecone integration is working

## 🛠️ Troubleshooting

### Common Issues:

1. **Build Fails**: Check build.sh permissions
   ```bash
   chmod +x build.sh
   git add build.sh
   git commit -m "Fix build.sh permissions"
   git push
   ```

2. **Static Files Not Loading**: Ensure `STATIC_ROOT` is set correctly in settings.py

3. **Database Issues**: SQLite is used for free deployment, should work out of the box

4. **Pinecone Connection**: Verify your API key and index name in environment variables

5. **Memory Issues**: Free tier has 512MB RAM limit, monitor usage

### Logs:
- Check Render logs in the dashboard for detailed error messages
- Use `print()` statements for debugging (they'll appear in logs)

## 🔧 Local vs Production Differences

| Feature | Local | Production |
|---------|-------|------------|
| Database | SQLite | SQLite |
| Vector Store | FAISS/Pinecone | Pinecone |
| Static Files | Django dev server | Collected static files |
| Debug Mode | True | False |
| SSL | HTTP | HTTPS (automatic) |

## 📊 Monitoring

- **Render Dashboard**: Monitor deployments, logs, and metrics
- **Pinecone Console**: Monitor vector operations and usage
- **OpenAI Dashboard**: Monitor API usage and costs

## 🎯 Post-Deployment

1. Test all functionality thoroughly
2. Share your deployed app URL
3. Consider adding a custom domain (paid feature)
4. Monitor usage and costs
5. Set up alerts for any issues

## 🔄 Updating Your App

```bash
# Make changes locally
git add .
git commit -m "Your update message"
git push origin main
# Render will automatically redeploy
```

## 💡 Tips for Free Tier

1. **Cold starts**: Free services sleep after 15 minutes of inactivity
2. **Build time**: ~5-10 minutes for initial deployment
3. **Storage**: Ephemeral - uploaded files don't persist across deployments
4. **Bandwidth**: 100GB/month included

---

🎉 **Congratulations!** Your NeuraRAG chatbot is now live on the web!
