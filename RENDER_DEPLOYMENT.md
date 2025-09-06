# NeuraRAG Render Deployment Guide

## 🚀 Ready to Deploy on Render!

Your Django RAG chatbot is now fully configured for Render deployment. Here's how to deploy it:

## Prerequisites

✅ **Already Done:**
- Django settings updated for production
- `render.yaml` configuration file created
- `build.sh` build script configured
- `requirements.txt` updated with production dependencies
- Static files and database configuration ready

## Step-by-Step Deployment

### 1. Push to GitHub

First, make sure all your changes are committed and pushed to GitHub:

```bash
git add .
git commit -m "Configure app for Render deployment"
git push origin main
```

### 2. Create a Render Account

1. Go to [render.com](https://render.com)
2. Sign up/sign in with your GitHub account
3. Connect your GitHub account to Render

### 3. Deploy Your Application (Web Service)

1. **Create Web Service:**
   - Click "New +" in your Render dashboard
   - Select "Web Service"
   - Connect your GitHub repository: `RAG_prototype`
   - Configure the following settings:
     - **Root Directory:** Leave BLANK
     - **Build Command:** `./build.sh`
     - **Start Command:** `cd RAG && gunicorn RAG.wsgi:application`
     - **Environment:** Python 3

2. **Create PostgreSQL Database:**
   - Click "New +" in your Render dashboard
   - Select "PostgreSQL"
   - Name it: `neurarag-db`
   - Choose your plan (free tier available)

3. **Configure Environment Variables:**
   
   Set these required environment variables in your web service:
   ```
   SECRET_KEY=generate-a-new-secret-key-here
   GEMINI_API_KEY=your-gemini-api-key
   ALLOWED_HOSTS=your-app-name.onrender.com
   DEBUG=False
   DATABASE_URL=copy-from-your-postgresql-database-internal-url
   ```
   
   **Important Security Notes:**
   - Generate a NEW secret key using: `python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"`
   - Get DATABASE_URL from your PostgreSQL database's "Internal Database URL"
   - Replace `your-app-name` with your actual service name
   
   **Note:** Replace `your-app-name` with the actual name you choose for your Render service.

4. **Deploy:**
   - Click "Create New Blueprint"
   - Render will automatically create:
     - A PostgreSQL database (neurarag-db)
     - A web service (neurarag)
   - The deployment will start automatically

### 4. Monitor Deployment

- Watch the build logs in your Render dashboard
- The initial deployment may take 5-10 minutes
- You'll get a URL like: `https://your-app-name.onrender.com`

## What Gets Deployed

### 🗄️ **Database**
- Managed PostgreSQL database
- Automatic backups
- Connection automatically configured

### 🌐 **Web Service**
- Your Django app running on Gunicorn
- Automatic HTTPS with SSL certificate
- Static files served via WhiteNoise

### 📁 **Persistent Storage**
- Your vector database (FAISS)
- Uploaded documents
- Persistent across deployments

## Post-Deployment Steps

### 1. Create Superuser (Optional)

Access the Django admin by creating a superuser:

```bash
# In Render's web service shell
python manage.py createsuperuser
```

### 2. Test Your Application

1. Visit your deployed URL
2. Test document upload functionality
3. Test chat interface
4. Verify all query modes work

### 3. Set Up Custom Domain (Optional)

1. In Render dashboard → Settings → Custom Domains
2. Add your domain
3. Update DNS records as instructed
4. Update `ALLOWED_HOSTS` environment variable

## Production Features

### ✨ **What Works in Production:**

- **Modern UI**: Glassmorphism design with animations
- **Loading Screen**: Smooth transitions between pages
- **Query Modes**: General, Summarize, Extract, Compare, Explain
- **Document Upload**: PDF, DOCX, TXT support
- **Vector Search**: FAISS-based similarity search
- **AI Chat**: Google Gemini integration
- **Responsive Design**: Works on all devices

### 🔐 **Security Features:**

- Environment-based configuration
- Secure headers enabled
- HTTPS enforced
- CSRF protection
- SQL injection protection
- XSS protection

### 📊 **Performance:**

- Static file compression
- Database query optimization
- Efficient vector search
- Responsive loading states

## Troubleshooting

### Common Issues:

1. **Build Fails:**
   - Check that all requirements are in `requirements.txt`
   - Verify `build.sh` has execute permissions

2. **Database Connection Issues:**
   - Ensure `DATABASE_URL` is automatically set by Render
   - Check PostgreSQL service is running

3. **Static Files Not Loading:**
   - Verify `collectstatic` runs in build script
   - Check WhiteNoise configuration

4. **Gemini API Errors:**
   - Verify `GEMINI_API_KEY` is set correctly
   - Check API key is valid and has proper permissions

### Getting Help:

- Check Render logs in your dashboard
- Review Django logs for detailed error messages
- Render has excellent documentation and support

## Cost Estimation

**Render Free Tier:**
- Web Service: Free (with some limitations)
- PostgreSQL: $7/month (after free trial)
- Total: ~$7/month for a production-ready deployment

**Benefits over other platforms:**
- ✅ Native Django support
- ✅ Persistent file storage
- ✅ Managed database
- ✅ Easy deployment
- ✅ Auto-scaling
- ✅ Free SSL

## Next Steps

Once deployed, you can:

1. **Monitor Usage**: Check analytics in Render dashboard
2. **Scale Up**: Upgrade to paid plans for more resources
3. **Add Features**: Redis for background tasks, email services
4. **Custom Domain**: Set up your own domain name
5. **Monitoring**: Add application monitoring services

Your NeuraRAG chatbot is now ready for production! 🎉
