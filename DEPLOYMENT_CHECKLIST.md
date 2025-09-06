# 🚀 NeuraRAG Deployment Checklist

## Pre-Deployment Checklist

- [ ] Code is committed to GitHub
- [ ] All dependencies are in `requirements.txt`
- [ ] `build.sh` has executable permissions (`chmod +x build.sh`)
- [ ] `.gitignore` excludes sensitive files
- [ ] Environment variables are documented
- [ ] Pinecone index is created and working locally
- [ ] OpenAI API key is valid and has credits

## Render Setup Checklist

- [ ] Render account created
- [ ] GitHub repository connected
- [ ] Web service configured:
  - [ ] Build Command: `./build.sh`
  - [ ] Start Command: `cd RAG && gunicorn RAG.wsgi:application`
  - [ ] Instance Type: Free

## Environment Variables Checklist

- [ ] `DEBUG=False`
- [ ] `DJANGO_SETTINGS_MODULE=RAG.settings`
- [ ] `GEMINI_API_KEY=...` (your actual key)
- [ ] `PINECONE_API_KEY=...` (your actual key)
- [ ] `PINECONE_ENVIRONMENT=us-east-1`
- [ ] `PINECONE_INDEX_NAME=neurarag-vectors`
- [ ] `USE_PINECONE=True`
- [ ] `DJANGO_SECRET_KEY=django-insecure-1x2i*f==tzwgb911!l*zuo%ou_3_hjr1%!p-cb$$$$4e(j1kt^8dc`

## Post-Deployment Testing

- [ ] App loads successfully
- [ ] Index page displays correctly
- [ ] Document upload works
- [ ] Chat functionality works
- [ ] Vector search returns relevant results
- [ ] No console errors in browser
- [ ] Mobile responsiveness works

## Performance Monitoring

- [ ] Check Render logs for errors
- [ ] Monitor Pinecone usage in console
- [ ] Monitor OpenAI API usage
- [ ] Test cold start behavior (after 15 min sleep)

---

## Quick Commands

### Generate Secret Key:
```bash
python generate_secret_key.py
```

### Deploy Commands:
```bash
git add .
git commit -m "Deploy to Render"
git push origin main
```

### Local Testing:
```bash
cd RAG
python manage.py runserver
```

---

✅ **Ready to deploy!** Follow the [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed instructions.
