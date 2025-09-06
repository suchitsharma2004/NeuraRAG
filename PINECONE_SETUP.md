# Pinecone Setup Guide for NeuraRAG (LlamaIndex v2 + NVIDIA + AWS)

## 🌲 Quick Pinecone Setup (Free Tier)

### Step 1: Create Pinecone Account
1. Go to [pinecone.io](https://www.pinecone.io/)
2. Sign up for free account
3. Verify your email

### Step 2: Create Index
1. In Pinecone console, click "Create Index"
2. **Index Name**: `neurarag-vectors`
3. **Dimensions**: `384` (for all-MiniLM-L6-v2 embeddings)
4. **Metric**: `cosine`
5. **Cloud Provider**: `AWS`
6. **Environment**: `us-east-1` (NVIDIA hosted on AWS)
7. **Pod Type**: `p1.x1` (free tier)
8. Click "Create Index"

### Step 3: Get API Key
1. Go to "API Keys" tab in Pinecone console
2. Copy your API key
3. Save it securely

### Step 4: Add to Render Environment Variables
```
USE_PINECONE=True
PINECONE_API_KEY=your-copied-api-key-here
PINECONE_INDEX_NAME=neurarag-vectors
PINECONE_ENVIRONMENT=us-east-1
```

## 🆚 Local Development vs Production

### **Local Development (FAISS)**
- Set `USE_PINECONE=False` in your `.env` file
- Uses local FAISS for fast development
- No internet required for vector searches

### **Production (Pinecone)**
- Set `USE_PINECONE=True` on Render
- Persistent vector storage in cloud
- Survives app restarts and redeployments

## 🎯 Free Tier Limits (AWS + NVIDIA Hosted)

**Pinecone Free Tier:**
- ✅ 1 million vectors
- ✅ 1 pod (p1.x1)
- ✅ 1 index
- ✅ Unlimited queries
- ✅ No time limit
- ✅ NVIDIA-accelerated performance on AWS
- ✅ Low latency with us-east-1 region

**Perfect for:**
- Personal projects
- Demos and portfolios
- Small-scale applications
- Learning and experimentation

## 🔄 Migration Path

Your app automatically detects the environment:
- **Local**: Uses FAISS (fast, offline)
- **Production**: Uses Pinecone (persistent, cloud)

No code changes needed! Just set the environment variables.

## 🚀 Ready to Deploy!

Once you've set up Pinecone:
1. Add environment variables to Render
2. Deploy your app
3. Upload documents - they'll be stored in Pinecone
4. Ask questions - vectors persist across restarts!

Your NeuraRAG chatbot will now have persistent memory! 🧠
