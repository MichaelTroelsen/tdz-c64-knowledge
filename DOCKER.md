# Docker Deployment Guide

This guide covers deployment options for the TDZ C64 Knowledge Base MCP server and Streamlit GUI.

## 🤔 Important Considerations

### MCP Servers Are Typically Local
- MCP servers communicate via **stdio** (standard input/output), not HTTP
- They're designed to run **locally** and connect to Claude Desktop/Code
- Hosting them in the cloud requires additional work (HTTP wrapper)
- **Best practice:** Run MCP server locally, optionally host GUI in cloud

### Two Deployment Scenarios

1. **Local Docker Deployment** - Easy setup for local use
2. **Cloud Deployment** - Host Streamlit GUI online for remote access

---

## 🐳 Option 1: Docker for Local Deployment (Recommended)

Create a Docker image for one-command setup.

### Benefits
- ✅ Consistent environment across machines
- ✅ Easy distribution to other users
- ✅ No cloud costs
- ✅ MCP servers work best locally anyway
- ✅ Full control over data

### Use Cases
- Personal knowledge base
- Sharing with other C64 enthusiasts
- Development and testing
- Offline usage

### Implementation Plan
```dockerfile
# Dockerfile example structure
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -e ".[dev,gui,semantic]"
EXPOSE 8501
CMD ["streamlit", "run", "admin_gui.py"]
```

---

## ☁️ Option 2: Cloud Deployment

Host the Streamlit GUI online for remote access.

### 🥇 Fly.io (Best for Full Deployment)

**Free Tier:**
- 3 shared-cpu VMs
- 3GB persistent storage
- 160GB bandwidth/month

**Pros:**
- ✅ Persistent volumes (SQLite database persists)
- ✅ Always-on (no cold starts)
- ✅ Can run both MCP server + GUI
- ✅ Simple deployment (`fly launch`)
- ✅ Custom domains supported

**Cons:**
- ⚠️ Free tier more limited than before
- ⚠️ Requires credit card for verification

**Best For:** Personal knowledge base with persistent data

**Deployment:**
```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Launch app
fly launch

# Deploy
fly deploy
```

---

### 🥈 Google Cloud Run (Best for GUI Only)

**Free Tier:**
- 2 million requests/month
- 360,000 GB-seconds/month
- 1GB network egress/month

**Pros:**
- ✅ Very generous free tier
- ✅ Auto-scaling (scales to zero)
- ✅ Fast deployment
- ✅ Google Cloud integration

**Cons:**
- ⚠️ Cold starts (first request after idle)
- ⚠️ Need Cloud Storage/Cloud SQL for persistence
- ⚠️ More complex for stateful apps

**Best For:** Public-facing GUI with separate database

**Deployment:**
```bash
# Build and deploy
gcloud run deploy tdz-c64-knowledge \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

---

### 🥉 Render

**Free Tier:**
- 750 hours/month
- 100GB bandwidth/month
- Automatic deploys from GitHub

**Pros:**
- ✅ Easy setup with GitHub integration
- ✅ Automatic HTTPS
- ✅ Free PostgreSQL database (for 90 days)

**Cons:**
- ⚠️ Spins down after 15min inactivity
- ⚠️ Slow cold starts (30-60 seconds)
- ⚠️ Limited to 750 hours/month

**Best For:** Demo deployments or low-traffic usage

**Deployment:**
- Connect GitHub repo
- Select "Web Service"
- Build command: `pip install -e ".[gui]"`
- Start command: `streamlit run admin_gui.py --server.port $PORT`

---

### 🏗️ AWS Free Tier (ECS/Fargate)

**Free Tier (12 months):**
- 750 hours/month of t2.micro
- 5GB storage
- Limited data transfer

**Pros:**
- ✅ Professional-grade infrastructure
- ✅ Great for learning AWS
- ✅ Many integration options

**Cons:**
- ⚠️ Complex setup
- ⚠️ Free tier expires after 12 months
- ⚠️ Easy to accidentally incur charges

**Best For:** Learning AWS or enterprise deployments

---

### 🔷 Azure Container Instances

**Free Tier:**
- Pay-as-you-go (very limited free tier)
- $200 credit for first 30 days

**Pros:**
- ✅ Simple container deployment
- ✅ Windows/Linux support
- ✅ Microsoft ecosystem integration

**Cons:**
- ⚠️ Not very generous free tier
- ⚠️ Can get expensive quickly

**Best For:** Microsoft-centric environments

---

### ⚡ Railway (Previously Popular)

**Free Tier:**
- $5/month credit (previously unlimited)
- Execution time limits

**Status:**
- ⚠️ Significantly reduced free tier in 2023
- ⚠️ $5/month doesn't go far
- ⚠️ No longer recommended for free hosting

---

## 📊 Comparison Table

| Service | Free Tier | Persistent Storage | Always-On | Best For |
|---------|-----------|-------------------|-----------|----------|
| **Fly.io** | 3 VMs, 3GB | ✅ Yes (volumes) | ✅ Yes | Personal use |
| **Google Cloud Run** | 2M req/mo | ⚠️ Separate service | ❌ No (scales to 0) | GUI only |
| **Render** | 750 hrs/mo | ⚠️ Spins down | ❌ No (15min idle) | Demos |
| **AWS** | 750 hrs/mo (12mo) | ✅ Yes (EBS) | ✅ Yes | Learning AWS |
| **Azure** | Very limited | ✅ Yes | ✅ Yes | Microsoft shops |
| **Railway** | $5/mo credit | ✅ Yes | ⚠️ Limited | Not recommended |
| **Local Docker** | Unlimited | ✅ Yes | ✅ Yes | **Best choice** |

---

## 💡 Recommendations by Use Case

### Personal Use (Recommended)
**→ Local Docker Deployment**
- No costs
- Full control
- Best performance
- MCP integration works perfectly

### Share with Friends
**→ Fly.io Deployment**
- Persistent storage
- Always available
- Simple URL to share
- Free tier sufficient

### Public Demo
**→ Google Cloud Run**
- Generous free tier
- Auto-scaling
- Professional appearance
- Can handle traffic spikes

### Production/Enterprise
**→ AWS or Azure**
- Professional infrastructure
- Compliance options
- Advanced features
- Support available

---

## 🚀 Next Steps

### For Local Docker Deployment:
1. Create Dockerfile
2. Create docker-compose.yml (with volume mounts)
3. Add .dockerignore
4. Build and test image
5. Publish to Docker Hub (optional)

### For Cloud Deployment:
1. Choose platform (Fly.io recommended)
2. Set up account
3. Configure persistent storage
4. Set environment variables
5. Deploy and test

### For Both:
1. Document deployment process
2. Create deployment scripts
3. Set up CI/CD (GitHub Actions)
4. Add health checks
5. Monitor and maintain

---

## 📝 Notes

- SQLite database needs persistent storage in cloud deployments
- Consider using PostgreSQL for cloud deployments (better concurrency)
- Environment variables needed: `USE_FTS5=1`, `USE_SEMANTIC_SEARCH=1`
- Semantic search embeddings can be large (factor into storage/memory)
- First-time semantic search will take time to build embeddings

---

## 🔗 Useful Resources

- [Fly.io Documentation](https://fly.io/docs/)
- [Google Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Render Documentation](https://render.com/docs)
- [Docker Documentation](https://docs.docker.com/)
- [Streamlit Deployment Guide](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app)

---

*Last updated: 2025-12-13*
