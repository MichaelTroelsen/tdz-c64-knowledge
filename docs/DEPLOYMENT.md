# TDZ C64 Knowledge Base - Production Deployment Guide

Complete guide for deploying the C64 Knowledge Base to production environments.

## Table of Contents
- [Deployment Options](#deployment-options)
- [Local Network Deployment](#local-network-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Docker Deployment](#docker-deployment)
- [Streamlit Cloud Deployment](#streamlit-cloud-deployment)
- [Security Hardening](#security-hardening)
- [Monitoring & Maintenance](#monitoring--maintenance)
- [Backup Automation](#backup-automation)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)

---

## Deployment Options

Choose the deployment option that best fits your needs:

| Option | Best For | Complexity | Cost | Scalability |
|--------|----------|------------|------|-------------|
| **Local Network** | Small teams, personal use | Low | Free | Low |
| **Cloud VM** | Production, teams | Medium | $5-50/mo | High |
| **Docker** | Dev/test, portability | Medium | Varies | High |
| **Streamlit Cloud** | Public demos, simple deployment | Low | Free tier | Medium |
| **Serverless** | API-only, auto-scaling | High | Pay-per-use | Very High |

---

## Local Network Deployment

Deploy on a Windows/Linux machine accessible to your local network.

### Prerequisites
- Windows 10/11 or Ubuntu 20.04+
- Python 3.10+
- Static IP or DNS name
- Open ports: 8501 (GUI), 8000 (API - optional)

### Setup Steps

#### 1. Install and Configure

```bash
# Clone repository
cd C:\deployment
git clone <your-repo-url> tdz-c64-knowledge
cd tdz-c64-knowledge

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux

# Install all dependencies
pip install -e ".[dev,gui]"
```

#### 2. Configure Environment

Create `.env` file:
```bash
# .env
TDZ_DATA_DIR=C:\c64kb-data
USE_FTS5=1
USE_SEMANTIC_SEARCH=1
USE_QUERY_PREPROCESSING=1
ALLOWED_DOCS_DIRS=C:\c64kb-data\uploads
```

#### 3. Import Initial Data

```bash
# Enable optimizations
.venv\Scripts\python.exe enable_fts5.py
.venv\Scripts\python.exe enable_semantic_search.py

# Import documents
.venv\Scripts\python.exe cli.py add-folder "pdf" --tags reference c64 --recursive
.venv\Scripts\python.exe cli.py add-folder "txt" --tags reference c64 --recursive

# Verify
.venv\Scripts\python.exe cli.py stats
```

#### 4. Start GUI for Network Access

```bash
# Start Streamlit with network binding
streamlit run admin_gui.py --server.address 0.0.0.0 --server.port 8501
```

Access from other devices: `http://<SERVER_IP>:8501`

#### 5. Run as Windows Service (Production)

Create `run_gui.bat`:
```batch
@echo off
cd C:\deployment\tdz-c64-knowledge
call .venv\Scripts\activate.bat
streamlit run admin_gui.py --server.address 0.0.0.0 --server.port 8501 --server.headless true
```

**Using NSSM (Non-Sucking Service Manager):**
```cmd
# Download NSSM from https://nssm.cc/download
nssm install C64KnowledgeBase "C:\deployment\tdz-c64-knowledge\run_gui.bat"
nssm set C64KnowledgeBase AppDirectory "C:\deployment\tdz-c64-knowledge"
nssm set C64KnowledgeBase DisplayName "C64 Knowledge Base GUI"
nssm set C64KnowledgeBase Description "Commodore 64 Documentation Knowledge Base"
nssm start C64KnowledgeBase
```

**Verify Service:**
- Open Services (services.msc)
- Find "C64 Knowledge Base GUI"
- Set startup type to "Automatic"

#### 6. Configure Firewall

**Windows Firewall:**
```cmd
# Allow Streamlit port
netsh advfirewall firewall add rule name="C64 Knowledge Base GUI" dir=in action=allow protocol=TCP localport=8501
```

**Linux (ufw):**
```bash
sudo ufw allow 8501/tcp
sudo ufw reload
```

---

## Cloud Deployment

Deploy on cloud platforms (AWS, Azure, GCP) for production-grade reliability.

### AWS EC2 Deployment

#### 1. Launch EC2 Instance

**Recommended Specs:**
- Instance type: `t3.medium` (2 vCPU, 4 GB RAM)
- Storage: 30 GB EBS (gp3)
- OS: Ubuntu 22.04 LTS
- Security group: Allow SSH (22), HTTP (80), HTTPS (443), Custom TCP (8501)

#### 2. Connect and Setup

```bash
# SSH into instance
ssh -i your-key.pem ubuntu@<EC2_PUBLIC_IP>

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.10+
sudo apt install python3.10 python3.10-venv python3-pip git -y

# Clone repository
cd /opt
sudo git clone <your-repo-url> tdz-c64-knowledge
sudo chown -R ubuntu:ubuntu tdz-c64-knowledge
cd tdz-c64-knowledge

# Setup virtual environment
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev,gui]"
```

#### 3. Configure Data Directory

```bash
# Create data directory
sudo mkdir -p /var/lib/c64kb-data
sudo chown ubuntu:ubuntu /var/lib/c64kb-data

# Set environment
export TDZ_DATA_DIR=/var/lib/c64kb-data
export USE_FTS5=1
export USE_SEMANTIC_SEARCH=1
```

#### 4. Import Data

```bash
# Enable optimizations
.venv/bin/python enable_fts5.py
.venv/bin/python enable_semantic_search.py

# Import documents (upload via SCP or S3)
.venv/bin/python cli.py add-folder "/path/to/docs" --tags reference c64 --recursive
```

#### 5. Setup Systemd Service

Create `/etc/systemd/system/c64kb-gui.service`:
```ini
[Unit]
Description=C64 Knowledge Base GUI
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/tdz-c64-knowledge
Environment="PATH=/opt/tdz-c64-knowledge/.venv/bin"
Environment="TDZ_DATA_DIR=/var/lib/c64kb-data"
Environment="USE_FTS5=1"
Environment="USE_SEMANTIC_SEARCH=1"
ExecStart=/opt/tdz-c64-knowledge/.venv/bin/streamlit run admin_gui.py --server.address 0.0.0.0 --server.port 8501 --server.headless true
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Start Service:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable c64kb-gui.service
sudo systemctl start c64kb-gui.service
sudo systemctl status c64kb-gui.service
```

#### 6. Setup Nginx Reverse Proxy (Optional but Recommended)

```bash
# Install Nginx
sudo apt install nginx -y

# Configure reverse proxy
sudo nano /etc/nginx/sites-available/c64kb
```

Add configuration:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }
}
```

**Enable site:**
```bash
sudo ln -s /etc/nginx/sites-available/c64kb /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

#### 7. Setup SSL with Let's Encrypt

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx -y

# Obtain certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal is configured automatically
sudo certbot renew --dry-run
```

**Access:** `https://your-domain.com`

### Azure / GCP Deployment

Similar steps as AWS:
1. Create VM instance (similar specs)
2. Configure networking (allow ports 80, 443, 8501)
3. Follow Ubuntu setup steps above
4. Use cloud-specific firewall rules
5. Consider using managed PostgreSQL for future database scaling

---

## Docker Deployment

Containerize the application for portability and consistency.

### 1. Create Dockerfile

Create `Dockerfile` in project root:
```dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy application files
COPY . /app/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[dev,gui]"

# Create data directory
RUN mkdir -p /data

# Set environment variables
ENV TDZ_DATA_DIR=/data
ENV USE_FTS5=1
ENV USE_SEMANTIC_SEARCH=1
ENV PYTHONUNBUFFERED=1

# Expose ports
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run GUI
CMD ["streamlit", "run", "admin_gui.py", "--server.address", "0.0.0.0", "--server.port", "8501", "--server.headless", "true"]
```

### 2. Create docker-compose.yml

```yaml
version: '3.8'

services:
  c64kb-gui:
    build: .
    image: tdz-c64-knowledge:latest
    container_name: c64kb-gui
    ports:
      - "8501:8501"
    volumes:
      - ./data:/data
      - ./backups:/backups
    environment:
      - TDZ_DATA_DIR=/data
      - USE_FTS5=1
      - USE_SEMANTIC_SEARCH=1
      - USE_QUERY_PREPROCESSING=1
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### 3. Build and Run

```bash
# Build image
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f c64kb-gui

# Stop services
docker-compose down
```

### 4. Import Data into Container

```bash
# Copy documents to container
docker cp ./pdf c64kb-gui:/app/pdf
docker cp ./txt c64kb-gui:/app/txt

# Import documents
docker exec -it c64kb-gui .venv/bin/python cli.py add-folder /app/pdf --tags reference c64 --recursive
docker exec -it c64kb-gui .venv/bin/python cli.py add-folder /app/txt --tags reference c64 --recursive

# Verify
docker exec -it c64kb-gui .venv/bin/python cli.py stats
```

### 5. Persistent Backups

```bash
# Create backup (inside container)
docker exec c64kb-gui .venv/bin/python -c "from server import KnowledgeBase; kb = KnowledgeBase(); kb.create_backup('/backups', compress=True); kb.close()"

# Backups are stored in ./backups (mounted volume)
ls -lh backups/
```

---

## Streamlit Cloud Deployment

Deploy the GUI to Streamlit Cloud for free public hosting.

### Prerequisites
- GitHub account
- Streamlit Cloud account (free tier available)
- Public or private GitHub repository

### Steps

#### 1. Prepare Repository

Ensure these files exist:
- `admin_gui.py` (main app)
- `server.py` (knowledge base)
- `requirements.txt` or `pyproject.toml`
- `.streamlit/config.toml` (optional, for customization)

Create `requirements.txt`:
```txt
mcp>=1.0.0
pypdf>=3.0.0
rank-bm25>=0.2.2
nltk>=3.8
sentence-transformers>=2.2.0
faiss-cpu>=1.7.4
streamlit>=1.28.0
pandas>=2.0.0
```

#### 2. Deploy to Streamlit Cloud

1. Go to https://share.streamlit.io/
2. Click "New app"
3. Connect your GitHub repository
4. Configure:
   - **Repository:** your-username/tdz-c64-knowledge
   - **Branch:** main
   - **Main file path:** admin_gui.py
   - **Python version:** 3.10
5. Advanced settings:
   - Add secrets (if needed):
     ```toml
     # .streamlit/secrets.toml
     TDZ_DATA_DIR = "/mount/data"
     USE_FTS5 = "1"
     USE_SEMANTIC_SEARCH = "1"
     ```

6. Click "Deploy"

#### 3. Limitations

**Streamlit Cloud Constraints:**
- 1 GB RAM limit (free tier)
- No persistent storage (resets on redeploy)
- Public URL (unless using private sharing)
- Limited to Streamlit apps (no MCP server)

**Workarounds:**
- Use smaller knowledge base (~50-100 docs)
- Pre-build embeddings and commit to repo
- Use cloud storage (S3, GCS) for data persistence
- Consider paid Streamlit tier for more resources

---

## Security Hardening

Essential security measures for production deployment.

### 1. Authentication

**Option A: Basic HTTP Authentication (Nginx)**

Add to Nginx config:
```nginx
location / {
    auth_basic "C64 Knowledge Base";
    auth_basic_user_file /etc/nginx/.htpasswd;

    proxy_pass http://localhost:8501;
    # ... rest of proxy config
}
```

Create password file:
```bash
sudo apt install apache2-utils
sudo htpasswd -c /etc/nginx/.htpasswd admin
sudo systemctl restart nginx
```

**Option B: OAuth2 Proxy**

Use OAuth2 Proxy for Google/GitHub authentication:
```bash
# Install oauth2-proxy
wget https://github.com/oauth2-proxy/oauth2-proxy/releases/download/v7.5.0/oauth2-proxy-v7.5.0.linux-amd64.tar.gz
tar xzf oauth2-proxy-v7.5.0.linux-amd64.tar.gz
sudo mv oauth2-proxy-v7.5.0.linux-amd64/oauth2-proxy /usr/local/bin/
```

See https://oauth2-proxy.github.io/oauth2-proxy/ for configuration.

### 2. Network Security

**Firewall Rules:**
```bash
# Only allow necessary ports
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

**SSH Hardening:**
```bash
# Disable password auth, use SSH keys only
sudo nano /etc/ssh/sshd_config
# Set: PasswordAuthentication no
sudo systemctl restart sshd
```

### 3. Directory Whitelisting

Enable path traversal protection:
```bash
# In .env or systemd service
ALLOWED_DOCS_DIRS=/var/lib/c64kb-data/uploads,/var/lib/c64kb-data/imports
```

### 4. Rate Limiting (Nginx)

```nginx
# Add to nginx.conf
limit_req_zone $binary_remote_addr zone=c64kb:10m rate=10r/s;

server {
    location / {
        limit_req zone=c64kb burst=20 nodelay;
        # ... proxy config
    }
}
```

### 5. HTTPS Only

Force HTTPS redirects:
```nginx
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}
```

### 6. Regular Updates

```bash
# Create update script: /opt/scripts/update_c64kb.sh
#!/bin/bash
cd /opt/tdz-c64-knowledge
git pull
source .venv/bin/activate
pip install --upgrade -e ".[dev,gui]"
sudo systemctl restart c64kb-gui
```

Schedule monthly via cron:
```bash
0 2 1 * * /opt/scripts/update_c64kb.sh >> /var/log/c64kb-update.log 2>&1
```

---

## Monitoring & Maintenance

### 1. Log Management

**Systemd Logs:**
```bash
# View GUI logs
sudo journalctl -u c64kb-gui.service -f

# Last 100 lines
sudo journalctl -u c64kb-gui.service -n 100

# Filter by time
sudo journalctl -u c64kb-gui.service --since "2025-12-13 10:00"
```

**Application Logs:**
```bash
# Check server.log
tail -f /opt/tdz-c64-knowledge/server.log

# Rotate logs with logrotate
sudo nano /etc/logrotate.d/c64kb
```

Add:
```
/opt/tdz-c64-knowledge/server.log {
    daily
    rotate 14
    compress
    delaycompress
    notifempty
    create 0644 ubuntu ubuntu
    sharedscripts
    postrotate
        systemctl reload c64kb-gui.service > /dev/null
    endscript
}
```

### 2. Health Monitoring

**Create Health Check Script:**
```bash
#!/bin/bash
# /opt/scripts/health_check.sh

URL="http://localhost:8501/_stcore/health"
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" $URL)

if [ $RESPONSE -ne 200 ]; then
    echo "Health check failed: $RESPONSE"
    sudo systemctl restart c64kb-gui.service
    # Send alert email (optional)
    echo "C64 Knowledge Base health check failed. Service restarted." | mail -s "Alert: C64KB Down" admin@example.com
fi
```

**Schedule via Cron:**
```bash
*/5 * * * * /opt/scripts/health_check.sh >> /var/log/c64kb-health.log 2>&1
```

### 3. Resource Monitoring

**Install monitoring tools:**
```bash
sudo apt install htop iotop -y
```

**Monitor usage:**
```bash
# CPU/Memory
htop

# Disk usage
df -h
du -sh /var/lib/c64kb-data/*

# Database size
ls -lh /var/lib/c64kb-data/knowledge_base.db
```

### 4. Performance Metrics

Enable analytics in GUI and review:
- Search query performance
- Popular queries
- Failed searches (knowledge gaps)
- Search mode usage

---

## Backup Automation

### Windows Task Scheduler

**Create Backup Script:** `C:\scripts\backup_c64kb.bat`
```batch
@echo off
set BACKUP_DIR=D:\backups\c64kb
set DATA_DIR=C:\c64kb-data
set RETENTION_DAYS=30

cd C:\deployment\tdz-c64-knowledge
call .venv\Scripts\activate.bat

python -c "from server import KnowledgeBase; kb = KnowledgeBase('%DATA_DIR%'); kb.create_backup('%BACKUP_DIR%', compress=True); kb.close()"

REM Delete backups older than 30 days
forfiles /p "%BACKUP_DIR%" /s /m *.zip /d -%RETENTION_DAYS% /c "cmd /c del @path"

echo Backup completed at %date% %time% >> C:\scripts\backup_c64kb.log
```

**Schedule Task:**
1. Open Task Scheduler
2. Create Basic Task
3. Name: "C64 Knowledge Base Backup"
4. Trigger: Daily, 2:00 AM
5. Action: Start Program
   - Program: `C:\scripts\backup_c64kb.bat`
6. Finish

### Linux Cron

**Create Backup Script:** `/opt/scripts/backup_c64kb.sh`
```bash
#!/bin/bash

BACKUP_DIR=/backups/c64kb
DATA_DIR=/var/lib/c64kb-data
RETENTION_DAYS=30
LOG_FILE=/var/log/c64kb-backup.log

cd /opt/tdz-c64-knowledge
source .venv/bin/activate

# Create backup
python -c "from server import KnowledgeBase; kb = KnowledgeBase('$DATA_DIR'); kb.create_backup('$BACKUP_DIR', compress=True); kb.close()"

# Delete old backups
find $BACKUP_DIR -name "kb_backup_*.zip" -mtime +$RETENTION_DAYS -delete

echo "$(date): Backup completed" >> $LOG_FILE
```

**Make executable:**
```bash
chmod +x /opt/scripts/backup_c64kb.sh
```

**Schedule via Cron:**
```bash
crontab -e
```

Add:
```cron
# Daily backup at 2 AM
0 2 * * * /opt/scripts/backup_c64kb.sh

# Weekly full backup to external storage (Sundays at 3 AM)
0 3 * * 0 /opt/scripts/backup_c64kb.sh && cp -r /backups/c64kb /mnt/external/c64kb-backups/
```

### Cloud Storage Sync

**AWS S3 Sync:**
```bash
# Install AWS CLI
sudo apt install awscli -y
aws configure

# Add to backup script
aws s3 sync /backups/c64kb s3://your-bucket/c64kb-backups/ --exclude "*" --include "kb_backup_*.zip"
```

**Azure Blob Storage:**
```bash
# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
az login

# Add to backup script
az storage blob upload-batch --destination c64kb-backups --source /backups/c64kb --account-name youraccount
```

---

## Performance Optimization

### 1. Database Optimization

**Enable FTS5:**
```bash
.venv/bin/python enable_fts5.py
```

**Enable Semantic Search:**
```bash
.venv/bin/python enable_semantic_search.py
```

**Verify Indexes:**
```bash
sqlite3 /var/lib/c64kb-data/knowledge_base.db "PRAGMA index_list('chunks');"
sqlite3 /var/lib/c64kb-data/knowledge_base.db "PRAGMA index_list('documents');"
```

### 2. System Resources

**Increase File Descriptors:**
```bash
# Edit /etc/security/limits.conf
ubuntu soft nofile 65536
ubuntu hard nofile 65536
```

**Optimize Python:**
```bash
# Use uvloop for async performance (if needed)
pip install uvloop
```

### 3. Caching

The knowledge base includes:
- 5-minute search result cache (automatic)
- BM25 index caching (automatic)
- Embeddings caching (persistent to disk)

No additional configuration needed.

### 4. Load Balancing (Future)

For high-traffic scenarios:
- Deploy multiple GUI instances
- Use Nginx upstream load balancing
- Share data directory via NFS/EFS
- Consider PostgreSQL migration for concurrent writes

---

## Troubleshooting

### Service Won't Start

**Check logs:**
```bash
sudo journalctl -u c64kb-gui.service -n 50
```

**Common issues:**
- Missing data directory: Create `/var/lib/c64kb-data`
- Permission denied: `sudo chown ubuntu:ubuntu /var/lib/c64kb-data`
- Port in use: Change port or stop conflicting service
- Python import errors: Reinstall dependencies

### High Memory Usage

**Symptoms:** GUI becomes slow, OOM errors

**Solutions:**
1. Increase instance RAM (4 GB minimum)
2. Disable semantic search if not needed:
   ```bash
   export USE_SEMANTIC_SEARCH=0
   ```
3. Reduce chunk size in search (decrease max_results)

### Slow Search Performance

**Check FTS5 enabled:**
```bash
sqlite3 /var/lib/c64kb-data/knowledge_base.db "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_fts';"
```

Should return `chunks_fts`. If not, run:
```bash
.venv/bin/python enable_fts5.py
```

### SSL Certificate Renewal Fails

**Test renewal:**
```bash
sudo certbot renew --dry-run
```

**Manual renewal:**
```bash
sudo certbot renew
sudo systemctl reload nginx
```

### Database Corruption

**Restore from backup:**
```bash
cd /opt/tdz-c64-knowledge
source .venv/bin/activate
python -c "from server import KnowledgeBase; kb = KnowledgeBase(); kb.restore_from_backup('/backups/c64kb/kb_backup_YYYYMMDD_HHMMSS.zip'); kb.close()"
```

---

## Production Checklist

Before going live, verify:

- [ ] All dependencies installed
- [ ] Data directory configured and writable
- [ ] FTS5 enabled and tested
- [ ] Semantic search enabled (optional)
- [ ] Documents imported successfully
- [ ] Health check passing
- [ ] Backups configured and tested
- [ ] SSL certificate installed (if public)
- [ ] Firewall rules configured
- [ ] Authentication enabled (if needed)
- [ ] Monitoring and logging setup
- [ ] Service auto-start enabled
- [ ] Restore procedure tested
- [ ] Update procedure documented
- [ ] Emergency contacts configured

---

## Support and Resources

- **Documentation:** See README.md, USER_GUIDE.md, CLAUDE.md
- **Logs:** `/var/log/` (Linux) or Event Viewer (Windows)
- **Health Check:** `http://localhost:8501/_stcore/health`
- **Database:** `/var/lib/c64kb-data/knowledge_base.db`
- **Backups:** Configure retention based on your requirements

---

**Version:** 2.5.0
**Last Updated:** 2025-12-13

Built with Claude Code
