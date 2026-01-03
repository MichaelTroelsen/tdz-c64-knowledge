# Security Guide

**Version:** 2.23.1
**Last Updated:** 2026-01-03

Security best practices and configuration for the TDZ C64 Knowledge Base.

---

## 📋 Table of Contents

- [Security Overview](#security-overview)
- [Path Traversal Protection](#path-traversal-protection)
- [API Security](#api-security)
- [Database Security](#database-security)
- [LLM API Security](#llm-api-security)
- [Production Deployment](#production-deployment)
- [Security Checklist](#security-checklist)

---

## 🔒 Security Overview

### Threat Model

The TDZ C64 Knowledge Base handles:
- **Local file access** - Reading documents from disk
- **Database operations** - SQLite read/write
- **External APIs** - LLM providers, web scraping
- **Network services** - Optional REST API

**Attack vectors to protect against:**
1. Path traversal attacks (accessing files outside allowed directories)
2. SQL injection (prevented by parameterized queries)
3. Unauthorized API access (REST API)
4. API key exposure (LLM providers)
5. XSS in web interfaces (Streamlit GUI)
6. Denial of service (resource exhaustion)

---

## 🛡️ Path Traversal Protection

### Overview

**Risk:** Malicious file paths could access sensitive files:
```python
# Attack attempt
add_document("../../../../etc/passwd")
add_document("C:\\Windows\\System32\\config\\SAM")
```

### Configuration

**Enable whitelist protection:**
```cmd
set ALLOWED_DOCS_DIRS=C:\safe\docs;D:\trusted\files;E:\backups
```

**Multiple directories (Windows):**
```cmd
set ALLOWED_DOCS_DIRS=C:\c64docs;C:\retro\manuals;D:\backups\c64
```

**Multiple directories (Linux/Mac):**
```bash
export ALLOWED_DOCS_DIRS=/data/c64docs:/backup/docs
```

**Disable (not recommended):**
```cmd
set ALLOWED_DOCS_DIRS=
```

---

### How It Works

1. **Path normalization:**
   - Resolves `..` and `.` components
   - Expands to absolute path
   - Normalizes path separators

2. **Whitelist check:**
   - Verifies path starts with an allowed directory
   - Prevents escaping with `../`

3. **Error on violation:**
   ```python
   SecurityError: Path outside allowed directories
   ```

---

### Best Practices

**✅ DO:**
- Set `ALLOWED_DOCS_DIRS` in production
- Use absolute paths in whitelist
- Include all legitimate document directories
- Review whitelist periodically

**❌ DON'T:**
- Disable protection in production
- Add overly broad paths (`C:\`, `/`)
- Trust user-provided paths without validation

---

## 🔐 API Security

### REST API Authentication

**Enable API keys:**
```cmd
set TDZ_API_KEYS=secret-key-1,secret-key-2,admin-key
```

**Use in requests:**
```bash
curl -H "X-API-Key: secret-key-1" http://localhost:8000/api/v1/search
```

**Production recommendations:**
- Use strong, random keys (32+ characters)
- Rotate keys periodically
- Use different keys for different clients
- Never commit keys to version control

---

### Generate Secure API Keys

**Python:**
```python
import secrets
api_key = secrets.token_urlsafe(32)
print(api_key)
# Example: 8f4d-KjNm_3Qp7RtYuW9Zx2CvBnM5aS
```

**PowerShell:**
```powershell
[System.Convert]::ToBase64String((1..32 | %{Get-Random -Minimum 0 -Maximum 256}))
```

---

### CORS Configuration

**Restrict origins:**
```cmd
set CORS_ORIGINS=https://app.example.com,https://admin.example.com
```

**Development (allow localhost):**
```cmd
set CORS_ORIGINS=http://localhost:3000,http://localhost:8501
```

**Production:**
```cmd
set CORS_ORIGINS=https://yourdomain.com
```

**Never use** `*` in production:
```cmd
# ❌ INSECURE
set CORS_ORIGINS=*
```

---

### HTTPS/TLS

**The REST API server doesn't include TLS.** Use a reverse proxy:

**nginx example:**
```nginx
server {
    listen 443 ssl;
    server_name api.example.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

### Rate Limiting

**Currently not implemented in application.**

**Workarounds:**
1. Use reverse proxy rate limiting (nginx, caddy)
2. Firewall rules
3. Cloud provider rate limiting (Cloudflare, AWS WAF)

**nginx example:**
```nginx
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

location /api/ {
    limit_req zone=api burst=20;
    proxy_pass http://127.0.0.1:8000;
}
```

---

## 💾 Database Security

### File Permissions

**Restrict database file access:**
```cmd
# Windows
icacls knowledge_base.db /inheritance:r /grant:r "%USERNAME%:F"

# Linux/Mac
chmod 600 knowledge_base.db
```

---

### Backup Security

**Encrypt backups:**
```bash
# Create encrypted backup
tar czf - knowledge_base.db | gpg --symmetric --cipher-algo AES256 > backup.tar.gz.gpg

# Restore
gpg --decrypt backup.tar.gz.gpg | tar xzf -
```

**Store securely:**
- Use encrypted storage
- Offsite backups
- Access control
- Regular backup rotation

---

### SQL Injection Prevention

**The codebase uses parameterized queries** (automatically safe):
```python
# ✅ SAFE - Parameterized query
cursor.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))

# ❌ NEVER DO THIS - String interpolation
cursor.execute(f"SELECT * FROM documents WHERE doc_id = '{doc_id}'")
```

**All database operations are safe by design.**

---

## 🤖 LLM API Security

### API Key Management

**Store keys in environment variables:**
```cmd
set ANTHROPIC_API_KEY=sk-ant-xxxxx
set OPENAI_API_KEY=sk-xxxxx
```

**Never:**
- ❌ Hardcode in source code
- ❌ Commit to version control
- ❌ Share in logs or error messages
- ❌ Include in client-side code

---

### Rotate Keys

**If key is compromised:**
1. Generate new key from provider dashboard
2. Update environment variable
3. Restart server
4. Revoke old key

---

### Cost Control

**Prevent excessive API usage:**
```cmd
# Limit bulk operations
set MAX_BULK_EXTRACTION_DOCS=100

# Use confidence thresholds to reduce calls
set ENTITY_CONFIDENCE_THRESHOLD=0.7

# Cache results
set LLM_CACHE_ENABLED=1
```

**Monitor usage:**
- Check provider dashboards
- Set up billing alerts
- Review extraction jobs regularly

---

## 🚀 Production Deployment

### Security Checklist

#### Required
- [ ] Enable `ALLOWED_DOCS_DIRS` whitelist
- [ ] Set `TDZ_API_KEYS` for REST API
- [ ] Configure `CORS_ORIGINS` (not `*`)
- [ ] Use HTTPS reverse proxy
- [ ] Secure database file permissions
- [ ] Store API keys in environment (not code)
- [ ] Enable logging (`LOG_LEVEL=INFO`)

#### Recommended
- [ ] Run as non-root/non-admin user
- [ ] Use dedicated data directory
- [ ] Implement regular backups
- [ ] Monitor server logs
- [ ] Set up intrusion detection
- [ ] Use firewall rules
- [ ] Rotate API keys periodically

#### Optional
- [ ] Enable two-factor auth for admin access
- [ ] Use VPN for remote access
- [ ] Implement audit logging
- [ ] Set up security scanning
- [ ] Use secrets management (Vault, AWS Secrets Manager)

---

### Environment Variable Security

**Development (.env file):**
```bash
# .env (add to .gitignore!)
TDZ_DATA_DIR=C:\dev\kb-test
ANTHROPIC_API_KEY=sk-ant-xxxxx
TDZ_API_KEYS=dev-key-1,dev-key-2
```

**Production (system environment):**
```cmd
# Windows - System environment variables
setx TDZ_DATA_DIR "C:\production\kb" /M
setx TDZ_API_KEYS "prod-key-xxxxx" /M

# Linux/Mac - /etc/environment or systemd
Environment="TDZ_DATA_DIR=/var/lib/tdz-c64-kb"
Environment="TDZ_API_KEYS=prod-key-xxxxx"
```

**Docker secrets:**
```yaml
services:
  tdz-kb:
    image: tdz-c64-kb:latest
    secrets:
      - anthropic_key
      - api_keys
    environment:
      ANTHROPIC_API_KEY_FILE: /run/secrets/anthropic_key
      TDZ_API_KEYS_FILE: /run/secrets/api_keys

secrets:
  anthropic_key:
    external: true
  api_keys:
    external: true
```

---

### Network Security

**Bind to localhost only:**
```cmd
# REST API - local only
set API_HOST=127.0.0.1
set API_PORT=8000
```

**Use reverse proxy for external access:**
```nginx
# Public access via proxy
server {
    listen 443 ssl;
    server_name api.example.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
    }
}
```

**Firewall rules:**
```cmd
# Windows Firewall - Block external access to port 8000
netsh advfirewall firewall add rule name="Block KB API" dir=in action=block protocol=TCP localport=8000

# Linux iptables
iptables -A INPUT -p tcp --dport 8000 -s 127.0.0.1 -j ACCEPT
iptables -A INPUT -p tcp --dport 8000 -j DROP
```

---

### Logging & Monitoring

**Enable security logging:**
```cmd
set LOG_LEVEL=INFO
set LOG_FILE=%TDZ_DATA_DIR%\server.log
```

**Monitor for:**
- Failed authentication attempts
- Path traversal attempts
- Unusual API usage patterns
- Database errors
- System resource usage

**Log rotation:**
```cmd
# Windows - Use Task Scheduler to run daily
powershell -Command "Get-Content -Path $env:TDZ_DATA_DIR\server.log -Tail 10000 | Set-Content -Path $env:TDZ_DATA_DIR\server.log"

# Linux - logrotate
/var/log/tdz-kb/*.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
}
```

---

### Update Management

**Keep dependencies updated:**
```cmd
# Check for updates
pip list --outdated

# Update specific packages
pip install --upgrade mcp pypdf

# Update all (test first!)
pip install --upgrade -r requirements.txt
```

**Security patches:**
- Subscribe to security advisories for dependencies
- Review CHANGELOG.md for security fixes
- Test updates in dev before production
- Have rollback plan

---

## 🔍 Security Auditing

### Regular Checks

**Monthly:**
- Review access logs
- Check API key usage
- Verify whitelist is current
- Scan for vulnerable dependencies

**Quarterly:**
- Rotate API keys
- Review and update firewall rules
- Test backup restoration
- Security patch updates

**Annually:**
- Full security audit
- Penetration testing (if applicable)
- Review and update security policies

---

### Vulnerability Scanning

**Check Python dependencies:**
```cmd
pip install safety
safety check
```

**Check for common issues:**
```cmd
pip install bandit
bandit -r server.py cli.py admin_gui.py
```

---

## 📚 Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CWE/SANS Top 25](https://cwe.mitre.org/top25/)
- [Python Security Best Practices](https://python.readthedocs.io/en/stable/library/security_warnings.html)

---

## 🆘 Security Incidents

### If Security Breach Suspected

1. **Isolate the system:**
   - Stop the server
   - Disconnect from network (if needed)

2. **Assess the damage:**
   - Check logs for unauthorized access
   - Review database for modifications
   - Check file system for changes

3. **Contain the breach:**
   - Rotate all API keys
   - Change passwords
   - Update firewall rules

4. **Recover:**
   - Restore from clean backup
   - Apply security patches
   - Update configurations

5. **Learn:**
   - Document incident
   - Improve security measures
   - Update procedures

---

**Version:** 2.23.1
**Platform:** Windows (with Linux/Mac notes)
**Last Updated:** 2026-01-03

**Remember:** Security is an ongoing process, not a one-time setup!
