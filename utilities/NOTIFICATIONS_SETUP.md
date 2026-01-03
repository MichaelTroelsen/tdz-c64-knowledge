# Webhook Notifications Setup Guide

**Version:** 2.23.15
**Last Updated:** 2026-01-03

Complete guide for setting up webhook notifications in the TDZ C64 Knowledge Base monitoring system.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Supported Channels](#supported-channels)
- [Quick Start](#quick-start)
- [Discord Setup](#discord-setup)
- [Slack Setup](#slack-setup)
- [Generic Webhook Setup](#generic-webhook-setup)
- [Email Setup](#email-setup)
- [Configuration Reference](#configuration-reference)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

The monitoring scripts (`monitor_daily.py` and `monitor_weekly.py`) can send notifications when:
- Documents have been updated
- New pages are discovered (weekly check)
- Pages become missing or inaccessible (weekly check)
- Check failures occur

Notifications are configured in `monitor_config.json` and support multiple channels simultaneously.

---

## 📡 Supported Channels

| Channel | Format | Rich Formatting | Best For |
|---------|--------|----------------|----------|
| **Discord** | Webhook (embeds) | ✅ Yes | Team chat, gaming communities |
| **Slack** | Webhook (blocks) | ✅ Yes | Enterprise teams |
| **Generic Webhook** | JSON POST | ❌ No | Custom integrations, IFTTT, Zapier |
| **Email** | SMTP (HTML + text) | ✅ Yes | Email alerts, formal reports |

---

## 🚀 Quick Start

### 1. Enable Notifications

Edit `monitor_config.json`:

```json
{
  "data_dir": "~/.tdz-c64-knowledge",
  "notifications": {
    "enabled": true,
    "discord": {
      "enabled": true,
      "webhook_url": "YOUR_DISCORD_WEBHOOK_URL_HERE"
    }
  }
}
```

### 2. Run with Notifications

```cmd
python utilities/monitor_daily.py --notify
```

---

## 💬 Discord Setup

### Create Discord Webhook

1. **Open Discord** and go to your server
2. **Right-click the channel** where you want notifications
3. Select **Edit Channel** → **Integrations** → **Webhooks**
4. Click **New Webhook**
5. **Customize:**
   - Name: "TDZ C64 Monitor"
   - Avatar: (optional)
6. **Copy Webhook URL**

### Configure

Add to `monitor_config.json`:

```json
{
  "notifications": {
    "enabled": true,
    "discord": {
      "enabled": true,
      "webhook_url": "https://discord.com/api/webhooks/123456789/abcdef...",
      "username": "TDZ C64 Monitor"
    }
  }
}
```

### Example Discord Message

![Discord notification with embed showing changed documents, new pages, and statistics]

**Features:**
- Color-coded embeds (green = all good, orange = attention needed, red = errors)
- Clickable links to changed documents
- Summary statistics
- Timestamp

---

## 📢 Slack Setup

### Create Slack Webhook

1. **Go to:** https://api.slack.com/apps
2. **Create New App** → From scratch
3. Name: "TDZ C64 Monitor"
4. Choose workspace
5. **Activate Incoming Webhooks** (toggle on)
6. **Add New Webhook to Workspace**
7. Select channel
8. **Copy Webhook URL**

### Configure

Add to `monitor_config.json`:

```json
{
  "notifications": {
    "enabled": true,
    "slack": {
      "enabled": true,
      "webhook_url": "https://hooks.slack.com/services/T00000000/B00000000/XXXX",
      "username": "TDZ C64 Monitor",
      "icon_emoji": ":robot_face:"
    }
  }
}
```

### Example Slack Message

**Features:**
- Block Kit formatting
- Clickable document links
- Summary fields
- Custom username and icon

---

## 🔗 Generic Webhook Setup

For custom integrations, IFTTT, Zapier, n8n, etc.

### Configure

```json
{
  "notifications": {
    "enabled": true,
    "webhook": {
      "enabled": true,
      "url": "https://your-server.com/webhook-endpoint",
      "headers": {
        "Authorization": "Bearer YOUR_SECRET_TOKEN",
        "X-Custom-Header": "value"
      }
    }
  }
}
```

### Payload Format

The webhook receives a JSON POST with this structure:

```json
{
  "type": "monitoring_check",
  "check_type": "daily",
  "timestamp": "2026-01-03T12:00:00.000000",
  "results": {
    "unchanged": 45,
    "changed": 2,
    "new_pages": 0,
    "missing_pages": 0,
    "failed": 0
  },
  "details": {
    "changed": [
      {
        "doc_id": "abc123...",
        "title": "VIC-II Programming Guide",
        "url": "https://example.com/vic-ii",
        "last_modified": "2026-01-03T11:30:00",
        "reason": "Last-Modified header changed"
      }
    ],
    "new_pages": [],
    "missing_pages": [],
    "failed": []
  }
}
```

### Example Integrations

**IFTTT:**
1. Create applet: "Webhooks → Your Action"
2. Use webhook URL from IFTTT
3. Process JSON in action filter

**Zapier:**
1. Create Zap: "Webhooks by Zapier (Catch Hook) → Your App"
2. Use webhook URL from Zapier
3. Map JSON fields in next step

**n8n:**
1. Add "Webhook" node (trigger)
2. Copy webhook URL
3. Add processing nodes

---

## 📧 Email Setup

### Gmail Setup (Recommended)

1. **Enable 2-Factor Authentication** on your Google account
2. **Create App Password:**
   - Go to: https://myaccount.google.com/apppasswords
   - Select app: "Mail"
   - Select device: "Other" → "TDZ Monitor"
   - **Copy the 16-character password**

### Configure

```json
{
  "notifications": {
    "enabled": true,
    "email": {
      "enabled": true,
      "smtp_server": "smtp.gmail.com",
      "smtp_port": 587,
      "from_address": "your-monitoring@gmail.com",
      "to_address": "admin@example.com",
      "username": "your-monitoring@gmail.com",
      "password": "your-16-char-app-password"
    }
  }
}
```

### Other Email Providers

**Outlook/Office 365:**
```json
{
  "smtp_server": "smtp.office365.com",
  "smtp_port": 587
}
```

**Yahoo:**
```json
{
  "smtp_server": "smtp.mail.yahoo.com",
  "smtp_port": 587
}
```

**Custom SMTP:**
```json
{
  "smtp_server": "mail.your-domain.com",
  "smtp_port": 465,  // or 587 for STARTTLS
}
```

---

## ⚙️ Configuration Reference

### Complete Example

```json
{
  "data_dir": "~/.tdz-c64-knowledge",
  "notifications": {
    "enabled": true,
    "discord": {
      "enabled": true,
      "webhook_url": "https://discord.com/api/webhooks/...",
      "username": "TDZ C64 Monitor"
    },
    "slack": {
      "enabled": true,
      "webhook_url": "https://hooks.slack.com/services/...",
      "username": "TDZ C64 Monitor",
      "icon_emoji": ":robot_face:"
    },
    "webhook": {
      "enabled": false,
      "url": "https://your-server.com/webhook",
      "headers": {
        "Authorization": "Bearer token"
      }
    },
    "email": {
      "enabled": false,
      "smtp_server": "smtp.gmail.com",
      "smtp_port": 587,
      "from_address": "monitor@example.com",
      "to_address": "admin@example.com",
      "username": "monitor@example.com",
      "password": "app-password-here"
    }
  }
}
```

### Options

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `enabled` | boolean | Yes | false | Master switch for all notifications |
| `discord.enabled` | boolean | No | false | Enable Discord notifications |
| `discord.webhook_url` | string | Yes* | - | Discord webhook URL |
| `discord.username` | string | No | "TDZ C64 Monitor" | Bot username |
| `slack.enabled` | boolean | No | false | Enable Slack notifications |
| `slack.webhook_url` | string | Yes* | - | Slack webhook URL |
| `slack.username` | string | No | "TDZ C64 Monitor" | Bot username |
| `slack.icon_emoji` | string | No | ":robot_face:" | Bot icon emoji |
| `webhook.enabled` | boolean | No | false | Enable generic webhook |
| `webhook.url` | string | Yes* | - | Webhook endpoint URL |
| `webhook.headers` | object | No | {} | Custom HTTP headers |
| `email.enabled` | boolean | No | false | Enable email notifications |
| `email.smtp_server` | string | Yes* | - | SMTP server hostname |
| `email.smtp_port` | number | Yes* | 587 | SMTP server port |
| `email.from_address` | string | Yes* | - | From email address |
| `email.to_address` | string | Yes* | - | To email address |
| `email.username` | string | No | - | SMTP authentication username |
| `email.password` | string | No | - | SMTP authentication password |

*Required if that channel is enabled

---

## 🧪 Testing

### Test Notifications

```cmd
# Test all enabled channels
python utilities/notifications.py

# Or using Python directly
python -c "from utilities.notifications import send_test_notification; import json; config = json.load(open('utilities/monitor_config.json')); send_test_notification(config)"
```

### Test Individual Channels

Edit `monitor_config.json` to enable only the channel you want to test, then run:

```cmd
python utilities/monitor_daily.py --notify --output test_daily.json
```

### Expected Output

```
[*] Sending notifications...
  ✓ discord
  ✗ slack
  ✓ email
```

- ✓ = Success
- ✗ = Failed (check error message above)

---

## 🔍 Troubleshooting

### Discord Issues

**Error:** "Invalid Webhook Token"
- **Solution:** Re-copy webhook URL from Discord (don't edit the URL)

**Error:** "Unknown Webhook"
- **Solution:** Webhook was deleted, create a new one

**No embeds showing:**
- **Solution:** Check channel permissions, ensure bot can send embeds

---

### Slack Issues

**Error:** "Invalid token"
- **Solution:** Re-create webhook from Slack app settings

**Error:** "Channel not found"
- **Solution:** Re-authorize webhook for the correct channel

**Formatting looks wrong:**
- **Solution:** Slack workspace may have restricted Block Kit, contact admin

---

### Email Issues

**Error:** "Authentication failed"
- **Solution:**
  - Gmail: Use App Password, not account password
  - Enable "Less secure apps" if using regular password (not recommended)

**Error:** "Connection refused"
- **Solution:** Check SMTP server and port are correct

**Error:** "Certificate verification failed"
- **Solution:**
  - Check SMTP server hostname is correct
  - Try port 465 instead of 587

---

### Generic Webhook Issues

**Error:** "Connection timeout"
- **Solution:** Check URL is accessible from your network

**Error:** "401 Unauthorized"
- **Solution:** Check Authorization header is correct

**No data received:**
- **Solution:** Check your endpoint is handling POST requests with JSON body

---

## 📝 Usage Examples

### Daily Monitoring with Notifications

```cmd
# Quick check, notify if changes
python utilities/monitor_daily.py --notify

# Quick check, auto-rescrape changes, notify
python utilities/monitor_daily.py --auto-rescrape --notify
```

### Weekly Monitoring with Notifications

```cmd
# Full check with structure discovery, notify if changes
python utilities/monitor_weekly.py --notify

# Full check, auto-rescrape, notify
python utilities/monitor_weekly.py --auto-rescrape --notify
```

### Windows Task Scheduler

Add `--notify` to scheduled tasks:

```cmd
C:\path\to\.venv\Scripts\python.exe C:\path\to\utilities\monitor_daily.py --notify --output %TEMP%\daily_check.json
```

---

## 🔒 Security Best Practices

### Webhook URLs
- **Never commit webhook URLs to git** (add `monitor_config.json` to `.gitignore`)
- Rotate webhooks periodically
- Use environment variables for sensitive data:

```python
# In monitor_config.json:
{
  "discord": {
    "webhook_url": "${DISCORD_WEBHOOK_URL}"
  }
}
```

### Email Passwords
- Use app-specific passwords, not account passwords
- Store in environment variables or password manager
- Never share SMTP credentials

### API Keys
- Rotate regularly (every 90 days)
- Use minimum required permissions
- Monitor usage for unauthorized access

---

## 📚 Additional Resources

- [Discord Webhooks Documentation](https://discord.com/developers/docs/resources/webhook)
- [Slack Incoming Webhooks](https://api.slack.com/messaging/webhooks)
- [Gmail App Passwords](https://support.google.com/accounts/answer/185833)
- [IFTTT Webhooks](https://ifttt.com/maker_webhooks)

---

**Version:** 2.23.15
**Last Updated:** 2026-01-03

**Need help?** See [TROUBLESHOOTING.md](../docs/TROUBLESHOOTING.md) or open a GitHub issue.
