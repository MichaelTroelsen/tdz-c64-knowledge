#!/usr/bin/env python3
"""
Notification Module for TDZ C64 Knowledge Base Monitoring

Supports multiple notification channels:
- Discord webhooks
- Slack webhooks
- Generic webhooks (POST JSON)
- Email (SMTP)

Usage:
    from notifications import NotificationManager

    manager = NotificationManager(config)
    manager.send_monitoring_alert(results, "daily")
"""

import json
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, List, Any, Optional


class NotificationManager:
    """Manages sending notifications through various channels."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize notification manager with configuration.

        Args:
            config: Configuration dictionary with notification settings
        """
        self.config = config
        self.notifications_config = config.get('notifications', {})
        self.enabled = self.notifications_config.get('enabled', False)

    def send_monitoring_alert(self, results: Dict[str, Any], check_type: str = "daily") -> Dict[str, bool]:
        """
        Send monitoring alerts through configured channels.

        Args:
            results: Monitoring results dictionary
            check_type: Type of check ("daily" or "weekly")

        Returns:
            Dictionary of {channel: success_bool}
        """
        if not self.enabled:
            return {"status": "disabled"}

        status = {}

        # Send to Discord
        if self.notifications_config.get('discord', {}).get('enabled'):
            try:
                self._send_discord(results, check_type)
                status['discord'] = True
            except Exception as e:
                print(f"[ERROR] Discord notification failed: {e}")
                status['discord'] = False

        # Send to Slack
        if self.notifications_config.get('slack', {}).get('enabled'):
            try:
                self._send_slack(results, check_type)
                status['slack'] = True
            except Exception as e:
                print(f"[ERROR] Slack notification failed: {e}")
                status['slack'] = False

        # Send to generic webhook
        if self.notifications_config.get('webhook', {}).get('enabled'):
            try:
                self._send_webhook(results, check_type)
                status['webhook'] = True
            except Exception as e:
                print(f"[ERROR] Webhook notification failed: {e}")
                status['webhook'] = False

        # Send email
        if self.notifications_config.get('email', {}).get('enabled'):
            try:
                self._send_email(results, check_type)
                status['email'] = True
            except Exception as e:
                print(f"[ERROR] Email notification failed: {e}")
                status['email'] = False

        return status

    def _send_discord(self, results: Dict[str, Any], check_type: str):
        """Send notification to Discord webhook."""
        discord_config = self.notifications_config['discord']
        webhook_url = discord_config.get('webhook_url')

        if not webhook_url:
            raise ValueError("Discord webhook_url not configured")

        # Build Discord embed
        embed = self._build_discord_embed(results, check_type)

        payload = {
            "username": discord_config.get('username', 'TDZ C64 Knowledge Monitor'),
            "embeds": [embed]
        }

        # Send with retry
        response = requests.post(
            webhook_url,
            json=payload,
            timeout=10
        )
        response.raise_for_status()

    def _build_discord_embed(self, results: Dict[str, Any], check_type: str) -> Dict[str, Any]:
        """Build Discord embed object."""
        changed = len(results.get('changed', []))
        failed = len(results.get('failed', []))
        new_pages = len(results.get('new_pages', []))
        missing = len(results.get('missing_pages', []))
        unchanged = len(results.get('unchanged', []))

        # Determine color based on results
        if changed > 0 or new_pages > 0 or missing > 0:
            color = 0xFFA500  # Orange - attention needed
        elif failed > 0:
            color = 0xFF0000  # Red - errors
        else:
            color = 0x00FF00  # Green - all good

        # Build description
        description = f"**{check_type.capitalize()} Monitoring Check Results**\n\n"
        description += f"✅ Unchanged: {unchanged}\n"

        if changed > 0:
            description += f"🔄 Changed: {changed}\n"
        if new_pages > 0:
            description += f"🆕 New Pages: {new_pages}\n"
        if missing > 0:
            description += f"❌ Missing: {missing}\n"
        if failed > 0:
            description += f"⚠️ Failed: {failed}\n"

        embed = {
            "title": f"TDZ C64 Knowledge Base - {check_type.capitalize()} Check",
            "description": description,
            "color": color,
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {
                "text": "TDZ C64 Knowledge Base Monitor"
            }
        }

        # Add fields for changed documents
        if changed > 0 and results.get('changed'):
            field_value = ""
            for doc in results['changed'][:5]:
                field_value += f"• [{doc['title']}]({doc.get('url', '')})\n"
            if changed > 5:
                field_value += f"... and {changed - 5} more\n"

            embed["fields"] = embed.get("fields", [])
            embed["fields"].append({
                "name": "🔄 Changed Documents",
                "value": field_value,
                "inline": False
            })

        # Add fields for new pages (weekly)
        if new_pages > 0 and results.get('new_pages'):
            field_value = ""
            for page in results['new_pages'][:5]:
                field_value += f"• {page['url']}\n"
            if new_pages > 5:
                field_value += f"... and {new_pages - 5} more\n"

            embed["fields"] = embed.get("fields", [])
            embed["fields"].append({
                "name": "🆕 New Pages Discovered",
                "value": field_value,
                "inline": False
            })

        return embed

    def _send_slack(self, results: Dict[str, Any], check_type: str):
        """Send notification to Slack webhook."""
        slack_config = self.notifications_config['slack']
        webhook_url = slack_config.get('webhook_url')

        if not webhook_url:
            raise ValueError("Slack webhook_url not configured")

        # Build Slack message
        message = self._build_slack_message(results, check_type)

        payload = {
            "username": slack_config.get('username', 'TDZ C64 Knowledge Monitor'),
            "icon_emoji": slack_config.get('icon_emoji', ':robot_face:'),
            "blocks": message
        }

        response = requests.post(
            webhook_url,
            json=payload,
            timeout=10
        )
        response.raise_for_status()

    def _build_slack_message(self, results: Dict[str, Any], check_type: str) -> List[Dict]:
        """Build Slack Block Kit message."""
        changed = len(results.get('changed', []))
        failed = len(results.get('failed', []))
        new_pages = len(results.get('new_pages', []))
        missing = len(results.get('missing_pages', []))
        unchanged = len(results.get('unchanged', []))

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"TDZ C64 Knowledge Base - {check_type.capitalize()} Check"
                }
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Unchanged:*\n{unchanged}"},
                    {"type": "mrkdwn", "text": f"*Changed:*\n{changed}"},
                ]
            }
        ]

        # Add weekly-specific fields
        if check_type == "weekly":
            blocks.append({
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*New Pages:*\n{new_pages}"},
                    {"type": "mrkdwn", "text": f"*Missing:*\n{missing}"},
                ]
            })

        # Add failed count
        if failed > 0:
            blocks.append({
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Failed Checks:*\n{failed}"}
                ]
            })

        # Add changed documents
        if changed > 0 and results.get('changed'):
            changed_text = "*Changed Documents:*\n"
            for doc in results['changed'][:5]:
                url = doc.get('url', '')
                changed_text += f"• <{url}|{doc['title']}>\n"
            if changed > 5:
                changed_text += f"... and {changed - 5} more\n"

            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": changed_text}
            })

        return blocks

    def _send_webhook(self, results: Dict[str, Any], check_type: str):
        """Send notification to generic webhook."""
        webhook_config = self.notifications_config['webhook']
        webhook_url = webhook_config.get('url')

        if not webhook_url:
            raise ValueError("Webhook URL not configured")

        # Build payload
        payload = {
            "type": "monitoring_check",
            "check_type": check_type,
            "timestamp": datetime.utcnow().isoformat(),
            "results": {
                "unchanged": len(results.get('unchanged', [])),
                "changed": len(results.get('changed', [])),
                "new_pages": len(results.get('new_pages', [])),
                "missing_pages": len(results.get('missing_pages', [])),
                "failed": len(results.get('failed', []))
            },
            "details": {
                "changed": results.get('changed', [])[:10],  # First 10
                "new_pages": results.get('new_pages', [])[:10],
                "missing_pages": results.get('missing_pages', [])[:10],
                "failed": results.get('failed', [])[:10]
            }
        }

        # Add custom headers if configured
        headers = webhook_config.get('headers', {})
        headers['Content-Type'] = 'application/json'

        response = requests.post(
            webhook_url,
            json=payload,
            headers=headers,
            timeout=10
        )
        response.raise_for_status()

    def _send_email(self, results: Dict[str, Any], check_type: str):
        """Send notification via email."""
        email_config = self.notifications_config['email']

        # Build email
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"TDZ C64 Knowledge - {check_type.capitalize()} Monitoring Report"
        msg['From'] = email_config.get('from_address')
        msg['To'] = email_config.get('to_address')

        # Build plain text version
        text_body = self._build_email_text(results, check_type)

        # Build HTML version
        html_body = self._build_email_html(results, check_type)

        # Attach both versions
        msg.attach(MIMEText(text_body, 'plain'))
        msg.attach(MIMEText(html_body, 'html'))

        # Send via SMTP
        smtp_server = email_config.get('smtp_server', 'smtp.gmail.com')
        smtp_port = email_config.get('smtp_port', 587)
        username = email_config.get('username')
        password = email_config.get('password')

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            if username and password:
                server.login(username, password)
            server.send_message(msg)

    def _build_email_text(self, results: Dict[str, Any], check_type: str) -> str:
        """Build plain text email body."""
        changed = len(results.get('changed', []))
        failed = len(results.get('failed', []))
        new_pages = len(results.get('new_pages', []))
        missing = len(results.get('missing_pages', []))
        unchanged = len(results.get('unchanged', []))

        body = f"TDZ C64 Knowledge Base - {check_type.capitalize()} Monitoring Report\n"
        body += "=" * 60 + "\n\n"
        body += f"Check completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        body += "SUMMARY:\n"
        body += f"  Unchanged:     {unchanged}\n"
        body += f"  Changed:       {changed}\n"

        if check_type == "weekly":
            body += f"  New Pages:     {new_pages}\n"
            body += f"  Missing Pages: {missing}\n"

        body += f"  Failed Checks: {failed}\n\n"

        if changed > 0:
            body += "-" * 60 + "\n"
            body += "CHANGED DOCUMENTS:\n"
            body += "-" * 60 + "\n"
            for doc in results.get('changed', [])[:10]:
                body += f"\n  • {doc['title']}\n"
                body += f"    URL: {doc.get('url', 'N/A')}\n"
            if changed > 10:
                body += f"\n  ... and {changed - 10} more\n"

        return body

    def _build_email_html(self, results: Dict[str, Any], check_type: str) -> str:
        """Build HTML email body."""
        changed = len(results.get('changed', []))
        failed = len(results.get('failed', []))
        new_pages = len(results.get('new_pages', []))
        missing = len(results.get('missing_pages', []))
        unchanged = len(results.get('unchanged', []))

        html = """
        <html>
          <head>
            <style>
              body { font-family: Arial, sans-serif; }
              .header { background-color: #4CAF50; color: white; padding: 20px; }
              .warning { background-color: #FFA500; }
              .error { background-color: #F44336; }
              .summary { margin: 20px; }
              .stat { display: inline-block; margin: 10px 20px; }
              .stat-label { font-weight: bold; }
              .stat-value { font-size: 24px; }
              .section { margin: 20px; }
              .doc-item { margin: 10px 0; padding: 10px; background-color: #f5f5f5; }
            </style>
          </head>
          <body>
        """

        # Header
        header_class = "header"
        if changed > 0 or new_pages > 0 or missing > 0:
            header_class += " warning"
        elif failed > 0:
            header_class += " error"

        html += f'<div class="{header_class}">'
        html += f'<h1>TDZ C64 Knowledge Base</h1>'
        html += f'<h2>{check_type.capitalize()} Monitoring Report</h2>'
        html += f'<p>{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>'
        html += '</div>'

        # Summary
        html += '<div class="summary">'
        html += f'<div class="stat"><div class="stat-label">Unchanged</div><div class="stat-value">{unchanged}</div></div>'
        html += f'<div class="stat"><div class="stat-label">Changed</div><div class="stat-value">{changed}</div></div>'

        if check_type == "weekly":
            html += f'<div class="stat"><div class="stat-label">New Pages</div><div class="stat-value">{new_pages}</div></div>'
            html += f'<div class="stat"><div class="stat-label">Missing</div><div class="stat-value">{missing}</div></div>'

        html += f'<div class="stat"><div class="stat-label">Failed</div><div class="stat-value">{failed}</div></div>'
        html += '</div>'

        # Changed documents
        if changed > 0:
            html += '<div class="section">'
            html += '<h3>Changed Documents</h3>'
            for doc in results.get('changed', [])[:10]:
                html += '<div class="doc-item">'
                html += f'<strong>{doc["title"]}</strong><br>'
                html += f'<a href="{doc.get("url", "#")}">{doc.get("url", "N/A")}</a>'
                html += '</div>'
            if changed > 10:
                html += f'<p>... and {changed - 10} more</p>'
            html += '</div>'

        html += '</body></html>'

        return html


def send_test_notification(config: Dict[str, Any], channel: str = "all"):
    """
    Send a test notification to verify configuration.

    Args:
        config: Configuration dictionary
        channel: Channel to test ("discord", "slack", "webhook", "email", or "all")
    """
    test_results = {
        'unchanged': [{'doc_id': 'test1'}],
        'changed': [
            {
                'doc_id': 'test2',
                'title': 'Test Changed Document',
                'url': 'https://example.com/test'
            }
        ],
        'new_pages': [],
        'missing_pages': [],
        'failed': []
    }

    manager = NotificationManager(config)

    print(f"\n[*] Sending test notification to {channel}...")
    status = manager.send_monitoring_alert(test_results, "test")

    print("\n[*] Test notification results:")
    for channel_name, success in status.items():
        status_str = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {channel_name}: {status_str}")


if __name__ == "__main__":
    # Example configuration
    example_config = {
        "notifications": {
            "enabled": True,
            "discord": {
                "enabled": True,
                "webhook_url": "https://discord.com/api/webhooks/YOUR_WEBHOOK_ID/YOUR_WEBHOOK_TOKEN",
                "username": "TDZ C64 Monitor"
            },
            "slack": {
                "enabled": False,
                "webhook_url": "https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
                "username": "TDZ C64 Monitor",
                "icon_emoji": ":robot_face:"
            },
            "webhook": {
                "enabled": False,
                "url": "https://your-server.com/webhook",
                "headers": {
                    "Authorization": "Bearer YOUR_TOKEN"
                }
            },
            "email": {
                "enabled": False,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "from_address": "monitor@example.com",
                "to_address": "admin@example.com",
                "username": "your_email@gmail.com",
                "password": "your_app_password"
            }
        }
    }

    print("Example notification configuration:")
    print(json.dumps(example_config, indent=2))
