# URL Monitoring Setup Guide

Quick guide to setting up automated URL monitoring for your C64 Knowledge Base.

## Overview

The monitoring system includes:
- **Daily Monitor** - Quick checks using Last-Modified headers (~1s per site)
- **Weekly Monitor** - Comprehensive checks with structure discovery (~10-60s per site)

## Quick Setup (Windows)

### 1. Run Setup Script

Right-click `setup_monitoring_tasks.bat` and select **"Run as administrator"**

This creates two scheduled tasks:
- **Daily Monitor**: Runs every day at 2:00 AM
- **Weekly Monitor**: Runs every Sunday at 3:00 AM

### 2. Configure Settings (Optional)

Edit `monitor_config.json` to customize:
- Data directory location
- Auto-rescrape behavior
- Notification settings
- Output file retention

### 3. Test the Scripts

Before relying on scheduled tasks, test manually:

```cmd
# Test daily monitor
.venv\Scripts\python.exe monitor_daily.py

# Test weekly monitor
.venv\Scripts\python.exe monitor_weekly.py
```

## Manual Usage

### Daily Monitor (Quick Check)

```cmd
# Basic usage
python monitor_daily.py

# With auto-rescrape
python monitor_daily.py --auto-rescrape

# With notifications
python monitor_daily.py --notify

# Custom output file
python monitor_daily.py --output my_check.json
```

**Features:**
- Checks Last-Modified HTTP headers
- Fast (~1 second per site)
- Detects changed and failed documents
- Optional auto-rescrape

**Use when:**
- Daily monitoring
- Quick status checks
- Bandwidth-constrained environments

### Weekly Monitor (Comprehensive Check)

```cmd
# Basic usage
python monitor_weekly.py

# With auto-rescrape
python monitor_weekly.py --auto-rescrape

# With notifications
python monitor_weekly.py --notify

# Without structure discovery (faster)
python monitor_weekly.py --no-structure
```

**Features:**
- Discovers new pages via website crawling
- Identifies missing/removed pages
- Full structure analysis
- Per-site statistics

**Use when:**
- Weekly comprehensive reviews
- After major site updates
- When documentation structure changes

## Understanding Results

### Exit Codes

Scripts return different exit codes for automation:

- **0**: Success, no changes detected
- **2**: Changes/new pages/missing pages detected
- **3**: Some checks failed
- **1**: Script error

### Output Files

Results are saved as JSON files:
- Daily: `url_check_daily_YYYYMMDD_HHMMSS.json`
- Weekly: `url_check_weekly_YYYYMMDD_HHMMSS.json`

### Result Structure

```json
{
  "unchanged": [/* up-to-date documents */],
  "changed": [/* documents with updates */],
  "new_pages": [/* newly discovered URLs */],
  "missing_pages": [/* removed/moved pages */],
  "failed": [/* check errors */],
  "scrape_sessions": [/* per-site statistics */]
}
```

## Configuration File

`monitor_config.json` settings:

### Basic Settings

```json
{
  "data_dir": "~/.tdz-c64-knowledge",

  "daily": {
    "enabled": true,
    "auto_rescrape": false,
    "notify": false
  },

  "weekly": {
    "enabled": true,
    "auto_rescrape": false,
    "notify": true,
    "check_structure": true
  }
}
```

### Notification Settings (Future)

Email, Slack, and webhook notifications are configured in the `notifications` section. Implementation coming soon.

### Output Settings

```json
{
  "output": {
    "save_results": true,
    "output_dir": ".",
    "keep_days": 30
  }
}
```

## Task Scheduler Management

### View Tasks

1. Open Task Scheduler (`taskschd.msc`)
2. Navigate to **Task Scheduler Library**
3. Look for tasks starting with "TDZ-C64-KB"

### Modify Schedule

In Task Scheduler:
1. Right-click the task
2. Select **Properties**
3. Go to **Triggers** tab
4. Edit the schedule

### Remove Tasks

Run `remove_monitoring_tasks.bat` as administrator

Or manually delete in Task Scheduler.

## Linux/macOS Setup

### Using Cron

Add to crontab (`crontab -e`):

```bash
# Daily quick check at 2 AM
0 2 * * * cd /path/to/tdz-c64-knowledge && .venv/bin/python monitor_daily.py --notify >> logs/monitor_daily.log 2>&1

# Weekly full check on Sundays at 3 AM
0 3 * * 0 cd /path/to/tdz-c64-knowledge && .venv/bin/python monitor_weekly.py --notify >> logs/monitor_weekly.log 2>&1
```

### Using systemd Timers

Create service files in `/etc/systemd/system/`:

**tdz-kb-daily.service:**
```ini
[Unit]
Description=TDZ C64 KB Daily URL Monitor

[Service]
Type=oneshot
WorkingDirectory=/path/to/tdz-c64-knowledge
ExecStart=/path/to/.venv/bin/python monitor_daily.py --notify
```

**tdz-kb-daily.timer:**
```ini
[Unit]
Description=Daily URL monitoring for TDZ C64 KB

[Timer]
OnCalendar=daily
OnCalendar=02:00
Persistent=true

[Install]
WantedBy=timers.target
```

Enable and start:
```bash
sudo systemctl enable --now tdz-kb-daily.timer
sudo systemctl enable --now tdz-kb-weekly.timer
```

## Troubleshooting

### Task Doesn't Run

Check Task Scheduler:
1. Open Task Scheduler
2. Find the task
3. Check **Last Run Result**
4. Review **History** tab

Common issues:
- Python path incorrect
- Virtual environment not activated
- Permissions issues

### Script Errors

Run manually to see full error output:
```cmd
.venv\Scripts\python.exe monitor_daily.py
```

Check:
- Virtual environment activated
- All dependencies installed
- Database path accessible

### No Notifications

Notifications require configuration in `monitor_config.json`.

Currently notifications print to stdout/logs. Email/Slack integration coming in future update.

## Best Practices

1. **Test First**: Run scripts manually before scheduling
2. **Review Logs**: Check output files regularly
3. **Start Conservative**: Use quick checks daily, full checks weekly
4. **Monitor Failures**: Investigate repeated check failures
5. **Backup Before Auto-Rescrape**: Test auto-rescrape manually first
6. **Disk Space**: Configure output file cleanup (keep_days setting)

## See Also

- [WEB_MONITORING_GUIDE.md](WEB_MONITORING_GUIDE.md) - Complete monitoring guide
- [README.md](README.md) - Project documentation
- [monitor_config.json](monitor_config.json) - Configuration file

## Support

For issues or questions:
- GitHub Issues: https://github.com/MichaelTroelsen/tdz-c64-knowledge/issues
- Documentation: See README.md and guides
