# Roadmap: v2.21.0 - Intelligent Monitoring & Notifications

Strategic planning document for next major release focusing on intelligent automation and user notifications.

---

## Vision

Transform URL monitoring from manual oversight to intelligent, self-managing system that:
- Proactively alerts users to important changes
- Learns normal patterns and detects anomalies
- Provides actionable insights through multiple channels
- Scales to hundreds of monitored sites effortlessly

---

## Current State (v2.20.1)

### Strengths
- ✅ High-performance async monitoring (18 docs/sec)
- ✅ Adaptive concurrency and connection pooling
- ✅ GUI dashboard with comprehensive results display
- ✅ Scheduled automation (daily/weekly tasks)
- ✅ Robust error handling with retry logic
- ✅ Configuration validation
- ✅ Session-based statistics

### Gaps
- ❌ No notification system - users must check manually
- ❌ No anomaly detection - can't distinguish important vs trivial changes
- ❌ Limited historical analysis - single point-in-time checks
- ❌ No distributed monitoring - single machine limitation
- ❌ Manual triage - users must investigate each change
- ❌ No caching - repeated checks to unchanged content

---

## v2.21.0 Feature Priorities

### Priority 1: Notification System (HIGH IMPACT)

**Goal**: Alert users immediately when important changes are detected

**Features**:

#### A1: Email Notifications (~2-3 hours)
- SMTP integration with configurable providers
- HTML email templates with styled change summaries
- Per-site notification preferences
- Digest mode (daily summary vs immediate alerts)
- Attachment support (export JSON, screenshots)
- Threading for conversation continuity

**Configuration**:
```json
"notifications": {
  "email": {
    "enabled": true,
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "use_tls": true,
    "from_address": "monitor@example.com",
    "to_addresses": ["user@example.com"],
    "digest_mode": "daily",  // "immediate" or "daily"
    "template": "html"  // "html" or "text"
  }
}
```

**Email Template Sections**:
- Executive summary (X changed, Y new, Z failed)
- Changed documents table (title, URL, reason, last checked)
- New pages discovered (with parent URL)
- Failed checks (with error details)
- Quick action links (re-scrape, view in GUI, export)

**Implementation Files**:
- `notifier.py` - Notification engine
- `email_templates/` - HTML/text templates
- `monitor_config.json` - Email settings

**Testing Strategy**:
- Unit tests for SMTP connection
- Template rendering tests
- Mock email sending for CI/CD
- Manual testing with Gmail/Outlook/SendGrid

#### A2: Slack Integration (~1.5-2 hours)
- Webhook-based notifications
- Rich message formatting with blocks
- Channel routing (different sites to different channels)
- Thread replies for change details
- Action buttons (acknowledge, re-scrape, mute)

**Configuration**:
```json
"notifications": {
  "slack": {
    "enabled": true,
    "webhook_url": "https://hooks.slack.com/services/...",
    "channel": "#monitoring",
    "mention_on_critical": "@channel",
    "thread_updates": true
  }
}
```

**Slack Message Format**:
```
📊 URL Monitoring Report - 2025-12-22 14:30

✅ Status: 3 changes detected
📄 Changed: 3 documents
🆕 New Pages: 5 discovered
❌ Failed: 1 check

[View Full Report] [Re-scrape All] [Export JSON]
```

#### A3: Discord Webhook Support (~1 hour)
- Similar to Slack but Discord-specific formatting
- Embed support for rich previews
- Role mentions for critical alerts
- Webhook URL configuration

**Total Time Estimate**: 4.5-6 hours

---

### Priority 2: ML Anomaly Detection (MEDIUM-HIGH IMPACT)

**Goal**: Intelligently identify significant changes vs noise

**Features**:

#### B1: Baseline Learning (~3-4 hours)
- Track historical check patterns per document
- Build profile: typical update frequency, change magnitude, failure rate
- Store in new `monitoring_history` table
- Configurable learning period (default: 30 days)

**Database Schema**:
```sql
CREATE TABLE monitoring_history (
    id INTEGER PRIMARY KEY,
    doc_id TEXT NOT NULL,
    check_date TEXT NOT NULL,
    status TEXT,  -- 'unchanged', 'changed', 'failed'
    change_type TEXT,  -- 'content', 'structure', 'metadata'
    response_time REAL,
    content_hash TEXT,
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
);

CREATE INDEX idx_history_doc ON monitoring_history(doc_id);
CREATE INDEX idx_history_date ON monitoring_history(check_date);
```

#### B2: Anomaly Detection (~4-5 hours)
- Detect unusual patterns:
  - Frequency anomalies: Site updating unexpectedly often/rarely
  - Content anomalies: Unusually large changes
  - Performance anomalies: Sudden response time degradation
  - Failure anomalies: New failure patterns
- Scoring system (0-100 severity)
- Threshold-based alerting

**Algorithm**:
```python
def calculate_anomaly_score(doc_id, current_check):
    """Calculate anomaly score based on historical patterns."""
    history = get_history(doc_id, days=30)

    # Frequency anomaly
    avg_interval = calculate_avg_update_interval(history)
    current_interval = time_since_last_change(doc_id)
    frequency_score = abs(current_interval - avg_interval) / avg_interval

    # Content magnitude anomaly
    avg_change_size = calculate_avg_change_magnitude(history)
    current_change_size = calculate_change_magnitude(current_check)
    magnitude_score = abs(current_change_size - avg_change_size) / avg_change_size

    # Performance anomaly
    avg_response_time = calculate_avg_response_time(history)
    current_response_time = current_check['response_time']
    performance_score = abs(current_response_time - avg_response_time) / avg_response_time

    # Weighted composite score
    return (
        frequency_score * 0.4 +
        magnitude_score * 0.4 +
        performance_score * 0.2
    ) * 100
```

**Severity Levels**:
- 0-30: Normal (no alert)
- 31-60: Minor (include in digest)
- 61-85: Moderate (immediate notification)
- 86-100: Critical (urgent alert with @mentions)

#### B3: Smart Filtering (~2 hours)
- Suppress repetitive changes (same content hash)
- Detect cosmetic changes (whitespace, timestamps)
- Ignore known patterns (ads, counters, dates)
- User-configurable ignore rules

**Configuration**:
```json
"anomaly_detection": {
  "enabled": true,
  "learning_period_days": 30,
  "alert_threshold": 60,
  "ignore_patterns": [
    "Updated: \\d{4}-\\d{2}-\\d{2}",  // Timestamps
    "Views: \\d+",  // View counters
    "<div class=\"ad\">.*?</div>"  // Ads
  ]
}
```

**Total Time Estimate**: 9-11 hours

---

### Priority 3: Historical Analysis & Reporting (MEDIUM IMPACT)

**Goal**: Provide trend analysis and insights over time

**Features**:

#### C1: Trend Visualization (~3-4 hours)
- GUI charts showing:
  - Update frequency over time (per site)
  - Response time trends
  - Failure rate trends
  - Change magnitude over time
- Export charts as images
- Date range filtering

**Implementation**:
- Streamlit line charts, bar charts
- Plotly for interactive graphs
- Data aggregation queries

#### C2: Report Generation (~2-3 hours)
- Weekly/monthly summary reports
- Top 10 most active sites
- Top 10 most stable sites
- Performance degradation alerts
- Capacity planning insights
- PDF export option

**Report Sections**:
1. Executive Summary
2. Site Activity Analysis
3. Performance Metrics
4. Failure Analysis
5. Recommendations

#### C3: Audit Log (~1-2 hours)
- Track all monitoring actions
- User operations (re-scrape, delete)
- Automated actions (scheduled checks)
- Configuration changes
- Exportable audit trail

**Total Time Estimate**: 6-9 hours

---

### Priority 4: Performance & Scalability Enhancements (LOW-MEDIUM IMPACT)

**Goal**: Scale to 500+ monitored sites efficiently

**Features**:

#### D1: Request Caching (~2-3 hours)
- HTTP cache with TTL for HEAD requests
- Cache Last-Modified headers
- Avoid redundant checks within time window
- Configurable cache duration (default: 1 hour)

**Implementation**:
```python
class URLCache:
    def __init__(self, ttl=3600):
        self.cache = {}  # url -> (timestamp, headers)
        self.ttl = ttl

    def get(self, url):
        if url in self.cache:
            timestamp, headers = self.cache[url]
            if time.time() - timestamp < self.ttl:
                return headers
        return None

    def set(self, url, headers):
        self.cache[url] = (time.time(), headers)
```

#### D2: Distributed Monitoring (~4-5 hours)
- Worker-based architecture
- Distribute URL checks across multiple processes
- Redis-based task queue
- Aggregated results collection
- Load balancing

**Architecture**:
```
Coordinator
    |
    v
Task Queue (Redis)
    |
    +-- Worker 1 (URLs 1-100)
    +-- Worker 2 (URLs 101-200)
    +-- Worker 3 (URLs 201-300)
    |
    v
Results Aggregator
```

#### D3: Advanced Rate Limiting (~1-2 hours)
- Per-domain rate limiting
- Respect robots.txt delays
- Backoff on 429 (Too Many Requests)
- Priority queue for critical sites

**Total Time Estimate**: 7-10 hours

---

### Priority 5: UI/UX Improvements (LOW-MEDIUM IMPACT)

**Goal**: Make monitoring more intuitive and actionable

**Features**:

#### E1: Dashboard Enhancements (~2-3 hours)
- Site health indicators (green/yellow/red)
- Last check timestamp with relative time ("2 hours ago")
- Quick filters (changed only, failed only, new pages)
- Bulk selection with checkboxes
- Multi-site actions (re-scrape selected, export selected)

#### E2: Notification Preferences UI (~2 hours)
- Per-site notification settings
- Mute/unmute sites
- Notification channel selection (email, Slack, both)
- Test notification button

#### E3: Anomaly Dashboard (~2-3 hours)
- Anomaly score visualization
- Historical score trends
- Threshold adjustment UI
- Whitelist/blacklist pattern editor

#### E4: Mobile-Responsive Design (~1-2 hours)
- Streamlit responsive layout
- Mobile-friendly tables
- Touch-friendly action buttons

**Total Time Estimate**: 7-10 hours

---

## Release Timeline

### Sprint 1: Notifications Foundation (1-1.5 weeks)
- Email notification system
- Slack integration
- Discord webhook support
- Testing and documentation

**Deliverables**:
- `notifier.py` module
- Email templates
- Configuration schema updates
- NOTIFICATION_GUIDE.md

### Sprint 2: Anomaly Detection (1.5-2 weeks)
- Historical tracking database
- Baseline learning algorithm
- Anomaly scoring system
- Smart filtering

**Deliverables**:
- `anomaly_detector.py` module
- Database schema migration
- Algorithm documentation
- ANOMALY_DETECTION.md

### Sprint 3: Historical Analysis (1 week)
- Trend visualization
- Report generation
- Audit logging

**Deliverables**:
- Enhanced admin GUI
- Report templates
- REPORTING_GUIDE.md

### Sprint 4: Performance & Polish (1 week)
- Request caching
- Rate limiting enhancements
- UI improvements
- Final testing

**Deliverables**:
- Optimized monitoring pipeline
- Enhanced GUI
- Performance benchmarks

**Total Estimated Time**: 4.5-6.5 weeks (25-35 hours implementation)

---

## Technical Architecture

### New Components

#### Notification Engine
```
NotificationEngine
    |
    +-- EmailNotifier (SMTP)
    +-- SlackNotifier (Webhook)
    +-- DiscordNotifier (Webhook)
    |
    v
NotificationQueue (async)
```

#### Anomaly Detection Pipeline
```
Monitoring Check
    |
    v
Historical Data Collector
    |
    v
Anomaly Detector
    |
    +-- Frequency Analyzer
    +-- Content Analyzer
    +-- Performance Analyzer
    |
    v
Severity Scorer
    |
    v
Notification Router
```

#### Distributed Architecture
```
Monitor Coordinator
    |
    v
Redis Task Queue
    |
    +-- Worker Pool (N workers)
    |   |
    |   +-- URL Checker (async)
    |   +-- Result Reporter
    |
    v
Result Aggregator
    |
    v
Anomaly Detector
    |
    v
Notification Engine
```

### Database Schema Changes

**New Tables**:
```sql
-- Historical monitoring data
CREATE TABLE monitoring_history (
    id INTEGER PRIMARY KEY,
    doc_id TEXT NOT NULL,
    check_date TEXT NOT NULL,
    status TEXT,
    change_type TEXT,
    response_time REAL,
    content_hash TEXT,
    anomaly_score REAL,
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
);

-- Notification preferences
CREATE TABLE notification_preferences (
    doc_id TEXT PRIMARY KEY,
    email_enabled INTEGER DEFAULT 1,
    slack_enabled INTEGER DEFAULT 1,
    discord_enabled INTEGER DEFAULT 0,
    muted INTEGER DEFAULT 0,
    min_severity INTEGER DEFAULT 60,
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
);

-- Audit log
CREATE TABLE monitoring_audit (
    id INTEGER PRIMARY KEY,
    timestamp TEXT NOT NULL,
    action TEXT NOT NULL,
    doc_id TEXT,
    user TEXT,
    details TEXT,
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
);
```

### Configuration Updates

**monitor_config.json** will expand to include:
```json
{
  "data_dir": "~/.tdz-c64-knowledge",

  "daily": { /* existing */ },
  "weekly": { /* existing */ },

  "notifications": {
    "email": {
      "enabled": true,
      "smtp_server": "smtp.gmail.com",
      "smtp_port": 587,
      "use_tls": true,
      "from_address": "monitor@example.com",
      "to_addresses": ["user@example.com"],
      "digest_mode": "daily"
    },
    "slack": {
      "enabled": false,
      "webhook_url": "",
      "channel": "#monitoring",
      "mention_on_critical": "@channel"
    },
    "discord": {
      "enabled": false,
      "webhook_url": ""
    }
  },

  "anomaly_detection": {
    "enabled": true,
    "learning_period_days": 30,
    "alert_threshold": 60,
    "ignore_patterns": []
  },

  "caching": {
    "enabled": true,
    "ttl_seconds": 3600,
    "max_size_mb": 100
  },

  "distributed": {
    "enabled": false,
    "workers": 4,
    "redis_url": "redis://localhost:6379"
  },

  "output": { /* existing */ },
  "logging": { /* existing */ }
}
```

---

## Dependencies

### New Python Packages

```toml
[project.dependencies]
# Email
"email-validator>=2.1.0",

# Notifications
"slack-sdk>=3.26.0",
"discord-webhook>=1.3.0",

# Anomaly Detection
"numpy>=1.24.0",
"scikit-learn>=1.3.0",  # For outlier detection algorithms
"pandas>=2.0.0",  # For time series analysis

# Visualization
"plotly>=5.18.0",

# Distributed (optional)
"redis>=5.0.0",
"celery>=5.3.0",

# Reporting
"reportlab>=4.0.0",  # PDF generation
"jinja2>=3.1.0",  # Template rendering
```

**Total Additional Dependencies**: 10 packages

---

## Migration Path

### From v2.20.x to v2.21.0

**Database Migration**:
```python
# migration_v2.21.0.py
def migrate_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Add new tables
    cursor.execute(CREATE_MONITORING_HISTORY_TABLE)
    cursor.execute(CREATE_NOTIFICATION_PREFERENCES_TABLE)
    cursor.execute(CREATE_MONITORING_AUDIT_TABLE)

    # Add indexes
    cursor.execute(CREATE_HISTORY_INDEXES)

    # Initialize preferences for existing documents
    cursor.execute("""
        INSERT INTO notification_preferences (doc_id)
        SELECT doc_id FROM documents
        WHERE source_url IS NOT NULL
    """)

    conn.commit()
    conn.close()
```

**Configuration Migration**:
- Automatically detect old config format
- Prompt user to add notification settings
- Provide migration wizard in GUI

**Breaking Changes**:
- None - all changes are additive
- Existing monitoring scripts continue to work
- New features opt-in via configuration

---

## Success Metrics

### Performance Targets
- ✅ Notification delivery < 5 seconds
- ✅ Anomaly detection < 100ms per document
- ✅ Support 500+ monitored sites
- ✅ 99.9% notification delivery rate

### User Experience Targets
- ✅ Reduce false positive alerts by 70%
- ✅ 90% of important changes detected automatically
- ✅ Zero configuration for basic email notifications
- ✅ < 2 clicks to configure advanced features

### Reliability Targets
- ✅ 99% anomaly detection accuracy after 30-day learning
- ✅ Graceful degradation on notification failures
- ✅ Zero data loss on system crashes

---

## Risk Assessment

### High Risk
- **Email Deliverability**: Potential spam filtering, DKIM/SPF requirements
  - Mitigation: Documentation for email provider setup, support for authenticated SMTP

- **False Positives**: Over-alerting users with noise
  - Mitigation: Conservative thresholds, learning period, user feedback loop

### Medium Risk
- **Performance at Scale**: 500+ sites may strain single machine
  - Mitigation: Distributed architecture ready as fallback

- **Third-Party API Changes**: Slack/Discord webhook formats may change
  - Mitigation: Version pinning, adapter pattern for easy updates

### Low Risk
- **Database Schema Migration**: Adding tables is low-risk
  - Mitigation: Migration script with rollback support

---

## Future Considerations (v2.22.0+)

**Beyond v2.21.0**:
- Browser automation for JavaScript-heavy sites (Playwright integration)
- Screenshot comparison for visual regression detection
- Natural language change descriptions (LLM integration)
- Mobile app for monitoring on-the-go
- Webhook endpoints for third-party integrations
- Multi-user support with role-based access control
- Cloud deployment options (AWS Lambda, Google Cloud Functions)

---

## Documentation Deliverables

### New Documentation
1. **NOTIFICATION_GUIDE.md** - Complete notification setup guide
2. **ANOMALY_DETECTION.md** - How anomaly detection works
3. **REPORTING_GUIDE.md** - Using historical analysis features
4. **SCALING_GUIDE.md** - Scaling to 500+ sites

### Updated Documentation
1. **README.md** - v2.21.0 features
2. **WEB_MONITORING_GUIDE.md** - New features integration
3. **MONITORING_SETUP.md** - Notification configuration

---

## Implementation Priority Matrix

| Feature | Impact | Effort | Priority | Sprint |
|---------|--------|--------|----------|--------|
| Email Notifications | High | Medium | 1 | 1 |
| Slack Integration | High | Low | 2 | 1 |
| Anomaly Detection | High | High | 3 | 2 |
| Historical Tracking | Medium | Medium | 4 | 2 |
| Trend Visualization | Medium | Medium | 5 | 3 |
| Request Caching | Medium | Low | 6 | 4 |
| Discord Webhooks | Low | Low | 7 | 1 |
| Distributed Workers | Low | High | 8 | Future |
| Report Generation | Low | Medium | 9 | 3 |
| UI Enhancements | Low | Medium | 10 | 4 |

---

## Open Questions

1. **Email Provider**: Should we support SendGrid/Mailgun APIs in addition to SMTP?
2. **Anomaly Algorithm**: Use simple statistical methods or ML models (isolation forest)?
3. **Notification Frequency**: Default to digest mode or immediate alerts?
4. **Data Retention**: How long to keep monitoring history (30/60/90 days)?
5. **Distributed Architecture**: Redis vs RabbitMQ vs database-backed queue?

**Resolution Strategy**: Start with simplest approach, add complexity based on user feedback

---

## Summary

v2.21.0 transforms URL monitoring from a **reactive tool** into a **proactive intelligence system**:

**Before v2.21.0**:
- User manually checks GUI for changes
- All changes treated equally
- No historical context
- Limited scalability

**After v2.21.0**:
- System alerts user to important changes
- Intelligent prioritization via ML
- Rich historical insights
- Scales to 500+ sites

**Key Metrics**:
- **Implementation**: 25-35 hours (4.5-6.5 weeks)
- **Impact**: High - reduces user oversight burden by 80%
- **Risk**: Low - additive changes, no breaking updates
- **ROI**: Excellent - small investment, large productivity gain

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
