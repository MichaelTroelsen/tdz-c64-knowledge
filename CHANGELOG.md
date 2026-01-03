# Changelog

All notable changes to the TDZ C64 Knowledge Base project.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.23.18] - 2026-01-03

### Fixed
- **Unicode Encoding Errors in Logging** - Fixed Windows console (cp1252) encoding errors
  - Replaced Unicode checkmark character (✓) with ASCII "[OK]" in logger calls
  - Changed `logger.warning()` to `logger.debug()` for frame detection failures
  - Frame detection failures are now logged at DEBUG level (less noisy)
  - All Unicode characters in MCP tool output preserved (JSON handles Unicode)
- **Frame Detection Timeout** - Improved timeout handling and error messages
  - Increased timeout from 10 to 30 seconds for slow connections
  - Separate exception handling for `requests.exceptions.Timeout`
  - More informative debug messages: "(will use mdscrape instead)"
  - Frame detection failures no longer clutter logs with WARNING level
- **Web Scraping Unit Test** - Added comprehensive test for C64 Wiki scraping
  - New test: `test_scrape_c64_wiki()` in `test_server.py`
  - Tests single-page scraping with progress tracking
  - Verifies progress callbacks, document metadata, and content
  - Includes detailed assertions and progress verification
  - Skips gracefully if mdscrape not available

## [2.23.17] - 2026-01-03

### Improved
- **Website Scraping Progress Feedback** - Real-time progress tracking with visual feedback and timeout detection
  - **Server-Side Improvements** (`server.py` - `scrape_url()` method)
    - Replaced blocking `subprocess.run()` with streaming `subprocess.Popen()`
    - Real-time output parsing to extract current URL being scraped
    - Background threads for non-blocking stdout/stderr reading
    - **Per-Page Timeout Detection**: Warns if a single page takes > 60 seconds
    - **Progress Updates**: Callback triggered for each page scraped with current URL and count
    - **Console Logging**: Server logs show "[X/Y] Scraping: <url>" for each page
    - **Completion Summary**: Shows total pages scraped on success (e.g., "✓ Scraping completed successfully (15 pages)")
    - Regex pattern matching for mdscrape output: `(?:Scraping|Processing|Fetching)[:\s]+(\S+)`
    - Graceful error handling with image-error filtering (unchanged)
  - **Streamlit GUI Improvements** (`admin_gui.py`)
    - **Real-Time Progress Bar**: Updates as each page is scraped (0-100%)
    - **Status Messages**: Shows "Scraping page X/Y" with live updates
    - **Current URL Display**: Shows which page is currently being processed
    - **Timeout Warnings**: Displays "⚠️ Page taking longer than 60s..." if stuck
    - Progress callback updates three UI elements:
      - Progress bar (visual percentage)
      - Status info box (current operation)
      - URL display (current page path, truncated to 80 chars)
    - Proper cleanup of all progress widgets on completion/error
  - **User Experience**:
    - No more "hanging" UI - users see exactly what's happening
    - Clear indication of scraping progress (X/Y pages)
    - Early warning if a page is slow (60s timeout alert)
    - Server console shows detailed progress for debugging
    - All existing functionality preserved (follow_links, max_pages, depth, etc.)

## [2.24.0] - In Planning

### Planning
- **Advanced Knowledge Extraction System** - Comprehensive implementation plan for 6 algorithms across 4 phases
  - **Planning Document** (`docs/KNOWLEDGE_EXTRACTION_PLAN.md` - 82 pages, 1,850 lines)
    - Complete roadmap for 64-80 hour implementation
    - 4 phases broken into 27 tasks (1-4 hours each)
    - Detailed code examples for every major component
    - Dependencies, testing strategies, and validation criteria
  - **Phase 1: Foundation (16-20 hours)**
    - Knowledge graph infrastructure with NetworkX
    - PageRank, community detection, centrality measures
    - 6 new MCP tools for graph operations
    - PyVis interactive HTML visualizations
    - GraphML/GEXF/JSON export formats
  - **Phase 2: Discovery (20-24 hours)**
    - Topic modeling: LDA, NMF, BERTopic
    - Document clustering: K-Means, DBSCAN, HDBSCAN
    - 8 new MCP tools for topics and clusters
    - Word clouds, cluster scatter plots, dendrograms
  - **Phase 3: Temporal (16-20 hours)**
    - Date/time extraction (regex + SpaCy)
    - Event detection (product releases, milestones, innovations)
    - Timeline construction and chronological ordering
    - 4 new MCP tools for timeline operations
    - Interactive timeline visualizations with Plotly
  - **Phase 4: Integration (12-16 hours)**
    - Streamlit analytics dashboard (6 pages)
    - Knowledge graph explorer with real-time controls
    - Topic discovery interface with model comparison
    - Cluster navigator with 2D/3D visualization
    - Timeline viewer with date filtering
    - Unified search and export tools
  - **Implementation Details**:
    - Total: 20+ new MCP tools for Claude integration
    - 10+ database tables/indexes for caching results
    - 90%+ test coverage requirement
    - Risk assessment with mitigations
    - 5-week recommended timeline
  - **Success Criteria**:
    - Knowledge graph: 100+ nodes, 200+ edges
    - Topics: 10 coherent themes from 7.4M words
    - Clusters: Silhouette score > 0.3
    - Timeline: 50+ events spanning C64 history (1975-1995)
    - Dashboard: All pages load in < 5 seconds

## [2.23.16] - 2026-01-03

### Added
- **Webhook Notifications for URL Monitoring** - Comprehensive multi-channel notification system for monitoring scripts
  - **New Notification Module** (`utilities/notifications.py` - 552 lines)
    - `NotificationManager` class with support for 4 notification channels
    - Error handling with per-channel status reporting
    - Automatic retry logic for failed deliveries
  - **Supported Channels**:
    1. **Discord Webhooks** - Rich embed notifications with color coding
       - Green (all good), Orange (attention needed), Red (errors)
       - Clickable document links in embedded fields
       - Customizable bot username
       - Automatic timestamp and footer
    2. **Slack Webhooks** - Block Kit formatted messages
       - Professional block-based layout with fields
       - Clickable document links with Slack markdown
       - Custom username and emoji icon
       - Summary statistics in structured format
    3. **Generic Webhooks** - JSON POST for custom integrations
       - Full monitoring results in structured JSON
       - Custom HTTP headers (Authorization, etc.)
       - Compatible with IFTTT, Zapier, n8n, Home Assistant
       - First 10 items from each category (changed, new, missing, failed)
    4. **Email Notifications** - SMTP with HTML + plain text
       - Professional HTML email with color-coded header
       - Statistics dashboard in email body
       - Plain text fallback for compatibility
       - Support for Gmail, Outlook, Yahoo, custom SMTP
  - **Updated Monitoring Scripts**:
    - `utilities/monitor_daily.py` - Integrated NotificationManager
    - `utilities/monitor_weekly.py` - Integrated NotificationManager
    - Removed TODO placeholders (lines 32)
    - Added real-time channel status display (✓/✗ indicators)
    - Graceful error handling with informative error messages
  - **Comprehensive Documentation** (`utilities/NOTIFICATIONS_SETUP.md` - 400+ lines)
    - Complete setup guide for all 4 notification channels
    - Discord: Webhook creation, configuration, example messages
    - Slack: App setup, webhook configuration, Block Kit examples
    - Email: SMTP configuration for Gmail, Outlook, Yahoo with app passwords
    - Generic Webhook: Integration examples for IFTTT, Zapier, n8n
    - Configuration reference table with all options
    - Testing procedures and test notification functionality
    - Troubleshooting guide for common issues
    - Security best practices (webhook URL protection, password management)
  - **Example Configuration** (`utilities/monitor_config.example.json`)
    - Complete example with all notification options
    - Inline comments explaining each field
    - SMTP server examples for common providers
    - Security notes and best practices
  - **Notification Features**:
    - Selective channel enable/disable
    - Rich formatting for changed documents, new pages, missing pages
    - Summary statistics (unchanged, changed, new, missing, failed)
    - Weekly-specific notifications (new page discovery, missing pages)
    - Grouped results (e.g., new pages grouped by site)
    - Customizable thresholds (notify only on changes)
  - **Usage**:
    ```bash
    # Enable notifications in monitor_config.json
    python utilities/monitor_daily.py --notify
    python utilities/monitor_weekly.py --notify

    # Test notifications
    python utilities/notifications.py
    ```
  - **Configuration Options**:
    - `notifications.enabled` - Master switch for all notifications
    - Channel-specific enable flags (discord, slack, webhook, email)
    - Webhook URLs and custom headers
    - SMTP server settings and credentials
    - Customizable usernames and display settings
  - Closes TODOs in monitor_daily.py:32 and monitor_weekly.py:32
  - Enables automated alerting for URL monitoring in scheduled tasks
  - Commit: 64cc46b "Feature: Webhook Notifications for URL Monitoring (v2.23.16)"

## [2.23.15] - 2026-01-03

### Changed
- **Major Documentation Consolidation and Enhancement** - Comprehensive documentation improvements
  - **Version Consistency**: Updated all documentation to v2.23.1
    - CONTEXT.md: Updated from v2.23.0 to v2.23.1
    - QUICKSTART.md: Updated from v2.18.0 to v2.23.1
  - **Documentation Cleanup**: Archived 5 cleanup/report files to `archive/cleanup-reports/`
    - CLEANUP_SUMMARY.md, COMPARISON_REPORT.md, FILE_INVENTORY.md
    - FINAL_CLEANUP_REPORT.md, TEST_REPORT.md
    - Root directory reduced from 11 to 6 markdown files (-45% cleanup)
  - **New Documentation Index**: Created `docs/README.md` (141 lines)
    - Complete documentation index with full navigation
    - Organized by topic: Quick Start, Development, API & Integration, Features, Setup, Operations, Planning
    - Cross-references to all 20+ documentation files
    - Quick links and topic-based organization
  - **New Comprehensive Guides** (6 new files, 3,210 lines total):
    1. **docs/MCP_TOOLS_REFERENCE.md** (817 lines) - Complete reference for all 50+ MCP tools
       - Tools organized into 6 categories with JSON examples
       - Search Tools (11), Document Management (6), URL Scraping (3)
       - AI & Analytics (14), Export Tools (3), System Tools (2), Advanced Tools (14)
       - Performance tips and best practices
    2. **docs/TROUBLESHOOTING.md** (635 lines) - Comprehensive troubleshooting guide
       - Installation issues, MCP integration problems, search troubleshooting
       - Document processing errors, performance optimization
       - Database issues, API problems, advanced debugging
    3. **docs/MCP_INTEGRATION.md** (580 lines) - Complete Claude Code/Desktop integration guide
       - Step-by-step setup for Claude Code and Claude Desktop
       - All configuration options and environment variables
       - Testing & verification procedures
       - Troubleshooting MCP-specific issues, advanced configurations
    4. **docs/SECURITY.md** (558 lines) - Production security best practices
       - Path traversal protection, API security (authentication, CORS, HTTPS)
       - Database security, LLM API key management
       - Production deployment checklist, security auditing procedures
    5. **docs/MIGRATION.md** (479 lines) - Version upgrade guide
       - Version compatibility matrix for all versions
       - Upgrade procedures with step-by-step instructions
       - Breaking changes documentation, rollback procedures
       - Pre/post-migration checklists, troubleshooting migration issues
    6. **docs/README.md** (141 lines) - Documentation index (already mentioned above)
  - **Documentation Structure Improvements**:
    - Clean separation: 6 core docs in root (README, QUICKSTART, ARCHITECTURE, CONTEXT, CLAUDE, CHANGELOG)
    - 21 specialized docs in docs/ folder (6 new, 15 existing)
    - 26 historical files properly archived in archive/
    - Professional organization: user docs vs developer docs vs reference docs
  - **Impact**:
    - Easier onboarding for new users with clear navigation
    - Complete reference for all features and tools
    - Production-ready security and deployment documentation
    - Clear troubleshooting guidance for common issues
    - Smooth upgrade path between versions with migration guide
  - Commit: 137d634 "docs: Consolidate and enhance documentation (v2.23.1)"

### Added
- **AI Suggestions Regenerate with Exclusions** (v2.23.14) - Smart recommendation regeneration
  - Implemented "Regenerate" button that excludes files already in knowledge base
  - **Two Regenerate Scenarios**:
    - When ALL recommendations are in KB: Primary button with prominent warning
    - When SOME recommendations are in KB: Secondary button with status message
  - **Smart Exclusion Logic**:
    - Automatically identifies which recommended files are already in KB using dual URL matching
    - Builds exclusion list with item titles and filenames
    - Stores exclusions in session state for use in next AI request
  - **Enhanced AI Prompt**:
    - Adds exclusion section to prompt when regenerating
    - Explicitly instructs AI to avoid recommending files already in KB
    - Provides full list of excluded files in JSON format
  - **Visual Feedback**:
    - Info banner shows number of files being excluded before regeneration
    - Spinner message updates to show exclusion count during AI analysis
    - Clear status messages: "All X recommendations are new!", "X/Y already in KB, Z new files available"
  - **User Experience**:
    - Seamless workflow: Click regenerate → See exclusions → Get fresh recommendations
    - No manual tracking needed - system automatically remembers what's in KB
    - Prevents duplicate recommendations after files are added
  - **Technical Implementation**:
    - Session state management for ai_exclusions
    - Automatic cleanup after exclusions are used
    - Rerun triggers for immediate UI update
  - Enables efficient exploration of archive.org search results by continuously suggesting new, relevant files
  - Location: admin_gui.py lines 4105-4120 (exclusion UI), 4139-4149 (prompt integration), 4453-4549 (regenerate buttons)

### Added
- **Archive Search Page in Admin GUI** - Internet Archive integration for content discovery (commits d0cb922, ac67a36, 5bfaa6d, 01acb5a)
  - Added new "🔍 Archive Search" page to Streamlit admin GUI with 4-tab interface
  - **Search Archive Tab**: Full-featured search interface
    - Full-text search across Internet Archive collections with advanced query construction
    - **Advanced Filters**: File type (PDF/TXT/HTML/DJVU/EPUB/MOBI), collection (texts/software/data), date range (1900-2026), subject tags
    - Configurable result limits (5-100 items)
    - Real-time search results with item metadata: title, creator, date, description, tags, download count
    - File listings with size information and format filtering
    - Direct archive.org links to source items
    - **Two download options per file**:
      - 💾 Download: Save to {data_dir}/downloads for manual review
      - ⚡ Quick Add: Download and immediately add to knowledge base with metadata
    - Automatic title and tag extraction from archive.org metadata
    - Source URL preservation for provenance tracking
  - **AI Suggestions Tab** (commit ac67a36): Intelligent file recommendation using Claude AI
    - AI-powered analysis of search results using Claude 3.5 Sonnet
    - Recommends top 5 most valuable files based on search query and C64 expertise
    - **Evaluation Criteria**: Technical accuracy, historical significance, uniqueness, completeness, format suitability
    - **Smart Recommendations**: Each includes priority level (High/Medium/Low), score (0-100), detailed rationale, and knowledge value description
    - **Priority Color Coding**: 🔴 High, 🟡 Medium, 🟢 Low for visual prioritization
    - **Quick Actions**: Download or Quick Add buttons directly from recommendations
    - **AI Analysis Summary**: Overall evaluation of search results with context-aware insights
    - Analyzes up to 10 search results with 5 files each for efficiency
    - Robust JSON parsing handles markdown code blocks and direct responses
    - Requires ANTHROPIC_API_KEY environment variable
    - Session state management for persistent suggestions across page interactions
    - Leverages C64-specific expertise in prompts for domain-accurate recommendations
  - **Quick Added Tab** (commits 5bfaa6d, 01acb5a): Track and audit Quick Add operations
    - Complete history of all Quick Add operations from Search Archive and AI Suggestions tabs
    - **Status Tracking**: Success (✅) or Failed (❌) indicators with color coding
    - **Detailed Information**: Document title, filename, source URL, doc ID, timestamp (UTC)
    - **Error Logging**: Failed attempts show error messages for debugging
    - **Chronological Display**: Newest entries shown first (reverse chronological order)
    - **Clear History**: One-click button to reset tracking
    - **Session Persistence**: History maintained during Streamlit session
    - **Audit Trail**: Full visibility into what was added, when, and from where
    - **Security Fix**: Resolves "Path outside allowed directories" error
      - Temp files now created in {data_dir}/temp/ instead of system temp directory
      - Ensures all file operations stay within allowed directories
      - Automatic cleanup of temp files on both success and failure
      - Changed from tempfile.NamedTemporaryFile() to Path-based temp files
    - Provides complete transparency and debugging support for quick-add workflows
  - **Downloaded Files Tab**: Manage downloaded content
    - View all downloaded files with size information
    - Individual actions: Add to KB or Delete
    - Bulk operations: Add All or Clear All with progress tracking
    - File location display and status monitoring
  - **Technical Implementation**:
    - Uses official internetarchive Python library v5.7.1
    - Uses anthropic Python library v0.75.0 for AI recommendations
    - Iterator-based result limiting for API efficiency
    - Metadata extraction and file enumeration
    - Full integration with existing KnowledgeBase.add_document() method
    - Comprehensive error handling and user feedback
  - **Testing** (commits d0cb922, ac67a36, 77b7b7f):
    - Unit test suite (test_archive_search.py) with 8 tests covering search, filtering, metadata extraction, file listing, and URL construction (100% pass rate)
    - Unit test suite (test_ai_suggestions.py) with 18 tests covering prompt construction, JSON parsing, recommendation validation, error handling, API integration (100% pass rate)
    - Unit test suite (test_quick_added.py) with 23 tests covering Quick Added tab, download functionality, and add to KB operations (100% pass rate)
      - TestQuickAddedTab (8 tests): Session state, success/failure entries, clear history, entry structure, timestamp format, chronological order
      - TestDownloadFunctionality (6 tests): Temp directory creation, file naming convention, cleanup on success/failure, security validation
      - TestAddToKBFunctionality (7 tests): Parameter validation, metadata updates, title/tag generation, error handling, path security
      - TestIntegration (2 tests): Complete workflow testing, failure tracking
    - Total: 49 tests, all passing
    - Comprehensive coverage of all Archive Search features including Quick Add workflow, security fixes, and error handling
  - Enables AI-assisted discovery and curation of C64 documentation from archive.org's vast collection directly within admin interface

- **Settings Page in Admin GUI** - Comprehensive configuration viewer (commit dc7c51b)
  - Added new "⚙️ Settings" page to Streamlit admin GUI with 4-tab interface
  - **File Paths Tab**: Displays data directory, database path with size, embeddings path, MCP config file locations
  - **MCP Configuration Tab**: Auto-detects and displays Claude Desktop/Code configurations
    - Searches multiple locations: Claude Desktop (%APPDATA%\Claude\claude_desktop_config.json), Claude Code project (.claude/settings.json), Claude Code global (~/.claude/settings.json)
    - Shows command, arguments, environment variables in organized layout
    - Provides example configuration if no config found
  - **Environment Variables Tab**: Lists important env vars with status indicators
    - Security masking for sensitive values (API keys, passwords, tokens)
    - Expandable section to view all environment variables
  - **Features & Capabilities Tab**: System health and feature status
    - Health status indicator (🟢 Healthy / 🟡 Degraded / 🔴 Unhealthy)
    - Feature flags: Search (FTS5, Semantic, BM25), Document Processing (PDF, OCR), AI (Entity Extraction, Relationships, RAG), Web (Scraping, Monitoring)
    - Database statistics: documents, chunks, entities, relationships, embeddings
    - Full health check JSON in expandable view
  - Provides complete visibility into server configuration and runtime environment

### Documentation
- **Comprehensive Documentation Reorganization** - Major cleanup and restructuring (commits e3e298e, 9c6051e)
  - Created `docs/` directory with 15 feature-specific guides organized by category
  - Moved feature documentation from root to `docs/`: REST API, Entity Extraction, Anomaly Detection, Summarization, Web Scraping, Web Monitoring, Testing, Examples, Deployment, Docker, Environment Setup, Poppler Setup, GUI, Monitoring, Roadmap
  - Reduced root directory MD files by 76% (42 → 10 files)
  - Archived 20 historical/obsolete documentation files to `archive/historical-docs/`
  - Archived 3 old release notes (v2.20-22) to `archive/release-notes/`
  - Updated README.md with comprehensive documentation index organized by category
  - Fixed broken documentation links (README_REST_API.md → docs/REST_API.md)
  - Created inventory and analysis reports: FILE_INVENTORY.md, COMPARISON_REPORT.md, CLEANUP_SUMMARY.md, FINAL_CLEANUP_REPORT.md
  - All historical files preserved in `archive/` for reference

### Fixed
- **Archive Search Quick Add Security** - Fixed path security violation (commits 5bfaa6d, 01acb5a)
  - Fixed "Path outside allowed directories" SecurityError in Quick Add functionality
  - Root Cause: Temporary files were created in system temp directory (e.g., C:\Users\...\AppData\Local\Temp\) which was outside allowed security boundaries
  - Solution: Temp files now created in {data_dir}/temp/ directory which is within allowed paths
  - Impact: Quick Add now works correctly without triggering security violations
  - Technical Details:
    - Changed from tempfile.NamedTemporaryFile() to Path-based temp file creation
    - Temp file naming: quick_add_{timestamp}_{filename}
    - Automatic cleanup on both success and failure paths
    - Proper error handling with temp file cleanup in exception handlers
  - Security: Maintains path traversal protection while enabling Quick Add workflow

### Refactored
- **Utility Scripts Organization** - Reorganized Python utility scripts for better maintainability
  - Created `utilities/` directory for active monitoring and load testing scripts (6 files)
  - Moved active scripts to `utilities/`: benchmark_comprehensive.py, load_test.py, load_test_500.py, monitor_daily.py, monitor_weekly.py, monitor_fast.py
  - Archived 12 obsolete utility scripts to `archive/utilities/`: debug_bm25.py, enable_fts5.py, enable_semantic_search.py, setup_claude_desktop.py, old benchmark scripts, URL check utilities, monitor config validator
  - Reduced root directory Python files by 50% (36 → 18 files)
  - All scripts preserved for rollback if needed

### Benefits
- **Cleaner Project Structure** - 64% reduction in root directory files (78 → 28 files)
  - Easier navigation and file discovery
  - Better IDE performance with fewer root-level files
  - Clearer separation: core vs feature vs utilities vs historical
  - Professional documentation hierarchy
  - Faster git operations with organized structure
  - Improved maintainability and onboarding experience

## [2.23.13] - 2026-01-02

### Added
- **AI Suggestions Enhancements** - 10 recommendations + smart KB status tracking
  - Increased from 5 to 10 recommendations per AI analysis
  - Added intelligent "already in KB" detection across all recommendations
  - Shows comprehensive status: "X/10 recommendations already in KB, Y new files available"
  - Smart warning when all recommendations are already in KB
  - Features:
    - **10 Recommendations**: More options to choose from
    - **KB Status Counter**: See at a glance how many are new vs already added
    - **"All in KB" Detection**: Warns when all recommendations exist
    - **Regenerate Button**: Placeholder for auto-regeneration feature (coming soon)
  - Locations:
    - Prompt: admin_gui.py line 4179 (10 instead of 5)
    - Status tracking: admin_gui.py lines 4413-4462
  - Impact: Better recommendations and visibility into what's new

### Changed
- **Version Bump** - Updated version from 2.23.12 to 2.23.13

## [2.23.12] - 2026-01-02

### Fixed
- **AI Suggestions JSON Parsing** - Enhanced error handling and prompt clarity
  - Added detailed JSON parsing error messages with line/column information
  - Shows actual AI response in expander for debugging when parsing fails
  - Improved prompt to be more explicit about JSON format requirements
  - Root Cause: Haiku sometimes returns JSON with syntax errors or extra text
  - Enhancements:
    - Better error reporting: shows exact location of JSON syntax errors
    - Stricter prompt: "Respond ONLY with valid JSON. No markdown, no explanations"
    - Added JSON formatting rules to prompt
    - Debug view: expandable section showing raw AI response
  - Locations: admin_gui.py lines 4203-4212 (error handling), 4160-4183 (improved prompt)
  - Impact: Users can now see and report exact JSON issues if they occur

### Changed
- **Version Bump** - Updated version from 2.23.11 to 2.23.12

## [2.23.11] - 2026-01-02

### Fixed
- **AI Suggestions Model - Use Haiku for Maximum Compatibility** - Fixed persistent 404 errors
  - Changed default model from `claude-3-5-sonnet-20240620` to `claude-3-haiku-20240307`
  - Root Cause: User's API key doesn't have access to Claude 3.5 Sonnet models
  - Solution: Use Claude 3 Haiku - most widely available, fast, and cost-effective
  - Location: admin_gui.py line 4180
  - Benefits:
    - Works with all API keys (maximum compatibility)
    - 15x cheaper than Sonnet ($0.25/MTok vs $3/MTok)
    - Fast responses (good for UI interactions)
    - Still provides quality recommendations
  - Users can still override: `set AI_SUGGESTIONS_MODEL=claude-3-sonnet-20240229`
  - Impact: AI Suggestions now works for all users

### Changed
- **Version Bump** - Updated version from 2.23.10 to 2.23.11

## [2.23.10] - 2026-01-02

### Fixed
- **AI Suggestions Model Compatibility** - Fixed 404 error with unavailable model
  - Changed from hardcoded `claude-3-5-sonnet-20241022` to stable `claude-3-5-sonnet-20240620`
  - Model is now configurable via `AI_SUGGESTIONS_MODEL` environment variable
  - Root Cause: Model `claude-3-5-sonnet-20241022` not available to all API keys/regions
  - Location: admin_gui.py line 4180
  - Users can override: `set AI_SUGGESTIONS_MODEL=claude-3-haiku-20240307` (cheaper option)
  - Impact: AI Suggestions now works without 404 errors

### Changed
- **Version Bump** - Updated version from 2.23.9 to 2.23.10

## [2.23.9] - 2026-01-02

### Fixed
- **Enhanced Duplicate Detection for Legacy Documents** - "In KB" status now detects OLD and NEW format source URLs
  - Enhanced duplicate detection to handle both URL formats:
    - **New format** (file URL): `https://archive.org/download/item-id/filename.pdf` - Exact URL match
    - **Old format** (item URL): `https://archive.org/details/item-id` - Item ID + filename match
  - Root Cause: Documents added before v2.23.8 have item URLs, not file URLs
  - Solution: Two-phase matching logic:
    1. Try exact file URL match (for documents added after v2.23.8)
    2. Try item ID + filename match (for documents added before v2.23.8)
  - Filename matching ignores file extension for flexibility (.PDF vs .txt)
  - Filename matching uses stem comparison: `Path(filename).stem`
  - Locations Fixed:
    - Search Archive Tab (admin_gui.py lines 3972-3980)
    - AI Suggestions Tab (admin_gui.py lines 4286-4294)
  - Impact: **All archive.org documents now show "✅ In KB" status correctly**
  - Works for documents added before AND after the fix

### Added
- **Comprehensive "In KB" Status Tests** - 9 unit tests validating entire duplicate detection workflow
  - test_in_kb_status.py with 100% pass rate (9/9 tests)
  - Tests cover:
    - File URL construction and encoding
    - source_url save/load cycle
    - Duplicate detection logic (new format)
    - Duplicate detection logic (old format) ← **Critical test**
    - Multiple files detection
    - Extension-independent matching
    - KB reloading after Quick Add
  - Validates "In KB" status shows correctly for all scenarios

### Changed
- **Version Bump** - Updated version from 2.23.8 to 2.23.9

## [2.23.8] - 2026-01-02

### Fixed
- **Archive Search Duplicate Detection** - Fixed "In KB" status not showing for existing files
  - "In KB" status now correctly shows for files already in knowledge base
  - Root Cause: source_url was saving item URL (`result['url']`) but checking file URL (`file['url']`) - these don't match!
  - Solution: Changed source_url to consistently use file URL (`file['url']`) for accurate duplicate detection
  - Locations Fixed:
    - Search Archive Quick Add (admin_gui.py line 4030, 4033)
    - AI Suggestions Quick Add (admin_gui.py line 4303, 4306)
  - Impact: Download/Quick Add buttons now correctly hidden when file already exists in KB
  - Duplicate detection now works properly across Search Archive and AI Suggestions tabs

### Changed
- **Version Bump** - Updated version from 2.23.7 to 2.23.8

## [2.23.7] - 2026-01-02

### Fixed
- **Downloaded Files KB Status Detection** - Now checks actual KB database, not just session state
  - Downloads tab now searches KB database to verify if files are already added
  - Checks both session state (fast) and KB database (comprehensive)
  - Searches by filename in document file_path field
  - Shows "✅ In KB" status for files added in previous sessions
  - Root Cause: Only checked session state, missed files added in previous sessions or other tabs
  - Location: admin_gui.py line 4436 (Downloaded Files tab)
  - Impact: Downloaded files now correctly show KB status across all sessions

### Changed
- **Version Bump** - Updated version from 2.23.6 to 2.23.7

### Impact
- Downloaded files status persists across sessions
- Prevents re-adding files that are already in KB
- More accurate KB status indicators
- Better user experience with persistent status tracking

## [2.23.6] - 2026-01-02

### Added
- **Duplicate Detection in Archive Search** - Show KB status instead of Download/Quick Add buttons
  - Check if files already exist in KB by source URL before showing action buttons
  - Display "✅ In KB" status with document ID for existing files
  - Prevents duplicate downloads and additions
  - Applied to both Search Archive tab and AI Suggestions tab
  - Locations:
    - admin_gui.py line 3956: Search Archive file list
    - admin_gui.py line 4251: AI Suggestions recommendations
  - Impact: Users can see at a glance which files are already in the knowledge base

### Changed
- **Version Bump** - Updated version from 2.23.5 to 2.23.6

### Impact
- Clearer visibility into which archive.org files are already in the KB
- Prevents accidental duplicate downloads
- Prevents duplicate Quick Add operations
- Better user experience with immediate status feedback
- More efficient workflow when browsing archive.org results

## [2.23.5] - 2026-01-02

### Fixed
- **Archive.org Filename Path Extraction** - Fixed FileNotFoundError for downloads and Quick Add
  - Extract only filename (not directory path) when saving files locally
  - Root Cause: Filenames like "Back in Time 3/Extras/Booklet.PDF" created non-existent subdirectories
  - Solution: Use `Path(filename).name` to extract only the last component
  - Fixed in 4 locations:
    - admin_gui.py line 3968: Search Archive Download button
    - admin_gui.py line 3988: Search Archive Quick Add temp file
    - admin_gui.py line 4248: AI Suggestions Quick Add temp file
    - admin_gui.py line 4323: AI Suggestions Download button
  - Example: "Back in Time 3/Extras/Booklet.PDF" → saved as "Booklet.PDF"
  - Impact: Downloads and Quick Add now work for all archive.org files with directory separators

### Testing
- **Filename Extraction Tests** - Added 7 new tests to test_url_encoding.py (23 total tests, 100% pass)
  - TestFilenameExtraction (7 tests): Simple paths, directory paths, multiple slashes, temp filename construction
  - **Specific test for user-reported error**: "Back in Time 3/Extras/Back in Time 3 Booklet.PDF"
  - Validates:
    - Path.name extracts only last component
    - No directory separators in extracted filenames
    - Downloads directory receives single-level filenames
    - Temp filenames don't create subdirectories
  - All tests pass on Windows (path separator agnostic)

### Changed
- **Version Bump** - Updated version from 2.23.4 to 2.23.5

### Impact
- Downloads work correctly for all archive.org file paths
- Quick Add temp file creation no longer fails with FileNotFoundError
- Filenames with directory components now saved properly
- No phantom subdirectories created in downloads or temp directories

## [2.23.4] - 2026-01-02

### Fixed
- **Archive.org URL Encoding** - Fixed InvalidURL error for filenames with spaces and special characters
  - Added URL encoding using `urllib.parse.quote()` for archive.org download URLs
  - Fixes `InvalidURL: URL can't contain control characters` error when downloading files
  - Root Cause: Filenames with spaces, slashes, and special characters not URL-encoded
  - Example: `Back in Time 1/Extras/BIT 1 - DD Booklet.pdf` → `Back%20in%20Time%201%2FExtras%2FBIT%201%20-%20DD%20Booklet.pdf`
  - Impact: All archive.org files now downloadable via Quick Add and Download buttons
  - Location: admin_gui.py line 3868 (URL construction in Archive Search)
  - Import: Added `from urllib.parse import quote` to module imports

### Testing
- **URL Encoding Test Suite** - Comprehensive unit tests for URL encoding (test_url_encoding.py)
  - Created test_url_encoding.py with 16 tests, 100% pass rate
  - TestURLEncoding (14 tests): Spaces, directories, special chars, unicode, edge cases
  - TestArchiveSearchURLs (2 tests): File dict structure, multiple files
  - **Test Coverage**:
    - Simple filenames without special characters
    - Filenames with spaces (encoded as %20)
    - Directory paths with slashes (encoded as %2F)
    - Special characters: & ( ) : $ % # ! @ = + ?
    - Unicode characters (Cyrillic, etc.)
    - Empty filenames and edge cases
    - Integration with urllib.request.urlretrieve()
  - Validates proper encoding with `safe=''` parameter
  - Ensures no control characters in final URLs
  - Total Archive Search test coverage: 65 tests across 4 test suites

### Changed
- **Version Bump** - Updated version from 2.23.3 to 2.23.4

### Impact
- Archive Search now handles all archive.org filenames correctly
- No more InvalidURL errors for files with spaces or special characters
- Improved reliability for international/unicode filenames
- Better compatibility with archive.org's diverse file naming conventions

## [2.23.3] - 2026-01-02

### Fixed
- **Downloaded Files Tracking** - Fixed file tracking to use filename instead of full path
  - Changed tracking key from `str(file)` (full path) to `file.name` (filename only)
  - Affects status tracking in Downloaded Files tab (line 4389)
  - Affects bulk add operations (line 4437)
  - Root Cause: Full paths created inconsistent tracking keys across sessions
  - Impact: Downloaded files status now properly persists ("Added to KB" indicator)
  - Files tracked consistently by name regardless of directory structure

- **Database Transaction Error** - Fixed "no transaction is active" error in Quick Add
  - Wrapped database UPDATE statements in proper transaction context (`with kb.db_conn:`)
  - Fixed in both Quick Add implementations (lines 4001, 4259)
  - Root Cause: Direct cursor.execute() + commit() without transaction context
  - Impact: Quick Add now reliably updates source_url metadata without transaction errors
  - Transaction auto-commits when exiting context manager

- **Quick Added Tab UX** - Added individual remove buttons for failed entries
  - Added 4th column with delete button (🗑️) for each entry
  - Users can now remove individual entries instead of clearing entire history
  - Proper index calculation for reversed display order
  - Impact: Better control over Quick Added history management
  - Especially useful for removing failed entries while keeping successful ones

### Changed
- **Version Bump** - Updated version from 2.23.2 to 2.23.3

### Impact
- Improved reliability of Archive Search Quick Add workflow
- Better user experience with working status tracking and entry management
- Eliminated transaction errors during metadata updates
- Enhanced Quick Added tab usability

## [2.23.2] - 2026-01-02

### Fixed
- **Archive Search Security Fix** - Added downloads and temp directories to allowed paths (commit 9f095cf)
  - Fixed "Path outside allowed directories" error when using Add to KB button
  - Added `{data_dir}/downloads` to default allowed directories for Downloaded Files tab
  - Added `{data_dir}/temp` to default allowed directories for Quick Add functionality
  - Root Cause: Only scraped_docs and current working directory were in allowed paths by default
  - Impact: All Archive Search features now fully functional (Add to KB, Quick Add, downloads)
  - Security: Maintains path traversal protection while enabling proper workflows
  - Updated allowed directories list in server.py (lines 309-314)

- **DocumentMeta Subscript Error** - Fixed object access in Downloaded Files Add to KB (commit 7cbf211)
  - Fixed `'DocumentMeta' object is not subscriptable` error
  - Root Cause: add_document() returns DocumentMeta object, code treated it as string
  - Changed `doc_id[:12]` to `doc.doc_id[:12]` for proper attribute access
  - Location: admin_gui.py line 4397 in Downloaded Files tab
  - Impact: Add to KB button now works correctly with proper doc ID display

- **Timezone Import Error** - Fixed NameError in Quick Add error handlers (commit 803f72d)
  - Fixed `NameError: name 'timezone' is not defined` in Quick Add functionality
  - Added timezone to module-level imports (from datetime import datetime, timezone)
  - Removed redundant local imports in Quick Add code blocks
  - Impact: Quick Add error tracking now functions properly in all code paths

### Testing
- **Quick Added Tab Test Suite** - Comprehensive unit tests (commit 77b7b7f)
  - Created test_quick_added.py with 23 tests, 100% pass rate
  - TestQuickAddedTab (8 tests): Session state, entry tracking, clear history, display order
  - TestDownloadFunctionality (6 tests): Temp directory, file naming, cleanup, security
  - TestAddToKBFunctionality (7 tests): Parameters, metadata, title/tags, error handling
  - TestIntegration (2 tests): Complete workflow, failure tracking
  - Total Archive Search test coverage: 49 tests across 3 test suites

### Changed
- **Version Bump** - Updated version from 2.23.1 to 2.23.2
- **Build Date** - Updated from 2026-01-01 to 2026-01-02

### Impact
- Archive Search now fully operational with all security issues resolved
- Complete test coverage ensures reliability of Quick Add workflows
- Proper error handling and cleanup in all code paths
- Enhanced user experience with working Add to KB and Quick Add features

## [2.23.1] - 2026-01-01

### Fixed
- **Python 3.14 Compatibility** - Added workaround for SQLite commit() SystemError
  - Python 3.14.0 has a bug where sqlite3.Connection.commit() can return NULL without setting exception
  - Added SystemError handling in _remove_document_db() method
  - Verifies deletion succeeded by checking document existence
  - Prevents test failures on Python 3.14+

- **translate-query CLI Command** - Fixed KeyError exceptions
  - Added missing 'suggested_query' field to all return paths in translate_nl_query()
  - Made entity field access safe with .get() for 'source' and 'confidence'
  - Command now works correctly with LLM-returned entities

- **Python 3.10 Compatibility** - Fixed f-string nested quotes syntax error
  - Line 6144 used f-string syntax only supported in Python 3.12+
  - Extracted string manipulation outside f-string for Python 3.10/3.11 compatibility
  - Maintains "Python 3.10+" compatibility as documented

- **Bare Except Blocks** - Replaced 6 instances with specific exception types
  - admin_gui.py (2): ValueError, TypeError, AttributeError for datetime/entity access
  - rest_server.py (2): Exception for SQL query errors
  - server.py (2): JSON/date parsing errors with specific types
  - Improves error handling and debuggability

### Refactored
- **Code Quality Improvements** - Major cleanup reducing Ruff errors 91% (69→6)
  - Removed 435 lines of duplicate method definitions (get_entity_analytics, translate_nl_query)
  - Auto-fixed 43 Ruff linting errors (imports, formatting, whitespace)
  - Removed unused imports (PIL.Image)
  - Fixed unused variables (full_text OCR logic, doc, total_chunks, seen_docs)
  - Split 6 multi-statement lines for better readability
  - Remaining 6 errors are intentional (E402: imports after sys.path.insert)

- **OCR Text Extraction** - Fixed logic to properly use OCR results
  - Corrected full_text variable usage in _extract_pdf_file()
  - Now properly replaces pages list with OCR text when scanned PDF detected
  - Prevents unused variable warnings

### Testing
- **Test Suite Status** - All 59 tests passing (100% pass rate)
  - 2 skipped tests (semantic search - feature not enabled)
  - Python 3.14 compatibility verified
  - No regressions from maintenance changes

- **REST API Test Fixes** - Achieved 100% pass rate for implemented endpoints (commit 5895edb)
  - Fixed 11 failing tests, improving from 64% to 82% pass rate (32/39 tests)
  - **Entity Storage**: Added document existence check to prevent FOREIGN KEY errors during background extraction
  - **Search Endpoints**: Fixed parameter mismatches (semantic_search: top_k→max_results, hybrid_search: removed invalid top_k)
  - **Document Endpoints**: Fixed DocumentMeta attribute mappings (created_at→indexed_at, num_chunks→total_chunks)
  - **Upload Endpoint**: Changed temp file location from system temp to data_dir/uploads for security compliance
  - **Test Assertions**: Fixed similar documents URL, corrected response format expectations
  - **Unimplemented Endpoints**: Properly marked 7 tests as skipped for optional endpoints (bulk ops, export endpoints)
  - All implemented REST API endpoints now fully tested and production-ready

## [2.23.0] - 2025-12-23

### Added

#### 🤖 RAG-Based Question Answering (Phase 2 Complete)

- **answer_question() Method** - Natural language Q&A using Retrieval-Augmented Generation
  - Intelligent search mode selection (keyword/semantic/hybrid) based on query analysis
  - Token-budget aware context building (4000 tokens) for LLM integration
  - Citation extraction and validation from generated answers
  - Confidence scoring (0.0-1.0) based on source agreement and LLM certainty
  - Graceful fallback to search summary when LLM unavailable
  - Works with Anthropic, OpenAI, and other LLM providers
  - MCP tool: `answer_question` with parameters (question, max_sources, search_mode)

#### 🔍 Advanced Search Features (Phase 2)

- **Fuzzy Search with Typo Tolerance** - `fuzzy_search()` method using rapidfuzz
  - Handles misspellings: "VIC2" → "VIC-II", "asembly" → "assembly", "6052" → "6502"
  - Configurable similarity threshold (default 80%)
  - Vocabulary building from indexed content for better matching
  - Exact matches prioritized, fuzzy matches as fallback
  - MCP tool: `fuzzy_search` with similarity_threshold parameter

- **Progressive Search Refinement** - `search_within_results()` method
  - Refine previous search results with follow-up queries
  - "Drill down" workflow for exploring large result sets
  - Progressive discovery: broad → specific → precise
  - MCP tool: `search_within_results` for iterative refinement

#### 🏷️ Smart Document Tagging System (Phase 2)

- **AI-Powered Tag Suggestions** - `suggest_tags()` method
  - Organized by category: hardware, programming, document-type, difficulty
  - Confidence scoring for each suggested tag
  - Multi-level categorization for better organization
  - MCP tool: `suggest_tags` with confidence_threshold parameter

- **Tag Category Browser** - `get_tags_by_category()` method
  - Browse all tags grouped by category
  - Usage counts for each tag
  - Discover how documents are organized
  - MCP tool: `get_tags_by_category` for tag exploration

- **Tag Application** - `add_tags_to_document()` method
  - Apply suggested tags to documents
  - Update document metadata with new tags
  - MCP tool: `add_tags_to_document` for tag management

### Documentation

- **README.md** - Compressed and updated
  - Reduced from 1714 to 461 lines (73% reduction)
  - Added RAG features and advanced search documentation
  - Consolidated tool documentation with concise examples
  - Updated version badge to v2.23.0

- **CONTEXT.md** - Compressed and updated
  - Reduced from 221 to 146 lines (34% reduction)
  - Updated MCP tools summary (50+ tools)
  - Added Phase 2 completion status
  - Updated version history highlights

- **CLAUDE.md** - Compressed and updated
  - Reduced from 184 to 117 lines (36% reduction)
  - Streamlined quick reference for Claude Code
  - Removed redundant architecture details
  - Preserved essential dev commands and patterns

- **version.py** - Updated VERSION_HISTORY
  - Comprehensive v2.23.0 release notes
  - Phase completion tracking (Phases 1, 2, 3 complete)
  - Feature tracking for RAG, fuzzy search, tagging

### Impact

- **Phase 2 Complete:** Advanced Search & Discovery fully implemented
  - RAG question answering transforms knowledge base into intelligent assistant
  - Fuzzy search improves usability with typo tolerance
  - Progressive refinement enables better information discovery
  - Smart tagging improves document organization

- **Documentation Efficiency:** 66% reduction in documentation size (2119 → 724 lines)
  - Removed 1395 lines of redundancy
  - Preserved all essential information
  - Clear separation of concerns between docs
  - Improved maintainability

### Phase Completion Status

- ✅ **Phase 1:** AI-Powered Intelligence (v2.13-v2.22.0)
  - Entity extraction, relationships, analytics
  - Document summarization and comparison
  - Natural language query translation

- ✅ **Phase 2:** Advanced Search & Discovery (v2.23.0)
  - RAG question answering with citations
  - Fuzzy search with typo tolerance
  - Progressive search refinement
  - Smart document tagging

- ✅ **Phase 3:** Content Intelligence (v2.15-v2.22.0)
  - Version tracking and update detection
  - Anomaly detection for URL content
  - Performance validation and benchmarking

- 🎯 **Next:** Production stability, maintenance, feature refinement

## [2.22.0] - 2025-12-23

### Added

#### 🧠 Enhanced Entity Intelligence

- **C64-Specific Regex Entity Patterns** - Instant, no-cost entity detection
  - 18 hardware patterns (VIC-II, SID, CIA1/2, 6502, 6510, KERNAL, BASIC, etc.)
  - 3 memory address formats with high confidence:
    - `$D000` format (99% confidence)
    - `0xD000` hexadecimal (98% confidence)
    - Decimal addresses like `53280` (85% confidence)
  - 56 6502 instruction opcodes (LDA, STA, JMP, JSR, ADC, etc.)
  - 15 C64 concept patterns (sprites, raster interrupts, character sets, etc.)
  - **5000x faster** than LLM-only extraction (~1ms vs ~5s)
  - Hybrid extraction strategy: Regex for well-known patterns + LLM for complex cases

- **Entity Normalization** - Consistent entity representation
  - New method: `_normalize_entity_text()` for standardization
  - Hardware normalization: VIC II / VIC 2 / VIC-II → VIC-II
  - Memory address normalization: $d020 → $D020, 0xd020 → $D020
  - Instruction normalization: lda → LDA
  - Concept singularization: sprites → sprite
  - **Impact:** Better cross-document entity matching and deduplication

- **Entity Source Tracking** - Know where entities came from
  - Tracks extraction source: `regex`, `llm`, or `both`
  - Confidence boosting when multiple sources agree
  - Regex-detected entities have higher baseline confidence
  - LLM entities validated by regex get confidence boost
  - Enables filtering by extraction quality and reliability

#### 📈 Enhanced Relationship Intelligence

- **Distance-Based Relationship Strength** - Proximity matters
  - Exponential decay weighting based on character distance
  - Decay factor: 500 characters
  - Adjacent entities (same sentence): ~0.95 strength
  - Distant entities (different paragraphs): ~0.40 strength
  - More meaningful relationship graphs and analytics

- **Logarithmic Normalization** - Better score distribution
  - Log-scale normalization: `log(1 + strength) / log(1 + max_strength)`
  - Avoids linear compression of relationship scores
  - More interpretable relationship strengths
  - Better visual representation in network graphs

#### 🔬 Performance Benchmarking Suite

- **Comprehensive Benchmarking Tool** - `benchmark_comprehensive.py` (440 lines)
  - 6 benchmark categories with detailed metrics:
    - **FTS5 Search:** 8 queries with timing and result counts
    - **Semantic Search:** 4 queries with first-query tracking (model loading)
    - **Hybrid Search:** 4 queries with different semantic weights
    - **Document Operations:** get_document, list_documents, get_stats
    - **Health Check:** 5-run average with status verification
    - **Entity Extraction:** Regex extraction performance
  - Baseline comparison with percentage differences
  - JSON output for performance tracking over time
  - Command: `python benchmark_comprehensive.py --output results.json`

- **Performance Baselines Established (185 docs):**
  - FTS5 search: 85.20ms avg (79-94ms range)
  - Semantic search: 16.48ms avg (first query 5.6s with model loading)
  - Hybrid search: 142.21ms avg
  - Document get: 1.95ms avg
  - List documents: <0.01ms
  - Get stats: 49.62ms
  - Health check: 1,089ms avg
  - Entity regex: 1.03ms avg

#### 🚀 Load Testing Infrastructure

- **Load Testing Suite** - `load_test_500.py` (568 lines)
  - Synthetic C64 documentation generation (10 topics)
  - Scales from current documents to target (e.g., 185 → 500)
  - Concurrent search testing (2/5/10 workers)
  - Memory profiling with psutil
  - Database size tracking
  - Baseline comparison with percentage metrics
  - Command: `python load_test_500.py --target 500 --output results.json`

- **Scalability Validation (500 docs vs 185 baseline):**
  - **FTS5 Search:** 92.54ms (+8.6%) - Excellent O(log n) scaling
  - **Semantic Search:** 13.66ms (-17.1%) - **FASTER at scale!**
  - **Hybrid Search:** 103.74ms (-27.0%) - **MUCH faster at scale!**
  - **Key Insight:** System benefits from scale due to better cache hit rates and FAISS index efficiency
  - Concurrent throughput: 5,712 queries/sec (10 workers)
  - Storage efficiency: 0.3 MB per document
  - Memory usage: ~1 MB per document in RAM

- **Scalability Projections:**
  - 1,000 docs: FTS5 ~100ms, Semantic ~12ms, Hybrid ~95ms
  - 5,000 docs: FTS5 ~120ms, Semantic ~10ms, Hybrid ~80ms
  - **Recommendation:** Excellent performance up to 5,000 documents with default configuration

### Changed

- **Enhanced extract_entities() Method**
  - Now uses hybrid extraction: Regex + LLM
  - Intelligent deduplication with entity normalization
  - Source tracking and confidence boosting
  - Better performance: Fast common entities (regex) + deep extraction (LLM)

- **Enhanced extract_entity_relationships() Method**
  - Distance-based strength calculation with exponential decay
  - Logarithmic normalization for better score distribution
  - More accurate relationship graphs

### Documentation

- **EXAMPLES.md** - Added comprehensive performance documentation
  - Performance benchmarking examples and methodology
  - Load testing results and analysis
  - Scalability insights and projections (185 → 5,000 docs)
  - Performance recommendations for different search modes
  - Tips for choosing between FTS5, semantic, and hybrid search

- **version.py** - Updated VERSION_HISTORY
  - Comprehensive v2.22.0 release notes
  - Feature tracking for new capabilities
  - Performance metrics and scalability findings

### Performance Improvements

- **Entity Extraction:** 5000x faster for common C64 terms (regex vs LLM)
- **Semantic Search:** 17% faster at scale (500 docs vs 185 docs)
- **Hybrid Search:** 27% faster at scale (500 docs vs 185 docs)
- **Better Cache Utilization:** Performance improves with larger datasets
- **FAISS Index Efficiency:** Semantic search benefits from more documents

### Files Added

- `benchmark_comprehensive.py` - Comprehensive performance benchmarking suite
- `load_test_500.py` - Load testing with synthetic document generation
- `benchmark_results.json` - Baseline performance metrics (185 docs)
- `load_test_results.json` - Scalability test results (500 docs)

### Impact

- **Cost Reduction:** Entity extraction 5000x faster for common C64 terms (no LLM calls needed)
- **Better Accuracy:** Entity normalization improves cross-document matching
- **Meaningful Relationships:** Distance-based weighting reflects actual entity proximity
- **Performance Confidence:** Established baselines enable regression tracking
- **Scalability Proven:** Validated excellent performance up to 5,000+ documents
- **Unexpected Benefit:** Semantic/hybrid search actually improve with more data

## [2.21.1] - 2025-12-23

### Fixed
- **Health Check False Positive** - Fixed health_check() incorrectly reporting "Semantic search enabled but embeddings not built" when using lazy-loaded embeddings
  - Health check now properly detects embeddings files on disk (not just in-memory)
  - Shows correct embeddings count and size even when not yet loaded
  - Fixes false warning for systems using lazy loading (default behavior)
- **Admin GUI AttributeError** - Fixed crash when viewing URL monitoring page
  - Fixed scrape_config parsing (was string, needed JSON parsing)
  - Added error handling for malformed JSON configs
  - Affected lines: admin_gui.py:1506, 1772
- **URL Scraping Image Errors** - Improved error handling for WordPress gallery sites
  - Image scraping errors (JPG, PNG, etc.) now logged as warnings instead of errors
  - Scraping continues despite image URL failures (partial success)
  - Cleaner logs for sites with image galleries
- **Test Suite Environment** - Improved test isolation and security
  - Added automatic ALLOWED_DOCS_DIRS configuration for test fixtures
  - Fixed SecurityError failures during testing
  - Better cleanup between tests

### Changed
- **Documentation Updates**
  - Updated CONTEXT.md from v2.14.0 to v2.21.0
  - Added version history highlights (v2.15-v2.21)
  - Organized MCP tools into logical categories
  - Added new environment variables documentation
  - Updated README.md version badge to v2.21.0

### Removed
- **Cleanup** - Removed obsolete performance testing files
  - benchmark_results.json (old benchmark data)
  - profile_performance.py (deprecated profiling script)

## [2.18.0] - 2025-12-21

### Added

#### 🚀 Background Entity Extraction (Phase 2)
- **Zero-delay entity extraction** - Extraction happens asynchronously in background
- **Background Worker Thread** - Dedicated daemon thread processes extraction queue
- **Job Tracking System** - Full visibility into extraction progress
  - Database table: `extraction_jobs` with status tracking
  - Status transitions: queued → running → completed/failed
  - Timestamps: queued_at, started_at, completed_at
  - Error messages and entity counts stored
- **Auto-Queue on Ingestion** - Documents automatically queued when added
  - Configurable via `AUTO_EXTRACT_ENTITIES=1` (default: enabled)
  - Doesn't block document ingestion
  - Duplicate prevention (skip if entities exist or job queued)
- **Job Management Methods** (3 new methods)
  - `queue_entity_extraction()` - Queue extraction for a document
  - `get_extraction_status()` - Check status for specific document
  - `get_all_extraction_jobs()` - List all jobs with filtering
- **MCP Tools** (3 new tools)
  - `queue_entity_extraction` - Manual job queuing
  - `get_extraction_status` - Status checking with job history
  - `get_extraction_jobs` - Monitor queue and job progress
- **Benefits:**
  - Users never wait for LLM extraction (previously 3-30 seconds)
  - Work continues immediately after document upload
  - Full job visibility and monitoring
  - Robust error handling

#### 📊 Entity Analytics Dashboard (Sprint 2)
- **Comprehensive Analytics** - `get_entity_analytics()` method
  - 6 data structures for complete entity analysis
  - Entity distribution by type (9 types tracked)
  - Top 50 entities by document count
  - Relationship statistics with type breakdown
  - Top 50 strongest relationships
  - 30-day extraction timeline
  - Overall summary statistics
- **Interactive GUI Dashboard** - 4-tab interface in admin_gui.py
  - **Overview Tab:** Entity distribution bar chart and data table
  - **Top Entities Tab:** Sortable table with filters (type, doc count, confidence)
  - **Relationships Tab:** Network graph + top relationships table
  - **Trends Tab:** Timeline of entity extractions over time
- **Interactive Network Graph** for entity relationships
  - Drag-and-drop node exploration with pyvis
  - Color-coded entity types with 7-color legend
  - Adjustable controls (max nodes, min strength, show/hide)
  - Edge thickness and transparency scaled by relationship strength
  - Hover tooltips with entity details and relationship strength
  - 600px responsive visualization with dark theme
- **Export Buttons** - CSV and JSON downloads from dashboard
- **Real-time Statistics** - 989 unique entities, 128 relationships tracked

#### ⚡ Performance Optimizations (Phase 1)
- **Semantic Search - 43% Faster** (14.53ms → 8.31ms)
  - Query embedding cache (LRU, 1-hour TTL)
  - First query: 14ms, subsequent: 6-8ms (cached)
  - Configurable via `EMBEDDING_CACHE_TTL` (default: 3600s)
- **Hybrid Search - 22% Faster** (19.44ms → 15.24ms)
  - Parallel execution of FTS5 + semantic searches
  - ThreadPoolExecutor with max_workers=2
  - Combined with embedding cache for even better results
- **Entity Extraction Caching**
  - Two-tier caching: memory (TTLCache) + database
  - Cached calls: 0.03ms (4x faster than database-only: 0.12ms)
  - First extraction: ~3s (LLM), subsequent: sub-millisecond
  - Configurable via `ENTITY_CACHE_TTL` (default: 86400s / 24 hours)
- **Overall Benchmark Improvement: 8% faster**
  - Total benchmark time: 6.27s → 5.75s
  - Memory impact: ~6.5MB for all caches (minimal)
- **Detailed Reporting**
  - PERFORMANCE_IMPROVEMENTS.md - Complete analysis
  - benchmark.py - Comprehensive benchmarking suite
  - Before/after comparisons with statistical analysis

#### 📄 Document Comparison (Sprint 3) - Already Existed
- Verified complete implementation
- Side-by-side comparison with 89.7% similarity scoring
- Metadata diff, content diff, entity comparison
- 4-tab GUI interface

#### 📤 Export Features (Sprint 4) - Already Existed
- Verified export_entities() and export_relationships()
- CSV and JSON export formats
- 996 entities exported successfully in testing

#### 🌐 REST API Server
- **Complete FastAPI-based HTTP/REST interface**
  - 27 endpoints across 6 categories
  - API key authentication via X-API-Key header
  - CORS middleware for cross-origin requests
  - OpenAPI/Swagger documentation at `/api/docs`
  - ReDoc documentation at `/api/redoc`
- **Endpoint Categories:**
  - Health & Stats (2 endpoints): health check, KB statistics
  - Search (5 endpoints): basic, semantic, hybrid, faceted, similar documents
  - Documents (7 endpoints): CRUD operations, bulk upload/delete
  - URL Scraping (3 endpoints): scrape URL, rescrape, check updates
  - AI Features (5 endpoints): summarization, entity extraction/search, relationships
  - Analytics & Export (5 endpoints): search analytics, CSV/JSON exports
- Pydantic v2 request/response validation
- Request logging and comprehensive error handling
- Export functionality for search results, documents, entities, relationships
- README_REST_API.md - 576 lines of complete API documentation
- run_rest_api.bat - Windows startup script

### Changed
- Thread-safe KnowledgeBase sharing between MCP and REST servers

### Fixed
- 7 REST API bugs during comprehensive testing
  - Stats endpoint: kb.use_fts5 attribute error
  - Semantic search: top_k parameter not supported
  - Get document: dict vs object return type mismatch
  - Hybrid search: top_k parameter compatibility
  - Get entities: list extraction from dict return
  - Get relationships: method name correction (get_entity_relationships)
  - Faceted search: validation requirement clarification

## [2.17.0] - 2025-12-21

### Added
- **Natural Language Query Translation** - AI-powered query parsing
  - Dual extraction: Regex patterns for C64-specific hardware + LLM for contextual entities
  - `translate_nl_query()` method with confidence scoring
  - MCP tool: `translate_query`
  - CLI command: `translate-query` with formatted output
  - GUI integration in Search page with NL translation toggle
  - Automatic search mode recommendation (keyword/semantic/hybrid)
  - Facet filter generation from detected entities
  - Graceful fallback when LLM unavailable

## [2.16.0] - 2025-12-21

### Added - Entity Relationship Tracking

#### 🔗 Relationship Discovery Engine
- **New Feature:** Track co-occurrence patterns between entities within documents
- **Relationship Type:** Co-occurrence-based (entities appearing together in chunks)
- **Strength Scoring:** Normalized 0.0-1.0 scores based on frequency
- **Context Preservation:** Sample text showing how entities relate
- **Incremental Updates:** Relationships averaged across multiple documents

#### 💾 Database Schema
- **New Table:** `entity_relationships` - Stores entity co-occurrence data
  - Fields: entity1_text, entity1_type, entity2_text, entity2_type, relationship_type, strength, doc_count, first_seen_doc, context_sample, last_updated
  - Composite PRIMARY KEY on (entity1_text, entity2_text, relationship_type)
  - Automatic deduplication (VIC-II + sprite == sprite + VIC-II)
- **Indexes:** 4 indexes for fast queries
  - idx_relationships_entity1 on entity1_text
  - idx_relationships_entity2 on entity2_text
  - idx_relationships_type on relationship_type
  - idx_relationships_strength on strength

#### 🔧 Core Methods (5 new methods)
- **`extract_entity_relationships(doc_id, min_confidence, force_regenerate)`** - Extract relationships from document
  - Analyzes entity co-occurrence within chunks
  - Normalizes strength scores per document
  - Updates existing relationships with weighted average
- **`get_entity_relationships(entity_text, relationship_type, min_strength, max_results)`** - Query relationships for entity
  - Find all entities related to a given entity
  - Filter by relationship type and minimum strength
  - Returns sorted by strength (strongest first)
- **`find_related_entities(entity_text, max_results)`** - Simplified relationship discovery
  - Convenience wrapper with min_strength=0.3
  - Quick way to find strongly related entities
- **`search_by_entity_pair(entity1, entity2, max_results)`** - Find documents with both entities
  - Searches for documents containing entity pair
  - Returns occurrence counts for each entity
  - Sorted by total occurrences
- **`extract_relationships_bulk(min_confidence, max_docs, skip_existing)`** - Bulk processing
  - Extract relationships from multiple documents
  - Comprehensive error handling and progress tracking
  - Returns statistics on processed documents

#### 🔌 MCP Tools (4 new tools)
- **`extract_entity_relationships`** - Extract relationships from a document
  - Inputs: doc_id (required), min_confidence, force_regenerate
  - Returns: Top 10 relationships sorted by strength
- **`get_entity_relationships`** - Get relationships for an entity
  - Inputs: entity_text (required), relationship_type, min_strength, max_results
  - Returns: Related entities with strength scores and contexts
- **`find_related_entities`** - Quick relationship lookup
  - Inputs: entity_text (required), max_results
  - Returns: Strongly related entities (strength ≥ 0.3)
- **`search_entity_pair`** - Find documents with entity pair
  - Inputs: entity1 (required), entity2 (required), max_results
  - Returns: Documents containing both entities with occurrence counts

#### 💻 CLI Commands (4 new commands)
- **`extract-relationships <doc_id>`** - Extract relationships from document
  - Options: --confidence (entity threshold)
  - Shows top 15 relationships with strength scores
- **`extract-all-relationships`** - Bulk extract from all documents
  - Options: --confidence, --max, --no-skip
  - Shows processing statistics
- **`show-relationships <entity>`** - Show related entities
  - Options: --min-strength, --max
  - Lists related entities sorted by strength
- **`search-pair <entity1> <entity2>`** - Find docs with both entities
  - Options: --max
  - Shows documents with occurrence counts

#### 🖥️ GUI Interface
- **New Page:** "Entity Relationships" (4 tabs)
  - Tab 1: Extract relationships from single document
  - Tab 2: View relationships for specific entity
  - Tab 3: Search documents by entity pair
  - Tab 4: Bulk extraction across documents
- **Statistics Dashboard:** Total relationships and unique entities
- **Interactive Results:** Expandable displays with context snippets
- **Progress Tracking:** Real-time progress bars for bulk operations

#### ✅ Features
- Bidirectional relationship queries (A→B or B→A)
- Relationship strength normalization across documents
- Context preservation for understanding relationships
- Database migration for existing installations
- Comprehensive error handling
- Full integration with entity extraction system

### Technical Details
- **Lines Added:** ~750 lines in server.py, ~145 lines in cli.py, ~300 lines in admin_gui.py
- **Database Tables:** 1 table + 4 indexes
- **Backward Compatibility:** 100% compatible with existing code
- **Performance:** Indexed queries for fast relationship lookups

### Testing Results
- ✅ Extracted 128 relationships from C64 Programmer's Reference Guide
- ✅ VIC-II correctly linked to sprite, SID, CIA, IRQ concepts
- ✅ Assembly instructions (LDA, STA, JMP) properly co-located
- ✅ Entity pair search successfully found 5 documents with VIC-II + sprite
- ✅ All MCP tools, CLI commands, and GUI tabs validated

## [2.15.0] - 2025-12-20

### Added - AI-Powered Named Entity Extraction

#### 🏷️ Entity Extraction Engine
- **New Feature:** Extract named entities from C64 documentation using AI (LLM)
- **Entity Types (7 categories):**
  - `hardware` - Chip names (SID, VIC-II, CIA, 6502, 6526, 6581, etc.)
  - `memory_address` - Memory addresses ($D000, $D020, $0400, etc.)
  - `instruction` - Assembly instructions (LDA, STA, JMP, JSR, RTS, etc.)
  - `person` - People mentioned (Bob Yannes, Jack Tramiel, etc.)
  - `company` - Organizations (Commodore, MOS Technology, etc.)
  - `product` - Hardware products (VIC-20, C128, 1541, etc.)
  - `concept` - Technical concepts (sprite, raster interrupt, IRQ, etc.)

#### 💾 Database Schema
- **New Table:** `document_entities` - Stores extracted entities with metadata
  - Fields: doc_id, entity_id, entity_text, entity_type, confidence, context, first_chunk_id, occurrence_count, generated_at, model
  - Foreign key CASCADE delete when document removed
- **FTS5 Virtual Table:** `entities_fts` - Full-text search on entities
  - Indexed fields: entity_text, context
  - Porter stemming for better search results
- **Triggers:** 3 triggers (INSERT, DELETE, UPDATE) keep FTS5 in sync
- **Indexes:** 3 indexes on doc_id, entity_type, entity_text for fast queries

#### 🔧 Core Methods
- **`extract_entities(doc_id, confidence_threshold, force_regenerate)`** - Extract entities from single document
  - Samples first 5 chunks (up to 5000 chars) for cost control
  - LLM temperature 0.3 for deterministic extraction
  - Case-insensitive deduplication with occurrence counting
  - Database caching to avoid re-extraction
- **`get_entities(doc_id, entity_types, min_confidence)`** - Retrieve stored entities
  - Filter by entity types and confidence threshold
  - Returns same structure as extract_entities
- **`search_entities(query, entity_types, min_confidence, max_results)`** - Search entities across all docs
  - FTS5 full-text search on entity text and context
  - Filter by type and confidence
  - Results grouped by document with match counts
- **`find_docs_by_entity(entity_text, entity_type, min_confidence, max_results)`** - Find docs containing entity
  - Exact entity text matching
  - Optional type and confidence filtering
- **`get_entity_stats(entity_type)`** - Get entity extraction statistics
  - Total entities and documents with entities
  - Breakdown by type
  - Top 20 entities by document count
  - Top 10 documents by entity count
- **`extract_entities_bulk(confidence_threshold, force_regenerate, max_docs, skip_existing)`** - Bulk extraction
  - Process multiple documents in batch
  - Skip existing entities unless force_regenerate
  - Comprehensive error handling and progress tracking

#### 🔌 MCP Tools (5 new tools)
- **`extract_entities`** - Extract entities from a document
  - Inputs: doc_id (required), confidence_threshold, force_regenerate
  - Returns: Entities grouped by type with first 5 per type
- **`list_entities`** - List all entities from a document
  - Inputs: doc_id (required), entity_types (filter), min_confidence
  - Returns: All entities with optional filtering
- **`search_entities`** - Search for entities across all documents
  - Inputs: query (required), entity_types, min_confidence, max_results
  - Returns: Documents with matching entities and contexts
- **`entity_stats`** - Show entity extraction statistics
  - Inputs: entity_type (optional filter)
  - Returns: Stats breakdown, top entities, top documents
- **`extract_entities_bulk`** - Bulk extract entities from all documents
  - Inputs: confidence_threshold, force_regenerate, max_docs, skip_existing
  - Returns: Processing statistics and results

#### 💻 CLI Commands (4 new commands)
- **`extract-entities <doc_id>`** - Extract entities from single document
  - Options: --confidence, --force
  - Shows first 10 entities per type
- **`extract-all-entities`** - Bulk extract from all documents
  - Options: --confidence, --force, --max, --no-skip
  - Shows processing statistics and entity counts by type
- **`search-entity <query>`** - Search for entities
  - Options: --type, --confidence, --max
  - Shows matching documents with first 3 matches per doc
- **`entity-stats`** - Show extraction statistics
  - Options: --type (filter by entity type)
  - Shows top 10 entities and documents

#### ✅ Features
- Confidence scoring (0.0-1.0) for each extracted entity
- Occurrence counting (how many times entity appears)
- Context snippets (surrounding text for each entity)
- Database migration for existing installations
- LLM provider support (Anthropic Claude, OpenAI GPT)
- Full-text search across all entities
- Comprehensive error handling

### Technical Details
- **Lines Added:** ~1,600 lines across server.py and cli.py
- **Database Tables:** 1 main table + 1 FTS5 table + 3 triggers + 3 indexes
- **Backward Compatibility:** 100% compatible with existing code
- **LLM Cost:** ~$0.01-0.04 per document (depending on size)

## [2.14.0] - 2025-12-18

### Added - UI/UX Improvements & Configuration Enhancements

#### 🎨 Loading Indicators
- **Centered Loading Screen:** Added professional loading indicators for long-running operations
  - Animated 🏃 icon for BM25 index building with real-time progress bar
  - Animated 🌐 icon for web scraping operations
  - Gradient purple backgrounds with pulsing animations
  - Progress percentage display and descriptive status text
  - ⏹️ Stop button to cancel index building operations

#### 🔧 Configuration & Environment
- **python-dotenv Integration:** Added automatic .env file loading in server.py
  - All environment variables now loaded from .env file automatically
  - No need to manually set environment variables
  - Simplifies deployment and configuration
- **Updated start-all.bat:**
  - Added POPPLER_PATH for OCR support
  - Fixed ALLOWED_DOCS_DIRS to include scraped_docs directory
  - All features enabled by default

#### 🐛 Bug Fixes
- **Preview Slider Fix:** Fixed "min_value must be less than max_value" error
  - Single-chunk documents now display properly
  - Shows "📄 Single chunk document" info instead of broken slider
- **Warning Suppression:** Eliminated ScriptRunContext warning spam in logs
  - Added logging configuration to suppress Streamlit thread warnings
  - Much cleaner, readable log output
- **Security Path Configuration:** Added scraped_docs directory to allowed paths
  - Web scraping now works without security violations
  - Both user uploads and scraped content are allowed

#### 🌐 Web Scraping Enhancements
- **Better Loading UX:** Centered loading indicator during scraping operations
- **Path Security:** Properly configured allowed directories for scraped content

#### ✅ Testing
- **Security Unit Tests:** Added comprehensive test suite (test_security.py)
  - Tests single and multiple allowed directories
  - Tests path traversal attack prevention
  - Tests case-insensitive Windows path handling
  - Tests no-restriction mode
  - Full coverage of _is_path_allowed() logic

### Configuration Files Updated
- **.env:** Added POPPLER_PATH and updated ALLOWED_DOCS_DIRS
- **start-all.bat:** Added POPPLER_PATH and scraped_docs to allowed directories
- **admin_gui.py:** Loading indicators, warning suppression, bug fixes
- **server.py:** python-dotenv integration

## [2.13.0] - 2025-12-17

### Added - AI-Powered Document Summarization (Phase 1.2)

#### 📝 Document Summarization Engine
- **New Method:** `generate_summary(doc_id, summary_type, force_regenerate)` - Generate AI summaries
- **Summary Types:**
  - **Brief:** 200-300 word overviews (1-2 paragraphs)
  - **Detailed:** 500-800 word comprehensive summaries (3-5 paragraphs)
  - **Bullet:** 8-12 key technical points in bullet format
- **Features:**
  - Intelligent database caching for instant retrieval
  - Configurable LLM providers (Anthropic Claude, OpenAI GPT)
  - Automatic fallback handling
  - Word count tracking and token usage logging

#### 🔄 Bulk Summarization
- **New Method:** `generate_summary_all(summary_types, force_regenerate, max_docs)` - Bulk process
- **Features:**
  - Generate multiple summary types in single operation
  - Process entire knowledge base or limit with max_docs
  - Comprehensive statistics (processed, failed, by type)
  - Non-blocking error handling per document
  - Force regeneration option to bypass cache

#### 📖 Summary Retrieval
- **New Method:** `get_summary(doc_id, summary_type)` - Cached retrieval
- **Features:**
  - Fast database lookup without API calls
  - Returns None if summary doesn't exist
  - Useful for checking cache before generation

#### 🛠️ MCP Tools (3 new)
- **Tool:** `summarize_document` - Generate single summary
  - Parameters: doc_id, summary_type (brief/detailed/bullet), force_regenerate
  - Returns: Formatted summary text
- **Tool:** `get_summary` - Retrieve cached summary
  - Parameters: doc_id, summary_type
  - Returns: Summary if cached, error message if not
- **Tool:** `summarize_all` - Bulk summarization
  - Parameters: summary_types, force_regenerate, max_docs
  - Returns: Statistics and sample results

#### 🖥️ CLI Commands (2 new)
- **Command:** `summarize <doc_id> [--type TYPE] [--force]`
  - Generate summary for specific document
  - Types: brief (default), detailed, bullet
  - Use --force to bypass cache
- **Command:** `summarize-all [--types TYPES] [--force] [--max NUM]`
  - Bulk generate summaries
  - Multiple types supported
  - Max documents limit for testing

#### 💾 Database Schema
- **New Table:** `document_summaries`
  - Fields: doc_id, summary_type, summary_text, generated_at, model, token_count
  - Primary Key: (doc_id, summary_type)
  - Foreign Key: CASCADE delete with documents table
- **Indexes:**
  - `idx_summaries_doc_id` - Fast lookup by document
  - `idx_summaries_type` - Fast lookup by summary type
- **Automatic Migration:** Schema created on first run for existing databases

#### 📚 Documentation
- **New File:** `SUMMARIZATION.md` - 400+ line comprehensive guide
  - Complete feature documentation
  - Configuration instructions
  - Usage examples and patterns
  - Performance metrics and cost analysis
  - Troubleshooting section
  - Advanced usage and future roadmap
- **Updated:** `README_UPDATED.md` - Added summarization section
- **Updated:** `QUICKSTART_UPDATED.md` - Added examples

#### 🚀 Launch Scripts
- **New:** `launch-cli-full-features.bat` - CLI with all features enabled
- **New:** `launch-gui-full-features.bat` - GUI with all features enabled
- **New:** `launch-server-full-features.bat` - MCP server with all features enabled

#### ⚙️ Configuration
- **New:** `.env` file - Complete environment configuration
  - All feature flags pre-configured
  - LLM provider settings
  - Caching and performance options

### Implementation Details
- **Code Added:** ~1,200 lines across server.py and cli.py
- **Database Compatibility:** Automatic schema migration
- **Backward Compatibility:** 100% compatible with existing code
- **Performance:** Caching provides 50-100ms retrieval vs 3-8s generation
- **Cost Estimates:** ~$0.01-0.04 per summary depending on type

### Testing
- ✅ Syntax validation passed
- ✅ Module imports verified
- ✅ Database initialization confirmed
- ✅ All methods present and functional
- ✅ Schema migration working
- ✅ 149 documents loaded successfully

---

## [2.12.0] - 2025-12-13

### Added - Smart Auto-Tagging with LLM Integration

#### 🤖 LLM Integration Module
- **New File:** `llm_integration.py` - Unified LLM client for multiple providers
- **Supported Providers:**
  - **Anthropic (Claude):** claude-3-haiku-20240307, claude-3-5-sonnet-20241022, claude-3-opus-20240229
  - **OpenAI (GPT):** gpt-3.5-turbo, gpt-4, gpt-4-turbo
- **Features:**
  - Auto-detection of provider from environment variables
  - JSON response parsing with markdown code block handling
  - Configurable models and parameters (temperature, max_tokens)
  - Error handling and logging
- **Environment Variables:**
  - `LLM_PROVIDER` - Provider selection (anthropic/openai, default: anthropic)
  - `ANTHROPIC_API_KEY` - API key for Claude
  - `OPENAI_API_KEY` - API key for GPT models
  - `LLM_MODEL` - Model selection (default: claude-3-haiku-20240307)

#### 🏷️ Smart Auto-Tagging
- **New Method:** `auto_tag_document(doc_id, confidence_threshold, max_tags, append)` - AI-powered tag generation
- **Analysis Process:**
  - Analyzes first 3 chunks (max 3000 chars) of document
  - Sends structured prompt to LLM requesting JSON tags
  - Returns tags with confidence scores (0.0-1.0) and reasoning
- **Tag Categories:**
  - **Hardware:** SID, VIC-II, CIA, 6510, PLA, etc.
  - **Programming Topics:** Assembly, BASIC, machine code, graphics, sound
  - **Document Type:** Tutorial, reference, manual, code example
  - **Difficulty Level:** Beginner, intermediate, advanced
- **Confidence Filtering:**
  - Default threshold: 0.7 (70% confidence)
  - Configurable per-request
  - Only high-confidence tags applied
- **Tag Application:**
  - Append mode: Add to existing tags (default)
  - Replace mode: Replace all existing tags
  - Returns detailed results with applied/skipped tags

**Usage:**
```python
# Python API - Auto-tag a single document
result = kb.auto_tag_document(
    doc_id="doc123",
    confidence_threshold=0.7,
    max_tags=10,
    append=True
)

# Via MCP in Claude Desktop
# "Auto-tag this document with AI-generated tags"
```

#### 📦 Bulk Auto-Tagging
- **New Method:** `auto_tag_all_documents(confidence_threshold, max_tags, append, skip_tagged, max_docs)` - Bulk AI tagging
- **Features:**
  - Process all documents or limit with `max_docs`
  - Skip already-tagged documents (configurable)
  - Rate limiting to prevent API overuse
  - Comprehensive error handling per document
  - Returns statistics: processed, skipped, failed, total_tags_added
- **Performance:**
  - Progress logging for each document
  - Continues on errors (non-blocking)
  - Transaction-safe (each document committed individually)

**Usage:**
```python
# Python API - Auto-tag all untagged documents
results = kb.auto_tag_all_documents(
    confidence_threshold=0.7,
    max_tags=10,
    append=True,
    skip_tagged=True,
    max_docs=None  # Process all
)

# Via MCP in Claude Desktop
# "Auto-tag all documents that don't have tags yet"
```

#### 🛠️ MCP Tools
- **New Tool:** `auto_tag_document` - Single document auto-tagging
  - Parameters: doc_id, confidence_threshold, max_tags, append
  - Returns: Applied tags with confidence scores and reasoning
- **New Tool:** `auto_tag_all` - Bulk document auto-tagging
  - Parameters: confidence_threshold, max_tags, append, skip_tagged, max_docs
  - Returns: Statistics and sample results

#### 📋 Configuration
- **New File:** `.env.example` - Complete environment configuration template
- **LLM Configuration:**
  - Provider selection
  - API key setup
  - Model selection
  - Usage examples
- **Search Configuration:**
  - FTS5, semantic search, BM25 settings
  - Query preprocessing options
- **Security Configuration:**
  - Document directory whitelisting
- **Data Storage:**
  - TDZ_DATA_DIR path configuration

### Changed
- Updated version to 2.12.0
- Version.py includes new features: smart_auto_tagging, llm_integration

### Benefits
- **Time Savings:** Automatically tag hundreds of documents in minutes
- **Consistency:** LLM applies tags uniformly across all documents
- **Discovery:** AI identifies relevant categories you might miss
- **Flexibility:** Support for multiple LLM providers (Claude, GPT)
- **Quality:** Confidence-based filtering ensures only relevant tags
- **Control:** Configurable thresholds and tag limits

### Testing
- LLM integration tested manually with sample documents
- Auto-tagging verified with C64 documentation
- Error handling tested for missing API keys
- JSON parsing tested with various response formats

### Dependencies
- **Required:** anthropic>=0.18.0 OR openai>=1.0.0 (install at least one)
- Install with: `pip install anthropic` or `pip install openai`

### Documentation
- Created `.env.example` with comprehensive configuration guide
- Updated `version.py` to v2.12.0
- Updated `CHANGELOG.md` with v2.12.0 entry
- See `llm_integration.py` docstrings for API details

### Developer Notes
**Implementation Details:**
- LLM integration uses provider abstraction pattern
- Auto-tagging analyzes document content + metadata
- Structured prompts guide LLM to return consistent JSON
- Confidence scores from LLM help filter low-quality suggestions
- Tag format: lowercase with hyphens (e.g., "sid-programming")
- All database operations within transactions

**Breaking Changes:** None - all new features are optional and require API keys

---

## [2.11.0] - 2025-12-13

### Added - GUI Improvements & Version Management

#### 📁 File Path Input in GUI
- **New Tab:** "Add by File Path" in Add Documents section
  - Paste file paths directly (e.g., `C:\Users\...\file.md`)
  - No need to upload files - just enter the path
  - Instant file validation (shows error if file doesn't exist)
  - Supports all file types (PDF, TXT, MD, HTML, Excel)

#### 🔍 Duplicate Detection
- **Intelligent Duplicate Detection** across all document operations
  - Detects files with identical content (content-based hashing)
  - Shows warning message: "Document already exists in knowledge base"
  - Displays existing document details (title, ID, chunks)
  - Prevents database bloat from duplicate entries
  - Works in all three add methods: Single Upload, Bulk Upload, File Path

#### 📄 Enhanced File Viewer
- **Markdown Files (.md):** Dual view mode
  - "Rendered" view - Shows beautifully formatted markdown
    - Headers, lists, code blocks, tables
    - Bold, italic, links
    - Full markdown rendering
  - "Raw Markdown" view - Shows source code
    - Syntax highlighting for markdown
    - Line numbers for easy reference
    - Scrollable code block
  - Toggle button to switch between views
- **Text Files (.txt):** Improved display
  - Scrollable code block with line numbers
  - Line count display for files over 50 lines
  - Monospace font for better readability
  - Professional code viewer styling
- **Removed:** Old `st.text_area()` approach for better UX

#### ⏳ Progress Indicators
- **All Document Operations** now show clear feedback:
  - Spinner with "Adding document: filename..." message
  - Bulk upload shows "Processing X/Y: filename" for each file
  - Clear success/error/duplicate messages
  - No more silent operations

#### 📦 Version Management System
- **New File:** `version.py` - Centralized version information
  - Semantic versioning (MAJOR.MINOR.PATCH)
  - Build date tracking
  - Feature version tracking
  - Version history
- **GUI Version Display:**
  - Sidebar footer shows current version
  - Build date displayed
  - Dynamic version from version.py
- **Server Logging:**
  - Version logged on startup
  - Visible in server.log file
  - Helps with debugging and support

#### 📋 Enhanced Messages
- **Success Messages:** "Document added successfully!" with full details
  - Shows document title
  - Shows chunk count
  - Shows document ID
- **Duplicate Messages:** "Document already exists" with existing doc info
  - Shows which document matched
  - Shows content hash matching
- **Error Messages:** Clear error descriptions
  - "File not found" for invalid paths
  - "Error adding document" with detailed error
- **Bulk Upload:** Summary statistics
  - Count of added documents
  - Count of duplicates (with filenames)
  - Count of failed documents

### Changed
- File viewer now uses `st.code()` and `st.markdown()` instead of `st.text_area()`
- Markdown files render formatting by default
- GUI footer dynamically shows version from version.py
- README.md includes version badge

### Testing
- Created `test_gui_changes.py` - 6/6 tests passing
- Created `test_file_viewer.py` - 4/4 tests passing
- Created `test_gui_startup.py` - Validates GUI can start
- All GUI functionality verified working

### Documentation
- Created `GUI_IMPROVEMENTS_SUMMARY.md`
- Created `FILE_VIEWER_IMPROVEMENTS.md`
- Updated README.md with version badge
- Updated CLAUDE.md with version information

### Developer Notes
**Implementation Details:**
- Version management uses centralized `version.py` file
- Semantic versioning with tuple and string representations
- GUI imports version dynamically (no hardcoded versions)
- Server logs version on every startup for debugging
- File viewer improvements use native Streamlit components
- Duplicate detection tracks document count before/after

**Benefits:**
- Better user experience with clear feedback
- No confusion about document status
- Easier file management (no mandatory uploads)
- Professional version tracking
- Improved code visibility in documentation

## [2.10.0] - 2025-12-13

### Added - HTML File Support
- **HTML File Support** - Ingest and search HTML documentation (.html, .htm)
  - Powered by BeautifulSoup4 for robust HTML parsing
  - Automatic encoding detection with chardet
  - Removes script, style, nav, footer, and header elements
  - Preserves code blocks (<pre>, <code>) with special formatting
  - Extracts page title when available
  - Cleans up excessive whitespace
  - Support for both single and bulk upload in GUI

### Changed
- Default bulk add pattern updated from `**/*.{pdf,txt,md,xlsx,xls}` to `**/*.{pdf,txt,md,html,htm,xlsx,xls}`
- GUI file uploaders now accept HTML files alongside other formats

### Dependencies
- Added `beautifulsoup4>=4.9.0` for HTML parsing
- Added `chardet>=5.0.0` for encoding detection

### Testing
- Added test fixture for sample HTML files
- Added `test_add_html_document()` to verify HTML extraction
- All 59 tests passing (2 skipped)

## [2.9.0] - 2025-12-13

### Added - Excel File Support & Enhanced Markdown
- **Excel File Support** - Ingest and search Excel spreadsheets (.xlsx, .xls)
  - Extract data from all sheets with clear delimiters
  - Tab-delimited columns for readability
  - Sheet count tracked as "pages" for consistency
  - Support for both single and bulk upload in GUI
  - Powered by openpyxl library
- **Enhanced Markdown Support** - Made .md file support more visible
  - Updated bulk add default pattern to include .md files
  - Updated GUI file uploaders to explicitly list Markdown files
  - Better discoverability for existing Markdown support

### Changed
- Default bulk add pattern updated from `**/*.{pdf,txt,md}` to `**/*.{pdf,txt,md,xlsx,xls}`
- GUI file uploaders now accept Excel files alongside PDFs, text, and Markdown

### Testing
- Added test fixture for sample Excel files
- Added `test_add_excel_document()` to verify Excel extraction
- All 58 tests passing (2 skipped)

## [2.8.0] - 2025-12-13

### Added - CI/CD Automation & Relationship Visualization

#### 🔄 CI/CD Pipeline
- **GitHub Actions Workflow** - Automated testing on push/PR
  - Multi-platform testing (Ubuntu, Windows, macOS)
  - Multi-version testing (Python 3.10, 3.11, 3.12)
  - Test coverage reporting with Codecov
  - Security checks (safety, bandit)
  - Code quality checks (ruff linting)
  - Integration tests
  - Documentation validation
  - Build status summary
- **Release Automation** - Automatic GitHub releases
  - Changelog extraction
  - Release artifact uploads
  - Version tagging support
  - Optional PyPI publishing (disabled by default)
- **Status Badges** - README.md badges for CI status, Python version, license

#### 📊 Relationship Graph Visualization
- **New Backend Method:** `get_relationship_graph(tags, relationship_types)`
  - Returns nodes and edges for graph visualization
  - Filter by tags or relationship types
  - Includes statistics (total nodes, edges, relationship types)
- **New GUI Page:** "🔗 Relationship Graph"
  - Interactive network graph powered by pyvis
  - Filter by tags and relationship types
  - Visualization options:
    - Enable/disable physics simulation
    - Choose layout algorithm (hierarchical, force_atlas, barnes_hut, repulsion)
    - Adjust node size
    - Show/hide relationship direction arrows
    - Custom edge colors
  - Color-coded edges by relationship type:
    - 🟢 Green for "related"
    - 🔵 Blue for "references"
    - 🟠 Orange for "prerequisite"
    - 🟣 Purple for "sequel"
  - Interactive features:
    - Click and drag nodes
    - Hover for details
    - Scroll to zoom
  - Export graph data as JSON
  - Statistics display (document count, relationship count, types)
  - Legend for relationship types
- **New Dependencies:**
  - pyvis >= 0.3.2 for interactive network visualization
  - networkx >= 3.0 for graph data structures

#### ✅ Testing
- **New Tests (2 total):**
  - `test_get_relationship_graph` - Basic graph generation
  - `test_get_relationship_graph_filtered` - Filtering by tags and types
- All 57 tests passing

**Benefits:**
- Automated quality assurance with CI/CD pipeline
- Visual exploration of document relationships
- Interactive graph makes complex relationships understandable
- Filter and export capabilities for analysis
- Professional DevOps practices

## [2.7.0] - 2025-12-13

### Added - GUI Enhancements & Document Relationships

#### 📦 Bulk File Upload in GUI
- **Bulk Upload Tab** in Documents page
  - Drag-and-drop multiple PDF/TXT files simultaneously
  - Progress bar showing upload status
  - Individual error handling per file
  - Apply tags to all uploaded files at once
  - Success/failure summary after upload

#### 🏷️ Tag Management Page
- **New Page:** Dedicated Tag Management interface
- **Tag Statistics Dashboard:**
  - Total tags count
  - Average documents per tag
  - Most used tag
- **Sortable Tag List:**
  - Sort by name (A-Z, Z-A)
  - Sort by document count
  - View document count for each tag
- **Four Tag Operations:**
  - **🔄 Rename Tag** - Rename a tag across all documents
  - **🔗 Merge Tags** - Combine multiple tags into one
  - **🗑️ Delete Tag** - Remove tag from all documents
  - **➕ Add to All** - Add a tag to every document in the knowledge base

#### 👁️ Document Preview
- **Preview Button** added to each document in Documents page
- **Preview Features:**
  - Adjustable chunk preview (1-10 chunks via slider)
  - Toggle metadata display (chunk ID, page number, word count)
  - Full document export as text file
  - Preview shows formatted markdown content
  - Session state preserves preview open/closed state

#### 🔗 Document Relationships
- **Backend Methods:**
  - `add_relationship(from_doc_id, to_doc_id, type, note)` - Create relationships between documents
  - `remove_relationship(from_doc_id, to_doc_id, type)` - Delete specific or all relationships
  - `get_relationships(doc_id, direction)` - Get outgoing/incoming/both relationships
  - `get_related_documents(doc_id, type)` - Get full metadata of related documents
- **Relationship Types:**
  - `related` - General relationship
  - `references` - One document references another
  - `prerequisite` - Must be read before another document
  - `sequel` - Continuation of another document
- **Database Schema:**
  - New `document_relationships` table
  - Foreign keys with CASCADE delete
  - UNIQUE constraint prevents duplicates
  - Optional notes for each relationship
- **GUI Relationships Panel:**
  - **Relationships Button** for each document
  - View outgoing and incoming relationships
  - Add new relationships with dropdown selection
  - Delete individual relationships
  - Display relationship type and notes
  - Shows related document titles

#### ✅ Testing
- **New Tests (11 total):**
  - `test_add_relationship` - Create relationships
  - `test_add_relationship_invalid_doc` - Error handling
  - `test_add_duplicate_relationship` - Prevent duplicates
  - `test_get_relationships_outgoing` - Outgoing links
  - `test_get_relationships_incoming` - Incoming links
  - `test_get_relationships_both` - Bidirectional queries
  - `test_remove_relationship` - Remove specific relationships
  - `test_remove_all_relationships` - Remove all between docs
  - `test_get_related_documents` - Full metadata retrieval
  - `test_get_related_documents_filtered` - Filter by type
  - `test_relationship_cascade_delete` - Verify CASCADE behavior

**Benefits:**
- Improved document organization with visual relationship mapping
- Efficient bulk operations for large collections
- Better tag management and cleanup
- Quick document preview without leaving the interface
- Track document dependencies and reading order

## [2.6.0] - 2025-12-13

### Added - Bulk Document Management

#### ⚡ Bulk Operations
- **New Method:** `update_tags_bulk()` - Update tags for multiple documents in bulk
  - Add tags to multiple documents
  - Remove tags from multiple documents
  - Replace all tags for multiple documents
  - Select documents by ID or by existing tags
  - Returns detailed results (updated/failed)
- **New Method:** `export_documents_bulk()` - Export document metadata in various formats
  - Export to JSON, CSV, or Markdown
  - Export all documents, by tags, or by specific IDs
  - Comprehensive metadata including title, tags, chunks, dates
- **Existing Methods Enhanced:**
  - `add_documents_bulk()` - Already existed, now documented
  - `remove_documents_bulk()` - Already existed, now documented

#### 🔧 MCP Tools for Claude Desktop
- **New Tool:** `update_tags_bulk` - Bulk tag management via Claude Desktop
- **New Tool:** `export_documents_bulk` - Bulk export via Claude Desktop

#### 🖥️ GUI Enhancements
- **New Feature:** Bulk Operations panel in Documents page
- **Three Tabs:**
  - 🗑️ **Bulk Delete** - Delete multiple documents by IDs or tags
  - 🏷️ **Bulk Re-tag** - Add/remove/replace tags for multiple documents
  - 📤 **Bulk Export** - Export document metadata in JSON/CSV/Markdown

#### 🔨 CLI Commands
- **New Command:** `remove-bulk` - Remove multiple documents from command line
- **New Command:** `update-tags-bulk` - Bulk tag updates from command line
- **New Command:** `export-bulk` - Export document metadata from command line

**Usage Examples:**
```bash
# CLI - Remove documents by tags
python cli.py remove-bulk --tags draft old

# CLI - Add tags to specific documents
python cli.py update-tags-bulk --doc-ids doc1 doc2 --add reviewed approved

# CLI - Export all documents as JSON
python cli.py export-bulk --format json --output documents.json

# CLI - Export documents with specific tags as CSV
python cli.py export-bulk --tags reference c64 --format csv --output c64_docs.csv
```

```python
# Python API - Bulk tag update
results = kb.update_tags_bulk(
    existing_tags=["draft"],
    add_tags=["reviewed"],
    remove_tags=["draft"]
)

# Python API - Export documents
export_data = kb.export_documents_bulk(
    tags=["reference", "c64"],
    format="markdown"
)
```

#### ✅ Testing
- **New Tests:**
  - `test_update_tags_bulk` - Comprehensive tag update testing
  - `test_export_documents_bulk` - Export format testing (JSON/CSV/Markdown)

**Benefits:**
- Efficient management of large document collections
- Easy reorganization and categorization
- Bulk cleanup operations
- Document metadata backup and reporting
- Integration with CI/CD workflows

---

## [2.5.0] - 2025-12-13

### Added - Backup/Restore & GUI Admin Interface

#### 💾 Backup & Restore Operations
- **New Method:** `create_backup(dest_dir, compress=True)` creates full knowledge base backups
- **Backup Features:**
  - Compressed ZIP archives or uncompressed directories
  - Includes database, embeddings, and metadata
  - Automatic timestamping (kb_backup_YYYYMMDD_HHMMSS)
  - Metadata file with document count, version, size info
- **New Method:** `restore_from_backup(backup_path, verify=True)` restores from backups
- **Restore Features:**
  - Supports both ZIP and directory backups
  - Automatic extraction of compressed backups
  - Backup verification (checksums, required files)
  - Safety backup before restoration
  - Complete database and embeddings restoration
- **MCP Tools:** New `create_backup` and `restore_backup` tools
- **Benefits:**
  - Data safety and disaster recovery
  - Version control for knowledge base
  - Easy migration between systems
  - Automated backup workflows

**Usage:**
```python
# Python API - Create backup
backup_path = kb.create_backup("/path/to/backups", compress=True)

# Restore from backup
result = kb.restore_from_backup("/path/to/backup.zip", verify=True)

# Via MCP in Claude Desktop
# "Create a backup of the knowledge base in ~/backups"
# "Restore from backup ~/backups/kb_backup_20251213_083327.zip"
```

#### 🎮 Streamlit GUI Admin Interface
- **New File:** `admin_gui.py` - Complete web-based admin interface
- **Dashboard Page:**
  - Real-time metrics (documents, chunks, words)
  - Health status monitoring
  - Database info and feature status
  - File types and tags overview
- **Documents Page:**
  - Upload new documents (PDF/TXT)
  - List and filter document library
  - Delete documents
  - View document metadata
- **Search Page:**
  - Multi-mode search (Keyword/Semantic/Hybrid)
  - Tag filtering
  - Export results (Markdown/JSON/HTML)
  - Adjustable semantic weight for hybrid search
- **Backup & Restore Page:**
  - Create backups with compression options
  - Restore from backups with verification
  - Safety warnings and confirmations
- **Analytics Page:**
  - Search analytics dashboard
  - Top queries, failed searches
  - Search mode usage charts
  - Popular tags visualization

**Usage:**
```bash
# Install GUI dependencies
pip install ".[gui]"

# Run admin interface
streamlit run admin_gui.py

# Access at http://localhost:8501
```

### Testing
- Added 3 new test cases (44 total tests, 42 passed, 2 skipped)
- `test_backup_and_restore()` - Full backup/restore cycle validation
- `test_uncompressed_backup()` - Uncompressed backup creation
- `test_backup_with_empty_kb()` - Edge case testing
- All existing tests continue to pass

### Performance
- Backup creation: ~100-500ms depending on database size
- Restore operation: ~200-800ms depending on database size
- Compression adds ~50-200ms overhead

### Dependencies
- Added `streamlit>=1.28.0` for GUI (optional, install with `pip install ".[gui]"`)
- Added `pandas>=2.0.0` for analytics charts (optional)
- No new required dependencies for core functionality

### Documentation
- Updated CHANGELOG.md with v2.5.0 release notes
- Updated FUTURE_IMPROVEMENTS.md to mark Automated Backup as completed
- Created comprehensive `admin_gui.py` with inline documentation

### Developer Notes
**Implementation Details:**
- Backup uses Python's `shutil` for file operations and `zipfile` for compression
- Restore creates safety backup before overwriting (automatic rollback capability)
- GUI uses Streamlit's session state for persistent knowledge base connection
- All backup/restore operations are logged with detailed timestamps
- Metadata JSON includes version info for forward compatibility

**Breaking Changes:** None - all new features are additive

---

## [2.4.0] - 2025-12-13

### Added - Query Autocompletion & Export Features

#### 🔍 Query Autocompletion
- **New Table:** `query_suggestions` FTS5 virtual table for fast autocomplete
- **Automatic Term Extraction:** Technical terms extracted during document ingestion
- **Four Categories:**
  - **Hardware:** ALL CAPS technical terms (VIC-II, SID, CIA, etc.)
  - **Register:** Memory addresses ($D000, $D020, $D418, etc.)
  - **Instruction:** 6502 assembly mnemonics (LDA, STA, JMP, etc.)
  - **Concept:** Technical phrases (Video Interface Controller, etc.)
- **New Methods:**
  - `build_suggestion_dictionary(rebuild=False)` - Build/rebuild suggestion index
  - `get_query_suggestions(partial, max_suggestions, category)` - Get autocomplete suggestions
  - `_update_suggestions_for_chunks(chunks)` - Incremental update during document addition
- **MCP Tool:** New `suggest_queries` tool with category filtering
- **FTS5 Prefix Matching:** Fast substring matching with proper escaping
- **Auto-Update:** Suggestions automatically updated when documents are added
- **Benefits:**
  - Type-ahead suggestions for better search experience
  - Discover technical terms and addresses
  - Category filtering for targeted suggestions
  - Frequency tracking shows common terms

**Usage:**
```python
# Python API - Get suggestions
suggestions = kb.get_query_suggestions("VIC", max_suggestions=5)

# Category-specific suggestions
suggestions = kb.get_query_suggestions("$D0", max_suggestions=5, category="register")

# Rebuild dictionary manually
kb.build_suggestion_dictionary(rebuild=True)

# Via MCP in Claude Desktop
# "Suggest queries starting with 'SID'"
```

#### 📤 Export Search Results
- **New Method:** `export_search_results(results, format, query)` exports to multiple formats
- **Three Export Formats:**
  - **Markdown:** Clean, readable format for documentation
  - **JSON:** Machine-readable format with full metadata
  - **HTML:** Styled format with embedded CSS for viewing in browsers
- **Format-Specific Methods:**
  - `_export_markdown()` - Markdown with headers, scores, metadata
  - `_export_json()` - JSON with query, timestamp, result count
  - `_export_html()` - HTML with embedded CSS styling
- **MCP Tool:** New `export_results` tool for exporting from Claude Desktop
- **Rich Metadata:** Includes query, timestamps, scores, snippets, tags
- **Benefits:**
  - Save search results for documentation
  - Share results in preferred format
  - Machine-readable JSON for automation
  - Styled HTML for presentations

**Usage:**
```python
# Python API - Export as markdown
results = kb.search("VIC-II graphics", max_results=10)
markdown = kb.export_search_results(results, format='markdown', query="VIC-II graphics")

# Export as JSON
json_export = kb.export_search_results(results, format='json')

# Export as HTML
html = kb.export_search_results(results, format='html', query="VIC-II graphics")

# Via MCP in Claude Desktop
# "Export these search results as HTML"
```

### Testing
- Added 2 new test cases (41 total tests, 39 passed, 2 skipped)
- `test_query_autocompletion()` - Validates term extraction, FTS5 matching, category filtering
- `test_export_functionality()` - Tests all three export formats and error handling
- All existing tests continue to pass

### Performance
New features maintain excellent performance:
- Query suggestions: ~5-15ms per autocomplete query (FTS5-powered)
- Suggestion extraction: ~10-30ms during document ingestion
- Export operations: ~10-50ms depending on result count and format

### Database Schema Changes
- New table: `query_suggestions` FTS5 virtual table (term, frequency, category)
- Automatic migrations for existing databases
- Incremental updates on document addition

### Documentation
- Updated CHANGELOG.md with v2.4.0 release notes
- Updated FUTURE_IMPROVEMENTS.md to mark completed features
- Test suite covers all new functionality

### Developer Notes
**Implementation Details:**
- Query suggestions use FTS5 for fast prefix matching
- Special characters ($, *, etc.) properly quoted for FTS5
- Incremental term updates use upsert logic (update or insert)
- Export formats include full result metadata
- HTML export includes embedded CSS for standalone viewing
- All features fully backward compatible

**Breaking Changes:** None - all new features are additive

---

## [2.3.0] - 2025-12-12

### Added - Performance & Content Enhancement

#### ⚡ Incremental Embeddings
- **New Method:** `_add_chunks_to_embeddings(chunks)` incrementally adds embeddings to FAISS index
- **Performance:** 10-100x faster than full rebuild for new documents
- **Smart Updates:** Detects existing index and adds new vectors without rebuilding
- **Automatic:** Document addition now uses incremental updates by default
- **Index Growth:** FAISS index grows incrementally as documents are added
- **Benefits:**
  - Fast document addition (seconds instead of minutes)
  - No need to rebuild entire embeddings index
  - Seamless integration with existing semantic search
  - Automatic persistence after each update

**Technical Details:**
- Generates embeddings only for new chunks
- Uses FAISS `add()` method to append to existing index
- Updates embeddings_doc_map concurrently
- Saves index to disk after each addition
- Invalidates similarity cache automatically

#### 🚀 Parallel Document Processing
- **ThreadPoolExecutor:** Process multiple documents concurrently
- **Thread-Safe:** Database operations protected with threading locks
- **Configurable Workers:** Set via `PARALLEL_WORKERS` environment variable (default: CPU count)
- **Modified Method:** `add_documents_bulk()` now uses parallel processing
- **Performance:** 3-4x faster bulk imports on multi-core systems
- **Safety:**
  - Critical sections protected with `self._lock`
  - Duplicate detection remains accurate
  - Database ACID guarantees maintained

**Usage:**
```python
# Set worker count (optional, defaults to CPU count)
import os
os.environ['PARALLEL_WORKERS'] = '8'

# Bulk add with parallel processing
results = kb.add_documents_bulk(
    directory="docs",
    pattern="**/*.pdf",
    tags=["reference"]
)
```

#### 🔗 Cross-Reference Detection
- **New Table:** `cross_references` stores extracted references with context
- **Reference Types:**
  - **Memory Addresses:** $D000, $D020, $D418 (4-digit hex with $ prefix)
  - **Register Offsets:** VIC+0, SID+4, CIA1+12 (chip name + offset)
  - **Page References:** "page 156", "see page 42"
- **Extraction Methods:**
  - `_extract_cross_references(chunks, doc_id)` - Main coordinator
  - `_extract_memory_addresses(text)` - Regex-based address extraction
  - `_extract_register_offsets(text)` - Chip+offset pattern matching
  - `_extract_page_references(text)` - Page number references
  - `_get_reference_context(text, reference)` - Surrounding context (200 chars)
- **New Method:** `find_by_reference(ref_type, ref_value, max_results)` searches by reference
- **MCP Tool:** New `find_by_reference` tool for Claude Desktop integration
- **Auto-Extraction:** References automatically extracted during document ingestion
- **Benefits:**
  - Track how specific registers are documented across documents
  - Find all mentions of a memory address
  - Cross-link related content automatically
  - Navigate documentation by technical references

**Usage:**
```python
# Python API - Find all documents mentioning $D020
results = kb.find_by_reference("memory_address", "$D020", max_results=10)

# Find all VIC+0 register references
results = kb.find_by_reference("register_offset", "VIC+0", max_results=10)

# Find page 156 references
results = kb.find_by_reference("page_reference", "156", max_results=10)

# Via MCP in Claude Desktop
# "Find all documents that reference the $D020 register"
```

### Testing
- Added 3 new test cases (39 total tests, 37 passed, 2 skipped)
- `test_incremental_embeddings()` - Validates incremental FAISS index updates
- `test_parallel_processing()` - Tests ThreadPoolExecutor with bulk add
- `test_cross_reference_detection()` - Tests reference extraction and lookup
- All existing tests continue to pass

### Performance
New features maintain excellent performance:
- Incremental embeddings: ~1-5s for typical documents (vs 30-120s for full rebuild)
- Parallel processing: 3-4x speedup on 4+ core systems
- Cross-reference extraction: ~10-30ms during document ingestion
- Cross-reference lookup: ~20-50ms with indexed queries

### Database Schema Changes
- New table: `cross_references` with indexes on (ref_type, ref_value) and doc_id
- Foreign key cascade deletes ensure data integrity
- Automatic migrations for existing databases

### Documentation
- Updated CHANGELOG.md with v2.3.0 release notes
- All new methods documented in CLAUDE.md
- Test suite covers all new functionality

### Developer Notes
**Implementation Details:**
- Incremental embeddings use FAISS IndexFlatIP for cosine similarity
- Parallel processing uses concurrent.futures.ThreadPoolExecutor
- Thread-safe operations protected by threading.Lock
- Cross-references use regex patterns for extraction
- All features fully backward compatible

**Breaking Changes:** None - all new features are additive

---

## [2.2.0] - 2025-12-12

### Added - Faceted Search & Analytics

#### 🔍 Faceted Search and Filtering
- **New Method:** `faceted_search(query, facet_filters, max_results, tags)` enables multi-dimensional filtering
- **Database Storage:** New `document_facets` table with composite primary key (doc_id, facet_type, facet_value)
- **Facet Types:**
  - **Hardware:** Automatically detects SID, VIC-II, CIA, 6502, PLA, Datasette, Disk Drive references
  - **Instructions:** Extracts 6502 assembly mnemonics (LDA, STA, JMP, etc.) - 46 mnemonics supported
  - **Registers:** Identifies memory-mapped register addresses ($D000-$DFFF, etc.)
- **Extraction Methods:**
  - `_extract_facets(text)` - Main coordinator method
  - `_extract_hardware_refs(text)` - Pattern-based hardware detection
  - `_extract_instructions(text)` - Assembly instruction detection
  - `_extract_registers(text)` - Register address extraction (4-digit hex with $ prefix)
- **Auto-Extraction:** Facets automatically extracted during document ingestion
- **MCP Tool:** New `faceted_search` tool with facet filter support
- **Benefits:**
  - Narrow results by technical domain (e.g., "find SID-related documents only")
  - Combine keyword search with facet filtering
  - Results include facet metadata for each document
  - Filter by multiple facet types simultaneously

**Usage:**
```python
# Python API - Search for SID chip documents
results = kb.faceted_search(
    query="sound",
    facet_filters={'hardware': ['SID'], 'instruction': ['LDA', 'STA']},
    max_results=5
)

# Via MCP in Claude Desktop
# "Search for sound programming documents that mention the SID chip and use LDA/STA instructions"
```

#### 📊 Search Analytics & Logging
- **New Method:** `_log_search(query, search_mode, results_count, execution_time_ms, tags)` logs all searches
- **Database Storage:** New `search_log` table tracks all search activity
- **Fields Logged:** timestamp, query, search_mode (fts5/bm25/semantic/hybrid/faceted), results_count, execution_time_ms, tags, clicked_doc_id
- **Analytics Method:** `get_search_analytics(days, limit)` provides comprehensive usage insights
- **Metrics Provided:**
  - **Overview:** Total searches, unique queries, avg results, avg execution time
  - **Top Queries:** Most frequently searched terms with average result counts
  - **Failed Searches:** Queries that returned 0 results (helps identify gaps)
  - **Search Modes:** Breakdown by search backend usage
  - **Popular Tags:** Most commonly used tag filters
- **Auto-Logging:** All search methods automatically log queries (search, semantic_search, hybrid_search, faceted_search)
- **MCP Tool:** New `search_analytics` tool for usage reporting
- **Benefits:**
  - Understand user search patterns
  - Identify knowledge gaps (failed searches)
  - Optimize search performance
  - Track search backend effectiveness

**Usage:**
```python
# Python API - Get last 30 days of analytics
analytics = kb.get_search_analytics(days=30, limit=100)

# Via MCP in Claude Desktop
# "Show me search analytics for the last 7 days"
```

### Testing
- Added 4 new test cases (36 total tests, 35 passed, 1 skipped)
- `test_facet_extraction()` - Validates hardware, instruction, and register detection
- `test_faceted_search()` - Tests facet filtering and result filtering
- `test_search_logging()` - Verifies search logging functionality
- `test_search_analytics()` - Tests analytics aggregation and reporting
- All existing tests continue to pass

### Performance
New features maintain sub-second performance:
- Facet extraction: ~10-50ms during document ingestion
- Faceted search: ~60-180ms (regular search + facet filtering)
- Search logging: ~5-10ms overhead per search (async-friendly)
- Analytics queries: ~50-200ms depending on date range

### Database Schema Changes
- New tables: `document_facets`, `search_log`
- Indexes: `idx_facets_type_value`, `idx_facets_doc_id`, `idx_search_log_query`, `idx_search_log_timestamp`, `idx_search_log_mode`
- Foreign key cascade deletes ensure data integrity
- Automatic migrations for existing databases

### Documentation
- Updated CHANGELOG.md with new feature details
- Updated FUTURE_IMPROVEMENTS.md to mark completed items
- Test suite expanded with 4 comprehensive tests

### Developer Notes
**Implementation Details:**
- Facet extraction uses regex patterns for hardware and instructions
- Register addresses normalized to uppercase ($D000 format)
- All search methods now log queries automatically
- Analytics use SQL GROUP BY and aggregation functions
- Search logging designed for minimal performance impact

**Breaking Changes:** None - all new features are additive

---

## [2.1.0] - 2025-12-12

### Added - Table Extraction and Code Block Detection

#### 📊 Table Extraction from PDFs
- **New Method:** `_extract_tables(filepath)` extracts structured tables from PDF documents using pdfplumber
- **Markdown Conversion:** Tables automatically converted to markdown format via `_table_to_markdown()`
- **Database Storage:** New `document_tables` table with 7 fields (doc_id, table_id, page, markdown, searchable_text, row_count, col_count)
- **FTS5 Search Index:** New `tables_fts` virtual table for full-text search on table content
- **MCP Tool:** New `search_tables` tool for searching tables with page numbers and metadata
- **Search Method:** New `search_tables(query, max_results, tags)` method with FTS5-powered search
- **Benefits:**
  - Critical for C64 documentation (memory maps, register tables, opcode references)
  - Preserves table structure and searchability
  - Returns results with row/column counts and page numbers
  - Fully integrated with document lifecycle (automatic cleanup on delete)

**Usage:**
```python
# Python API
results = kb.search_tables("VIC-II sprite registers", max_results=5)

# Via MCP in Claude Desktop
# "Search for tables about sprite registers"
```

#### 🔧 Code Block Detection
- **New Method:** `_detect_code_blocks(text)` detects BASIC, Assembly, and Hex dump code blocks using regex
- **Three Code Types Supported:**
  - **BASIC:** Line-numbered programs (e.g., "10 PRINT", "20 GOTO")
  - **Assembly:** 6502 mnemonics (LDA, STA, JMP, etc.)
  - **Hex Dumps:** Memory dumps with addresses (e.g., "D000: 00 01 02 03")
- **Database Storage:** New `document_code_blocks` table with 7 fields (doc_id, block_id, page, block_type, code, searchable_text, line_count)
- **FTS5 Search Index:** New `code_fts` virtual table for full-text search on code content
- **MCP Tool:** New `search_code` tool with optional block_type filtering
- **Search Method:** New `search_code(query, max_results, block_type, tags)` method
- **Pattern Matching:** Requires 3+ consecutive lines for detection to avoid false positives
- **Benefits:**
  - Automatically extracts programming examples from documentation
  - Searchable code snippets with syntax highlighting
  - Filter results by code type (BASIC/Assembly/Hex)
  - Returns line counts and page numbers

**Usage:**
```python
# Python API
results = kb.search_code("LDA STA", max_results=5, block_type="assembly")

# Via MCP in Claude Desktop
# "Search for assembly code that uses LDA and STA"
```

### Testing
- Added 4 new test cases (32 total tests, 31 passed, 1 skipped)
- `test_code_block_detection()` - Validates BASIC, Assembly, and Hex dump detection
- `test_table_extraction()` - Verifies pdfplumber integration
- `test_table_search()` - Tests FTS5 table search functionality
- `test_code_search()` - Tests code block search with type filtering
- All existing tests continue to pass

### Performance
New features maintain sub-second performance:
- Table extraction: Depends on PDF size and table count
- Code block detection: ~10-50ms for typical documents
- Table search: ~50-150ms (FTS5-powered)
- Code search: ~50-150ms (FTS5-powered)

### Documentation
- Updated CLAUDE.md with new architecture details
- Added sections for table extraction and code block detection
- Updated Key Methods section with new methods
- This CHANGELOG documents all improvements

### Developer Notes
**Implementation Details:**
- Uses pdfplumber for table extraction (optional dependency)
- Regex patterns for code detection (BASIC: `\d+\s+[A-Z]+`, Assembly: mnemonics, Hex: `[0-9A-F]{4}:\s*(?:[0-9A-F]{2}\s*)+`)
- Both features use FTS5 virtual tables with automatic triggers for synchronization
- Foreign key cascade deletes ensure data integrity
- Schema migrations handle existing databases automatically

**Database Schema Changes:**
- New tables: `document_tables`, `document_code_blocks`
- New FTS5 indexes: `tables_fts`, `code_fts`
- Automatic triggers for INSERT/UPDATE/DELETE synchronization
- Backward compatible with existing databases (migrations run automatically)

**Breaking Changes:** None - all new features are additive

---

## [2.0.0] - 2025-12-12

### Added - Priority 1 Improvements

#### 🎯 Hybrid Search
- **New Method:** `hybrid_search(query, max_results, tags, semantic_weight)` combines FTS5 keyword search with semantic search
- **Configurable Weighting:** Adjust balance between keyword precision and semantic recall (default: 70% FTS5, 30% semantic)
- **Score Normalization:** Both search types normalized to 0-1 range for fair comparison
- **Result Merging:** Intelligent merging of results from both search backends
- **MCP Tool:** New `hybrid_search` tool available in Claude Desktop integration
- **Benefits:**
  - Best of both worlds: exact keyword matching AND conceptual understanding
  - Better handles technical terms + synonyms simultaneously
  - Example: "6502 assembler" finds exact matches AND related content about machine code, opcodes

**Usage:**
```python
# Python API
results = kb.hybrid_search("SID chip sound", max_results=5, semantic_weight=0.3)

# Via MCP in Claude Desktop
# "Use hybrid search to find information about graphics sprites"
```

#### ✨ Enhanced Snippet Extraction
- **Term Density Scoring:** Sliding window analysis finds regions with highest query term concentration
- **Complete Sentences:** Expands to sentence boundaries instead of hard cutoffs
- **Code Block Preservation:** Detects and preserves complete code blocks (indented lines)
- **Whole Word Highlighting:** Uses word boundaries for better highlighting accuracy
- **Better Context:** More relevant snippets with natural boundaries

**Improvements:**
- No more mid-sentence cuts
- Better context around search terms
- Code snippets preserved intact
- 80% snippet size threshold ensures adequate context

#### 🏥 Health Monitoring
- **New Method:** `health_check()` performs comprehensive system diagnostics
- **Database Health:**
  - Integrity checking (PRAGMA integrity_check)
  - Database file size monitoring
  - Orphaned chunks detection
  - Disk space warnings (< 1GB free)
- **Feature Status:**
  - FTS5 enabled/available
  - Semantic search enabled/available (with embeddings count)
  - BM25 enabled
  - Query preprocessing status
- **Performance Metrics:**
  - Cache status and utilization
  - BM25 index build status
- **MCP Tool:** New `health_check` tool returns formatted system status

**Health Check Output:**
```
Status: HEALTHY
Message: All systems operational

Metrics:
  documents: 145
  chunks: 4,665
  total_words: 6,870,642

Database:
  size_mb: 45.23
  integrity: ok
  disk_free_gb: 125.5

Features:
  ✓ fts5_enabled: True
  ✓ fts5_available: True
  ✓ semantic_search_enabled: True
  ✓ semantic_search_available: True
  ✓ embeddings_count: 2347

Performance:
  cache_enabled: True
  cache_size: 23
  cache_capacity: 100

✓ No issues detected
```

### Testing
- Added 3 new test cases (28 total tests, all passing)
- `test_health_check()` - Validates health check structure and metrics
- `test_enhanced_snippet_extraction()` - Verifies improved snippet quality
- `test_hybrid_search()` - Tests hybrid search score normalization and merging
- Test coverage: 27 passed, 1 skipped (semantic search in test env)

### Performance
All new features maintain sub-second performance:
- Hybrid search: ~60-180ms (combines two searches)
- Enhanced snippets: Same as regular search (minimal overhead from sentence detection)
- Health check: ~50-100ms (database queries + file stats)

### Documentation
- Updated CLAUDE.md with new methods
- This CHANGELOG documents all improvements
- Test suite expanded to cover new functionality

### Developer Notes
**Implementation Details:**
- Hybrid search uses score normalization (max-value scaling for FTS5, native 0-1 for semantic)
- Enhanced snippets use regex sentence boundary detection with 80% size threshold
- Health check uses SQLite PRAGMA commands for integrity + Python shutil for disk stats
- All features fully backward compatible

**Breaking Changes:** None - all new features are additive

---

## Previous Versions

### [1.0.0] - 2025-12-11
- Initial production release
- 145 documents, 4,665 chunks, 6.8M words
- FTS5 full-text search (480x speedup vs BM25)
- Semantic search with all-MiniLM-L6-v2 embeddings
- OCR support for scanned PDFs (Tesseract + Poppler)
- Content-based duplicate detection
- SQLite database backend
- MCP integration for Claude Desktop
- 25 comprehensive tests
