# TDZ C64 Knowledge - File Inventory & Cleanup Report

**Generated:** 2026-01-02
**Purpose:** Comprehensive inventory of all documentation and Python files with cleanup recommendations

---

## Executive Summary

- **42 Markdown files** in root directory
- **36 Python files** in root directory
- **Recommendations:**
  - Archive 15 obsolete/utility Python files → `archive/`
  - Consolidate 20+ documentation files → reduce to ~12 core docs
  - Remove 8 duplicate/outdated MD files

---

## 1. CORE PRODUCTION FILES (Keep As-Is)

### Core Python Files (6 files - 900KB total)

| File | Size | Purpose | Status |
|------|------|---------|--------|
| **server.py** | 594 KB | Main MCP server, KnowledgeBase class, 50+ tools | ✅ ACTIVE |
| **cli.py** | 40 KB | Command-line interface | ✅ ACTIVE |
| **admin_gui.py** | 172 KB | Streamlit web UI | ✅ ACTIVE |
| **rest_server.py** | 29 KB | FastAPI REST API (27 endpoints) | ✅ ACTIVE |
| **rest_models.py** | 12 KB | Pydantic models for REST API | ✅ ACTIVE |
| **version.py** | 27 KB | Centralized version management | ✅ ACTIVE |

### Supporting Python Files (3 files - 55KB)

| File | Size | Purpose | Status |
|------|------|---------|--------|
| **llm_integration.py** | 7 KB | LLM provider abstraction (Anthropic/OpenAI) | ✅ ACTIVE (imported by server.py) |
| **anomaly_detector.py** | 24 KB | ML-based anomaly detection for URL monitoring | ✅ ACTIVE |
| **migration_v2_21_0.py** | 11 KB | Database migration script for v2.21.0 | ⚠️ KEEP (may be needed for upgrades) |

### Core Documentation (12 files - KEEP)

| File | Purpose | Status |
|------|---------|--------|
| **README.md** | Main project documentation | ✅ PRIMARY |
| **ARCHITECTURE.md** | Technical architecture & developer guide | ✅ PRIMARY |
| **CONTEXT.md** | Project status, quick reference | ✅ PRIMARY |
| **CLAUDE.md** | Quick reference for Claude Code | ✅ PRIMARY |
| **CHANGELOG.md** | Version history | ✅ PRIMARY |
| **QUICKSTART.md** | Fast setup guide | ✅ KEEP |
| **README_REST_API.md** | REST API documentation | ✅ KEEP |
| **ANOMALY_DETECTION.md** | Anomaly detection guide | ✅ KEEP |
| **ENTITY_EXTRACTION.md** | Entity extraction guide | ✅ KEEP |
| **SUMMARIZATION.md** | AI summarization guide | ✅ KEEP |
| **WEB_SCRAPING_GUIDE.md** | Web scraping documentation | ✅ KEEP |
| **EXAMPLES.md** | Performance analysis examples | ✅ KEEP |

---

## 2. TEST FILES (9 files - Keep for CI/CD)

| File | Size | Purpose | Status |
|------|------|---------|--------|
| **test_server.py** | 76 KB | Main test suite (59 tests) | ✅ ACTIVE |
| **test_rest_api.py** | 19 KB | REST API tests (39 tests) | ✅ ACTIVE |
| **test_rest_smoke.py** | 4 KB | REST API smoke tests | ✅ ACTIVE |
| **test_security.py** | 6 KB | Security path validation tests | ✅ ACTIVE |
| **test_anomaly_detector.py** | 14 KB | Anomaly detection tests | ✅ ACTIVE |
| **test_e2e_integration.py** | 15 KB | End-to-end integration tests | ✅ ACTIVE |
| **test_performance_regression.py** | 9 KB | Performance regression tests | ✅ ACTIVE |
| **test_debug_endpoints.py** | 2 KB | Debug endpoint tests | ✅ ACTIVE |
| **validate_docs.py** | 8 KB | Documentation validation | ✅ KEEP |

**Recommendation:** ✅ KEEP ALL - Used by CI/CD pipeline

---

## 3. OBSOLETE/UTILITY FILES → ARCHIVE

### Python Utility Scripts (15 files - 128KB)

**Recommendation:** Move to `archive/utilities/` directory

| File | Last Modified | Size | Purpose | Action |
|------|--------------|------|---------|--------|
| debug_bm25.py | 2025-12-17 | 3 KB | BM25 debugging (obsolete - FTS5 is primary) | 📦 ARCHIVE |
| enable_fts5.py | 2025-12-17 | 5 KB | One-time FTS5 migration | 📦 ARCHIVE |
| enable_semantic_search.py | 2025-12-17 | 3 KB | One-time semantic setup | 📦 ARCHIVE |
| setup_claude_desktop.py | 2025-12-17 | 5 KB | Config generator (one-time use) | 📦 ARCHIVE |
| benchmark.py | 2025-12-21 | 15 KB | Old benchmark (superseded) | 📦 ARCHIVE |
| benchmark_final.py | 2025-12-22 | 7 KB | Duplicate benchmark | 📦 ARCHIVE |
| benchmark_comprehensive.py | 2025-12-23 | 16 KB | Comprehensive bench (keep if needed) | ⚠️ MAYBE KEEP |
| benchmark_health_stats.py | 2025-12-23 | 6 KB | Health stats benchmark | 📦 ARCHIVE |
| benchmark_url_monitoring.py | 2025-12-22 | 11 KB | URL monitoring bench | 📦 ARCHIVE |
| load_test.py | 2025-12-22 | 13 KB | Load testing utility | ⚠️ MAYBE KEEP |
| load_test_500.py | 2025-12-23 | 20 KB | 500-doc load test | ⚠️ MAYBE KEEP |
| monitor_config_validator.py | 2025-12-22 | 9 KB | Config validation utility | 📦 ARCHIVE |
| monitor_daily.py | 2025-12-22 | 7 KB | Daily monitoring script | ⚠️ MAYBE KEEP (if used in cron) |
| monitor_weekly.py | 2025-12-22 | 10 KB | Weekly monitoring script | ⚠️ MAYBE KEEP (if used in cron) |
| monitor_fast.py | 2025-12-22 | 24 KB | Fast monitoring implementation | ⚠️ MAYBE KEEP |
| profile_anomaly.py | 2025-12-22 | 4 KB | Anomaly profiling | 📦 ARCHIVE |
| run_full_url_check.py | 2025-12-22 | 5 KB | URL checking utility | 📦 ARCHIVE |
| run_quick_url_check.py | 2025-12-22 | 3 KB | Quick URL check | 📦 ARCHIVE |

**Note:** Check if monitoring scripts are used in scheduled tasks before archiving.

---

## 4. DUPLICATE/OUTDATED DOCUMENTATION → REMOVE/CONSOLIDATE

### Duplicate Documentation (8 files - CONSOLIDATE)

| File | Issue | Recommendation |
|------|-------|----------------|
| **README_UPDATED.md** | Duplicate of README.md | 🗑️ DELETE (outdated version) |
| **QUICKSTART_UPDATED.md** | Duplicate of QUICKSTART.md | 🗑️ DELETE (outdated version) |
| **FUTURE_IMPROVEMENTS.md** | Status: 100% complete, obsolete | 🗑️ DELETE (all done) |
| **FUTURE_IMPROVEMENTS_2025.md** | Next-gen roadmap | ⚠️ MERGE into ROADMAP.md? |
| **PERFORMANCE.md** | Duplicate/overlap with below | 🔄 MERGE with others |
| **PERFORMANCE_ANALYSIS.md** | Performance docs | 🔄 MERGE into EXAMPLES.md |
| **PERFORMANCE_IMPROVEMENTS.md** | Phase 1 improvements | 🔄 MERGE into CHANGELOG.md |
| **PERFORMANCE_OPTIMIZATIONS_PHASE2.md** | Phase 2 improvements | 🔄 MERGE into CHANGELOG.md |

### Obsolete/Specific Docs (12 files - ARCHIVE or CONSOLIDATE)

| File | Issue | Recommendation |
|------|-------|----------------|
| **DEPLOYMENT_GUIDE.md** | Deployment instructions | ⚠️ MERGE into README.md deployment section? |
| **DOCKER.md** | Docker setup | ⚠️ MERGE into README.md or standalone? |
| **ENVIRONMENT_SETUP.md** | Environment config | 🔄 MERGE into QUICKSTART.md |
| **FEATURES.md** | Feature list | 🔄 MERGE into README.md |
| **FILE_VIEWER_IMPROVEMENTS.md** | v2.11.0 feature notes | 📦 ARCHIVE (historical) |
| **GUI_IMPROVEMENTS_SUMMARY.md** | v2.11.0 GUI changes | 📦 ARCHIVE (historical) |
| **GUI_README.md** | GUI documentation | 🔄 MERGE into README.md |
| **IMPROVEMENTS.md** | General improvements | 🔄 MERGE into CHANGELOG.md |
| **INNOVATION_ROADMAP.md** | Future innovation ideas | ⚠️ KEEP or MERGE with ROADMAP? |
| **MONITORING_SETUP.md** | Monitoring configuration | ⚠️ KEEP (if monitoring is used) |
| **WEB_MONITORING_GUIDE.md** | URL monitoring docs | ⚠️ KEEP (active feature) |
| **POPPLER_SETUP.md** | Poppler installation | 🔄 MERGE into QUICKSTART.md |

### Release Notes (3 files - ARCHIVE)

| File | Recommendation |
|------|----------------|
| **RELEASE_NOTES_v2.20.0.md** | 📦 ARCHIVE (info in CHANGELOG.md) |
| **RELEASE_NOTES_v2.21.0.md** | 📦 ARCHIVE (info in CHANGELOG.md) |
| **RELEASE_NOTES_v2.22.0.md** | 📦 ARCHIVE (info in CHANGELOG.md) |

### Status/Summary Docs (4 files - REMOVE/MERGE)

| File | Recommendation |
|------|----------------|
| **PROJECT_STATUS.md** | 🔄 MERGE into CONTEXT.md |
| **QUICK_WINS_SUMMARY.md** | 📦 ARCHIVE (completed phase) |
| **ROADMAP_v2.21.0.md** | 📦 ARCHIVE (old roadmap) |
| **TODO.md** | ⚠️ CHECK if still used, else DELETE |

### Test/Report Docs (2 files - KEEP)

| File | Recommendation |
|------|----------------|
| **TEST_REPORT.md** | ✅ KEEP (current test status) |
| **TESTING.md** | ✅ KEEP (testing guide) |

### User Documentation (2 files - CONSOLIDATE)

| File | Recommendation |
|------|----------------|
| **USER_GUIDE.md** | 🔄 MERGE into README.md |
| **DEPLOYMENT_GUIDE.md** | 🔄 MERGE into README.md |

---

## 5. PROPOSED DIRECTORY STRUCTURE

```
tdz-c64-knowledge/
│
├── README.md                    # Main documentation
├── QUICKSTART.md                # Fast setup
├── ARCHITECTURE.md              # Technical details
├── CONTEXT.md                   # Project status
├── CLAUDE.md                    # Claude Code reference
├── CHANGELOG.md                 # Version history
│
├── docs/                        # Feature-specific documentation
│   ├── REST_API.md             # REST API guide
│   ├── ANOMALY_DETECTION.md    # Anomaly detection
│   ├── ENTITY_EXTRACTION.md    # Entity features
│   ├── SUMMARIZATION.md        # AI summarization
│   ├── WEB_SCRAPING.md         # Web scraping
│   ├── EXAMPLES.md             # Performance examples
│   ├── TESTING.md              # Testing guide
│   ├── DEPLOYMENT.md           # Deployment (merged)
│   └── MONITORING.md           # Monitoring (if active)
│
├── archive/                     # Archived obsolete files
│   ├── utilities/              # Old utility scripts
│   │   ├── debug_bm25.py
│   │   ├── enable_fts5.py
│   │   ├── setup_claude_desktop.py
│   │   ├── benchmark_*.py
│   │   ├── run_*_url_check.py
│   │   └── profile_anomaly.py
│   │
│   ├── release-notes/          # Old release notes
│   │   ├── v2.20.0.md
│   │   ├── v2.21.0.md
│   │   └── v2.22.0.md
│   │
│   └── historical-docs/        # Historical documentation
│       ├── FILE_VIEWER_IMPROVEMENTS.md
│       ├── GUI_IMPROVEMENTS_SUMMARY.md
│       ├── FUTURE_IMPROVEMENTS.md (completed)
│       ├── QUICK_WINS_SUMMARY.md
│       └── ROADMAP_v2.21.0.md
│
├── server.py                    # Core MCP server
├── cli.py                       # CLI interface
├── admin_gui.py                 # Web UI
├── rest_server.py               # REST API server
├── rest_models.py               # API models
├── version.py                   # Version management
├── llm_integration.py           # LLM abstraction
├── anomaly_detector.py          # Anomaly detection
│
├── test_*.py                    # Test files (9 files)
│
└── utilities/                   # Active utility scripts (if any)
    ├── benchmark_comprehensive.py  # If still needed
    ├── load_test_500.py            # If still needed
    └── monitor_*.py                # If actively used
```

---

## 6. CLEANUP ACTION PLAN

### Phase 1: Archive Utility Scripts (Low Risk)

```bash
mkdir -p archive/utilities
mv debug_bm25.py archive/utilities/
mv enable_fts5.py archive/utilities/
mv enable_semantic_search.py archive/utilities/
mv setup_claude_desktop.py archive/utilities/
mv benchmark.py archive/utilities/
mv benchmark_final.py archive/utilities/
mv benchmark_health_stats.py archive/utilities/
mv benchmark_url_monitoring.py archive/utilities/
mv profile_anomaly.py archive/utilities/
mv run_full_url_check.py archive/utilities/
mv run_quick_url_check.py archive/utilities/
mv monitor_config_validator.py archive/utilities/
```

### Phase 2: Archive Historical Documentation (Low Risk)

```bash
mkdir -p archive/release-notes
mkdir -p archive/historical-docs

# Release notes
mv RELEASE_NOTES_v2.20.0.md archive/release-notes/
mv RELEASE_NOTES_v2.21.0.md archive/release-notes/
mv RELEASE_NOTES_v2.22.0.md archive/release-notes/

# Historical docs
mv FILE_VIEWER_IMPROVEMENTS.md archive/historical-docs/
mv GUI_IMPROVEMENTS_SUMMARY.md archive/historical-docs/
mv FUTURE_IMPROVEMENTS.md archive/historical-docs/
mv QUICK_WINS_SUMMARY.md archive/historical-docs/
mv ROADMAP_v2.21.0.md archive/historical-docs/
```

### Phase 3: Remove Duplicates (Medium Risk - Review First)

```bash
# Remove outdated duplicates
rm README_UPDATED.md
rm QUICKSTART_UPDATED.md

# Check TODO.md first
# If empty or unused: rm TODO.md
```

### Phase 4: Consolidate Documentation (High Risk - Manual Merge)

Create `docs/` directory and consolidate:

1. **MERGE PERFORMANCE DOCS:**
   - Combine PERFORMANCE.md + PERFORMANCE_ANALYSIS.md → docs/EXAMPLES.md
   - Add PERFORMANCE_IMPROVEMENTS.md + PERFORMANCE_OPTIMIZATIONS_PHASE2.md to CHANGELOG.md
   - Delete originals after verification

2. **MERGE USER DOCS:**
   - Merge USER_GUIDE.md content into README.md
   - Merge DEPLOYMENT_GUIDE.md into README.md or separate docs/DEPLOYMENT.md
   - Merge ENVIRONMENT_SETUP.md into QUICKSTART.md
   - Merge POPPLER_SETUP.md into QUICKSTART.md

3. **CONSOLIDATE FEATURE DOCS:**
   - Move README_REST_API.md → docs/REST_API.md
   - Move ANOMALY_DETECTION.md → docs/ANOMALY_DETECTION.md
   - Move ENTITY_EXTRACTION.md → docs/ENTITY_EXTRACTION.md
   - Move SUMMARIZATION.md → docs/SUMMARIZATION.md
   - Move WEB_SCRAPING_GUIDE.md → docs/WEB_SCRAPING.md
   - Move EXAMPLES.md → docs/EXAMPLES.md

4. **PROJECT STATUS:**
   - Merge PROJECT_STATUS.md into CONTEXT.md
   - Merge FEATURES.md into README.md

---

## 7. FINAL RECOMMENDED FILE COUNT

### After Cleanup:

**Root Directory:**
- 6 core documentation files (README, QUICKSTART, ARCHITECTURE, CONTEXT, CLAUDE, CHANGELOG)
- 9 core Python files (server, cli, admin_gui, rest_server, rest_models, version, llm_integration, anomaly_detector, migration)
- 9 test files (test_*.py, validate_docs.py)
- 1 inventory file (FILE_INVENTORY.md - this document)

**docs/ Directory:**
- 8-10 feature-specific guides

**archive/ Directory:**
- 15+ utility scripts
- 5+ release notes
- 10+ historical docs

**Total Reduction:**
- From **42 MD files** → **15 MD files** (64% reduction)
- From **36 Python files** → **18-21 active files** (40% reduction)

---

## 8. MONITORING SCRIPTS - DECISION NEEDED

**Question for User:** Are these monitoring scripts actively used in cron jobs or scheduled tasks?

| File | Action if YES | Action if NO |
|------|---------------|--------------|
| monitor_daily.py | KEEP in utilities/ | ARCHIVE |
| monitor_weekly.py | KEEP in utilities/ | ARCHIVE |
| monitor_fast.py | KEEP in utilities/ | ARCHIVE |

If YES: Create `utilities/` directory for active scripts
If NO: Archive all

---

## 9. LOAD TESTING - DECISION NEEDED

**Question for User:** Are these load testing scripts needed for future benchmarking?

| File | Action if YES | Action if NO |
|------|---------------|--------------|
| load_test.py | KEEP in utilities/ | ARCHIVE |
| load_test_500.py | KEEP in utilities/ | ARCHIVE |
| benchmark_comprehensive.py | KEEP in utilities/ | ARCHIVE |

---

## 10. NEXT STEPS

1. **Review this inventory report**
2. **Answer monitoring/load testing questions** (section 8 & 9)
3. **Execute Phase 1** (archive utility scripts - safest)
4. **Execute Phase 2** (archive historical docs - safe)
5. **Execute Phase 3** (remove duplicates - needs review)
6. **Execute Phase 4** (consolidate docs - manual merge work)
7. **Create docs/ directory** and organize feature documentation
8. **Update README.md** with new documentation structure
9. **Commit changes** with detailed commit message

---

**End of Inventory Report**
