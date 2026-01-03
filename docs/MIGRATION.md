# Migration Guide

**Version:** 2.23.1
**Last Updated:** 2026-01-03

Guide for migrating between versions of the TDZ C64 Knowledge Base.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Version Compatibility](#version-compatibility)
- [Upgrading from v2.22.x to v2.23.x](#upgrading-from-v222x-to-v223x)
- [Upgrading from v2.21.x to v2.22.x](#upgrading-from-v221x-to-v222x)
- [Upgrading from v2.20.x to v2.21.x](#upgrading-from-v220x-to-v221x)
- [Major Version Migrations](#major-version-migrations)
- [Breaking Changes](#breaking-changes)
- [Rollback Procedures](#rollback-procedures)

---

## 🎯 Overview

### Semantic Versioning

The project follows semantic versioning: `MAJOR.MINOR.PATCH`

- **MAJOR**: Breaking changes (rare)
- **MINOR**: New features, backwards compatible
- **PATCH**: Bug fixes, fully compatible

**Example:** v2.23.1
- Major: 2 (second major version)
- Minor: 23 (23rd minor release)
- Patch: 1 (1st patch)

---

### When to Migrate

**✅ Safe to upgrade:**
- Same MAJOR version (e.g., v2.22.0 → v2.23.1)
- PATCH updates (e.g., v2.23.0 → v2.23.1)

**⚠️ Review carefully:**
- MAJOR version changes (e.g., v1.x → v2.x)
- Large MINOR jumps (e.g., v2.10.0 → v2.23.0)

**💡 Best practice:**
- Test in development first
- Backup database before upgrading
- Read CHANGELOG.md for changes

---

## 📊 Version Compatibility

### Database Compatibility

| From Version | To Version | Database Migration | Notes |
|--------------|------------|-------------------|-------|
| v2.22.x | v2.23.x | ✅ Automatic | No action needed |
| v2.21.x | v2.22.x | ✅ Automatic | No action needed |
| v2.20.x | v2.21.x | ⚠️ Migration script | Run migration_v2_21_0.py |
| v2.15.x | v2.20.x | ✅ Automatic | Entity tables added |
| v2.13.x | v2.15.x | ✅ Automatic | Summary tables added |
| v2.0.x | v2.13.x | ✅ Automatic | No breaking changes |
| v1.x | v2.x | ❌ Manual | See Major Version Migration |

### Configuration Compatibility

| Feature | Introduced | Config Changes |
|---------|-----------|----------------|
| RAG Q&A | v2.23.0 | Add LLM_PROVIDER, API keys |
| Fuzzy Search | v2.23.0 | Auto-enabled |
| Smart Tagging | v2.23.0 | Add LLM_PROVIDER, API keys |
| Anomaly Detection | v2.21.0 | Auto-enabled |
| Entity Extraction | v2.15.0 | Add LLM_PROVIDER, API keys |
| REST API | v2.18.0 | Optional TDZ_API_KEYS |
| Semantic Search | v2.0.0 | Add USE_SEMANTIC_SEARCH=1 |
| FTS5 Search | v2.0.0 | Add USE_FTS5=1 |

---

## 🚀 Upgrading from v2.22.x to v2.23.x

**Changes in v2.23.x:**
- ✨ RAG question answering
- ✨ Fuzzy search
- ✨ Progressive search refinement
- ✨ Smart document tagging
- 🔧 Performance improvements
- 📚 Documentation updates

### Upgrade Steps

**1. Backup your database:**
```cmd
copy %TDZ_DATA_DIR%\knowledge_base.db %TDZ_DATA_DIR%\knowledge_base.db.v2.22.bak
```

**2. Update code:**
```cmd
cd C:\path\to\tdz-c64-knowledge
git pull
# Or download latest release
```

**3. Update dependencies:**
```cmd
.venv\Scripts\activate
pip install --upgrade -r requirements.txt
```

**4. Update configuration (optional):**
```json
{
  "env": {
    "USE_FTS5": "1",
    "USE_FUZZY_SEARCH": "1",
    "LLM_PROVIDER": "anthropic",
    "ANTHROPIC_API_KEY": "sk-ant-xxxxx"
  }
}
```

**5. Test the upgrade:**
```cmd
.venv\Scripts\python.exe cli.py stats
.venv\Scripts\python.exe cli.py search "test" --max 5
```

**6. Restart Claude Code/Desktop**

### New Features to Try

**RAG Question Answering:**
```
Ask Claude: "How do I program sprites on the VIC-II?"
# Uses answer_question tool
```

**Fuzzy Search:**
```
Ask Claude: "Search for VIC2 asembly programming"
# Autocorrects to "VIC-II assembly"
```

**Smart Tagging:**
```cmd
.venv\Scripts\python.exe cli.py suggest-tags <doc_id>
```

---

## 🔧 Upgrading from v2.21.x to v2.22.x

**Changes in v2.22.x:**
- 🚀 5000x faster entity extraction (regex patterns)
- 📊 Enhanced entity analytics
- 🔗 Distance-based relationship scoring
- ⚡ Health check optimization (93% faster)
- 🐛 Bug fixes

### Upgrade Steps

**1. Backup:**
```cmd
copy %TDZ_DATA_DIR%\knowledge_base.db %TDZ_DATA_DIR%\knowledge_base.db.v2.21.bak
```

**2. Update code and dependencies:**
```cmd
git pull
.venv\Scripts\activate
pip install --upgrade -r requirements.txt
```

**3. No database migration needed** - Fully backwards compatible

**4. Test:**
```cmd
.venv\Scripts\python.exe cli.py stats
```

**5. Restart services**

### Performance Improvements

**Re-extract entities** to benefit from regex speedup:
```cmd
# Optional - only if you want faster extraction
.venv\Scripts\python.exe cli.py extract-all-entities --force --confidence 0.6
```

---

## 🗄️ Upgrading from v2.20.x to v2.21.x

**Changes in v2.21.x:**
- 🤖 Anomaly detection system
- 📈 1500x performance improvement
- 🏥 Health check enhancements
- 🐛 Bug fixes

### Upgrade Steps

**1. Backup:**
```cmd
copy %TDZ_DATA_DIR%\knowledge_base.db %TDZ_DATA_DIR%\knowledge_base.db.v2.20.bak
```

**2. Update code:**
```cmd
git pull
.venv\Scripts\activate
pip install --upgrade -r requirements.txt
```

**3. Run migration script:**
```cmd
# Dry run first
.venv\Scripts\python.exe migration_v2_21_0.py --dry-run

# Apply migration
.venv\Scripts\python.exe migration_v2_21_0.py

# Verify
.venv\Scripts\python.exe migration_v2_21_0.py --verify
```

**4. Test:**
```cmd
.venv\Scripts\python.exe cli.py stats
```

**5. Restart services**

### What the Migration Adds

- `monitoring_history` table
- `monitoring_config` table
- Indexes for anomaly detection
- Baseline data structures

---

## 📦 Major Version Migrations

### v1.x to v2.x (Historical)

**Breaking changes:**
- JSON to SQLite database
- New chunk storage format
- FTS5 full-text search
- MCP protocol update

**Migration (manual):**
1. Export v1.x documents
2. Fresh install of v2.x
3. Re-import documents

**Not recommended** - v1.x is deprecated

---

## ⚠️ Breaking Changes

### Version History

**v2.23.0:**
- None - Fully backwards compatible

**v2.22.0:**
- None - Fully backwards compatible

**v2.21.0:**
- Database schema changes (automatic migration)
- New required tables for monitoring

**v2.20.0:**
- None - Fully backwards compatible

**v2.18.0:**
- REST API introduced (optional, no breaking changes)

**v2.15.0:**
- Entity extraction tables added (automatic migration)
- LLM provider configuration required for entity features

**v2.13.0:**
- Summary tables added (automatic migration)

**v2.0.0:**
- Major rewrite
- JSON → SQLite migration
- MCP protocol update

---

## 🔙 Rollback Procedures

### Quick Rollback

**If upgrade fails:**

1. **Stop the server:**
   - Restart Claude Code/Desktop to kill MCP connection
   - Stop REST API/GUI if running

2. **Restore backup:**
   ```cmd
   copy %TDZ_DATA_DIR%\knowledge_base.db.v2.22.bak %TDZ_DATA_DIR%\knowledge_base.db
   ```

3. **Revert code:**
   ```cmd
   git checkout v2.22.0
   # Or restore from backup
   ```

4. **Verify:**
   ```cmd
   .venv\Scripts\python.exe cli.py stats
   ```

5. **Restart services**

---

### Complete Rollback

**If database corrupted:**

1. **Stop all services**

2. **Restore from backup:**
   ```cmd
   copy C:\backups\knowledge_base.db %TDZ_DATA_DIR%\knowledge_base.db
   ```

3. **Restore code:**
   ```cmd
   cd C:\path\to\tdz-c64-knowledge
   git checkout <previous-version-tag>
   ```

4. **Reinstall dependencies:**
   ```cmd
   .venv\Scripts\activate
   pip install --force-reinstall -r requirements.txt
   ```

5. **Test:**
   ```cmd
   .venv\Scripts\python.exe cli.py stats
   ```

---

## 📝 Pre-Migration Checklist

Before upgrading:

- [ ] Read CHANGELOG.md for version changes
- [ ] Backup database file
- [ ] Backup source code (if modified)
- [ ] Test upgrade in development environment
- [ ] Document current configuration
- [ ] Note current version: `cli.py --version`
- [ ] Check disk space (database may grow)
- [ ] Schedule maintenance window
- [ ] Notify users (if shared system)

---

## ✅ Post-Migration Checklist

After upgrading:

- [ ] Verify version: `cli.py --version`
- [ ] Run health check: `cli.py health-check`
- [ ] Test search: `cli.py search "test" --max 5`
- [ ] Test MCP integration in Claude
- [ ] Check logs for errors
- [ ] Verify all documents present
- [ ] Test new features
- [ ] Update documentation (if custom setup)
- [ ] Create new backup
- [ ] Monitor performance

---

## 🛠️ Troubleshooting Migrations

### Migration Script Fails

**Error:** "Migration already applied"

**Solution:** Already migrated, safe to proceed

---

**Error:** "Database locked"

**Solution:**
1. Stop all connections
2. Wait 30 seconds
3. Retry migration

---

**Error:** "Disk full"

**Solution:**
1. Free up space
2. Database may grow during migration
3. Need ~2x current size free

---

### After Migration Issues

**Search returns no results:**
```cmd
# Rebuild FTS5 index
.venv\Scripts\python.exe -c "from server import KnowledgeBase; kb = KnowledgeBase(); kb._rebuild_fts5_indexes()"
```

**Embeddings missing:**
```cmd
# Rebuild embeddings
.venv\Scripts\python.exe -c "from server import KnowledgeBase; kb = KnowledgeBase(); kb._build_embeddings()"
```

**Entities gone:**
```cmd
# Re-extract entities
.venv\Scripts\python.exe cli.py extract-all-entities
```

---

## 📚 Additional Resources

- [CHANGELOG.md](../CHANGELOG.md) - Detailed version history
- [README.md](../README.md) - Current features
- [Troubleshooting Guide](TROUBLESHOOTING.md) - Common issues
- [GitHub Releases](https://github.com/MichaelTroelsen/tdz-c64-knowledge/releases) - Release notes

---

## 💡 Best Practices

**Always:**
- Backup before upgrading
- Test in development first
- Read the changelog
- Check compatibility table

**Consider:**
- Upgrading during low-usage times
- Testing rollback procedure
- Documenting your specific setup
- Keeping backups for 30+ days

**Never:**
- Skip backups
- Upgrade in production without testing
- Ignore migration errors
- Delete old backups immediately

---

**Version:** 2.23.1
**Last Updated:** 2026-01-03

**Need help?** See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) or open a GitHub issue.
