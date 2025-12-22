# TODO: v2.20.0 Completion Tasks

Tasks remaining for v2.20.0 final release.

## Status: Documentation Complete ✅

**Completed:**
- ✅ v2.20.0 release (datetime bug fix, enhanced URL checking, security fix)
- ✅ README.md updated with new features  
- ✅ WEB_MONITORING_GUIDE.md created (comprehensive guide)
- ✅ .gitignore cleanup and temp file exclusions
- ✅ Commits pushed to GitHub

**Current Version:** v2.20.0
**Last Commit:** ec9cd41 - Documentation updates
**Branch:** master (synced with origin)

---

## Completed in Current Session ✅

### B4: GUI URL Monitor Tab 🖥️ (~1 hour) - COMPLETED ✅
Added comprehensive monitoring dashboard to admin_gui.py with:
- ✅ Sites overview table with scrape session grouping
- ✅ Check buttons (quick/full mode)
- ✅ Results display with 5 tabs (Changed, New Pages, Missing, Unchanged, Sessions)
- ✅ Bulk actions (re-scrape all, remove all, export JSON)
- ✅ Individual actions (re-scrape, view stats, remove)
- ✅ Metrics dashboard (monitored sites, unique sources, last check time)

## Remaining Tasks for Next Session

### D: Performance & Polish 🔧 (~1.5 hours)
1. Benchmark URL discovery (20 min)
2. Optimize with async/await (45 min)
3. Add progress indicators (15 min)

### B2: Scheduled Monitoring 🕒 (~45 min)
Create automated monitoring scripts:
- monitor_daily.py (quick check)
- monitor_weekly.py (full check)
- monitor_config.json
- Setup scripts for cron/Task Scheduler

---

## Time Estimate: ~3.5-4 hours total

## Priority
1. B4 (GUI) - High user value
2. B2 (Scheduler) - Automation value
3. D (Performance) - Enhancement

See full details in this file for implementation plans.
