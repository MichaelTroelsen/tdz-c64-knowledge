# TDZ C64 Knowledge Base - GUI Admin Interface

Web-based admin interface for managing the C64 Knowledge Base.

## Installation

### 1. Install GUI Dependencies

```bash
# Install streamlit and pandas
pip install ".[gui]"

# Or install individually
pip install streamlit>=1.28.0 pandas>=2.0.0
```

### 2. Run the Admin Interface

```bash
streamlit run admin_gui.py
```

The interface will open automatically in your default web browser at `http://localhost:8501`.

## Features

### 📊 Dashboard
- **Real-time Metrics:** View document count, chunks, total words, and system status
- **Health Monitoring:** Database integrity, disk space, feature status
- **Overview:** File types, tags, and system configuration

### 📚 Documents
- **Upload Documents:** Drag-and-drop PDF or TXT files
- **Document Library:** Browse all indexed documents with filtering
- **Manage Documents:** View metadata, delete documents
- **Tags:** Assign tags during upload for organization

### 🔍 Search
- **Three Search Modes:**
  - **Keyword (FTS5):** Fast exact matching
  - **Semantic:** Conceptual/meaning-based search
  - **Hybrid:** Combines keyword + semantic (adjustable weighting)
- **Tag Filtering:** Filter results by document tags
- **Export Results:** Download search results as Markdown, JSON, or HTML
- **Result Preview:** View snippets, scores, and metadata

### 💾 Backup & Restore
- **Create Backups:**
  - Choose destination directory
  - Option to compress to ZIP
  - Automatic timestamping
  - Includes database and embeddings
- **Restore Backups:**
  - Restore from ZIP or directory
  - Verification before restoration
  - Automatic safety backup
  - Shows restore statistics

### 📈 Analytics
- **Search Analytics Dashboard:**
  - Total searches and unique queries
  - Average results and execution time
  - Top queries with frequency
  - Failed searches (0 results)
  - Search mode usage charts
  - Popular tags

## Usage Tips

### Uploading Documents
1. Navigate to **📚 Documents** page
2. Click "➕ Add New Document" expander
3. Upload PDF or TXT file
4. (Optional) Enter title and tags
5. Click "Add Document"

### Searching
1. Navigate to **🔍 Search** page
2. Select search mode (Keyword/Semantic/Hybrid)
3. Enter query
4. Adjust settings (max results, semantic weight, tags)
5. Click "Search"
6. Export results if needed

### Creating Backups
1. Navigate to **💾 Backup & Restore** page
2. Enter backup directory path
3. Choose compression option
4. Click "Create Backup"
5. Note the backup path for restoration

### Viewing Analytics
1. Navigate to **📈 Analytics** page
2. Select time range (7/14/30/60/90 days)
3. Click "Generate Report"
4. View metrics, charts, and insights

## Configuration

The GUI uses the same data directory as the MCP server:

- **Default:** `~/.tdz-c64-knowledge`
- **Custom:** Set `TDZ_DATA_DIR` environment variable

```bash
# Use custom data directory
export TDZ_DATA_DIR="/path/to/data"
streamlit run admin_gui.py
```

## Search Modes

### Keyword (FTS5)
- **Best for:** Exact terms, register addresses, specific commands
- **Speed:** Very fast (~50-140ms)
- **Accuracy:** High precision for exact matches

### Semantic
- **Best for:** Conceptual queries, "how to" questions
- **Speed:** Fast (~12-25ms)
- **Accuracy:** Good for understanding meaning

### Hybrid
- **Best for:** General search, balanced results
- **Speed:** Medium (~60-180ms)
- **Accuracy:** Best of both worlds
- **Tip:** Adjust semantic weight (0-1) to tune balance

## Troubleshooting

### GUI Won't Start
```bash
# Check if streamlit is installed
streamlit --version

# Reinstall if needed
pip install --force-reinstall streamlit
```

### Can't Upload Documents
- Check file format (only PDF and TXT supported)
- Verify data directory has write permissions
- Check disk space

### Search Returns No Results
- Try different search modes
- Remove tag filters
- Check if documents are indexed (Dashboard page)

### Backup Fails
- Verify destination directory exists and is writable
- Check available disk space
- Ensure knowledge base is not being modified during backup

## Performance

- **Dashboard:** Updates in real-time
- **Search:** Results typically within 100ms
- **Backup:** ~100-500ms depending on database size
- **Upload:** Depends on file size and content

## Security

- **Local Only:** GUI runs on localhost by default
- **No Authentication:** Suitable for single-user/trusted environments
- **Data Safety:** All operations are logged
- **Backup Safety:** Automatic safety backup before restore

## Advanced Features

### Batch Operations
- Use Python API or CLI for bulk operations
- GUI is optimized for interactive use

### Scheduled Backups
- Use OS task scheduler (cron/Task Scheduler)
- Call `kb.create_backup()` from Python script

### Remote Access
If you need to access the GUI remotely:

```bash
streamlit run admin_gui.py --server.address 0.0.0.0
```

**Warning:** Only do this on trusted networks!

## Support

For issues or questions:
1. Check the logs in `server.log`
2. Review error messages in the GUI
3. Verify data directory permissions
4. Ensure all dependencies are installed

## Version

**Current Version:** 2.5.0
**Requires:** Python 3.10+, Streamlit 1.28+

---

Built with ❤️ using Streamlit and Claude Code
