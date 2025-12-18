@echo off
REM Start All - Launch TDZ C64 Knowledge Base GUI
REM This launches the Streamlit admin interface with all features enabled
REM The MCP server runs automatically when needed by Claude Desktop

echo ==========================================
echo  TDZ C64 Knowledge Base - Start All
echo ==========================================
echo.
echo Starting Admin GUI with full features...
echo.
echo Features enabled:
echo  ✓ FTS5 full-text search (480x faster)
echo  ✓ Semantic search (conceptual matching)
echo  ✓ BM25 ranking algorithm
echo  ✓ Query preprocessing (stemming + stopwords)
echo  ✓ Fuzzy search (typo tolerance)
echo  ✓ OCR for scanned PDFs
echo  ✓ Search caching
echo  ✓ Web scraping (mdscrape integration)
echo.
echo ==========================================
echo.

REM Set all environment variables for advanced features
set USE_FTS5=1
set USE_SEMANTIC_SEARCH=1
set USE_BM25=1
set USE_QUERY_PREPROCESSING=1
set USE_FUZZY_SEARCH=1
set FUZZY_THRESHOLD=80
set USE_OCR=1
set SEARCH_CACHE_SIZE=100
set SEARCH_CACHE_TTL=300
set LLM_PROVIDER=anthropic
set LLM_MODEL=claude-3-haiku-20240307
set ALLOWED_DOCS_DIRS=C:\Users\mit\Downloads\tdz-c64-knowledge-input,C:\Users\mit\.tdz-c64-knowledge\scraped_docs
set POPPLER_PATH=C:\Users\mit\claude\c64server\tdz-c64-knowledge\poppler-25.12.0\Library\bin

echo Starting Streamlit GUI on http://localhost:8501
echo.
echo To stop the GUI, close this window or press Ctrl+C
echo.
echo NOTE: The MCP server for Claude Desktop runs separately
echo       through your Claude Desktop configuration.
echo ==========================================
echo.

REM Activate virtual environment and run Streamlit
call .venv\Scripts\activate.bat
python -m streamlit run admin_gui.py

REM Keep window open if there's an error
if errorlevel 1 (
    echo.
    echo ==========================================
    echo ERROR: Failed to start GUI
    echo ==========================================
    echo.
    echo Possible fixes:
    echo 1. Make sure virtual environment is set up
    echo 2. Run: .venv\Scripts\activate
    echo 3. Install dependencies: pip install -r requirements.txt
    echo.
    pause
)
