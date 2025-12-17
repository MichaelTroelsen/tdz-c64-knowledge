@echo off
REM Launch Streamlit GUI with all advanced features enabled

echo.
echo ==========================================
echo TDZ C64 Knowledge Base - GUI Launcher
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

echo Features enabled:
echo  ✓ FTS5 full-text search (480x faster)
echo  ✓ Semantic search (conceptual matching)
echo  ✓ BM25 ranking algorithm
echo  ✓ Query preprocessing (stemming + stopwords)
echo  ✓ Fuzzy search (typo tolerance)
echo  ✓ OCR for scanned PDFs
echo  ✓ Search caching (100 entries, 5-min TTL)
echo.
echo Starting Streamlit GUI on http://localhost:8501
echo.

.venv\Scripts\python.exe -m streamlit run admin_gui.py
