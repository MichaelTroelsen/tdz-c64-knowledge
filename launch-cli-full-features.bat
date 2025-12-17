@echo off
REM Launch CLI with all advanced features enabled

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

REM Launch CLI with arguments passed through
.venv\Scripts\python.exe cli.py %*
