@echo off
echo ================================================================================
echo PHASE 2 COMPREHENSIVE TEST SUITE
echo ================================================================================
echo.

echo [1/5] Running LDA Topic Modeling Tests...
.venv\Scripts\python.exe test_lda_implementation.py
if errorlevel 1 (
    echo [ERROR] LDA tests failed
    exit /b 1
)
echo.

echo [2/5] Running NMF and BERTopic Tests...
.venv\Scripts\python.exe test_nmf_bertopic.py
if errorlevel 1 (
    echo [ERROR] NMF/BERTopic tests failed
    exit /b 1
)
echo.

echo [3/5] Running Clustering Tests...
.venv\Scripts\python.exe test_clustering.py
if errorlevel 1 (
    echo [ERROR] Clustering tests failed
    exit /b 1
)
echo.

echo [4/5] Running MCP Tools Tests...
.venv\Scripts\python.exe test_mcp_tools_phase2.py
if errorlevel 1 (
    echo [ERROR] MCP tools tests failed
    exit /b 1
)
echo.

echo [5/5] Running Visualization Tests...
.venv\Scripts\python.exe test_visualizations.py
if errorlevel 1 (
    echo [ERROR] Visualization tests failed
    exit /b 1
)
echo.

echo ================================================================================
echo ALL PHASE 2 TESTS PASSED!
echo ================================================================================
echo.
echo Test Summary:
echo - LDA Topic Modeling: PASSED
echo - NMF and BERTopic: PASSED
echo - Document Clustering: PASSED
echo - MCP Tools: PASSED
echo - Visualizations: PASSED
echo.
echo Phase 2 is complete and fully tested.
echo ================================================================================
