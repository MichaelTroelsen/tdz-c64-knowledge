# Poppler + OCR Setup Documentation

## Summary

Poppler has been successfully configured for OCR functionality in your C64 Knowledge Base MCP server!

## What Was Done

1. **Located Poppler Installation**
   - Path: `C:\Users\mit\claude\c64server\tdz-c64-knowledge\poppler-25.12.0\Library\bin`
   - Version: 25.12.0
   - Contains all necessary binaries (pdftoppm.exe, etc.)

2. **Updated server.py**
   - Added support for `POPPLER_PATH` environment variable
   - Automatically uses Poppler when available for OCR operations
   - Falls back gracefully if Poppler path not specified

3. **Verified OCR Functionality**
   - Successfully extracted text from scanned PDFs
   - Tested with: `40_Best_Machine_Code_Routines_for_C64.pdf` (175 pages, 135,451 characters)
   - Tested with: `Commodore_64_BASIC_Quick_Reference_Guide.pdf` (2 pages, 22,579 characters)

## How to Use

### Option 1: Set Environment Variables (Recommended)

When running CLI commands or MCP server, set these environment variables:

```cmd
set USE_OCR=1
set POPPLER_PATH=C:\Users\mit\claude\c64server\tdz-c64-knowledge\poppler-25.12.0\Library\bin
```

### Option 2: Configure in MCP Settings

Add to your MCP configuration (e.g., `.claude/settings.local.json` or Claude Desktop config):

```json
{
  "mcpServers": {
    "tdz-c64-knowledge": {
      "command": "C:\\Users\\mit\\claude\\c64server\\tdz-c64-knowledge\\.venv\\Scripts\\python.exe",
      "args": ["C:\\Users\\mit\\claude\\c64server\\tdz-c64-knowledge\\server.py"],
      "env": {
        "TDZ_DATA_DIR": "C:\\Users\\mit\\.tdz-c64-knowledge",
        "USE_OCR": "1",
        "POPPLER_PATH": "C:\\Users\\mit\\claude\\c64server\\tdz-c64-knowledge\\poppler-25.12.0\\Library\\bin",
        "USE_FTS5": "1"
      }
    }
  }
}
```

### Option 3: Add to System PATH

Add Poppler to your system PATH environment variable:
```
C:\Users\mit\claude\c64server\tdz-c64-knowledge\poppler-25.12.0\Library\bin
```

This makes Poppler available system-wide without needing to set POPPLER_PATH.

## Testing

### Quick Test
```cmd
cd C:\Users\mit\claude\c64server\tdz-c64-knowledge
.venv\Scripts\activate
python test_poppler_ocr.py
```

### End-to-End Test
```cmd
python test_ocr_end_to_end.py
```

## Adding Scanned PDFs with OCR

### Using CLI
```cmd
set USE_OCR=1
set POPPLER_PATH=C:\Users\mit\claude\c64server\tdz-c64-knowledge\poppler-25.12.0\Library\bin

.venv\Scripts\python.exe cli.py add "path\to\scanned.pdf" --title "Document Title" --tags reference c64
```

### Using MCP
When configured properly, the MCP server will automatically use OCR for scanned PDFs when you add documents through the `add_document` tool.

## How It Works

1. **PDF Detection**: When adding a PDF, the system first tries to extract text normally
2. **OCR Fallback**: If the PDF has little or no extractable text (< 10 characters), it's treated as scanned
3. **Image Conversion**: Poppler's `pdftoppm` converts PDF pages to images
4. **Text Extraction**: Tesseract OCR extracts text from each image
5. **Indexing**: Extracted text is chunked and indexed in the knowledge base

## Log Output

When OCR is working correctly, you'll see log messages like:
```
INFO - OCR enabled (Tesseract found)
INFO - Using Poppler from: C:\Users\mit\claude\c64server\tdz-c64-knowledge\poppler-25.12.0\Library\bin
INFO - PDF appears to be scanned (0 chars extracted), falling back to OCR
INFO - Using OCR to extract text from scanned PDF: document.pdf
INFO - OCR extraction complete: 22579 characters from 2 pages
```

## Troubleshooting

### Issue: "Unable to get page count. Is poppler installed and in PATH?"

**Solution**: Set the `POPPLER_PATH` environment variable as described above.

### Issue: OCR not working

**Checklist**:
1. Verify `USE_OCR=1` is set
2. Verify Tesseract is installed: `.venv\Scripts\python.exe -c "import pytesseract; print(pytesseract.get_tesseract_version())"`
3. Verify Poppler path is correct: Check that `pdftoppm.exe` exists in the specified path
4. Run `python check_ocr.py` to diagnose issues

### Issue: OCR is slow

This is normal. OCR processing is CPU-intensive:
- Small documents (2-10 pages): 10-30 seconds
- Medium documents (50-100 pages): 2-5 minutes
- Large documents (200+ pages): 5-15 minutes

## Additional Resources

- Poppler documentation: https://poppler.freedesktop.org/
- Tesseract OCR: https://github.com/tesseract-ocr/tesseract
- pdf2image: https://github.com/Belval/pdf2image

## Changes Made to Code

### server.py

**Added** (lines 285-292):
```python
self.poppler_path = os.getenv('POPPLER_PATH', None)
if self.use_ocr:
    # ... existing code ...
    if self.poppler_path:
        self.logger.info(f"Using Poppler from: {self.poppler_path}")
```

**Modified** `_extract_pdf_with_ocr()` method (lines 850-853):
```python
if self.poppler_path:
    images = convert_from_path(filepath, poppler_path=self.poppler_path)
else:
    images = convert_from_path(filepath)
```

## Status

✅ **Poppler installed and configured**
✅ **server.py updated to support POPPLER_PATH**
✅ **OCR tested and working with scanned PDFs**
✅ **Ready for production use**

---

Last Updated: 2025-12-12
