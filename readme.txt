What's included:
FilePurposeserver.pyMain MCP server with search, add, remove toolscli.pyCommand-line tool for bulk importing docssetup.batOne-click setup (creates venv, installs deps)run.batRun server standalone for testingtdz.batCLI shortcut (e.g., tdz add file.pdf)README.mdFull documentation
Quick start on Windows:

Extract the zip to C:\Users\YourName\mcp-servers\tdz-c64-knowledge
Double-click setup.bat (installs Python deps)
Add to Claude Code:

   claude mcp add tdz-c64-knowledge -- "C:\path\to\.venv\Scripts\python.exe" "C:\path\to\server.py"
Bulk import your PDFs:
cmdtdz add-folder "C:\c64docs" --tags reference --recursive
Tools available to Claude:

search_docs — Search across all docs
add_document — Add a PDF or text file
list_docs — Show what's indexed
get_chunk / get_document — Read specific content
kb_stats — Show stats

Once set up, you can ask Claude Code things like "search the C64 docs for VIC-II sprite registers" and it'll query your library.