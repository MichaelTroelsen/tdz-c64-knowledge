"""Tool schemas for the MCP server.

Extracted verbatim from server.py's list_tools() (R12): ~2,300 lines of
pure data that made the dispatch module hard to navigate and had no
reason to sit next to executable code.
"""
from mcp.types import Tool

TOOL_SCHEMAS: list[Tool] = [
        Tool(
            name="search_docs",
            description="Search the C64 knowledge base for information. Use this to find documentation about memory maps, opcodes, BASIC commands, SID, VIC-II, CIA chips, etc.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (keywords or phrases)"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 5)",
                        "default": 5
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by document tags (optional)"
                    },
                    "include_superseded": {
                        "type": "boolean",
                        "description": "Include superseded knowledge-card versions in results (default: false). Superseded cards are retracted/replaced content excluded from search by default.",
                        "default": False
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="translate_query",
            description="Translate natural language query to structured search parameters. Parses queries like 'find sprite information' or 'how does the SID chip work' into search terms, entity mentions, and facet filters. Returns recommended search mode (keyword/semantic/hybrid) and confidence score.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query to translate (e.g., 'find information about VIC-II sprites')"
                    },
                    "confidence_threshold": {
                        "type": "number",
                        "description": "Minimum confidence for entity extraction (0.0-1.0, default: 0.7)",
                        "default": 0.7
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_chunk",
            description="Get the full content of a specific document chunk. Use after search_docs to read more context.",
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "Document ID from search results"
                    },
                    "chunk_id": {
                        "type": "integer",
                        "description": "Chunk ID from search results"
                    }
                },
                "required": ["doc_id", "chunk_id"]
            }
        ),
        Tool(
            name="get_document",
            description="Get the full content of a document by ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "Document ID"
                    }
                },
                "required": ["doc_id"]
            }
        ),
        Tool(
            name="list_docs",
            description="List all documents in the C64 knowledge base.",
            inputSchema={
                "type": "object",
                "properties": {
                    "include_superseded": {
                        "type": "boolean",
                        "description": "Include superseded knowledge-card versions (default: false)",
                        "default": False
                    }
                }
            }
        ),
        Tool(
            name="get_document_by_card_id",
            description="Resolve a knowledge card's logical id (the `id` field in its ```json block, e.g. 'mon-deenen') to its live document. Use this instead of search when you already know a card's id, e.g. to follow an edges.derives_from/successor_of/shares_routine_with reference.",
            inputSchema={
                "type": "object",
                "properties": {
                    "card_id": {
                        "type": "string",
                        "description": "The card's logical id"
                    },
                    "include_superseded": {
                        "type": "boolean",
                        "description": "If no live card matches, fall back to the most recently indexed superseded version (default: false)",
                        "default": False
                    }
                },
                "required": ["card_id"]
            }
        ),
        Tool(
            name="add_document",
            description="Add a PDF or text file to the knowledge base. For a knowledge card (a markdown file whose content has a fenced ```json block with an `id` field), this refuses by default if a live card with that id already exists - use update_document to replace it, or pass replace=true here.",
            inputSchema={
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Full path to the PDF or text file"
                    },
                    "title": {
                        "type": "string",
                        "description": "Document title (optional, defaults to filename)"
                    },
                    "replace": {
                        "type": "boolean",
                        "description": "If the file is a knowledge card and a live card with the same id already exists, replace it (supersede the old one) instead of refusing (default: false)",
                        "default": False
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for categorization (e.g., 'memory-map', 'sid', 'basic', 'assembly')"
                    }
                },
                "required": ["filepath"]
            }
        ),
        Tool(
            name="update_document",
            description="Replace an existing knowledge card's content (and refresh everything derived from it - chunks, embeddings, entities, entity relationships) at its stable logical id. Resolves card_id_or_doc_id to the live document, replaces it with filepath's content, and marks the old version superseded rather than leaving two live copies. Refuses if the new file declares a different card id than the one being updated.",
            inputSchema={
                "type": "object",
                "properties": {
                    "card_id_or_doc_id": {
                        "type": "string",
                        "description": "The card's logical id (from its json block) or the exact doc_id of the existing live document to replace"
                    },
                    "filepath": {
                        "type": "string",
                        "description": "Full path to the new content to replace it with"
                    },
                    "title": {
                        "type": "string",
                        "description": "New title (optional, defaults to the existing document's title)"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "New tags (optional, defaults to the existing document's tags)"
                    }
                },
                "required": ["card_id_or_doc_id", "filepath"]
            }
        ),
        Tool(
            name="scrape_url",
            description="Scrape a documentation website and add all pages to the knowledge base. Supports recursive scraping of entire sites by following links. Great for ingesting online documentation like http://www.sidmusic.org/sid/. Converts HTML to searchable markdown.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Starting URL to scrape (e.g., http://www.sidmusic.org/sid/)"
                    },
                    "title": {
                        "type": "string",
                        "description": "Base title for scraped documents (optional, defaults to page titles)"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for scraped documents (domain name auto-added)"
                    },
                    "follow_links": {
                        "type": "boolean",
                        "description": "Follow links to scrape sub-pages (default: true). Set to false to scrape only the single page.",
                        "default": True
                    },
                    "same_domain_only": {
                        "type": "boolean",
                        "description": "Only follow links on the same domain (default: true). Prevents scraping external sites.",
                        "default": True
                    },
                    "max_pages": {
                        "type": "integer",
                        "description": "Maximum number of pages to scrape (default: 50)",
                        "default": 50,
                        "minimum": 1,
                        "maximum": 500
                    },
                    "depth": {
                        "type": "integer",
                        "description": "Maximum link depth to follow (default: 3). Depth of 1=single page, 2=linked pages, 3=two levels deep.",
                        "default": 3,
                        "minimum": 1,
                        "maximum": 10
                    },
                    "limit": {
                        "type": "string",
                        "description": "Advanced: Only scrape URLs with this prefix (overrides same_domain_only)"
                    },
                    "threads": {
                        "type": "integer",
                        "description": "Number of concurrent download threads (default: 3 - keep this low, these sources are small volunteer-run sites)",
                        "default": 3,
                        "minimum": 1,
                        "maximum": 20
                    },
                    "delay": {
                        "type": "integer",
                        "description": "Delay between requests in milliseconds (default: 500)",
                        "default": 500,
                        "minimum": 0,
                        "maximum": 5000
                    },
                    "selector": {
                        "type": "string",
                        "description": "CSS selector for main content (optional, auto-detected)"
                    }
                },
                "required": ["url"]
            }
        ),
        Tool(
            name="rescrape_document",
            description="Re-scrape a URL-sourced document to check for updates. Removes the old version and re-scrapes the original URL with the same configuration.",
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "Document ID to re-scrape (must be a URL-sourced document)"
                    }
                },
                "required": ["doc_id"]
            }
        ),
        Tool(
            name="check_url_updates",
            description="Check all URL-sourced documents for updates by comparing Last-Modified headers. Detects when source URLs have been modified since last scrape.",
            inputSchema={
                "type": "object",
                "properties": {
                    "auto_rescrape": {
                        "type": "boolean",
                        "description": "Automatically re-scrape changed URLs (default: false)",
                        "default": False
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="repoint_document",
            description=(
                "Point an existing document at a relocated source file without reingesting. "
                "Use when a document's recorded filepath no longer exists on disk (see the "
                "missing_source_files metric in health_check). The candidate file is verified "
                "against the document's recorded content hash and the ALLOWED_DOCS_DIRS "
                "whitelist before anything is written; a hash mismatch is refused unless force "
                "is set, because binding a document to the wrong file is worse than a missing path."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "Document ID whose filepath should be repaired"
                    },
                    "new_filepath": {
                        "type": "string",
                        "description": "Path the document now lives at (must be inside ALLOWED_DOCS_DIRS)"
                    },
                    "force": {
                        "type": "boolean",
                        "description": "Re-point even when the file content does not match the recorded hash (default false)",
                        "default": False
                    }
                },
                "required": ["doc_id", "new_filepath"]
            }
        ),
        Tool(
            name="add_deepsid_document",
            description=(
                "Ingest one SID tune's metadata from DeepSID's JSON API as a searchable "
                "document. Takes the tune's collection path, NOT a URL and NOT a local file: "
                "the format is the collection folder with a leading underscore and no leading "
                "slash, e.g. '_High Voltage SID Collection/MUSICIANS/H/Hubbard_Rob/Commando.sid'. "
                "A path that matches no tune is refused rather than stored as an empty card. "
                "The card carries title, author, release, chip model, video standard, subtune "
                "count and addresses, plus the player routine, per-subtune song lengths and the "
                "full STIL entry - the last three exist nowhere in a .sid file's own header, "
                "which is the reason to call the API at all."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "fullname": {
                        "type": "string",
                        "description": (
                            "Tune's collection path, leading underscore and no leading slash, "
                            "e.g. '_High Voltage SID Collection/MUSICIANS/H/Hubbard_Rob/Commando.sid'"
                        )
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional tags to attach to the created document"
                    }
                },
                "required": ["fullname"]
            }
        ),
        Tool(
            name="add_deepsid_folder",
            description=(
                "Ingest every tune in one DeepSID folder from a SINGLE request, as "
                "searchable documents - because one music.php listing already carries "
                "each tune's author, release, chip model, lengths, player and STIL "
                "entry, so this makes exactly one request no matter how many tunes the "
                "folder holds (a 96-tune folder still costs one call - do not loop "
                "add_deepsid_document per tune). Does NOT walk subfolders; lists one "
                "directory only. "
                "WARNING: `folder` uses a DIFFERENT path format than "
                "add_deepsid_document's `fullname`, and swapping them fails silently "
                "(music.php answers HTTP 500 with an empty body; info.php returns an "
                "empty record). `folder` takes a LEADING SLASH and no other prefix "
                "change, e.g. '/_High Voltage SID Collection/MUSICIANS/H/Hubbard_Rob' "
                "- contrast add_deepsid_document's `fullname`, which has a leading "
                "underscore and NO leading slash, e.g. "
                "'_High Voltage SID Collection/MUSICIANS/H/Hubbard_Rob/Commando.sid'."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "folder": {
                        "type": "string",
                        "description": (
                            "Folder path, LEADING SLASH required, e.g. "
                            "'/_High Voltage SID Collection/MUSICIANS/H/Hubbard_Rob'"
                        )
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional tags to attach to each created document"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Optional cap on how many of the folder's tunes to ingest"
                    }
                },
                "required": ["folder"]
            }
        ),
        Tool(
            name="remove_document",
            description="Remove a document from the knowledge base.",
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "Document ID to remove"
                    }
                },
                "required": ["doc_id"]
            }
        ),
        Tool(
            name="semantic_search",
            description="Search the knowledge base using semantic/conceptual similarity (requires USE_SEMANTIC_SEARCH=1). Finds documents based on meaning, not just keywords. Example: searching for 'movable objects' can find 'sprites'.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (natural language)"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 5)",
                        "default": 5
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by document tags (optional)"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="find_similar",
            description="Find documents similar to a given document. Uses semantic embeddings if available, falls back to TF-IDF. Great for discovering related content.",
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "Document ID to find similar documents for"
                    },
                    "chunk_id": {
                        "type": "integer",
                        "description": "Optional chunk ID (if omitted, uses entire document)"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 5)",
                        "default": 5
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by document tags (optional)"
                    }
                },
                "required": ["doc_id"]
            }
        ),
        Tool(
            name="hybrid_search",
            description="Perform hybrid search combining FTS5 keyword search and semantic search. Best of both worlds - finds exact keyword matches AND conceptually related content. Returns results ranked by weighted combination of both scores.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 5)",
                        "default": 5
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by document tags (optional)"
                    },
                    "semantic_weight": {
                        "type": "number",
                        "description": "Relative weight of the semantic ranking, 0.0-1.0 (default: 0.7). Higher values favor conceptual matches, lower values favor exact keyword matches.",
                        "default": 0.7
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="fuzzy_search",
            description="Search with typo tolerance using fuzzy string matching. Handles misspellings and variations like 'VIC2' → 'VIC-II', 'asembly' → 'assembly', '6052' → '6502'. Returns exact matches first, then fuzzy matches if needed.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (may contain typos)"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 5)",
                        "default": 5
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by document tags (optional)"
                    },
                    "similarity_threshold": {
                        "type": "integer",
                        "description": "Minimum similarity score 0-100 (default: 80). Lower values are more forgiving of typos.",
                        "default": 80,
                        "minimum": 0,
                        "maximum": 100
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="search_within_results",
            description="Refine previous search results with an additional query. Useful for progressive search refinement: first search broadly (e.g., 'VIC-II'), then refine (e.g., 'sprite collision'). Returns filtered and re-ranked results from previous search set.",
            inputSchema={
                "type": "object",
                "properties": {
                    "previous_results": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Results from a previous search (pass the full result objects)"
                    },
                    "refinement_query": {
                        "type": "string",
                        "description": "Query to refine the previous results"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of refined results (default: 5)",
                        "default": 5
                    }
                },
                "required": ["previous_results", "refinement_query"]
            }
        ),
        Tool(
            name="suggest_tags",
            description="Get tag suggestions for a document based on content analysis. Detects hardware components (SID, VIC-II, CIA), programming topics (assembly, BASIC, graphics), document types (reference, tutorial), and difficulty levels. Useful for organizing documents.",
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "Document ID to analyze"
                    },
                    "confidence_threshold": {
                        "type": "number",
                        "description": "Minimum confidence for suggestions 0.0-1.0 (default: 0.6)",
                        "default": 0.6,
                        "minimum": 0.0,
                        "maximum": 1.0
                    }
                },
                "required": ["doc_id"]
            }
        ),
        Tool(
            name="get_tags_by_category",
            description="Browse all tags organized by category (hardware, programming, document-type, difficulty, custom). Shows tag usage count and sample documents. Useful for discovering and organizing content.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="answer_question",
            description="Answer questions about C64 documentation using RAG (Retrieval-Augmented Generation). Synthesizes information from multiple sources with citations. Returns answer text with source references and confidence score.",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to answer about C64 documentation"
                    },
                    "max_sources": {
                        "type": "integer",
                        "description": "Maximum number of documentation sources to use for context (default: 5)",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20
                    },
                    "search_mode": {
                        "type": "string",
                        "enum": ["auto", "keyword", "semantic", "hybrid"],
                        "description": "Search strategy to use (default: auto for intelligent selection)",
                        "default": "auto"
                    }
                },
                "required": ["question"]
            }
        ),
        Tool(
            name="kb_stats",
            description="Get statistics about the knowledge base.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="health_check",
            description="Perform health check on the knowledge base system. Returns status, metrics, feature availability, and any issues detected.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="reconcile_chunk_cache",
            description="Reconcile the in-memory chunk cache against the database on demand, without restarting the server. Fixes cases where search_docs still returns content for a document that get_document/list_docs already report as removed (stale in-memory cache from before a fix, or any future cache/DB divergence). Also invalidates the BM25 index, embeddings, and search caches so they rebuild from the reconciled data.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="reconcile_embeddings",
            description="Backfill embeddings for documents that have chunks but no embedded vectors. The embeddings rebuild only triggers when the index is completely empty, so documents added while USE_SEMANTIC_SEARCH was off (or before the model was loaded) can be permanently invisible to semantic_search/find_similar with no error - health_check's embeddings_doc_coverage_pct feature flags this. Safe to run concurrently with other agent sessions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "max_docs": {
                        "type": "integer",
                        "description": "Optional cap on how many missing documents to backfill in this call, for processing a large gap incrementally instead of one long-running call."
                    }
                }
            }
        ),
        Tool(
            name="detect_anomalies",
            description="Detect anomalies in URL monitoring history. Analyzes patterns to identify unusual update frequencies, performance degradation, or unexpected content changes. Returns anomalies with severity scores (normal/minor/moderate/critical) based on learned baselines.",
            inputSchema={
                "type": "object",
                "properties": {
                    "min_severity": {
                        "type": "string",
                        "description": "Minimum severity level to include ('normal', 'minor', 'moderate', 'critical', default: 'moderate')",
                        "enum": ["normal", "minor", "moderate", "critical"],
                        "default": "moderate"
                    },
                    "days": {
                        "type": "integer",
                        "description": "Number of days of history to analyze (default: 7)",
                        "default": 7,
                        "minimum": 1,
                        "maximum": 90
                    }
                }
            }
        ),
        Tool(
            name="check_updates",
            description="Check all indexed documents for updates. Detects files that have been modified since indexing and optionally re-indexes them automatically.",
            inputSchema={
                "type": "object",
                "properties": {
                    "auto_update": {
                        "type": "boolean",
                        "description": "Automatically re-index changed documents (default: false)",
                        "default": False
                    }
                }
            }
        ),
        Tool(
            name="add_documents_bulk",
            description="Add multiple documents from a directory at once. Supports glob patterns for file matching.",
            inputSchema={
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Directory to search for documents"
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern (default: **/*.{pdf,txt})",
                        "default": "**/*.{pdf,txt}"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags to apply to all documents (optional)"
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Search subdirectories (default: true)",
                        "default": True
                    },
                    "skip_duplicates": {
                        "type": "boolean",
                        "description": "Skip files with duplicate content (default: true)",
                        "default": True
                    }
                },
                "required": ["directory"]
            }
        ),
        Tool(
            name="remove_documents_bulk",
            description="Remove multiple documents by IDs or tags. Useful for cleaning up the knowledge base.",
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of document IDs to remove (optional)"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Remove all documents with these tags (optional)"
                    }
                }
            }
        ),
        Tool(
            name="update_tags_bulk",
            description="Update tags for multiple documents in bulk. Add, remove, or replace tags for documents selected by ID or existing tags. Useful for reorganizing and categorizing the knowledge base.",
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of document IDs to update (optional, use existing_tags to find documents)"
                    },
                    "existing_tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Find documents with any of these tags (alternative to doc_ids)"
                    },
                    "add_tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags to add to the documents"
                    },
                    "remove_tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags to remove from the documents"
                    },
                    "replace_tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Replace all tags with these tags"
                    }
                }
            }
        ),
        Tool(
            name="export_documents_bulk",
            description="Export metadata for multiple documents in JSON, CSV, or Markdown format. Useful for creating reports, backups, or sharing document lists.",
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of document IDs to export (optional, defaults to all or filtered by tags)"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Export documents with any of these tags (optional)"
                    },
                    "format": {
                        "type": "string",
                        "description": "Export format (default: json)",
                        "enum": ["json", "csv", "markdown"],
                        "default": "json"
                    }
                }
            }
        ),
        Tool(
            name="search_tables",
            description="Search for tables in PDF documents. Tables contain structured data like memory maps, register definitions, and command references. Returns tables in markdown format with page numbers.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for table content"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 5)",
                        "default": 5
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by document tags (optional)"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="search_code",
            description="Search for code blocks in documents (BASIC, Assembly, Hex dumps). Finds programming examples and code snippets. Returns code with type (basic/assembly/hex), line count, and page numbers.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for code content"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 5)",
                        "default": 5
                    },
                    "block_type": {
                        "type": "string",
                        "description": "Filter by code type: 'basic', 'assembly', or 'hex' (optional)",
                        "enum": ["basic", "assembly", "hex"]
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by document tags (optional)"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="faceted_search",
            description="Search with faceted filtering. Filter results by hardware components (SID, VIC-II, CIA, etc.), assembly instructions (LDA, STA, etc.), or memory registers ($D000, etc.). Great for narrowing down search results to specific technical domains.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "facet_filters": {
                        "type": "object",
                        "description": "Facet filters as dict of facet_type -> list of values. Example: {'hardware': ['SID', 'VIC-II'], 'instruction': ['LDA', 'STA']}",
                        "additionalProperties": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 5)",
                        "default": 5
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by document tags (optional)"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="search_analytics",
            description="Get search analytics and insights. Shows popular queries, failed searches, search mode usage, and performance metrics.",
            inputSchema={
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "description": "Number of days to analyze (default: 30)",
                        "default": 30
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results for top queries (default: 100)",
                        "default": 100
                    }
                }
            }
        ),
        Tool(
            name="find_by_reference",
            description="Find documents by cross-reference. Search for documents containing specific memory addresses ($D020), register offsets (VIC+0, SID+4), or page references (page 156). Great for tracking how specific registers or memory locations are documented.",
            inputSchema={
                "type": "object",
                "properties": {
                    "ref_type": {
                        "type": "string",
                        "description": "Type of reference to search for",
                        "enum": ["memory_address", "register_offset", "page_reference"]
                    },
                    "ref_value": {
                        "type": "string",
                        "description": "The reference value (e.g., '$D020', 'VIC+0', '156')"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 10)",
                        "default": 10
                    }
                },
                "required": ["ref_type", "ref_value"]
            }
        ),
        Tool(
            name="suggest_queries",
            description="Get autocomplete suggestions for partial queries. Suggests technical terms, memory addresses, instructions, and concepts based on indexed content. Great for discovering searchable content and learning proper terminology.",
            inputSchema={
                "type": "object",
                "properties": {
                    "partial": {
                        "type": "string",
                        "description": "Partial query string (e.g., 'VIC', 'SID', '$D0')"
                    },
                    "max_suggestions": {
                        "type": "integer",
                        "description": "Maximum number of suggestions (default: 5)",
                        "default": 5
                    },
                    "category": {
                        "type": "string",
                        "description": "Optional category filter",
                        "enum": ["hardware", "register", "instruction", "concept"]
                    }
                },
                "required": ["partial"]
            }
        ),
        Tool(
            name="export_results",
            description="Export search results to various formats (markdown, json, html). Use this to save search results for offline use, sharing, or creating custom reference guides.",
            inputSchema={
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "description": "Search results array (from any search method)"
                    },
                    "format": {
                        "type": "string",
                        "description": "Export format",
                        "enum": ["markdown", "json", "html"],
                        "default": "markdown"
                    },
                    "query": {
                        "type": "string",
                        "description": "Optional query string to include in export"
                    }
                },
                "required": ["results"]
            }
        ),
        Tool(
            name="create_backup",
            description="Create a backup of the knowledge base. Backs up database and embeddings to a zip file. Use this regularly for data safety and before making major changes.",
            inputSchema={
                "type": "object",
                "properties": {
                    "dest_dir": {
                        "type": "string",
                        "description": "Destination directory for backup"
                    },
                    "compress": {
                        "type": "boolean",
                        "description": "Whether to compress backup to zip file (default: true)",
                        "default": True
                    }
                },
                "required": ["dest_dir"]
            }
        ),
        Tool(
            name="restore_backup",
            description="Restore knowledge base from a backup. WARNING: This will replace the current database. A safety backup is created automatically before restoration.",
            inputSchema={
                "type": "object",
                "properties": {
                    "backup_path": {
                        "type": "string",
                        "description": "Path to backup file or directory"
                    },
                    "verify": {
                        "type": "boolean",
                        "description": "Whether to verify backup integrity before restoring (default: true)",
                        "default": True
                    }
                },
                "required": ["backup_path"]
            }
        ),
        Tool(
            name="auto_tag_document",
            description="Automatically generate tags for a document using AI analysis. Analyzes document content and suggests relevant tags across categories: hardware (sid, vic-ii), programming (assembly, basic), document type (tutorial, reference), and difficulty level (beginner, advanced). Requires LLM configuration (set LLM_PROVIDER and API key).",
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "Document ID to tag"
                    },
                    "confidence_threshold": {
                        "type": "number",
                        "description": "Minimum confidence to accept tag (0.0-1.0, default: 0.7)",
                        "default": 0.7,
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "max_tags": {
                        "type": "integer",
                        "description": "Maximum number of tags to suggest (default: 10)",
                        "default": 10
                    },
                    "append": {
                        "type": "boolean",
                        "description": "If true, append to existing tags; if false, replace (default: true)",
                        "default": True
                    }
                },
                "required": ["doc_id"]
            }
        ),
        Tool(
            name="auto_tag_all",
            description="Bulk auto-tag multiple documents using AI. Analyzes content and suggests relevant tags for all documents (or subset). Useful for initial organization or re-tagging collections. Can skip already-tagged documents and limit processing count. Requires LLM configuration.",
            inputSchema={
                "type": "object",
                "properties": {
                    "confidence_threshold": {
                        "type": "number",
                        "description": "Minimum confidence to accept tag (0.0-1.0, default: 0.7)",
                        "default": 0.7,
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "max_tags": {
                        "type": "integer",
                        "description": "Maximum tags per document (default: 10)",
                        "default": 10
                    },
                    "append": {
                        "type": "boolean",
                        "description": "If true, append to existing tags; if false, replace (default: true)",
                        "default": True
                    },
                    "skip_tagged": {
                        "type": "boolean",
                        "description": "If true, skip documents that already have tags (default: true)",
                        "default": True
                    },
                    "max_docs": {
                        "type": "integer",
                        "description": "Maximum number of documents to process (optional, for testing or rate limiting)"
                    }
                }
            }
        ),
        Tool(
            name="summarize_document",
            description="Generate an AI-powered summary of a document. Supports brief (200-300 words), detailed (500-800 words), or bullet-point summaries.",
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "Document ID to summarize"
                    },
                    "summary_type": {
                        "type": "string",
                        "enum": ["brief", "detailed", "bullet"],
                        "description": "Type of summary: 'brief' (default), 'detailed', or 'bullet'",
                        "default": "brief"
                    },
                    "force_regenerate": {
                        "type": "boolean",
                        "description": "If true, regenerate summary even if cached version exists (default: false)",
                        "default": False
                    }
                },
                "required": ["doc_id"]
            }
        ),
        Tool(
            name="get_summary",
            description="Retrieve a cached summary of a document without regenerating it.",
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "Document ID"
                    },
                    "summary_type": {
                        "type": "string",
                        "enum": ["brief", "detailed", "bullet"],
                        "description": "Type of summary (default: 'brief')",
                        "default": "brief"
                    }
                },
                "required": ["doc_id"]
            }
        ),
        Tool(
            name="summarize_all",
            description="Bulk generate summaries for all documents in the knowledge base.",
            inputSchema={
                "type": "object",
                "properties": {
                    "summary_types": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["brief", "detailed", "bullet"]},
                        "description": "Types of summaries to generate (default: ['brief'])"
                    },
                    "force_regenerate": {
                        "type": "boolean",
                        "description": "If true, regenerate all summaries (default: false)",
                        "default": False
                    },
                    "max_docs": {
                        "type": "integer",
                        "description": "Maximum number of documents to process (optional, for testing)"
                    }
                }
            }
        ),
        Tool(
            name="extract_entities",
            description="Extract named entities from a C64 document using AI. Identifies hardware (SID, VIC-II, CIA, 6502), memory addresses ($D000), assembly instructions (LDA, STA), people, companies, products, and technical concepts. Returns entities with type, confidence score, and context. Requires LLM configuration.",
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "Document ID to extract entities from"
                    },
                    "confidence_threshold": {
                        "type": "number",
                        "description": "Minimum confidence to include entity (0.0-1.0, default: 0.6)",
                        "default": 0.6,
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "force_regenerate": {
                        "type": "boolean",
                        "description": "Force re-extraction even if entities already exist (default: false)",
                        "default": False
                    }
                },
                "required": ["doc_id"]
            }
        ),
        Tool(
            name="list_entities",
            description="List all entities extracted from a document, grouped by type (hardware, memory_address, instruction, person, company, product, concept). Great for getting an overview of what a document covers.",
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "Document ID"
                    },
                    "entity_types": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["hardware", "memory_address", "instruction", "person", "company", "product", "concept"]
                        },
                        "description": "Filter by entity types (optional, returns all types if omitted)"
                    },
                    "min_confidence": {
                        "type": "number",
                        "description": "Minimum confidence threshold (0.0-1.0, default: 0.0)",
                        "default": 0.0,
                        "minimum": 0.0,
                        "maximum": 1.0
                    }
                },
                "required": ["doc_id"]
            }
        ),
        Tool(
            name="search_entities",
            description="Search for entities across all documents using full-text search. Find all documents mentioning specific hardware, addresses, instructions, people, companies, products, or concepts.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g., 'VIC-II', 'sprite', '$D000')"
                    },
                    "entity_types": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["hardware", "memory_address", "instruction", "person", "company", "product", "concept"]
                        },
                        "description": "Filter by entity types (optional)"
                    },
                    "min_confidence": {
                        "type": "number",
                        "description": "Minimum confidence threshold (0.0-1.0, default: 0.0)",
                        "default": 0.0,
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of documents to return (default: 20)",
                        "default": 20,
                        "minimum": 1,
                        "maximum": 100
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="entity_stats",
            description="Get statistics about extracted entities in the knowledge base. Shows breakdown by type, top entities, and documents with most entities. Useful for understanding the knowledge base content.",
            inputSchema={
                "type": "object",
                "properties": {
                    "entity_type": {
                        "type": "string",
                        "enum": ["hardware", "memory_address", "instruction", "person", "company", "product", "concept"],
                        "description": "Filter statistics by entity type (optional, shows all types if omitted)"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="get_entity_analytics",
            description="Get comprehensive entity analytics for dashboard visualization. Provides entity distribution by type, top entities by document count, relationship statistics, top entity relationships, and extraction timeline trends over time.",
            inputSchema={
                "type": "object",
                "properties": {
                    "time_range_days": {
                        "type": "integer",
                        "description": "Number of days to include in timeline analysis (default: 30)",
                        "default": 30,
                        "minimum": 1,
                        "maximum": 365
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="extract_entities_bulk",
            description="Bulk extract entities from multiple documents in the knowledge base. Processes documents in batch, skips documents that already have entities (unless force_regenerate). Returns statistics about processed documents and extracted entities.",
            inputSchema={
                "type": "object",
                "properties": {
                    "confidence_threshold": {
                        "type": "number",
                        "description": "Minimum confidence to include entity (0.0-1.0, default: 0.6)",
                        "default": 0.6,
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "force_regenerate": {
                        "type": "boolean",
                        "description": "Force re-extraction even if entities already exist (default: false)",
                        "default": False
                    },
                    "max_docs": {
                        "type": "integer",
                        "description": "Maximum number of documents to process (optional, for testing)",
                        "minimum": 1
                    },
                    "skip_existing": {
                        "type": "boolean",
                        "description": "Skip documents that already have entities (default: true)",
                        "default": True
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="extract_entity_relationships",
            description="Extract co-occurrence relationships between entities in a document. Analyzes how entities appear together (e.g., VIC-II + raster interrupt, SID + sound programming). Returns entity pairs with relationship strength and context snippets.",
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "Document ID to extract relationships from"
                    },
                    "min_confidence": {
                        "type": "number",
                        "description": "Minimum confidence threshold for entities (0.0-1.0, default: 0.6)",
                        "default": 0.6,
                        "minimum": 0.0,
                        "maximum": 1.0
                    }
                },
                "required": ["doc_id"]
            }
        ),
        Tool(
            name="get_entity_relationships",
            description="Get all entities related to a specific entity. Shows which other entities frequently co-occur with the target entity, sorted by relationship strength. Great for discovering related concepts, hardware, and techniques.",
            inputSchema={
                "type": "object",
                "properties": {
                    "entity_text": {
                        "type": "string",
                        "description": "The entity to find relationships for (e.g., 'VIC-II', 'SID', 'LDA')"
                    },
                    "min_strength": {
                        "type": "number",
                        "description": "Minimum relationship strength (0.0-1.0, default: 0.0)",
                        "default": 0.0,
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of related entities to return (default: 20)",
                        "default": 20,
                        "minimum": 1,
                        "maximum": 100
                    }
                },
                "required": ["entity_text"]
            }
        ),
        Tool(
            name="find_related_entities",
            description="Discover entities related to a given entity (simplified version of get_entity_relationships). Returns top related entities for quick exploration and discovery.",
            inputSchema={
                "type": "object",
                "properties": {
                    "entity_text": {
                        "type": "string",
                        "description": "The entity to find related entities for"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of related entities (default: 10)",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 50
                    }
                },
                "required": ["entity_text"]
            }
        ),
        Tool(
            name="search_entity_pair",
            description="Find documents that contain both entities. Useful for finding documentation about specific combinations (e.g., 'VIC-II' AND 'raster interrupt'). Returns documents with both entity counts and context snippets.",
            inputSchema={
                "type": "object",
                "properties": {
                    "entity1": {
                        "type": "string",
                        "description": "First entity to search for"
                    },
                    "entity2": {
                        "type": "string",
                        "description": "Second entity to search for"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of documents to return (default: 10)",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 50
                    }
                },
                "required": ["entity1", "entity2"]
            }
        ),
        Tool(
            name="compare_documents",
            description="Compare two documents side-by-side with similarity scoring, metadata diff, content diff, and entity comparison. Perfect for finding differences between document versions, comparing related documents, or analyzing document similarity. Returns comprehensive comparison with cosine similarity score (0.0-1.0).",
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_id_1": {
                        "type": "string",
                        "description": "First document ID to compare"
                    },
                    "doc_id_2": {
                        "type": "string",
                        "description": "Second document ID to compare"
                    },
                    "comparison_type": {
                        "type": "string",
                        "description": "Type of comparison: 'full' (all), 'metadata' (metadata + entities), 'content' (metadata + similarity + diff)",
                        "enum": ["full", "metadata", "content"],
                        "default": "full"
                    }
                },
                "required": ["doc_id_1", "doc_id_2"]
            }
        ),
        Tool(
            name="export_entities",
            description="Export all extracted entities to CSV or JSON format. Includes entity text, type, confidence scores, document counts, and occurrence counts. Useful for data analysis, reporting, or importing into other tools.",
            inputSchema={
                "type": "object",
                "properties": {
                    "format": {
                        "type": "string",
                        "description": "Export format: 'csv' or 'json'",
                        "enum": ["csv", "json"],
                        "default": "csv"
                    },
                    "entity_types": {
                        "type": "array",
                        "description": "Filter by entity types (e.g., ['hardware', 'instruction']). Leave empty for all types.",
                        "items": {
                            "type": "string",
                            "enum": ["hardware", "memory_address", "instruction", "person", "company", "product", "concept"]
                        }
                    },
                    "min_confidence": {
                        "type": "number",
                        "description": "Minimum confidence threshold (0.0-1.0)",
                        "default": 0.0,
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Optional file path to save export (if not provided, returns as string)"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="export_relationships",
            description="Export all entity relationships to CSV or JSON format. Includes entity pairs, types, relationship strength scores (0.0-1.0), and document counts. Perfect for network analysis, visualization, or data export.",
            inputSchema={
                "type": "object",
                "properties": {
                    "format": {
                        "type": "string",
                        "description": "Export format: 'csv' or 'json'",
                        "enum": ["csv", "json"],
                        "default": "csv"
                    },
                    "min_strength": {
                        "type": "number",
                        "description": "Minimum relationship strength (0.0-1.0)",
                        "default": 0.0,
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "entity_types": {
                        "type": "array",
                        "description": "Filter by entity types (applies to either entity in pair). Leave empty for all types.",
                        "items": {
                            "type": "string",
                            "enum": ["hardware", "memory_address", "instruction", "person", "company", "product", "concept"]
                        }
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Optional file path to save export (if not provided, returns as string)"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="queue_entity_extraction",
            description="Queue a document for background entity extraction. Extraction happens asynchronously without blocking. Use this to extract entities from documents without waiting for LLM processing. Returns job ID for tracking progress.",
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "Document ID to extract entities from"
                    },
                    "confidence_threshold": {
                        "type": "number",
                        "description": "Minimum confidence threshold (0.0-1.0, default: 0.6)",
                        "default": 0.6,
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "skip_if_exists": {
                        "type": "boolean",
                        "description": "Skip if entities already exist or job is queued (default: true)",
                        "default": True
                    }
                },
                "required": ["doc_id"]
            }
        ),
        Tool(
            name="get_extraction_status",
            description="Get the entity extraction status for a document. Shows whether entities exist, extraction job status (queued/running/completed/failed), timestamps, and error messages if any. Use this to check if extraction is complete before querying entities.",
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "Document ID to check extraction status for"
                    }
                },
                "required": ["doc_id"]
            }
        ),
        Tool(
            name="get_extraction_jobs",
            description="Get all entity extraction jobs with optional status filtering. Shows job queue, running jobs, completed extractions, and failed jobs. Useful for monitoring background extraction progress across all documents.",
            inputSchema={
                "type": "object",
                "properties": {
                    "status_filter": {
                        "type": "string",
                        "description": "Filter by status: 'queued', 'running', 'completed', or 'failed'. Leave empty for all jobs.",
                        "enum": ["queued", "running", "completed", "failed"]
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of jobs to return (default: 100)",
                        "default": 100,
                        "minimum": 1,
                        "maximum": 1000
                    }
                },
                "required": []
            }
        ),
        # ============================================================
        # Figure OCR Tools
        # ============================================================
        Tool(
            name="batch_ocr_figures",
            description="Queue every PDF in the knowledge base for background figure OCR. Extracts embedded images (diagrams, memory maps, register tables, schematics) from each PDF and OCRs them into searchable text. Runs in the background - returns immediately with the jobs queued. Ingest-time OCR only covers fully-scanned PDFs, so this is what makes figures inside normal text PDFs searchable. Use figure_ocr_status to monitor progress and search_figures to query the results.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of documents to queue this call (default: all PDFs)",
                        "minimum": 1
                    },
                    "reprocess": {
                        "type": "boolean",
                        "description": "Re-OCR documents that already have extracted figures (default: false)",
                        "default": False
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="ocr_document_figures",
            description="Queue a single document for background figure OCR. Use batch_ocr_figures to do the whole knowledge base instead.",
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "Document ID (must be a PDF)"
                    },
                    "reprocess": {
                        "type": "boolean",
                        "description": "Re-OCR even if figures were already extracted (default: false)",
                        "default": False
                    }
                },
                "required": ["doc_id"]
            }
        ),
        Tool(
            name="figure_ocr_status",
            description="Report figure-OCR coverage across the knowledge base: how many PDFs have been processed, how many figures were extracted, how many yielded text, and how many jobs are still pending. Also reports whether figure OCR is available at all (needs PyMuPDF, Tesseract and USE_OCR=1).",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        Tool(
            name="search_figures",
            description="Search text that was OCR'd out of document figures. Finds content that lives only inside images - memory-map diagrams, register tables, pinout drawings, schematics - which plain document search cannot reach. Returns the document, page number and figure index for each hit.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (keywords or phrases)"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 10)",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 100
                    },
                    "doc_id": {
                        "type": "string",
                        "description": "Restrict the search to one document (optional)"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_document_figures",
            description="List every figure extracted from one document, in page order, with its OCR text and the path to the extracted image file.",
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "Document ID"
                    },
                    "with_text_only": {
                        "type": "boolean",
                        "description": "Only return figures whose OCR produced text (default: false)",
                        "default": False
                    }
                },
                "required": ["doc_id"]
            }
        ),
        # ============================================================
        # Knowledge Graph Tools (v2.24.0 - Phase 1, Task 1.4)
        # ============================================================
        Tool(
            name="build_knowledge_graph",
            description="Build a knowledge graph from extracted entities and relationships. Returns graph statistics including node count, edge count, density, and connected components. The graph can be filtered by entity types, minimum occurrence counts, and relationship strength thresholds.",
            inputSchema={
                "type": "object",
                "properties": {
                    "entity_types": {
                        "type": "array",
                        "description": "Filter to specific entity types (e.g., ['hardware', 'instruction']). Leave empty for all types.",
                        "items": {
                            "type": "string",
                            "enum": ["hardware", "memory_address", "instruction", "person", "company", "product", "concept"]
                        }
                    },
                    "min_occurrences": {
                        "type": "integer",
                        "description": "Minimum entity occurrences to include in graph",
                        "default": 2,
                        "minimum": 1
                    },
                    "min_relationship_strength": {
                        "type": "number",
                        "description": "Minimum relationship strength threshold (0.0-1.0)",
                        "default": 0.3,
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "use_cache": {
                        "type": "boolean",
                        "description": "Use cached graph if available",
                        "default": True
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="analyze_graph_pagerank",
            description="Calculate PageRank scores for entities in the knowledge graph. PageRank identifies the most 'important' or 'central' entities based on their connections. Higher scores indicate entities that are more connected and influential in the knowledge network.",
            inputSchema={
                "type": "object",
                "properties": {
                    "entity_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter graph to specific entity types (optional)"
                    },
                    "min_occurrences": {
                        "type": "integer",
                        "description": "Minimum entity occurrences (default: 2)",
                        "default": 2
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Number of top entities to return (default: 20)",
                        "default": 20,
                        "minimum": 1,
                        "maximum": 100
                    },
                    "alpha": {
                        "type": "number",
                        "description": "Damping parameter for PageRank (default: 0.85)",
                        "default": 0.85,
                        "minimum": 0.0,
                        "maximum": 1.0
                    }
                }
            }
        ),
        Tool(
            name="detect_graph_communities",
            description="Detect communities (clusters) in the knowledge graph. Communities are groups of entities that are more densely connected to each other than to the rest of the graph. This helps identify topic clusters and thematic groupings.",
            inputSchema={
                "type": "object",
                "properties": {
                    "entity_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter graph to specific entity types (optional)"
                    },
                    "min_occurrences": {
                        "type": "integer",
                        "description": "Minimum entity occurrences (default: 2)",
                        "default": 2
                    },
                    "algorithm": {
                        "type": "string",
                        "description": "Detection algorithm: 'louvain' (best for large graphs), 'label_propagation' (fast), or 'greedy_modularity'",
                        "default": "louvain",
                        "enum": ["louvain", "label_propagation", "greedy_modularity"]
                    }
                }
            }
        ),
        Tool(
            name="calculate_graph_centrality",
            description="Calculate centrality measures for entities in the knowledge graph. Returns betweenness, closeness, and degree centrality. These measures identify entities that bridge different parts of the graph, are close to all others, or have many connections.",
            inputSchema={
                "type": "object",
                "properties": {
                    "entity_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter graph to specific entity types (optional)"
                    },
                    "min_occurrences": {
                        "type": "integer",
                        "description": "Minimum entity occurrences (default: 2)",
                        "default": 2
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Number of top entities per measure (default: 10)",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 50
                    }
                }
            }
        ),
        Tool(
            name="get_graph_statistics",
            description="Get statistical overview of the knowledge graph including node count, edge count, density, connected components, and degree distribution. Provides insight into the overall structure and complexity of the knowledge network.",
            inputSchema={
                "type": "object",
                "properties": {
                    "entity_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter graph to specific entity types (optional)"
                    },
                    "min_occurrences": {
                        "type": "integer",
                        "description": "Minimum entity occurrences (default: 2)",
                        "default": 2
                    }
                }
            }
        ),
        Tool(
            name="compute_graph_metrics",
            description="Compute comprehensive graph analysis metrics including PageRank (importance), betweenness centrality (bridge nodes), degree centrality (connections), and community detection. Returns detailed metrics for all entities and graph-level statistics. Metrics are stored in the database for later retrieval.",
            inputSchema={
                "type": "object",
                "properties": {
                    "entity_types": {
                        "type": "array",
                        "description": "Filter graph to specific entity types",
                        "items": {
                            "type": "string",
                            "enum": ["hardware", "memory_address", "instruction", "person", "company", "product", "concept"]
                        }
                    },
                    "min_occurrences": {
                        "type": "integer",
                        "description": "Minimum entity occurrences for graph building",
                        "default": 2,
                        "minimum": 1
                    },
                    "min_relationship_strength": {
                        "type": "number",
                        "description": "Minimum relationship strength",
                        "default": 0.3,
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "store_results": {
                        "type": "boolean",
                        "description": "Store computed metrics to database",
                        "default": True
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="get_entity_metrics",
            description="Retrieve stored graph metrics for a specific entity. Returns PageRank score, betweenness centrality, degree centrality, community ID, entity type, and computation timestamp. Useful for understanding an entity's importance and role in the knowledge graph.",
            inputSchema={
                "type": "object",
                "properties": {
                    "entity_text": {
                        "type": "string",
                        "description": "Entity to retrieve metrics for (e.g., 'VIC-II', 'SID', 'Commodore 64')"
                    },
                    "metric_types": {
                        "type": "array",
                        "description": "Specific metrics to retrieve. Leave empty for all metrics.",
                        "items": {
                            "type": "string",
                            "enum": ["pagerank", "betweenness", "degree", "community"]
                        }
                    }
                },
                "required": ["entity_text"]
            }
        ),
        Tool(
            name="find_entity_path",
            description="Find the shortest path between two entities in the knowledge graph. Returns the complete path, path length, and relationship details for each connection. Useful for discovering how concepts are related and understanding knowledge connections.",
            inputSchema={
                "type": "object",
                "properties": {
                    "entity1": {
                        "type": "string",
                        "description": "Source entity (e.g., 'STA')"
                    },
                    "entity2": {
                        "type": "string",
                        "description": "Target entity (e.g., 'VIC-II')"
                    },
                    "max_path_length": {
                        "type": "integer",
                        "description": "Maximum path length to search",
                        "default": 6,
                        "minimum": 1,
                        "maximum": 10
                    },
                    "store_result": {
                        "type": "boolean",
                        "description": "Store computed path to database",
                        "default": True
                    }
                },
                "required": ["entity1", "entity2"]
            }
        ),
        Tool(
            name="get_entity_community",
            description="Get all entities in the same community as the specified entity. Communities are groups of closely related entities detected through graph analysis. Returns community ID, member count, and list of community members with their types.",
            inputSchema={
                "type": "object",
                "properties": {
                    "entity_text": {
                        "type": "string",
                        "description": "Entity to find community for (e.g., 'SID')"
                    },
                    "max_members": {
                        "type": "integer",
                        "description": "Maximum number of community members to return",
                        "default": 50,
                        "minimum": 1
                    }
                },
                "required": ["entity_text"]
            }
        ),
        Tool(
            name="get_top_entities",
            description="Get top-ranked entities by a specific metric (PageRank, betweenness, or degree centrality). Returns ranked list of entities with their scores, types, and other metrics. Useful for discovering the most important, central, or well-connected entities in the knowledge graph.",
            inputSchema={
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "description": "Metric to rank by",
                        "enum": ["pagerank", "betweenness", "degree"],
                        "default": "pagerank"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of top entities to return",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 100
                    },
                    "entity_types": {
                        "type": "array",
                        "description": "Filter to specific entity types",
                        "items": {
                            "type": "string",
                            "enum": ["hardware", "memory_address", "instruction", "person", "company", "product", "concept"]
                        }
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="visualize_graph",
            description="Generate interactive HTML visualization of the knowledge graph using PyVis. Creates a beautiful network diagram with customizable node colors (by entity type or community), node sizes (by PageRank/betweenness/degree), and interactive physics simulation. Perfect for exploring entity relationships and discovering patterns in the knowledge base.",
            inputSchema={
                "type": "object",
                "properties": {
                    "output_path": {
                        "type": "string",
                        "description": "Output file path for HTML visualization (relative to data directory or absolute path)",
                        "default": "knowledge_graph.html"
                    },
                    "entity_types": {
                        "type": "array",
                        "description": "Filter to specific entity types",
                        "items": {
                            "type": "string",
                            "enum": ["hardware", "memory_address", "instruction", "person", "company", "product", "concept"]
                        }
                    },
                    "min_occurrences": {
                        "type": "integer",
                        "description": "Minimum entity occurrences to include",
                        "default": 2,
                        "minimum": 1
                    },
                    "min_relationship_strength": {
                        "type": "number",
                        "description": "Minimum relationship strength",
                        "default": 0.3,
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "color_by": {
                        "type": "string",
                        "description": "Node coloring scheme",
                        "enum": ["entity_type", "community", "uniform"],
                        "default": "entity_type"
                    },
                    "size_by": {
                        "type": "string",
                        "description": "Node sizing metric",
                        "enum": ["pagerank", "degree", "betweenness", "uniform"],
                        "default": "pagerank"
                    },
                    "physics_enabled": {
                        "type": "boolean",
                        "description": "Enable physics simulation for dynamic layout",
                        "default": True
                    },
                    "height": {
                        "type": "string",
                        "description": "Visualization height (CSS format)",
                        "default": "800px"
                    },
                    "width": {
                        "type": "string",
                        "description": "Visualization width (CSS format)",
                        "default": "100%"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="train_lda_topics",
            description="Train LDA topic model on all documents to discover latent topics. Uses Latent Dirichlet Allocation to find topics based on word co-occurrence patterns.",
            inputSchema={
                "type": "object",
                "properties": {
                    "num_topics": {
                        "type": "integer",
                        "description": "Number of topics to discover",
                        "default": 10,
                        "minimum": 2
                    },
                    "max_iter": {
                        "type": "integer",
                        "description": "Maximum training iterations",
                        "default": 100
                    },
                    "max_features": {
                        "type": "integer",
                        "description": "Maximum vocabulary size",
                        "default": 1000
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="train_nmf_topics",
            description="Train NMF topic model on all documents. Non-negative Matrix Factorization often produces more coherent topics than LDA and is faster.",
            inputSchema={
                "type": "object",
                "properties": {
                    "num_topics": {
                        "type": "integer",
                        "description": "Number of topics to discover",
                        "default": 10,
                        "minimum": 2
                    },
                    "max_iter": {
                        "type": "integer",
                        "description": "Maximum training iterations",
                        "default": 200
                    },
                    "max_features": {
                        "type": "integer",
                        "description": "Maximum vocabulary size",
                        "default": 1000
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="train_bertopic",
            description="Train BERTopic model using embeddings + UMAP + HDBSCAN. State-of-the-art transformer-based topic modeling that automatically discovers topics.",
            inputSchema={
                "type": "object",
                "properties": {
                    "num_topics": {
                        "type": "integer",
                        "description": "Target number of topics (actual may vary)",
                        "default": 10,
                        "minimum": 2
                    },
                    "min_topic_size": {
                        "type": "integer",
                        "description": "Minimum documents per topic",
                        "default": 5,
                        "minimum": 2
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="get_document_topics",
            description="Get topic assignments for a specific document or all documents. Shows which topics each document belongs to with probability scores.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_type": {
                        "type": "string",
                        "description": "Topic model type",
                        "enum": ["lda", "nmf", "bertopic"]
                    },
                    "doc_id": {
                        "type": "string",
                        "description": "Document ID (optional, omit to get all documents)"
                    },
                    "min_probability": {
                        "type": "number",
                        "description": "Minimum topic probability to include",
                        "default": 0.1,
                        "minimum": 0.0,
                        "maximum": 1.0
                    }
                },
                "required": ["model_type"]
            }
        ),
        Tool(
            name="cluster_documents_kmeans",
            description="Cluster documents using K-Means algorithm on embeddings. Partitions documents into K clusters based on similarity.",
            inputSchema={
                "type": "object",
                "properties": {
                    "num_clusters": {
                        "type": "integer",
                        "description": "Number of clusters (K)",
                        "default": 10,
                        "minimum": 2
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="cluster_documents_dbscan",
            description="Cluster documents using DBSCAN (density-based clustering). Automatically discovers clusters and identifies outliers without needing to specify number of clusters.",
            inputSchema={
                "type": "object",
                "properties": {
                    "eps": {
                        "type": "number",
                        "description": "Maximum distance between neighbors",
                        "default": 0.5,
                        "minimum": 0.0
                    },
                    "min_samples": {
                        "type": "integer",
                        "description": "Minimum samples in neighborhood",
                        "default": 3,
                        "minimum": 2
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="cluster_documents_hdbscan",
            description="Cluster documents using HDBSCAN (hierarchical density-based clustering). Improved version of DBSCAN with better handling of varying density clusters.",
            inputSchema={
                "type": "object",
                "properties": {
                    "min_cluster_size": {
                        "type": "integer",
                        "description": "Minimum cluster size",
                        "default": 5,
                        "minimum": 2
                    },
                    "min_samples": {
                        "type": "integer",
                        "description": "Minimum samples in neighborhood",
                        "default": 3,
                        "minimum": 1
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="get_cluster_documents",
            description="Get documents in a specific cluster or all clusters. Shows cluster membership, distances, and cluster characteristics.",
            inputSchema={
                "type": "object",
                "properties": {
                    "algorithm": {
                        "type": "string",
                        "description": "Clustering algorithm",
                        "enum": ["kmeans", "dbscan", "hdbscan"]
                    },
                    "cluster_number": {
                        "type": "integer",
                        "description": "Cluster number (optional, omit to get all clusters)"
                    }
                },
                "required": ["algorithm"]
            }
        ),
        # Phase 2: Visualization Tools
        Tool(
            name="generate_topic_wordcloud",
            description="Generate word cloud visualization for a topic. Creates an image file showing the most important words in a topic with size proportional to their weights.",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic_id": {
                        "type": "string",
                        "description": "Topic ID to visualize"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to save the word cloud image (PNG)"
                    },
                    "width": {
                        "type": "integer",
                        "description": "Image width in pixels (default: 800)",
                        "default": 800
                    },
                    "height": {
                        "type": "integer",
                        "description": "Image height in pixels (default: 400)",
                        "default": 400
                    },
                    "background_color": {
                        "type": "string",
                        "description": "Background color (default: 'white')",
                        "default": "white"
                    }
                },
                "required": ["topic_id", "output_path"]
            }
        ),
        Tool(
            name="visualize_cluster_scatter",
            description="Generate 2D scatter plot of document clusters using UMAP projection. Shows how documents are distributed across clusters in 2D space.",
            inputSchema={
                "type": "object",
                "properties": {
                    "algorithm": {
                        "type": "string",
                        "description": "Clustering algorithm",
                        "enum": ["kmeans", "dbscan", "hdbscan"]
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to save the scatter plot image (PNG)"
                    },
                    "width": {
                        "type": "integer",
                        "description": "Image width in pixels (default: 1200)",
                        "default": 1200
                    },
                    "height": {
                        "type": "integer",
                        "description": "Image height in pixels (default: 800)",
                        "default": 800
                    },
                    "n_neighbors": {
                        "type": "integer",
                        "description": "UMAP n_neighbors parameter (default: 15)",
                        "default": 15
                    },
                    "min_dist": {
                        "type": "number",
                        "description": "UMAP min_dist parameter (default: 0.1)",
                        "default": 0.1
                    }
                },
                "required": ["algorithm", "output_path"]
            }
        ),
        Tool(
            name="generate_topic_heatmap",
            description="Generate heatmap showing document-topic probability matrix. Visualizes which topics are most prominent in which documents.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_type": {
                        "type": "string",
                        "description": "Topic model type",
                        "enum": ["lda", "nmf", "bertopic"]
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to save the heatmap image (PNG)"
                    },
                    "max_topics": {
                        "type": "integer",
                        "description": "Maximum number of topics to include (default: 20)",
                        "default": 20
                    },
                    "max_documents": {
                        "type": "integer",
                        "description": "Maximum number of documents to include (default: 50)",
                        "default": 50
                    }
                },
                "required": ["model_type", "output_path"]
            }
        ),
        Tool(
            name="visualize_cluster_distribution",
            description="Generate bar chart showing cluster size distribution. Shows how many documents are in each cluster.",
            inputSchema={
                "type": "object",
                "properties": {
                    "algorithm": {
                        "type": "string",
                        "description": "Clustering algorithm",
                        "enum": ["kmeans", "dbscan", "hdbscan"]
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to save the bar chart image (PNG)"
                    },
                    "width": {
                        "type": "integer",
                        "description": "Image width in pixels (default: 1000)",
                        "default": 1000
                    },
                    "height": {
                        "type": "integer",
                        "description": "Image height in pixels (default: 600)",
                        "default": 600
                    }
                },
                "required": ["algorithm", "output_path"]
            }
        ),
        # Phase 3: Temporal Analysis Tools
        Tool(
            name="extract_document_events",
            description="Extract temporal events from a document (product releases, company milestones, technical innovations, cultural events). Detects event patterns and dates, then stores to database.",
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "Document ID to extract events from"
                    },
                    "min_confidence": {
                        "type": "number",
                        "description": "Minimum confidence threshold (0.0-1.0)",
                        "default": 0.5,
                        "minimum": 0.0,
                        "maximum": 1.0
                    }
                },
                "required": ["doc_id"]
            }
        ),
        Tool(
            name="get_timeline",
            description="Get chronological timeline of events with optional filtering. Returns timeline entries sorted by date with event details.",
            inputSchema={
                "type": "object",
                "properties": {
                    "start_year": {
                        "type": "integer",
                        "description": "Filter events from this year onwards (inclusive)"
                    },
                    "end_year": {
                        "type": "integer",
                        "description": "Filter events up to this year (inclusive)"
                    },
                    "category": {
                        "type": "string",
                        "description": "Filter by category (e.g., '1980s-release', '1970s-innovation')"
                    },
                    "min_importance": {
                        "type": "integer",
                        "description": "Minimum importance level (1-5, where 5 is highest)",
                        "default": 1,
                        "minimum": 1,
                        "maximum": 5
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of entries to return"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="search_events_by_date",
            description="Search for events within a date range. Filter by year range, event type (release, milestone, innovation, cultural, update), and confidence.",
            inputSchema={
                "type": "object",
                "properties": {
                    "start_year": {
                        "type": "integer",
                        "description": "Start year (inclusive)"
                    },
                    "end_year": {
                        "type": "integer",
                        "description": "End year (inclusive)"
                    },
                    "event_type": {
                        "type": "string",
                        "description": "Event type filter",
                        "enum": ["release", "milestone", "innovation", "cultural", "update"]
                    },
                    "min_confidence": {
                        "type": "number",
                        "description": "Minimum confidence threshold",
                        "default": 0.5,
                        "minimum": 0.0,
                        "maximum": 1.0
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="get_historical_context",
            description="Get historical context for a specific year. Returns events from the target year plus surrounding years to provide temporal context.",
            inputSchema={
                "type": "object",
                "properties": {
                    "year": {
                        "type": "integer",
                        "description": "Target year to get context for"
                    },
                    "context_years": {
                        "type": "integer",
                        "description": "Number of years before/after to include",
                        "default": 2,
                        "minimum": 0,
                        "maximum": 10
                    }
                },
                "required": ["year"]
            }
        ),
    ]
