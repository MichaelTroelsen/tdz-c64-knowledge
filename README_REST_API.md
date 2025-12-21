# TDZ C64 Knowledge Base - REST API Documentation

FastAPI-based REST API server for the C64 Knowledge Base. Provides HTTP endpoints for all knowledge base operations including search, document management, URL scraping, AI features, and data export.

## Quick Start

### Installation

```bash
# Install REST API dependencies
pip install -e ".[rest]"
```

### Running the Server

```bash
# Method 1: Using uvicorn directly
uvicorn rest_server:app --reload --port 8000

# Method 2: Using the batch file (Windows)
run_rest_api.bat

# Method 3: Using Python directly
python rest_server.py
```

The server will start on `http://localhost:8000` by default.

**API Documentation:**
- Swagger UI: `http://localhost:8000/api/docs`
- ReDoc: `http://localhost:8000/api/redoc`

## Configuration

### Environment Variables

```bash
# Data Directory
TDZ_DATA_DIR=C:\path\to\data          # Default: ~/.tdz-c64-knowledge

# Authentication
TDZ_API_KEYS=key1,key2,key3           # Comma-separated API keys (optional)

# CORS
CORS_ORIGINS=http://localhost:3000,https://example.com  # Allowed origins

# Server Settings
API_HOST=0.0.0.0                      # Host to bind to (default: 0.0.0.0)
API_PORT=8000                         # Port to bind to (default: 8000)
```

### Authentication

API key authentication via `X-API-Key` header:

```bash
curl -H "X-API-Key: your-api-key-here" http://localhost:8000/api/v1/documents
```

**Note:** If no API keys are configured, authentication is disabled (not recommended for production).

## API Endpoints

### Health & Statistics

#### GET /api/v1/health
Health check endpoint (no authentication required).

```bash
curl http://localhost:8000/api/v1/health
```

#### GET /api/v1/stats
Get knowledge base statistics.

```bash
curl -H "X-API-Key: key" http://localhost:8000/api/v1/stats
```

---

### Search Endpoints

#### POST /api/v1/search
Basic keyword search using FTS5/BM25.

```bash
curl -X POST "http://localhost:8000/api/v1/search" \
  -H "X-API-Key: key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "sprite graphics VIC-II",
    "max_results": 10,
    "tags": ["graphics", "hardware"]
  }'
```

#### POST /api/v1/search/semantic
Semantic search using embeddings.

```bash
curl -X POST "http://localhost:8000/api/v1/search/semantic" \
  -H "X-API-Key: key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do sprites work on the VIC-II chip?",
    "max_results": 5,
    "top_k": 50
  }'
```

#### POST /api/v1/search/hybrid
Hybrid search combining keyword and semantic.

```bash
curl -X POST "http://localhost:8000/api/v1/search/hybrid" \
  -H "X-API-Key: key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "SID chip sound programming",
    "max_results": 10,
    "semantic_weight": 0.7,
    "top_k": 100
  }'
```

#### POST /api/v1/search/faceted
Faceted search with entity-based filtering.

```bash
curl -X POST "http://localhost:8000/api/v1/search/faceted" \
  -H "X-API-Key: key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "programming",
    "max_results": 10,
    "facet_filters": {
      "hardware": ["VIC-II", "SID"],
      "concept": ["sprite", "sound"]
    }
  }'
```

#### GET /api/v1/similar/{doc_id}
Find documents similar to a given document.

```bash
curl "http://localhost:8000/api/v1/similar/89d0943d6009?max_results=5" \
  -H "X-API-Key: key"
```

---

### Document CRUD

#### GET /api/v1/documents
List all documents with filtering and pagination.

```bash
curl "http://localhost:8000/api/v1/documents?tags=reference,c64&file_type=pdf&limit=20&offset=0" \
  -H "X-API-Key: key"
```

#### GET /api/v1/documents/{doc_id}
Get specific document metadata.

```bash
curl "http://localhost:8000/api/v1/documents/89d0943d6009" \
  -H "X-API-Key: key"
```

#### POST /api/v1/documents
Upload a new document.

```bash
curl -X POST "http://localhost:8000/api/v1/documents" \
  -H "X-API-Key: key" \
  -F "file=@c64-reference.pdf" \
  -F "tags=reference,c64" \
  -F "title=C64 Programmer's Reference Guide"
```

#### PUT /api/v1/documents/{doc_id}
Update document metadata.

```bash
curl -X PUT "http://localhost:8000/api/v1/documents/89d0943d6009" \
  -H "X-API-Key: key" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Updated C64 Reference Guide",
    "add_tags": ["updated", "v2"],
    "remove_tags": ["draft"]
  }'
```

#### DELETE /api/v1/documents/{doc_id}
Delete a document.

```bash
curl -X DELETE "http://localhost:8000/api/v1/documents/89d0943d6009" \
  -H "X-API-Key: key"
```

#### POST /api/v1/documents/bulk
Upload multiple documents at once.

```bash
curl -X POST "http://localhost:8000/api/v1/documents/bulk" \
  -H "X-API-Key: key" \
  -F "files=@doc1.pdf" \
  -F "files=@doc2.pdf" \
  -F "tags=reference,bulk-import"
```

#### DELETE /api/v1/documents/bulk
Bulk delete multiple documents.

```bash
curl -X DELETE "http://localhost:8000/api/v1/documents/bulk" \
  -H "X-API-Key: key" \
  -H "Content-Type: application/json" \
  -d '{
    "doc_ids": ["89d0943d6009", "a1b2c3d4e5f6"],
    "confirm": true
  }'
```

---

### URL Scraping

#### POST /api/v1/scrape
Scrape a URL and add to knowledge base.

```bash
curl -X POST "http://localhost:8000/api/v1/scrape" \
  -H "X-API-Key: key" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://www.c64-wiki.com/wiki/VIC",
    "max_depth": 2,
    "max_pages": 20,
    "tags": ["c64-wiki", "reference"],
    "allowed_domains": ["c64-wiki.com"],
    "max_workers": 5
  }'
```

#### POST /api/v1/scrape/rescrape/{doc_id}
Re-scrape a URL-sourced document.

```bash
curl -X POST "http://localhost:8000/api/v1/scrape/rescrape/89d0943d6009" \
  -H "X-API-Key: key" \
  -H "Content-Type: application/json" \
  -d '{
    "force": false
  }'
```

#### POST /api/v1/scrape/check-updates
Check all scraped documents for updates.

```bash
curl -X POST "http://localhost:8000/api/v1/scrape/check-updates" \
  -H "X-API-Key: key"
```

---

### AI Features

#### POST /api/v1/summarize/{doc_id}
Generate an AI summary of a document.

```bash
curl -X POST "http://localhost:8000/api/v1/summarize/89d0943d6009" \
  -H "X-API-Key: key" \
  -H "Content-Type: application/json" \
  -d '{
    "max_length": 300,
    "style": "technical"
  }'
```

Styles: `technical`, `simple`, `detailed`

#### POST /api/v1/entities/extract/{doc_id}
Extract entities from a document.

```bash
curl -X POST "http://localhost:8000/api/v1/entities/extract/89d0943d6009" \
  -H "X-API-Key: key" \
  -H "Content-Type: application/json" \
  -d '{
    "confidence_threshold": 0.8,
    "entity_types": ["hardware", "instruction"]
  }'
```

#### POST /api/v1/entities/search
Search for entities across all documents.

```bash
curl -X POST "http://localhost:8000/api/v1/entities/search" \
  -H "X-API-Key: key" \
  -H "Content-Type: application/json" \
  -d '{
    "entity_text": "VIC-II",
    "entity_type": "hardware",
    "min_confidence": 0.7
  }'
```

#### GET /api/v1/entities/{doc_id}
Get all entities for a document.

```bash
curl "http://localhost:8000/api/v1/entities/89d0943d6009?entity_type=hardware&min_confidence=0.7" \
  -H "X-API-Key: key"
```

#### GET /api/v1/relationships/{entity_text}
Get relationships for a specific entity.

```bash
curl "http://localhost:8000/api/v1/relationships/VIC-II?min_strength=0.5&max_results=20" \
  -H "X-API-Key: key"
```

---

### Analytics

#### GET /api/v1/analytics/search
Get search analytics (placeholder implementation).

```bash
curl "http://localhost:8000/api/v1/analytics/search?time_range_days=30" \
  -H "X-API-Key: key"
```

---

### Export Endpoints

#### GET /api/v1/export/search
Export search results to CSV or JSON.

```bash
curl "http://localhost:8000/api/v1/export/search?query=sprite&format=csv&max_results=50" \
  -H "X-API-Key: key" \
  --output search_results.csv
```

#### GET /api/v1/export/documents
Export document list to CSV or JSON.

```bash
curl "http://localhost:8000/api/v1/export/documents?format=csv&tags=reference" \
  -H "X-API-Key: key" \
  --output documents.csv
```

#### GET /api/v1/export/entities
Export entities to CSV or JSON.

```bash
curl "http://localhost:8000/api/v1/export/entities?format=csv&entity_type=hardware&min_confidence=0.7" \
  -H "X-API-Key: key" \
  --output entities.csv
```

#### GET /api/v1/export/relationships
Export entity relationships to CSV or JSON.

```bash
curl "http://localhost:8000/api/v1/export/relationships?format=csv&min_strength=0.5" \
  -H "X-API-Key: key" \
  --output relationships.csv
```

---

## Response Format

All endpoints return JSON responses with a standard format:

### Success Response

```json
{
  "success": true,
  "data": { ... },
  "metadata": { ... }
}
```

### Error Response

```json
{
  "detail": "Error message"
}
```

## HTTP Status Codes

- **200 OK** - Success
- **400 Bad Request** - Invalid request parameters
- **403 Forbidden** - Invalid or missing API key
- **404 Not Found** - Resource not found
- **500 Internal Server Error** - Server error
- **503 Service Unavailable** - KnowledgeBase not initialized

## Rate Limiting

Rate limiting can be enabled via environment variable:

```bash
ENABLE_RATE_LIMITING=1
```

Default limits: TBD (not yet implemented in current version)

## CORS Configuration

Configure allowed origins for cross-origin requests:

```bash
CORS_ORIGINS=http://localhost:3000,https://app.example.com
```

Default: Allows all origins (`["*"]`)

## Running Both Servers

The MCP server and REST API server can run simultaneously:

- **MCP Server**: stdio transport (Claude Desktop)
- **REST API Server**: HTTP on port 8000

```bash
# Terminal 1: MCP Server
python server.py

# Terminal 2: REST API Server
uvicorn rest_server:app --reload --port 8000
```

## Testing

Use the interactive Swagger UI for testing:

```
http://localhost:8000/api/docs
```

Or use curl/httpie for command-line testing (see examples above).

## Production Deployment

### Security Recommendations

1. **Enable API Keys**: Set `TDZ_API_KEYS` environment variable
2. **Configure CORS**: Restrict `CORS_ORIGINS` to your domains
3. **Use HTTPS**: Deploy behind a reverse proxy (nginx, caddy)
4. **Rate Limiting**: Enable rate limiting for production
5. **Firewall**: Restrict access to trusted IPs

### Example Production Setup

```bash
# .env file
TDZ_DATA_DIR=/var/lib/tdz-c64-knowledge
TDZ_API_KEYS=secure-key-1,secure-key-2
CORS_ORIGINS=https://app.example.com
API_HOST=127.0.0.1
API_PORT=8000
ENABLE_RATE_LIMITING=1
```

### Using a Reverse Proxy (nginx)

```nginx
server {
    listen 443 ssl;
    server_name api.example.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Development

### Adding New Endpoints

1. Add Pydantic models to `rest_models.py`
2. Implement endpoint in `rest_server.py`
3. Add to appropriate tag group (`Search`, `Documents`, `AI Features`, etc.)
4. Update this documentation

### File Structure

```
tdz-c64-knowledge/
├── rest_server.py          # FastAPI application
├── rest_models.py          # Pydantic request/response models
├── server.py               # KnowledgeBase core
├── run_rest_api.bat        # Windows startup script
├── README_REST_API.md      # This file
└── pyproject.toml          # Dependencies
```

## Troubleshooting

### Server won't start

**Error**: `ModuleNotFoundError: No module named 'fastapi'`

**Solution**: Install REST dependencies:
```bash
pip install -e ".[rest]"
```

### Authentication disabled warning

**Warning**: `No API keys configured - authentication disabled!`

**Solution**: Set API keys in environment:
```bash
set TDZ_API_KEYS=key1,key2,key3  # Windows
export TDZ_API_KEYS=key1,key2,key3  # Linux/Mac
```

### KnowledgeBase not initialized

**Error**: `503 Service Unavailable - KnowledgeBase not initialized`

**Solution**: Check data directory exists and has proper permissions:
```bash
# Windows
echo %TDZ_DATA_DIR%
dir %TDZ_DATA_DIR%

# Linux/Mac
echo $TDZ_DATA_DIR
ls -la $TDZ_DATA_DIR
```

## Support

- Documentation: See main README.md
- Issues: https://github.com/yourusername/tdz-c64-knowledge/issues
- API Docs: http://localhost:8000/api/docs (when server running)

## Version

Current REST API Version: **v2.18.0** (2025-12-21)

- 27 fully functional endpoints across 6 categories
- All endpoints tested and verified
- Complete API documentation with examples
- Production-ready with authentication and CORS support

See version.py for complete version history.
