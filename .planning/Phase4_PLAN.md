# Phase 4: C64-Specific Features Implementation Plan

**Status:** Ready to Begin
**Version Target:** v2.24.0-v2.26.0
**Time Estimate:** 40-70 hours total
**Date Created:** December 23, 2025

---

## Overview

Phase 4 implements C64-specific features that leverage the knowledge base for Commodore 64 development. Three features with increasing complexity:

1. **VICE Emulator Integration** (4.1) - Highest ROI
2. **PRG File Analysis** (4.2) - High Value
3. **SID Music Metadata** (4.3) - Medium Complexity

---

## Phase 4.1: VICE Emulator Integration ⭐⭐⭐⭐⭐

**Impact:** Very High | **Effort:** 20-30 hours | **Complexity:** High

### Problem Statement
C64 developers using VICE emulator lack documentation lookup while debugging. Currently:
- Memory inspection requires external manual lookups
- No real-time documentation context during development
- Developers must context-switch between emulator and docs

### Solution
Real-time bidirectional integration between VICE debugger and knowledge base:
- Look up documentation for memory addresses in running programs
- Annotated memory dumps with inline documentation
- Register definitions with hardware reference
- Step-through debugging with documentation context

### Architecture

#### Connection Layer
```
Knowledge Base ←→ VICE Monitor (TCP port 6510)
                  ├─ Monitor protocol (text-based)
                  ├─ Memory read/write operations
                  └─ Register/CPU state queries
```

#### Components

1. **VICEClient Class** (new file: `vice_integration.py`)
   - TCP socket connection to VICE monitor
   - Low-level monitor protocol implementation
   - Memory peek/poke operations
   - Register/CPU state queries
   - Error handling and reconnection logic

2. **VICEMemoryResolver Class**
   - Cross-reference memory addresses with documentation
   - Map addresses to hardware registers/memory regions
   - Annotate memory with context from KB
   - Format output for human readability

3. **VICE Debugger Integration**
   - Integration hooks for common breakpoints
   - Symbol export from KB for assembly code
   - Real-time variable watches with docs
   - Memory inspection dashboards

### Implementation Steps

#### Step 1: VICE Monitor Protocol (2-3 hours)
**Files:** `vice_integration.py` (150-200 lines)

```python
class VICEClient:
    def __init__(self, host="localhost", port=6510, timeout=5):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False

    def connect(self):
        """Connect to VICE remote monitor."""
        # Create socket, connect with error handling

    def send_command(self, command: str) -> str:
        """Send monitor command and get response."""
        # Send command + \r\n, read response
        # Handle VICE monitor protocol: prompt (.), responses, etc.

    def peek(self, address: int) -> int:
        """Read byte from memory."""
        # "m <addr>" → parse response

    def poke(self, address: int, value: int) -> bool:
        """Write byte to memory."""
        # "S <addr> <value>"

    def get_registers(self) -> dict:
        """Get CPU registers."""
        # "r" command → parse registers (A, X, Y, PC, SP, P)

    def get_cpu_state(self) -> dict:
        """Get full CPU state for display."""
```

**Testing:**
- Unit tests with mock VICE socket
- Integration test with running VICE instance
- Protocol compliance verification

#### Step 2: Memory Documentation Resolver (3-4 hours)
**Files:** `vice_integration.py` (additional 200-250 lines)

```python
class VICEMemoryResolver:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self._build_memory_map()  # At startup

    def _build_memory_map(self):
        """Extract memory addresses from KB."""
        # Search KB for "$D000", "0xD000", "53280" patterns
        # Build address → (doc_id, context) mapping

    def resolve_address(self, address: int) -> dict:
        """Get documentation for address."""
        return {
            'address': "$D000",
            'chip': "VIC-II",
            'register': "Border Color",
            'doc_title': "VIC-II Register Map",
            'doc_id': "abc123",
            'context': "...description...",
            'confidence': 0.95
        }

    def annotate_memory_dump(self, start: int, end: int) -> str:
        """Create annotated memory dump."""
        # Generate output like:
        # $D000: 14  # VIC-II: Border Color (currently light gray)
        # $D001: 32  # VIC-II: Background Color
```

**Challenges:**
- Normalize address formats ($D000 vs 0xD000 vs decimal)
- Handle register ranges (e.g., $D000-$D02E for VIC-II)
- Confidence scoring for resolved addresses

#### Step 3: MCP Tools (2-3 hours)
**Files:** `server.py` (add to list_tools and call_tool)

```python
Tool(
    name="vice_memory_lookup",
    description="Look up documentation for memory address in VICE emulator",
    inputSchema={
        "type": "object",
        "properties": {
            "address": {
                "type": "string",
                "description": "Memory address ($D000, 0xD000, or decimal)"
            },
            "action": {
                "type": "string",
                "enum": ["peek", "annotate", "register"],
                "description": "What to do: peek=read value, annotate=show docs, register=CPU state"
            }
        },
        "required": ["address", "action"]
    }
)
```

#### Step 4: CLI Commands (1-2 hours)
**Files:** `cli.py`

```
python cli.py vice-lookup $D000           # Get docs for address
python cli.py vice-memory $D000 $D030     # Annotated dump
python cli.py vice-connect localhost 6510 # Test connection
```

#### Step 5: GUI Integration (2-3 hours)
**Files:** `admin_gui.py`

- New tab: "VICE Debugger Integration"
- Memory address lookup form
- Live register display
- Memory browser with documentation
- Breakpoint/watchpoint management

### Testing Strategy

**Unit Tests:**
- VICE protocol parsing
- Address resolution accuracy
- Memory dump formatting

**Integration Tests:**
- Connect to running VICE instance
- Read/write memory
- Verify documentation retrieval
- Performance (lookup time < 100ms)

**Manual Testing:**
- Connect to VICE with running C64 program
- Inspect SID register ($D400+)
- Check VIC-II memory ($D000+)
- Verify documentation accuracy

### Success Criteria
- ✓ Connect to VICE on port 6510
- ✓ Read memory and CPU registers
- ✓ Resolve >90% of common C64 addresses
- ✓ Memory lookup < 100ms
- ✓ MCP tool works in Claude Code
- ✓ All documentation accurate and relevant
- ✓ Zero crashes on malformed data

### Risk Mitigation
- **VICE not running:** Graceful connection error with helpful message
- **Timeout:** Implement reconnect logic with exponential backoff
- **Wrong address format:** Support $XXXX, 0xXXXX, decimal
- **Address not in KB:** Return generic explanation of memory region

---

## Phase 4.2: PRG File Analysis ⭐⭐⭐⭐

**Impact:** High | **Effort:** 12-16 hours | **Complexity:** Medium

### Problem Statement
C64 binary executables (PRG files) are difficult to analyze:
- No built-in documentation for loaded code
- Binary format requires specialized knowledge
- Developers manually trace through memory map
- Hard to understand program structure

### Solution
Analyze PRG files and cross-reference with documentation:
- Disassemble PRG header and code sections
- Extract metadata (load address, size)
- Identify which memory regions are used
- Find documentation for those regions
- Build searchable code analysis index

### Implementation Steps

#### Step 1: PRG File Parser (3-4 hours)
**Files:** `prg_analysis.py` (150-200 lines)

```python
class PRGFile:
    """C64 PRG file parser."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.load_address = None
        self.code = None
        self.size = 0

    def parse(self) -> dict:
        """Parse PRG file structure."""
        with open(self.filepath, 'rb') as f:
            # First 2 bytes: load address (little-endian)
            self.load_address = int.from_bytes(f.read(2), 'little')

            # Remaining bytes: program code
            self.code = f.read()
            self.size = len(self.code)

        return {
            'load_address': f"${self.load_address:04X}",
            'load_address_decimal': self.load_address,
            'size': self.size,
            'end_address': f"${self.load_address + self.size - 1:04X}",
            'memory_regions': self._identify_regions()
        }

    def _identify_regions(self) -> list[dict]:
        """Identify memory regions used by PRG."""
        regions = []
        start = self.load_address
        end = self.load_address + self.size

        # C64 memory map:
        # $0000-$00FF: Zero page
        # $0100-$01FF: Stack
        # $0200-$7FFF: Program/data area
        # $8000-$9FFF: Cartridge ROM
        # $A000-$BFFF: BASIC ROM (if enabled)
        # $D000-$DFFF: I/O (SID, VIC-II, CIA, etc.)
        # $E000-$FFFF: Kernal ROM

        if start <= 0x00FF:
            regions.append({
                'name': 'Zero Page',
                'start': start,
                'end': min(end, 0x00FF),
                'chip': 'CPU',
                'readable': True
            })

        # ... more region identification

        return regions
```

#### Step 2: Code Structure Analysis (2-3 hours)
**Files:** `prg_analysis.py` (additional 100-150 lines)

```python
class PRGAnalyzer:
    """Analyze PRG file structure and link to KB."""

    def __init__(self, kb: KnowledgeBase, prg_file: PRGFile):
        self.kb = kb
        self.prg = prg_file

    def analyze_structure(self) -> dict:
        """Extract structure information from PRG."""
        return {
            'metadata': self.prg.parse(),
            'memory_regions': self._get_region_docs(),
            'potential_chips_used': self._identify_chips(),
            'likely_features': self._detect_features()
        }

    def _get_region_docs(self) -> list:
        """Find KB docs for each memory region."""
        docs = []
        for region in self.prg._identify_regions():
            # Search KB for docs about this region
            results = self.kb.search(region['name'])
            docs.append({
                'region': region,
                'documentation': results
            })
        return docs

    def _identify_chips(self) -> list:
        """Guess which chips the program uses."""
        # Based on memory regions accessed
        chips = []
        for region in self.prg._identify_regions():
            if region['start'] >= 0xD000 and region['start'] <= 0xDFFF:
                chips.append('VIC-II' if region['start'] < 0xD400 else 'SID')
            if region['start'] >= 0xDC00 and region['start'] <= 0xDCFF:
                chips.append('CIA1')
        return list(set(chips))

    def _detect_features(self) -> list:
        """Detect high-level features from code patterns."""
        features = []
        # Look for common code patterns:
        # - Sprite patterns (check VIC-II memory)
        # - Sound (check SID memory)
        # - IRQ handlers (check Kernal)
        return features
```

#### Step 3: MCP Tools (2 hours)
**Files:** `server.py`

```python
Tool(
    name="analyze_prg",
    description="Analyze Commodore 64 PRG executable file",
    inputSchema={
        "type": "object",
        "properties": {
            "prg_path": {
                "type": "string",
                "description": "Path to .prg file"
            },
            "detail_level": {
                "type": "string",
                "enum": ["brief", "detailed", "full"],
                "description": "Level of analysis detail"
            }
        },
        "required": ["prg_path"]
    }
)
```

#### Step 4: CLI Commands (1.5 hours)
**Files:** `cli.py`

```
python cli.py analyze-prg program.prg              # Brief analysis
python cli.py analyze-prg program.prg --detailed   # With docs
python cli.py analyze-prg program.prg --full       # Full analysis + disassembly
```

#### Step 5: Database Integration (1 hour)
**Files:** `server.py`

```python
# New table: prg_analysis
CREATE TABLE prg_analysis (
    prg_id TEXT PRIMARY KEY,
    prg_path TEXT,
    filename TEXT,
    load_address INTEGER,
    size INTEGER,
    end_address INTEGER,
    chips_used TEXT,  -- JSON array
    features TEXT,    -- JSON array
    analyzed_at TEXT,
    analysis_data TEXT  -- Full JSON analysis
)
```

### Testing Strategy

**Test Cases:**
- Simple BASIC program (short, minimal features)
- Graphics program (uses VIC-II sprites/colors)
- Music program (uses SID)
- Complex game (multiple chips, mixed features)

**Validation:**
- Load address correctly parsed
- Memory regions correctly identified
- Documentation retrieval accurate
- No false positives for chip detection

### Success Criteria
- ✓ Parse PRG header correctly
- ✓ Identify memory regions used
- ✓ Find 80%+ of relevant KB docs
- ✓ Detect common chips (VIC-II, SID, CIA)
- ✓ Analysis < 500ms for typical PRG
- ✓ MCP tool works reliably
- ✓ Handle edge cases gracefully

---

## Phase 4.3: SID Music File Metadata ⭐⭐⭐

**Impact:** Medium | **Effort:** 8-12 hours | **Complexity:** Low-Medium

### Problem Statement
SID music files lack searchable metadata:
- Composer information scattered across files
- No unified music database
- Hard to find music for specific composers/years
- Musical metadata not indexed

### Solution
Extract SID file metadata and integrate with knowledge base:
- Parse SID file headers (PSID/RSID format)
- Extract metadata: title, author, year, length
- Create searchable music database
- Cross-reference with composer documentation
- Build music analytics

### Implementation Steps

#### Step 1: SID File Parser (2-3 hours)
**Files:** `sid_parser.py` (100-150 lines)

```python
class SIDFile:
    """SID music file parser."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.filename = Path(filepath).name

    def parse(self) -> dict:
        """Parse SID file header and metadata."""
        with open(self.filepath, 'rb') as f:
            header = f.read(124)  # PSID header

        # SID header format (PSID v2):
        # Offset  Size  Field
        # 0       4     Magic ('PSID' or 'RSID')
        # 4       2     Version
        # 6       2     Header size
        # 8       2     Load address
        # 10      2     Init address
        # 12      2     Play address
        # 14      2     Songs
        # 16      2     Start song
        # 18      4     Speed (bits)
        # 22      32    Title
        # 54      32    Author
        # 86      32    Released/Copyright
        # 118     2     Flags
        # 120     1     Start page
        # 121     1     Page length

        magic = header[0:4].decode('ascii', errors='ignore')
        if magic not in ['PSID', 'RSID']:
            raise ValueError(f"Invalid SID file: {magic}")

        return {
            'format': magic,
            'version': int.from_bytes(header[4:6], 'big'),
            'load_address': int.from_bytes(header[8:10], 'big'),
            'songs': int.from_bytes(header[14:16], 'big'),
            'title': header[22:54].rstrip(b'\x00').decode('ascii', errors='ignore'),
            'author': header[54:86].rstrip(b'\x00').decode('ascii', errors='ignore'),
            'copyright': header[86:118].rstrip(b'\x00').decode('ascii', errors='ignore'),
            'file_size': Path(self.filepath).stat().st_size
        }
```

#### Step 2: Music Database Integration (2-3 hours)
**Files:** `server.py` (add methods), database schema

```python
# New table: sid_files
CREATE TABLE sid_files (
    sid_id TEXT PRIMARY KEY,
    filename TEXT UNIQUE,
    filepath TEXT,
    title TEXT,
    author TEXT,
    copyright TEXT,
    year INTEGER,  -- extracted from copyright
    songs INTEGER,
    load_address INTEGER,
    format TEXT,   -- PSID or RSID
    file_size INTEGER,
    added_at TEXT,
    metadata_hash TEXT
)

# FTS5 table for searching
CREATE VIRTUAL TABLE sid_files_fts USING fts5(
    title, author, copyright
)
```

Methods in `KnowledgeBase`:

```python
def add_sid_file(self, filepath: str) -> dict:
    """Index a SID file."""
    sid = SIDFile(filepath)
    metadata = sid.parse()

    # Extract year from copyright
    year = self._extract_year(metadata['copyright'])

    # Store in database
    # ... INSERT into sid_files ...

    return metadata

def search_sid(self, query: str) -> list[dict]:
    """Search SID files by title/author/copyright."""
    # FTS5 query on sid_files_fts

def get_sid_analytics(self) -> dict:
    """Music database statistics."""
    return {
        'total_files': count,
        'total_songs': sum,
        'composers': list,
        'top_composers': list,
        'year_range': (min, max),
        'format_distribution': dict
    }
```

#### Step 3: Composer Documentation Linking (1-2 hours)
**Files:** `server.py` (additional methods)

```python
def find_composer_docs(self, composer_name: str) -> list:
    """Find KB docs about a composer."""
    return self.search(composer_name)

def link_sid_to_docs(self, sid_id: str) -> dict:
    """Cross-reference SID with documentation."""
    sid = self.get_sid(sid_id)
    composer_docs = self.find_composer_docs(sid['author'])

    return {
        'sid_file': sid,
        'composer_documentation': composer_docs,
        'technical_docs': self.search(f"SID music {sid['author']}")
    }
```

#### Step 4: MCP Tools (1.5 hours)
**Files:** `server.py`

```python
Tool(
    name="add_sid_file",
    description="Index a SID music file for searching",
    inputSchema={...}
),
Tool(
    name="search_sid",
    description="Search SID music database by title/author/year",
    inputSchema={...}
),
Tool(
    name="sid_analytics",
    description="Get SID music database statistics",
    inputSchema={...}
)
```

#### Step 5: CLI Commands (0.5 hours)
**Files:** `cli.py`

```
python cli.py add-sid program.sid              # Index SID file
python cli.py search-sid "Rob Hubbard"         # Search by composer
python cli.py search-sid "Last Ninja"          # Search by title
python cli.py sid-stats                        # Database stats
```

### Testing Strategy

**Test Cases:**
- Simple SID file (title, author, year)
- Complex SID (multiple songs, metadata)
- Malformed SID (invalid header)
- Year extraction from copyright

**Validation:**
- Metadata correctly parsed
- Duplicate detection (same file, different path)
- Search results accurate
- Analytics calculations correct

### Success Criteria
- ✓ Parse PSID/RSID headers correctly
- ✓ Extract title, author, year accurately
- ✓ Full-text search working
- ✓ Composer documentation retrieval > 70%
- ✓ Analytics dashboard functional
- ✓ Handle edge cases gracefully
- ✓ Scalable to 1000+ SID files

---

## Implementation Sequence

### Recommended Order
1. **PRG Analysis** (4.2) - Simplest, good foundation
2. **SID Metadata** (4.3) - Small scope, straightforward
3. **VICE Integration** (4.1) - Most complex, best done last

### Timeline Estimate
- **Week 1:** PRG Analysis (3-4 days) + SID Metadata (2-3 days)
- **Week 2:** VICE Integration (4-5 days)
- **Week 3:** Integration, testing, documentation

### Version Releases
- **v2.24.0:** PRG File Analysis
- **v2.25.0:** SID Music Metadata + Analytics
- **v2.26.0:** VICE Emulator Integration

---

## Critical Success Factors

1. **Robustness:** Handle malformed files gracefully
2. **Performance:** All operations < 1 second
3. **Accuracy:** >90% of searches return relevant results
4. **Documentation:** Clear examples and usage guides
5. **Testing:** Comprehensive test suite for edge cases

---

## Out of Scope (Future Phases)

- Disassembler for PRG code analysis (Phase 5)
- VICE breakpoint automation (Phase 5)
- SID synthesizer emulation (Phase 6)
- Multi-user music sharing (Phase 6)
- VST plugin integration (Phase 6)

---

## Success Metrics

**Phase 4.1 (VICE):**
- ✓ Memory lookups < 100ms
- ✓ >90% address resolution
- ✓ Zero crashes on edge cases
- ✓ Works with VICE 3.5+

**Phase 4.2 (PRG):**
- ✓ Parse 100% of valid PRG files
- ✓ Analysis < 500ms
- ✓ >80% doc retrieval rate
- ✓ Correct memory region identification

**Phase 4.3 (SID):**
- ✓ Parse 100% of PSID/RSID files
- ✓ Search < 100ms
- ✓ >70% composer doc retrieval
- ✓ Scalable to 5000+ files

---

## Ready to Begin ✅

All three features are well-defined with clear implementation steps.
Phase 4 will add significant C64-specific value to the knowledge server.

**Next Step:** Choose which feature to implement first and begin Phase 4.1, 4.2, or 4.3.
