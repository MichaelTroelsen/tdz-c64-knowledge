# 🚀 Wiki Next-Level Improvements

Building on the 6 quick-start enhancements already implemented, here are the next wave of improvements organized by effort and impact.

---

## 🎯 Phase 1: Quick Wins (< 1 Day Each)

### 1. 📱 Mobile-Optimized Navigation
**Problem:** Current nav may not be touch-friendly
**Solution:** Hamburger menu for mobile devices

**Implementation:**
```javascript
// Add to enhancements.js
function setupMobileNav() {
  if (window.innerWidth < 768) {
    const nav = document.querySelector('.main-nav');
    const burger = document.createElement('button');
    burger.className = 'burger-menu';
    burger.innerHTML = '☰';
    burger.onclick = () => nav.classList.toggle('mobile-open');
    nav.parentElement.insertBefore(burger, nav);
  }
}
```

**CSS:**
```css
@media (max-width: 768px) {
  .burger-menu {
    display: block;
    font-size: 2em;
    background: var(--accent-color);
    color: white;
    border: none;
    padding: 10px 20px;
  }
  .main-nav {
    display: none;
    position: fixed;
    top: 60px;
    left: 0;
    right: 0;
    background: var(--card-bg);
    flex-direction: column;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
  }
  .main-nav.mobile-open {
    display: flex;
  }
}
```

**Effort:** 2 hours
**Impact:** ⭐⭐⭐⭐ Better mobile experience

---

### 2. 🔖 Bookmark & Favorites System
**Problem:** Users can't save articles for later
**Solution:** LocalStorage-based bookmarking

**Features:**
- ⭐ Bookmark button on each article/document
- 📚 Bookmarks page showing saved items
- 🗑️ Remove bookmarks
- 💾 Persist across sessions
- 📊 Show bookmark count

**Implementation:**
```javascript
class BookmarkManager {
  constructor() {
    this.bookmarks = JSON.parse(localStorage.getItem('bookmarks') || '[]');
  }

  add(id, title, url, type) {
    this.bookmarks.push({ id, title, url, type, date: Date.now() });
    this.save();
    this.updateUI();
  }

  remove(id) {
    this.bookmarks = this.bookmarks.filter(b => b.id !== id);
    this.save();
  }

  save() {
    localStorage.setItem('bookmarks', JSON.stringify(this.bookmarks));
  }

  isBookmarked(id) {
    return this.bookmarks.some(b => b.id === id);
  }
}
```

**Effort:** 4 hours
**Impact:** ⭐⭐⭐⭐ Improves return visits

---

### 3. 📊 Article Table of Contents (Auto-Generated)
**Problem:** Long articles hard to navigate
**Solution:** Auto-generate TOC from headings

**Implementation:**
```javascript
function generateTOC() {
  const article = document.querySelector('.article-content');
  if (!article) return;

  const headings = article.querySelectorAll('h2, h3');
  if (headings.length < 3) return; // Skip short articles

  const toc = document.createElement('div');
  toc.className = 'article-toc';
  toc.innerHTML = '<h3>Table of Contents</h3><ul></ul>';

  const list = toc.querySelector('ul');
  headings.forEach((heading, i) => {
    const id = `section-${i}`;
    heading.id = id;

    const li = document.createElement('li');
    li.className = heading.tagName.toLowerCase();
    li.innerHTML = `<a href="#${id}">${heading.textContent}</a>`;
    list.appendChild(li);
  });

  article.insertBefore(toc, article.firstChild);
}
```

**CSS:**
```css
.article-toc {
  background: var(--bg-color);
  border-left: 4px solid var(--accent-color);
  padding: 20px;
  margin: 20px 0;
  border-radius: 8px;
}

.article-toc ul {
  list-style: none;
  padding-left: 0;
}

.article-toc .h2 {
  font-weight: bold;
  margin: 10px 0;
}

.article-toc .h3 {
  padding-left: 20px;
  color: var(--primary-color);
}

.article-toc a {
  text-decoration: none;
  color: var(--text-color);
}

.article-toc a:hover {
  color: var(--accent-color);
}
```

**Effort:** 3 hours
**Impact:** ⭐⭐⭐⭐ Better navigation

---

### 4. 🔎 Search Autocomplete & Suggestions
**Problem:** Users don't know what to search for
**Solution:** Show popular searches and autocomplete

**Implementation:**
```javascript
function setupSearchAutocomplete() {
  const searchInput = document.getElementById('search-input');
  const suggestions = document.createElement('div');
  suggestions.className = 'search-suggestions';
  searchInput.parentElement.appendChild(suggestions);

  // Popular searches from entities
  const popularSearches = [
    'SID chip', 'VIC-II', 'sprites', 'raster interrupts',
    'music tracker', '6502 assembly', 'memory map', 'CIA timer'
  ];

  searchInput.addEventListener('input', (e) => {
    const query = e.target.value.toLowerCase();
    if (query.length < 2) {
      suggestions.innerHTML = '';
      return;
    }

    // Filter matching suggestions
    const matches = popularSearches
      .filter(s => s.toLowerCase().includes(query))
      .slice(0, 5);

    suggestions.innerHTML = matches
      .map(s => `<div class="suggestion" onclick="searchFor('${s}')">${s}</div>`)
      .join('');
  });
}
```

**Effort:** 3 hours
**Impact:** ⭐⭐⭐⭐ Better search UX

---

### 5. 📈 Popular Articles / Trending Topics
**Problem:** Users don't know what's worth reading
**Solution:** Show most-referenced articles

**Implementation:**
```javascript
// In wiki_export.py, track article popularity by doc_count
function generatePopularSection() {
  const articles = articlesData
    .sort((a, b) => b.doc_count - a.doc_count)
    .slice(0, 10);

  return `
    <section class="popular-articles">
      <h2>🔥 Most Referenced Topics</h2>
      <div class="popular-grid">
        ${articles.map((a, i) => `
          <div class="popular-card">
            <span class="rank">#${i+1}</span>
            <a href="articles/${a.filename}">${a.title}</a>
            <span class="refs">${a.doc_count} references</span>
          </div>
        `).join('')}
      </div>
    </section>
  `;
}
```

**Effort:** 2 hours
**Impact:** ⭐⭐⭐ Content discovery

---

### 6. 🎨 Syntax Highlighting for Code
**Problem:** Code blocks are monochrome
**Solution:** Add syntax highlighting for assembly

**Implementation:**
```javascript
// Add Prism.js or highlight.js
function highlightCode() {
  document.querySelectorAll('pre code').forEach(block => {
    // Simple 6502 assembly highlighter
    let code = block.textContent;

    // Highlight opcodes
    code = code.replace(/\b(LDA|STA|LDX|STX|LDY|STY|JMP|JSR|RTS|BNE|BEQ|CMP|ADC|SBC|INC|DEC|AND|ORA|EOR|ASL|LSR|ROL|ROR|BIT|NOP|PHA|PLA|PHP|PLP|TAX|TXA|TAY|TYA|TSX|TXS|INX|INY|DEX|DEY|CLC|SEC|CLI|SEI|CLV|CLD|SED)\b/g,
      '<span class="opcode">$&</span>');

    // Highlight hex values
    code = code.replace(/\$[0-9A-Fa-f]+/g, '<span class="hex">$&</span>');

    // Highlight comments
    code = code.replace(/;.*/g, '<span class="comment">$&</span>');

    block.innerHTML = code;
  });
}
```

**CSS:**
```css
.opcode { color: #ff79c6; font-weight: bold; }
.hex { color: #50fa7b; }
.comment { color: #6272a4; font-style: italic; }
```

**Effort:** 4 hours
**Impact:** ⭐⭐⭐⭐ Readability

---

## 🎯 Phase 2: Medium Effort (1-3 Days Each)

### 7. 🗺️ Visual Memory Map Explorer
**The Essential Reference Tool**

**Features:**
- Interactive C64 memory map ($0000-$FFFF)
- Color-coded regions (RAM, ROM, I/O, Free)
- Click regions for detailed info
- Expandable I/O chip sections (VIC-II, SID, CIA)
- Hover tooltips with register descriptions
- Search memory addresses
- Copy address to clipboard

**Data Structure:**
```javascript
const memoryMap = {
  regions: [
    { start: 0x0000, end: 0x00FF, name: 'Zero Page', color: '#ff6b6b', type: 'RAM',
      description: 'Fast access page for variables and pointers' },
    { start: 0x0100, end: 0x01FF, name: 'Stack', color: '#4ecdc4', type: 'RAM',
      description: 'System stack for subroutine calls' },
    { start: 0x0400, end: 0x07FF, name: 'Screen RAM', color: '#95e1d3', type: 'RAM',
      description: 'Default screen memory (1000 bytes)' },
    { start: 0xD000, end: 0xD3FF, name: 'VIC-II', color: '#f38181', type: 'I/O',
      registers: [
        { addr: 0xD000, name: 'M0X', desc: 'Sprite 0 X Position' },
        { addr: 0xD001, name: 'M0Y', desc: 'Sprite 0 Y Position' },
        // ... all 47 VIC-II registers
      ]
    },
    { start: 0xD400, end: 0xD7FF, name: 'SID', color: '#aa96da', type: 'I/O',
      registers: [
        { addr: 0xD400, name: 'FRELO1', desc: 'Voice 1 Frequency Low' },
        // ... all 29 SID registers
      ]
    },
    // ... complete memory map
  ]
};
```

**Implementation:**
```javascript
class MemoryMapExplorer {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.canvas = document.createElement('canvas');
    this.ctx = this.canvas.getContext('2d');
    this.scale = 4; // pixels per memory location
    this.setupCanvas();
    this.render();
    this.setupInteraction();
  }

  render() {
    const width = 256 * this.scale;
    const height = 256 * this.scale;

    // Draw memory map
    memoryMap.regions.forEach(region => {
      const startY = Math.floor(region.start / 256) * this.scale;
      const startX = (region.start % 256) * this.scale;
      const size = (region.end - region.start + 1);

      this.ctx.fillStyle = region.color;
      this.ctx.fillRect(startX, startY, size * this.scale, this.scale);
    });
  }

  onClick(x, y) {
    const addr = Math.floor(y / this.scale) * 256 + Math.floor(x / this.scale);
    const region = this.findRegion(addr);
    this.showDetails(region, addr);
  }

  showDetails(region, addr) {
    // Display detailed panel
    const details = document.getElementById('memory-details');
    details.innerHTML = `
      <h3>${region.name}</h3>
      <p><strong>Address:</strong> $${addr.toString(16).toUpperCase()}</p>
      <p><strong>Type:</strong> ${region.type}</p>
      <p>${region.description}</p>
      ${this.renderRegisters(region)}
    `;
  }
}
```

**Effort:** 2-3 days
**Impact:** ⭐⭐⭐⭐⭐ Game changer for reference

---

### 8. 📚 Assembly Instruction Reference Card
**Quick lookup for all 6502 opcodes**

**Features:**
- Filterable table of all 56 6502 instructions
- Search by opcode name (LDA, STA, etc.)
- Filter by category (Load/Store, Arithmetic, Branch, etc.)
- Show all addressing modes with cycle counts
- Example code for each instruction
- Flags affected (N, Z, C, V, etc.)
- Copy examples to clipboard

**Implementation:**
```javascript
const instructions = [
  {
    opcode: 'LDA',
    name: 'Load Accumulator',
    description: 'Loads a byte into the accumulator register',
    flags: 'N Z',
    modes: [
      { mode: 'Immediate', syntax: 'LDA #$nn', bytes: 2, cycles: 2, hex: 'A9' },
      { mode: 'Zero Page', syntax: 'LDA $nn', bytes: 2, cycles: 3, hex: 'A5' },
      { mode: 'Zero Page,X', syntax: 'LDA $nn,X', bytes: 2, cycles: 4, hex: 'B5' },
      { mode: 'Absolute', syntax: 'LDA $nnnn', bytes: 3, cycles: 4, hex: 'AD' },
      { mode: 'Absolute,X', syntax: 'LDA $nnnn,X', bytes: 3, cycles: 4, hex: 'BD' },
      { mode: 'Absolute,Y', syntax: 'LDA $nnnn,Y', bytes: 3, cycles: 4, hex: 'B9' },
      { mode: 'Indirect,X', syntax: 'LDA ($nn,X)', bytes: 2, cycles: 6, hex: 'A1' },
      { mode: 'Indirect,Y', syntax: 'LDA ($nn),Y', bytes: 2, cycles: 5, hex: 'B1' }
    ],
    examples: [
      { code: 'LDA #$FF', comment: 'Load 255 into accumulator' },
      { code: 'LDA $D020', comment: 'Load border color value' }
    ],
    category: 'Load/Store'
  },
  // ... all 56 instructions
];
```

**HTML Template:**
```html
<div class="instruction-card">
  <div class="instruction-header">
    <h3>LDA - Load Accumulator</h3>
    <div class="flags">Flags: <span class="flag">N</span> <span class="flag">Z</span></div>
  </div>

  <p class="description">Loads a byte into the accumulator register</p>

  <table class="addressing-modes">
    <tr>
      <th>Mode</th>
      <th>Syntax</th>
      <th>Bytes</th>
      <th>Cycles</th>
      <th>Hex</th>
    </tr>
    <!-- rows for each addressing mode -->
  </table>

  <div class="examples">
    <h4>Examples:</h4>
    <pre><code>LDA #$FF    ; Load 255 into accumulator
STA $D020   ; Store in border color</code></pre>
  </div>
</div>
```

**Effort:** 2 days (data entry + UI)
**Impact:** ⭐⭐⭐⭐⭐ Essential reference

---

### 9. 🎯 Guided Learning Paths
**Structured tutorials from beginner to advanced**

**Paths:**
1. **"Your First C64 Program"** (30 min)
   - Understanding the C64
   - BASIC basics
   - Your first POKE
   - Running code in VICE

2. **"6502 Assembly Fundamentals"** (2 hours)
   - What is assembly?
   - Registers and memory
   - Your first assembly program
   - Compiling and running

3. **"Graphics Programming"** (3 hours)
   - VIC-II basics
   - Screen modes
   - Sprites
   - Raster effects

4. **"Sound & Music"** (2 hours)
   - SID chip architecture
   - Playing notes
   - Using music trackers
   - Sound effects

**Implementation:**
```javascript
const learningPaths = {
  beginner: {
    id: 'first-program',
    title: 'Your First C64 Program',
    duration: '30 min',
    difficulty: 'Beginner',
    prerequisites: [],
    steps: [
      {
        id: 1,
        title: 'Understanding the C64',
        duration: '5 min',
        content: 'docs/introduction.html',
        quiz: [
          { q: 'What CPU does the C64 use?', a: '6510', choices: ['6502', '6510', 'Z80'] }
        ]
      },
      // ... more steps
    ]
  }
};

class LearningPathTracker {
  constructor() {
    this.progress = JSON.parse(localStorage.getItem('learning-progress') || '{}');
  }

  completeStep(pathId, stepId) {
    if (!this.progress[pathId]) this.progress[pathId] = [];
    this.progress[pathId].push(stepId);
    this.save();
    this.updateUI();
  }

  getProgress(pathId) {
    const completed = this.progress[pathId]?.length || 0;
    const total = learningPaths[pathId].steps.length;
    return Math.round((completed / total) * 100);
  }
}
```

**Effort:** 3 days (content + implementation)
**Impact:** ⭐⭐⭐⭐⭐ Transforms learning

---

### 10. 🔗 Related Content Recommendations
**"You might also like..."**

**Features:**
- Show related articles based on entity overlap
- "Similar documents" based on content similarity
- "Next steps" suggestions
- "Prerequisites" for advanced topics
- Visual connection graph

**Implementation:**
```javascript
// Already have entity co-occurrence data!
function getRelatedContent(currentArticle) {
  // Use existing entity relationships
  const relatedEntities = currentArticle.related_entities;

  // Find articles that share these entities
  const recommendations = articles
    .filter(a => a.id !== currentArticle.id)
    .map(a => ({
      article: a,
      score: calculateSimilarity(currentArticle, a)
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 5);

  return recommendations;
}

function calculateSimilarity(a1, a2) {
  const entities1 = new Set(a1.entities.map(e => e.text));
  const entities2 = new Set(a2.entities.map(e => e.text));

  const intersection = new Set([...entities1].filter(x => entities2.has(x)));
  const union = new Set([...entities1, ...entities2]);

  return intersection.size / union.size; // Jaccard similarity
}
```

**HTML:**
```html
<section class="related-content">
  <h3>Related Topics</h3>
  <div class="related-grid">
    <div class="related-card">
      <a href="articles/sprites.html">Sprite Programming</a>
      <span class="similarity">85% similar</span>
      <p>Learn about sprite multiplexing and collision detection</p>
    </div>
    <!-- more cards -->
  </div>
</section>
```

**Effort:** 2 days
**Impact:** ⭐⭐⭐⭐ Content discovery

---

## 🎯 Phase 3: Advanced Features (1+ Week)

### 11. 🎮 Interactive Code Playground
**The ultimate learning tool**

**Features:**
- In-browser 6502 emulator
- Live code editor with syntax highlighting
- Step-through debugger
- Memory & register viewer
- Screen output (simulated C64 display)
- Pre-loaded examples from articles
- Share code via URL
- Export to PRG files

**Technology Stack:**
- **Emulator:** 6502.js or Easy6502
- **Editor:** CodeMirror with 6502 mode
- **UI:** Split pane (code | output)

**Implementation Overview:**
```javascript
class CodePlayground {
  constructor() {
    this.emulator = new Emulator6502();
    this.editor = CodeMirror(document.getElementById('editor'), {
      mode: 'text/x-6502',
      theme: 'monokai',
      lineNumbers: true
    });
    this.setupControls();
  }

  compile() {
    const code = this.editor.getValue();
    const assembled = this.assemble(code);
    this.emulator.load(assembled);
  }

  run() {
    this.compile();
    this.emulator.run();
    this.updateDisplay();
  }

  step() {
    this.emulator.step();
    this.updateRegisters();
    this.updateMemory();
  }

  updateDisplay() {
    // Render screen memory to canvas
    const screen = this.emulator.getMemory(0x0400, 0x07FF);
    this.renderScreen(screen);
  }
}
```

**Effort:** 1-2 weeks
**Impact:** ⭐⭐⭐⭐⭐ Revolutionary

---

### 12. 💬 AI Assistant / Chatbot
**Ask questions, get instant answers**

**Features:**
- Natural language questions
- Answer from knowledge base
- Code generation
- Explain concepts
- Debug help
- Link to sources

**Implementation:**
```javascript
// Use existing KB semantic search + LLM API
async function askQuestion(question) {
  // 1. Search knowledge base
  const context = await searchKB(question);

  // 2. Generate answer with LLM
  const prompt = `
    Question: ${question}

    Context from C64 knowledge base:
    ${context.map(c => c.content).join('\n\n')}

    Answer the question based on the context above.
    Include code examples if relevant.
    Cite sources.
  `;

  const answer = await callLLM(prompt);

  // 3. Return with sources
  return {
    answer: answer,
    sources: context.map(c => ({ title: c.doc_title, url: c.doc_url }))
  };
}
```

**UI:**
```html
<div class="chat-widget">
  <div class="chat-messages" id="messages"></div>
  <div class="chat-input">
    <input type="text" id="question" placeholder="Ask about the C64...">
    <button onclick="ask()">Ask</button>
  </div>
</div>
```

**Effort:** 1-2 weeks (requires LLM API)
**Impact:** ⭐⭐⭐⭐⭐ Beginner-friendly

---

### 13. 📴 Progressive Web App (PWA)
**Install as app, works offline**

**Features:**
- Install to home screen
- Works completely offline
- Background sync for updates
- Push notifications for new content
- Lightning fast (cached)
- Native-like experience

**Implementation:**
```javascript
// service-worker.js
const CACHE_NAME = 'c64-kb-v1';
const urlsToCache = [
  '/',
  '/index.html',
  '/assets/css/style.css',
  '/assets/js/main.js',
  '/assets/js/enhancements.js',
  // ... all static assets
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(urlsToCache))
  );
});

self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request)
      .then(response => response || fetch(event.request))
  );
});
```

**manifest.json:**
```json
{
  "name": "C64 Knowledge Base",
  "short_name": "C64 KB",
  "icons": [
    { "src": "/icon-192.png", "sizes": "192x192", "type": "image/png" },
    { "src": "/icon-512.png", "sizes": "512x512", "type": "image/png" }
  ],
  "start_url": "/",
  "display": "standalone",
  "background_color": "#3f51b5",
  "theme_color": "#3f51b5"
}
```

**Effort:** 1 week
**Impact:** ⭐⭐⭐⭐ Accessibility

---

## 📊 Priority Matrix

| Feature | Effort | Impact | Priority |
|---------|--------|--------|----------|
| Mobile Navigation | 2h | ⭐⭐⭐⭐ | HIGH |
| Bookmarks | 4h | ⭐⭐⭐⭐ | HIGH |
| Auto TOC | 3h | ⭐⭐⭐⭐ | HIGH |
| Search Autocomplete | 3h | ⭐⭐⭐⭐ | HIGH |
| Syntax Highlighting | 4h | ⭐⭐⭐⭐ | HIGH |
| Popular Articles | 2h | ⭐⭐⭐ | MEDIUM |
| Memory Map | 2-3d | ⭐⭐⭐⭐⭐ | MEDIUM |
| Instruction Reference | 2d | ⭐⭐⭐⭐⭐ | MEDIUM |
| Learning Paths | 3d | ⭐⭐⭐⭐⭐ | MEDIUM |
| Related Content | 2d | ⭐⭐⭐⭐ | MEDIUM |
| Code Playground | 1-2w | ⭐⭐⭐⭐⭐ | LONG-TERM |
| AI Chatbot | 1-2w | ⭐⭐⭐⭐⭐ | LONG-TERM |
| PWA | 1w | ⭐⭐⭐⭐ | LONG-TERM |

---

## 🎯 Recommended Implementation Order

### Week 1: Polish & Mobile
1. Mobile Navigation (2h)
2. Syntax Highlighting (4h)
3. Auto TOC (3h)
4. Search Autocomplete (3h)
5. Bookmarks (4h)
6. Popular Articles (2h)

**Total:** ~18 hours, 6 features ✅

### Week 2-3: Major Features
1. Memory Map Explorer (3 days)
2. Assembly Reference (2 days)
3. Related Content (2 days)

**Total:** 7 days, 3 game-changing features ✅

### Week 4+: Advanced
1. Learning Paths (3 days)
2. PWA (1 week)
3. Code Playground (2 weeks)
4. AI Chatbot (2 weeks)

---

## 💡 Quick Implementation Tips

### For Phase 1 (Quick Wins):
- All can be done in `enhancements.js` + CSS
- No server-side changes needed
- LocalStorage for persistence
- Progressive enhancement

### For Phase 2 (Medium Effort):
- May need data preparation in `wiki_export.py`
- Static JSON data files
- Client-side rendering
- No backend required

### For Phase 3 (Advanced):
- May require external libraries
- Consider API integration for AI features
- Plan for scalability
- Test thoroughly

---

## 🚀 Ready to Start?

Pick features from Phase 1 and I can implement them immediately!

Which ones interest you most?
1. Mobile-optimized navigation?
2. Bookmark/favorites system?
3. Auto-generated table of contents?
4. Search autocomplete?
5. Syntax highlighting?
6. Popular articles section?

Or jump to a Phase 2/3 feature for bigger impact! 🎯
