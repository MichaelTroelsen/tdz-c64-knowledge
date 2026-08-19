// Wiki Enhancements - UX Improvements
// Theme switcher, copy buttons, reading progress, back-to-top

// ===== THEME SWITCHER =====
const themes = ['light', 'dark', 'c64'];
let currentTheme = localStorage.getItem('theme') || 'light';

function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
    currentTheme = theme;
    updateThemeIcon();
}

function cycleTheme() {
    const currentIndex = themes.indexOf(currentTheme);
    const nextIndex = (currentIndex + 1) % themes.length;
    setTheme(themes[nextIndex]);
}

function updateThemeIcon() {
    const button = document.getElementById('theme-toggle');
    if (button) {
        const icons = { 'light': '☀️', 'dark': '🌙', 'c64': '💙' };
        const labels = { 'light': 'Light Theme', 'dark': 'Dark Theme', 'c64': 'C64 Classic' };
        button.textContent = icons[currentTheme];
        button.title = labels[currentTheme];
    }
}

// ===== COPY BUTTONS =====
function addCopyButtons() {
    document.querySelectorAll('pre code').forEach(block => {
        // Skip if already wrapped
        if (block.parentNode.classList.contains('code-block-wrapper')) return;

        const wrapper = document.createElement('div');
        wrapper.className = 'code-block-wrapper';

        const button = document.createElement('button');
        button.className = 'copy-button';
        button.textContent = '📋 Copy';
        button.title = 'Copy code to clipboard';

        button.onclick = async () => {
            try {
                await navigator.clipboard.writeText(block.textContent);
                button.textContent = '✅ Copied!';
                button.classList.add('copied');
                setTimeout(() => {
                    button.textContent = '📋 Copy';
                    button.classList.remove('copied');
                }, 2000);
            } catch (err) {
                console.error('Failed to copy:', err);
                button.textContent = '❌ Failed';
                setTimeout(() => {
                    button.textContent = '📋 Copy';
                }, 2000);
            }
        };

        const pre = block.parentNode;
        pre.parentNode.insertBefore(wrapper, pre);
        wrapper.appendChild(pre);
        wrapper.appendChild(button);
    });
}

// ===== READING PROGRESS =====
function addReadingProgress() {
    const progress = document.createElement('div');
    progress.className = 'reading-progress';
    document.body.appendChild(progress);

    function updateProgress() {
        const scrolled = window.pageYOffset;
        const height = document.documentElement.scrollHeight - window.innerHeight;
        const percent = height > 0 ? (scrolled / height) * 100 : 0;
        progress.style.width = percent + '%';
    }

    window.addEventListener('scroll', updateProgress);
    updateProgress();
}

// ===== BACK TO TOP =====
function addBackToTop() {
    const button = document.createElement('button');
    button.className = 'back-to-top';
    button.textContent = '↑';
    button.title = 'Back to top';
    button.onclick = () => {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    };
    document.body.appendChild(button);

    function updateBackToTop() {
        if (window.pageYOffset > 300) {
            button.classList.add('visible');
        } else {
            button.classList.remove('visible');
        }
    }

    window.addEventListener('scroll', updateBackToTop);
    updateBackToTop();
}

// ===== THEME TOGGLE BUTTON =====
function addThemeToggle() {
    const button = document.createElement('button');
    button.id = 'theme-toggle';
    button.className = 'theme-toggle';
    button.onclick = cycleTheme;
    document.body.appendChild(button);
    updateThemeIcon();
}

// ===== SEARCH HIGHLIGHTING =====
function highlightSearchTerms() {
    const urlParams = new URLSearchParams(window.location.search);
    const query = urlParams.get('q');
    if (!query) return;

    const terms = query.toLowerCase().split(/\s+/).filter(t => t.length >= 3);
    if (terms.length === 0) return;

    // Highlight in paragraphs, list items, and table cells
    document.querySelectorAll('main p, main li, main td, .chunk-content').forEach(element => {
        let html = element.innerHTML;
        let modified = false;

        terms.forEach(term => {
            const regex = new RegExp(`(${escapeRegex(term)})`, 'gi');
            if (regex.test(html)) {
                html = html.replace(regex, '<mark class="highlight">$1</mark>');
                modified = true;
            }
        });

        if (modified) {
            element.innerHTML = html;
        }
    });
}

function escapeRegex(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// ===== SYNTAX HIGHLIGHTING =====
function highlightSyntax() {
    document.querySelectorAll('pre code').forEach(block => {
        // Skip if already highlighted
        if (block.classList.contains('highlighted')) return;

        let code = block.textContent;

        // Detect if this is BASIC or Assembly code
        const isBASIC = /^\s*\d+\s+(PRINT|REM|FOR|NEXT|IF|THEN|GOTO|GOSUB|INPUT|LET|DIM|DATA|READ|POKE|PEEK|SYS)/im.test(code);

        if (isBASIC) {
            // BASIC syntax highlighting

            // 1. Highlight line numbers
            code = code.replace(/^\s*(\d+)\s+/gm, match => `<span class="basic-linenum">${match}</span>`);

            // 2. Highlight BASIC keywords
            const basicKeywords = 'PRINT|REM|FOR|NEXT|IF|THEN|ELSE|GOTO|GOSUB|RETURN|INPUT|LET|DIM|DATA|READ|RESTORE|POKE|PEEK|SYS|END|STOP|LOAD|SAVE|RUN|LIST|NEW|CLR|GET|TAB|SPC|AND|OR|NOT|TO|STEP|FN|DEF|ON';
            const basicRegex = new RegExp(`\b(${basicKeywords})\b`, 'gi');
            code = code.replace(basicRegex, match => `<span class="basic-keyword">${match.toUpperCase()}</span>`);

            // 3. Highlight strings
            code = code.replace(/"[^"]*"/g, match => `<span class="basic-string">${match}</span>`);

            // 4. Highlight BASIC functions
            const basicFunctions = 'CHR\$|ASC|LEN|LEFT\$|RIGHT\$|MID\$|STR\$|VAL|INT|ABS|SIN|COS|TAN|ATN|SQR|RND|SGN|LOG|EXP|FRE|POS|USR';
            const funcRegex = new RegExp(`\b(${basicFunctions})`, 'gi');
            code = code.replace(funcRegex, match => `<span class="basic-function">${match.toUpperCase()}</span>`);

            // 5. Highlight REM comments
            code = code.replace(/\bREM\s+.*/gi, match => `<span class="basic-comment">${match}</span>`);

        } else {
            // 6502/6510 Assembly syntax highlighting

            // 1. Highlight comments (;)
            code = code.replace(/;.*/g, match => `<span class="asm-comment">${match}</span>`);

            // 2. Highlight hex values ($xx, $xxxx)
            code = code.replace(/\$[0-9A-Fa-f]+/g, match => `<span class="asm-hex">${match}</span>`);

            // 3. Highlight binary values (%)
            code = code.replace(/%[01]+/g, match => `<span class="asm-binary">${match}</span>`);

            // 4. Highlight decimal numbers
            code = code.replace(/\b\d+\b/g, match => `<span class="asm-number">${match}</span>`);

            // 5. Highlight opcodes (all 56 6502 instructions)
            const opcodes = 'ADC|AND|ASL|BCC|BCS|BEQ|BIT|BMI|BNE|BPL|BRK|BVC|BVS|CLC|CLD|CLI|CLV|CMP|CPX|CPY|DEC|DEX|DEY|EOR|INC|INX|INY|JMP|JSR|LDA|LDX|LDY|LSR|NOP|ORA|PHA|PHP|PLA|PLP|ROL|ROR|RTI|RTS|SBC|SEC|SED|SEI|STA|STX|STY|TAX|TAY|TSX|TXA|TXS|TYA';
            const opcodeRegex = new RegExp(`\b(${opcodes})\b`, 'gi');
            code = code.replace(opcodeRegex, match => `<span class="asm-opcode">${match.toUpperCase()}</span>`);

            // 6. Highlight labels (word followed by :)
            code = code.replace(/^(\w+):/gm, match => `<span class="asm-label">${match}</span>`);

            // 7. Highlight directives (.byte, .word, etc.)
            code = code.replace(/\.\w+/g, match => `<span class="asm-directive">${match}</span>`);
        }

        block.innerHTML = code;
        block.classList.add('highlighted');
    });
}

// ===== AUTO-GENERATED TABLE OF CONTENTS =====
function generateTableOfContents() {
    // Only generate TOC for article pages or long documents
    const article = document.querySelector('.article-content, main');
    if (!article) return;

    // Find all headings
    const headings = article.querySelectorAll('h2, h3');
    if (headings.length < 3) return; // Skip if too few headings

    // Create TOC container
    const toc = document.createElement('div');
    toc.className = 'auto-toc';
    toc.innerHTML = `
        <div class="toc-header">
            <h3>📑 Table of Contents</h3>
            <button class="toc-toggle" onclick="this.parentElement.parentElement.classList.toggle('collapsed')">−</button>
        </div>
        <nav class="toc-nav"></nav>
    `;

    const nav = toc.querySelector('.toc-nav');
    const tocList = document.createElement('ul');

    // Generate TOC entries
    headings.forEach((heading, index) => {
        // Create unique ID for heading
        const id = heading.id || `toc-section-${index}`;
        heading.id = id;

        // Create TOC item
        const li = document.createElement('li');
        li.className = `toc-${heading.tagName.toLowerCase()}`;

        const link = document.createElement('a');
        link.href = `#${id}`;
        link.textContent = heading.textContent;
        link.onclick = (e) => {
            e.preventDefault();
            heading.scrollIntoView({ behavior: 'smooth', block: 'start' });
            // Highlight the clicked section temporarily
            heading.classList.add('toc-highlight');
            setTimeout(() => heading.classList.remove('toc-highlight'), 2000);
        };

        li.appendChild(link);
        tocList.appendChild(li);
    });

    nav.appendChild(tocList);

    // Insert TOC at the beginning of the article
    const insertPoint = article.querySelector('h1, .article-meta');
    if (insertPoint && insertPoint.nextSibling) {
        insertPoint.parentNode.insertBefore(toc, insertPoint.nextSibling);
    } else {
        article.insertBefore(toc, article.firstChild);
    }

    // Add active section highlighting on scroll
    window.addEventListener('scroll', () => updateActiveTocSection(headings, tocList));
}

function updateActiveTocSection(headings, tocList) {
    const scrollPos = window.pageYOffset + 100;

    let activeIndex = 0;
    headings.forEach((heading, index) => {
        if (heading.offsetTop <= scrollPos) {
            activeIndex = index;
        }
    });

    // Update active class
    const tocItems = tocList.querySelectorAll('li');
    tocItems.forEach((item, index) => {
        item.classList.toggle('active', index === activeIndex);
    });
}

// ===== SEARCH AUTOCOMPLETE =====
function setupSearchAutocomplete() {
    const searchInput = document.getElementById('search-input');
    if (!searchInput) return;

    // Create suggestions container
    const suggestions = document.createElement('div');
    suggestions.className = 'search-autocomplete';
    suggestions.id = 'search-autocomplete';
    searchInput.parentElement.appendChild(suggestions);

    // Popular/common search terms
    const popularSearches = [
        'SID chip', 'VIC-II', 'VIC-II registers', 'sprites', 'sprite multiplexing',
        'raster interrupts', 'raster bars', 'music tracker', 'music editor',
        '6502 assembly', '6510 assembly', 'memory map', 'zero page',
        'CIA timer', 'CIA chip', 'joystick input', 'keyboard input',
        'screen RAM', 'color RAM', 'character set', 'bitmap mode',
        'sound effects', 'SID music', 'border color', 'background color',
        'BASIC programming', 'machine code', 'assembler', 'VICE emulator'
    ];

    let currentFocus = -1;

    searchInput.addEventListener('input', (e) => {
        const query = e.target.value.trim().toLowerCase();
        suggestions.innerHTML = '';
        currentFocus = -1;

        if (query.length < 2) {
            suggestions.style.display = 'none';
            return;
        }

        // Filter matching suggestions
        const matches = popularSearches
            .filter(term => term.toLowerCase().includes(query))
            .slice(0, 8);

        if (matches.length === 0) {
            suggestions.style.display = 'none';
            return;
        }

        // Display suggestions
        matches.forEach((term, index) => {
            const div = document.createElement('div');
            div.className = 'autocomplete-item';
            div.setAttribute('data-index', index);

            // Highlight matching part
            const termLower = term.toLowerCase();
            const startIdx = termLower.indexOf(query);
            const endIdx = startIdx + query.length;

            const before = term.substring(0, startIdx);
            const match = term.substring(startIdx, endIdx);
            const after = term.substring(endIdx);

            div.innerHTML = `${before}<strong>${match}</strong>${after}`;

            div.onclick = () => selectSuggestion(term);
            suggestions.appendChild(div);
        });

        suggestions.style.display = 'block';
    });

    function selectSuggestion(term) {
        searchInput.value = term;
        suggestions.style.display = 'none';
        // Trigger search
        const event = new KeyboardEvent('keyup', { key: 'Enter' });
        searchInput.dispatchEvent(event);
    }

    // Keyboard navigation
    searchInput.addEventListener('keydown', (e) => {
        const items = suggestions.querySelectorAll('.autocomplete-item');
        if (items.length === 0) return;

        if (e.key === 'ArrowDown') {
            e.preventDefault();
            currentFocus++;
            if (currentFocus >= items.length) currentFocus = 0;
            setActive(items);
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            currentFocus--;
            if (currentFocus < 0) currentFocus = items.length - 1;
            setActive(items);
        } else if (e.key === 'Enter' && currentFocus > -1) {
            e.preventDefault();
            items[currentFocus].click();
        } else if (e.key === 'Escape') {
            suggestions.style.display = 'none';
            currentFocus = -1;
        }
    });

    function setActive(items) {
        items.forEach((item, index) => {
            item.classList.toggle('active', index === currentFocus);
        });
        if (currentFocus >= 0) {
            searchInput.value = items[currentFocus].textContent;
        }
    }

    // Close suggestions when clicking outside
    document.addEventListener('click', (e) => {
        if (e.target !== searchInput) {
            suggestions.style.display = 'none';
        }
    });
}

// ===== POPULAR ARTICLES SECTION =====
function displayPopularArticles() {
    const popularContainer = document.getElementById('popular-articles');
    if (!popularContainer) return;

    // This data should be generated by wiki_export.py
    // For now, we'll use the articles.json if available
    fetch('assets/data/articles.json')
        .then(response => response.json())
        .then(data => {
            if (!data || !data.articles) return;

            // Sort by doc_count (most referenced)
            const popular = data.articles
                .sort((a, b) => b.doc_count - a.doc_count)
                .slice(0, 6);

            popularContainer.innerHTML = `
                <h2>🔥 Most Referenced Topics</h2>
                <div class="popular-grid">
                    ${popular.map((article, index) => `
                        <div class="popular-card">
                            <div class="popular-rank">#${index + 1}</div>
                            <h3><a href="articles/${article.filename}">${article.title}</a></h3>
                            <div class="popular-meta">
                                <span class="doc-count">📚 ${article.doc_count} references</span>
                                <span class="related-count">🔗 ${article.related_count || 0} related</span>
                            </div>
                        </div>
                    `).join('')}
                </div>
            `;
        })
        .catch(err => {
            console.error('Failed to load popular articles:', err);
        });
}

// ===== LAZY LOADING IMAGES =====
function setupLazyLoading() {
    // Check if IntersectionObserver is supported
    if ('IntersectionObserver' in window) {
        setupIntersectionObserver();
    } else {
        // Fallback for older browsers - load all images immediately
        loadAllImages();
    }
}

function setupIntersectionObserver() {
    const imageObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const img = entry.target;
                loadImage(img);
                observer.unobserve(img);
            }
        });
    }, {
        // Start loading when image is 50px from viewport
        rootMargin: '50px',
        threshold: 0.01
    });

    // Find all images with loading="lazy" attribute
    const lazyImages = document.querySelectorAll('img[loading="lazy"]');
    lazyImages.forEach(img => {
        imageObserver.observe(img);
    });

    // Also observe dynamically added images
    const mutationObserver = new MutationObserver(mutations => {
        mutations.forEach(mutation => {
            mutation.addedNodes.forEach(node => {
                if (node.nodeName === 'IMG' && node.getAttribute('loading') === 'lazy') {
                    imageObserver.observe(node);
                } else if (node.querySelectorAll) {
                    const lazyImgs = node.querySelectorAll('img[loading="lazy"]');
                    lazyImgs.forEach(img => imageObserver.observe(img));
                }
            });
        });
    });

    mutationObserver.observe(document.body, {
        childList: true,
        subtree: true
    });

    console.log(`🖼️ Lazy loading enabled for ${lazyImages.length} images`);
}

function loadImage(img) {
    // Check if we have a data-src attribute (common lazy loading pattern)
    const src = img.getAttribute('data-src') || img.getAttribute('src');

    if (!src) return;

    // Create a new image to preload
    const tempImg = new Image();

    tempImg.onload = () => {
        img.src = src;
        img.classList.add('loaded');

        // Remove placeholder if it exists
        const placeholder = img.previousElementSibling;
        if (placeholder && placeholder.classList.contains('lazy-placeholder')) {
            placeholder.remove();
        }
    };

    tempImg.onerror = () => {
        console.error('Failed to load image:', src);
        img.classList.add('error');
    };

    // Start loading
    if (img.getAttribute('data-src')) {
        tempImg.src = img.getAttribute('data-src');
    } else {
        tempImg.src = src;
    }
}

function loadAllImages() {
    // Fallback for browsers without IntersectionObserver
    const lazyImages = document.querySelectorAll('img[loading="lazy"]');
    lazyImages.forEach(img => {
        const src = img.getAttribute('data-src') || img.getAttribute('src');
        if (src) {
            img.src = src;
            img.classList.add('loaded');
        }
    });

    console.log(`🖼️ Loaded ${lazyImages.length} images (fallback mode)`);
}

// Helper function to add lazy loading to dynamically created images
function makeLazyImage(src, alt = '', className = '') {
    const img = document.createElement('img');
    img.setAttribute('data-src', src);
    img.setAttribute('loading', 'lazy');
    img.alt = alt;
    if (className) img.className = className;

    // Add placeholder
    const placeholder = document.createElement('div');
    placeholder.className = 'lazy-placeholder';

    return { placeholder, img };
}

// Export for use in other scripts
window.lazyLoadingUtils = {
    makeLazyImage,
    loadImage
};

// ===== RELATED DOCUMENTS SIDEBAR =====
async function loadRelatedDocuments() {
    const sidebar = document.getElementById('related-docs-sidebar');
    if (!sidebar) return;

    // Get current document ID from page
    const currentDocId = sidebar.getAttribute('data-doc-id');
    if (!currentDocId) {
        sidebar.innerHTML = '<div class="no-related-docs">No document ID found</div>';
        return;
    }

    try {
        // Load similarities data
        const response = await fetch('../assets/data/similarities.json');
        const similarities = await response.json();

        // Get related docs for current document
        const relatedDocs = similarities[currentDocId] || [];

        if (relatedDocs.length === 0) {
            sidebar.innerHTML = '<div class="no-related-docs">No related documents found</div>';
            return;
        }

        // Build HTML for related docs
        const html = relatedDocs.slice(0, 5).map((doc, index) => {
            const scorePercent = Math.round(doc.score * 100);

            return `
                <div class="related-doc-item" style="animation: fadeIn 0.3s ease-out ${index * 0.1}s both">
                    <a href="${doc.filename}">${escapeHtml(doc.title)}</a>
                    <div class="related-doc-score">
                        <span>${scorePercent}%</span>
                        <div class="similarity-bar">
                            <div class="similarity-fill" style="width: ${scorePercent}%"></div>
                        </div>
                    </div>
                    <div class="related-doc-meta">
                        ${doc.common_entities > 0 ? `<span>🏷️ ${doc.common_entities} shared topics</span>` : ''}
                        ${doc.common_tags > 0 ? `<span>📁 ${doc.common_tags} shared tags</span>` : ''}
                    </div>
                </div>
            `;
        }).join('');

        sidebar.innerHTML = html;

    } catch (error) {
        console.error('Failed to load related documents:', error);
        sidebar.innerHTML = '<div class="no-related-docs">Unable to load related documents</div>';
    }
}

// Fade-in animation for related docs
const fadeInStyle = document.createElement('style');
fadeInStyle.textContent = `
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
`;
document.head.appendChild(fadeInStyle);

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ===== BOOKMARKS SYSTEM =====
class BookmarkManager {
    constructor() {
        this.bookmarks = JSON.parse(localStorage.getItem('c64-bookmarks') || '[]');
        this.init();
    }

    init() {
        // Add bookmark button to articles and documents
        this.addBookmarkButtons();
        // Show bookmark indicator if current page is bookmarked
        this.updateBookmarkStatus();
    }

    addBookmarkButtons() {
        // Add to article pages
        const article = document.querySelector('.article-content h1');
        if (article) {
            const pageId = this.getCurrentPageId();
            const button = this.createBookmarkButton(pageId);
            article.parentElement.insertBefore(button, article.nextSibling);
        }

        // Add to document pages
        const docTitle = document.querySelector('main h1');
        if (docTitle && !article) {
            const pageId = this.getCurrentPageId();
            const button = this.createBookmarkButton(pageId);
            docTitle.parentElement.insertBefore(button, docTitle.nextSibling);
        }
    }

    createBookmarkButton(pageId) {
        const button = document.createElement('button');
        button.className = 'bookmark-btn';
        button.setAttribute('data-page-id', pageId);

        const isBookmarked = this.isBookmarked(pageId);
        button.innerHTML = isBookmarked ? '⭐ Bookmarked' : '☆ Bookmark';
        button.classList.toggle('bookmarked', isBookmarked);

        button.onclick = (e) => {
            e.preventDefault();
            this.toggleBookmark(pageId);
        };

        return button;
    }

    getCurrentPageId() {
        const path = window.location.pathname;
        return path.split('/').pop().replace('.html', '') || 'index';
    }

    getCurrentPageTitle() {
        return document.querySelector('h1')?.textContent || 'Untitled';
    }

    getCurrentPageUrl() {
        return window.location.pathname;
    }

    isBookmarked(pageId) {
        return this.bookmarks.some(b => b.id === pageId);
    }

    toggleBookmark(pageId) {
        if (this.isBookmarked(pageId)) {
            this.removeBookmark(pageId);
        } else {
            this.addBookmark(pageId);
        }
    }

    addBookmark(pageId) {
        const bookmark = {
            id: pageId,
            title: this.getCurrentPageTitle(),
            url: this.getCurrentPageUrl(),
            date: Date.now()
        };

        this.bookmarks.push(bookmark);
        this.save();
        this.updateUI(pageId, true);
        this.showNotification('Bookmark added!');
    }

    removeBookmark(pageId) {
        this.bookmarks = this.bookmarks.filter(b => b.id !== pageId);
        this.save();
        this.updateUI(pageId, false);
        this.showNotification('Bookmark removed');
    }

    save() {
        localStorage.setItem('c64-bookmarks', JSON.stringify(this.bookmarks));
    }

    updateUI(pageId, isBookmarked) {
        const button = document.querySelector(`[data-page-id="${pageId}"]`);
        if (button) {
            button.innerHTML = isBookmarked ? '⭐ Bookmarked' : '☆ Bookmark';
            button.classList.toggle('bookmarked', isBookmarked);
        }
    }

    updateBookmarkStatus() {
        const pageId = this.getCurrentPageId();
        const button = document.querySelector(`[data-page-id="${pageId}"]`);
        if (button) {
            const isBookmarked = this.isBookmarked(pageId);
            button.classList.toggle('bookmarked', isBookmarked);
        }
    }

    showNotification(message) {
        const notification = document.createElement('div');
        notification.className = 'bookmark-notification';
        notification.textContent = message;
        document.body.appendChild(notification);

        setTimeout(() => notification.classList.add('show'), 10);
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => notification.remove(), 300);
        }, 2000);
    }

    getBookmarks() {
        return this.bookmarks;
    }

    exportBookmarks() {
        const data = JSON.stringify(this.bookmarks, null, 2);
        const blob = new Blob([data], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'c64-bookmarks.json';
        a.click();
    }
}

// ===== AI CHATBOT =====
class AIChatbot {
    constructor() {
        this.isOpen = false;
        this.messages = [];
        this.init();
    }

    init() {
        this.createChatWidget();
        this.loadHistory();
    }

    createChatWidget() {
        const widget = document.createElement('div');
        widget.className = 'chat-widget';
        widget.innerHTML = `
            <button class="chat-toggle" id="chat-toggle" title="Ask AI Assistant">
                <span class="bot-icon">🤖</span>
                <span class="bot-label">Ask AI</span>
            </button>
            <div class="chat-container" id="chat-container">
                <div class="chat-header">
                    <h3>🤖 C64 AI Assistant</h3>
                    <button class="chat-close" id="chat-close">×</button>
                </div>
                <div class="chat-messages" id="chat-messages">
                    <div class="chat-message bot">
                        <div class="message-content">
                            👋 Hi! I'm your C64 knowledge assistant. Ask me anything about:
                            <ul>
                                <li>6502/6510 assembly programming</li>
                                <li>VIC-II graphics & sprites</li>
                                <li>SID sound & music</li>
                                <li>Hardware (CIA, memory map, I/O)</li>
                                <li>Development tools & techniques</li>
                            </ul>
                            Try: "How do I change the border color?" or "Explain sprite multiplexing"
                        </div>
                    </div>
                </div>
                <div class="chat-input-container">
                    <input type="text" id="chat-input" placeholder="Ask about the C64..." autocomplete="off">
                    <button id="chat-send">Send</button>
                </div>
                <div class="chat-examples">
                    <small>Quick examples:</small>
                    <button class="example-btn" onclick="chatbot.askExample('How do I play a sound on the SID?')">🎵 Play sound</button>
                    <button class="example-btn" onclick="chatbot.askExample('Show me sprite example code')">🎮 Sprite code</button>
                </div>
            </div>
        `;

        document.body.appendChild(widget);
        this.attachEventListeners();
    }

    attachEventListeners() {
        document.getElementById('chat-toggle').onclick = () => this.toggle();
        document.getElementById('chat-close').onclick = () => this.toggle();
        document.getElementById('chat-send').onclick = () => this.sendMessage();

        const input = document.getElementById('chat-input');
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.sendMessage();
        });
    }

    toggle() {
        this.isOpen = !this.isOpen;
        document.getElementById('chat-container').classList.toggle('open', this.isOpen);

        if (this.isOpen) {
            document.getElementById('chat-input').focus();
        }
    }

    async sendMessage() {
        const input = document.getElementById('chat-input');
        const question = input.value.trim();

        if (!question) return;

        // Add user message
        this.addMessage('user', question);
        input.value = '';

        // Show typing indicator
        this.showTyping();

        // Get answer (simulate for now, would use KB search + LLM in production)
        const answer = await this.getAnswer(question);

        // Remove typing indicator and add bot response
        this.hideTyping();
        this.addMessage('bot', answer.text, answer.sources);

        // Save to history
        this.saveHistory();
    }

    addMessage(role, content, sources = []) {
        const messagesContainer = document.getElementById('chat-messages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${role}`;

        let messageHTML = `<div class="message-content">${this.formatMessage(content)}</div>`;

        if (sources && sources.length > 0) {
            messageHTML += '<div class="message-sources"><strong>Sources:</strong><ul>';
            sources.forEach(source => {
                messageHTML += `<li><a href="${source.url}">${source.title}</a></li>`;
            });
            messageHTML += '</ul></div>';
        }

        messageDiv.innerHTML = messageHTML;
        messagesContainer.appendChild(messageDiv);

        // Scroll to bottom
        messagesContainer.scrollTop = messagesContainer.scrollHeight;

        // Store message
        this.messages.push({ role, content, sources, timestamp: Date.now() });
    }

    formatMessage(text) {
        // Convert code blocks
        text = text.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');

        // Convert inline code
        text = text.replace(/`([^`]+)`/g, '<code>$1</code>');

        // Convert line breaks
        text = text.replace(/\n/g, '<br>');

        return text;
    }

    showTyping() {
        const messagesContainer = document.getElementById('chat-messages');
        const typing = document.createElement('div');
        typing.className = 'chat-message bot typing';
        typing.id = 'typing-indicator';
        typing.innerHTML = '<div class="message-content"><div class="typing-dots"><span></span><span></span><span></span></div></div>';
        messagesContainer.appendChild(typing);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    hideTyping() {
        const typing = document.getElementById('typing-indicator');
        if (typing) typing.remove();
    }

    async getAnswer(question) {
        // Simple pattern matching for demo (would use actual KB search + LLM in production)
        const answers = this.getPatternAnswers();

        const lowerQ = question.toLowerCase();

        // Check for pattern matches
        for (const [pattern, answer] of Object.entries(answers)) {
            if (lowerQ.includes(pattern)) {
                return {
                    text: answer.text,
                    sources: answer.sources || []
                };
            }
        }

        // Default response with search suggestion
        return {
            text: `I'll help you with that! While I'm in demo mode, I can provide information about C64 topics.

Try these resources:
• Search the wiki for "${question}"
• Browse the Articles section for comprehensive guides
• Check the Memory Map for hardware details

For the full AI assistant, we'd integrate:
1. Semantic search through the knowledge base
2. LLM API (Claude/GPT) for natural responses
3. Code generation from examples
4. Citation of source documents`,
            sources: [
                { title: 'Search Results', url: 'index.html?q=' + encodeURIComponent(question) },
                { title: 'Articles', url: 'articles.html' }
            ]
        };
    }

    getPatternAnswers() {
        return {
            'border color': {
                text: `To change the border color on the C64, write to address $D020 (53280 decimal).

**In BASIC:**
\`\`\`
POKE 53280, 0   ; Black border
POKE 53280, 1   ; White border
\`\`\`

**In Assembly:**
\`\`\`
LDA #$00        ; Black
STA $D020       ; Write to border color register
\`\`\`

Color values range from 0-15. The VIC-II chip controls this register.`,
                sources: [
                    { title: 'VIC-II Article', url: 'articles/VIC-II.html' },
                    { title: 'Color Article', url: 'articles/Color.html' }
                ]
            },
            'sprite': {
                text: `**Sprites on the C64:**

The VIC-II chip provides 8 hardware sprites (24x21 pixels each).

**Basic sprite setup:**
\`\`\`
; Enable sprite 0
LDA #$01
STA $D015       ; Sprite enable register

; Set X position
LDA #100
STA $D000       ; Sprite 0 X position

; Set Y position
LDA #50
STA $D001       ; Sprite 0 Y position

; Set sprite pointer (shape)
LDA #13
STA $07F8       ; Sprite 0 pointer
\`\`\`

Sprite data is 63 bytes (24x21 bits + 1 padding).`,
                sources: [
                    { title: 'Sprite Article', url: 'articles/Sprite.html' },
                    { title: 'VIC-II Article', url: 'articles/VIC-II.html' }
                ]
            },
            'sound': {
                text: `**Playing sounds on the SID chip:**

The SID chip ($D400-$D7FF) has 3 independent voices.

**Simple beep on Voice 1:**
\`\`\`
LDA #$0F
STA $D418       ; Max volume

LDA #$20
STA $D405       ; Attack/Decay
STA $D406       ; Sustain/Release

LDA #$10
STA $D401       ; Frequency high

LDA #$00
STA $D400       ; Frequency low

LDA #$11
STA $D404       ; Triangle wave + gate on
\`\`\``,
                sources: [
                    { title: 'SID Article', url: 'articles/SID.html' },
                    { title: 'Sound Article', url: 'articles/Sound.html' }
                ]
            },
            'memory map': {
                text: `**C64 Memory Map:**

\`\`\`
$0000-$00FF  Zero Page (256 bytes)
$0100-$01FF  Stack (256 bytes)
$0200-$03FF  OS/BASIC workspace
$0400-$07FF  Screen RAM (1000 bytes)
$0800-$9FFF  BASIC program & variables
$A000-$BFFF  BASIC ROM (8K)
$C000-$CFFF  RAM (4K)
$D000-$DFFF  I/O area (4K)
  $D000-$D3FF  VIC-II registers
  $D400-$D7FF  SID registers
  $DC00-$DCFF  CIA #1 registers
  $DD00-$DDFF  CIA #2 registers
$E000-$FFFF  Kernal ROM (8K)
\`\`\`

Total: 64K addressable memory with bank switching.`,
                sources: [
                    { title: 'Memory Article', url: 'articles/Memory.html' }
                ]
            }
        };
    }

    askExample(question) {
        document.getElementById('chat-input').value = question;
        this.sendMessage();
    }

    saveHistory() {
        localStorage.setItem('c64-chat-history', JSON.stringify(this.messages.slice(-50))); // Keep last 50
    }

    loadHistory() {
        const history = localStorage.getItem('c64-chat-history');
        if (history) {
            try {
                this.messages = JSON.parse(history);
                // Could restore messages to UI here if desired
            } catch (e) {
                console.error('Failed to load chat history:', e);
            }
        }
    }

    clearHistory() {
        this.messages = [];
        localStorage.removeItem('c64-chat-history');
        document.getElementById('chat-messages').innerHTML = '';
    }
}

// Global instances
let bookmarkManager;
let chatbot;

// ===== KEYBOARD SHORTCUTS =====
function setupKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        // Alt+T = Toggle theme
        if (e.altKey && e.key === 't') {
            e.preventDefault();
            cycleTheme();
        }
        // Alt+H = Go to home
        if (e.altKey && e.key === 'h') {
            e.preventDefault();
            window.location.href = '/';
        }
        // Alt+B = Toggle bookmarks view
        if (e.altKey && e.key === 'b') {
            e.preventDefault();
            showBookmarksPanel();
        }
        // Alt+C = Toggle chatbot
        if (e.altKey && e.key === 'c') {
            e.preventDefault();
            if (chatbot) chatbot.toggle();
        }
    });
}

function showBookmarksPanel() {
    if (!bookmarkManager) return;

    const bookmarks = bookmarkManager.getBookmarks();

    if (bookmarks.length === 0) {
        alert('No bookmarks yet! Use the ⭐ Bookmark button on articles to save them.');
        return;
    }

    const panel = document.createElement('div');
    panel.className = 'bookmarks-panel';
    panel.innerHTML = `
        <div class="bookmarks-content">
            <div class="bookmarks-header">
                <h2>📚 Your Bookmarks (${bookmarks.length})</h2>
                <button class="close-btn" onclick="this.parentElement.parentElement.parentElement.remove()">×</button>
            </div>
            <div class="bookmarks-list">
                ${bookmarks.map(b => `
                    <div class="bookmark-item">
                        <a href="${b.url}">${b.title}</a>
                        <button class="remove-bookmark" onclick="bookmarkManager.removeBookmark('${b.id}'); this.parentElement.remove();">🗑️</button>
                    </div>
                `).join('')}
            </div>
            <div class="bookmarks-footer">
                <button onclick="bookmarkManager.exportBookmarks()">💾 Export Bookmarks</button>
            </div>
        </div>
    `;

    document.body.appendChild(panel);
}

// ===== INITIALIZE ALL =====
document.addEventListener('DOMContentLoaded', () => {
    // Initialize theme first (before anything renders)
    setTheme(currentTheme);

    // Add all enhancements
    addThemeToggle();
    addCopyButtons();
    addReadingProgress();
    addBackToTop();
    highlightSearchTerms();
    setupKeyboardShortcuts();

    // NEW FEATURES
    highlightSyntax();  // Syntax highlighting for code blocks
    // generateTableOfContents();  // Auto TOC for articles - DISABLED per user request
    setupSearchAutocomplete();  // Search suggestions
    displayPopularArticles();  // Popular articles on homepage
    bookmarkManager = new BookmarkManager();  // Bookmarks system
    chatbot = new AIChatbot();  // AI chatbot
    setupLazyLoading();  // Lazy load images for performance
    loadRelatedDocuments();  // Load related documents sidebar

    console.log('✅ Wiki enhancements loaded');
    console.log('💡 Keyboard shortcuts:');
    console.log('   Alt+T = Toggle theme');
    console.log('   Alt+H = Go to home');
    console.log('   Alt+B = View bookmarks');
    console.log('   Alt+C = Toggle AI chatbot');
});

// Export for use in other scripts
window.wikiEnhancements = {
    setTheme,
    cycleTheme,
    addCopyButtons
};
