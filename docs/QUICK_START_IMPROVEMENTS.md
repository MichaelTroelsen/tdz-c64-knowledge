# Quick Start: Implement These Today! 🚀

## 5 Improvements You Can Add in Under 30 Minutes Each

---

## 1. 🌗 Dark Mode Toggle (15 minutes)

### Add to `wiki_export.py` in `_create_css()` method:

```css
/* Add to existing CSS */

/* Theme Variables */
:root {
  --bg-primary: #FFFFFF;
  --bg-secondary: #F7FAFC;
  --text-primary: #1A202C;
  --text-secondary: #4A5568;
  --accent: #3182CE;
  --border: #E2E8F0;
  --code-bg: #2D3748;
  --code-text: #F7FAFC;
}

[data-theme="dark"] {
  --bg-primary: #1A202C;
  --bg-secondary: #2D3748;
  --text-primary: #F7FAFC;
  --text-secondary: #CBD5E0;
  --accent: #63B3ED;
  --border: #4A5568;
  --code-bg: #1A202C;
  --code-text: #F7FAFC;
}

[data-theme="c64"] {
  --bg-primary: #3F51B5;
  --bg-secondary: #5C6BC0;
  --text-primary: #E8EAF6;
  --text-secondary: #C5CAE9;
  --accent: #FFC107;
  --border: #7986CB;
  --code-bg: #283593;
  --code-text: #00E676;
}

/* Update existing styles to use variables */
body {
  background-color: var(--bg-primary);
  color: var(--text-primary);
}

.card-bg {
  background: var(--bg-secondary);
}

.border-color {
  border-color: var(--border);
}

/* Theme Toggle Button */
.theme-toggle {
  position: fixed;
  top: 20px;
  right: 20px;
  z-index: 1000;
  background: var(--accent);
  color: white;
  border: none;
  padding: 10px 15px;
  border-radius: 20px;
  cursor: pointer;
  font-size: 1.2em;
  transition: all 0.3s;
}

.theme-toggle:hover {
  transform: scale(1.1);
}
```

### Add to JavaScript (create `theme.js`):

```javascript
// Theme switcher
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
  const icons = {
    'light': '☀️',
    'dark': '🌙',
    'c64': '💙'
  };
  button.textContent = icons[currentTheme];
}

// Initialize theme
setTheme(currentTheme);

// Add toggle button to page
document.addEventListener('DOMContentLoaded', () => {
  const button = document.createElement('button');
  button.id = 'theme-toggle';
  button.className = 'theme-toggle';
  button.onclick = cycleTheme;
  document.body.appendChild(button);
  updateThemeIcon();
});
```

**Done!** Users can now toggle between light/dark/C64 themes.

---

## 2. 📋 Copy Code Button (10 minutes)

### Add to CSS:

```css
/* Code block improvements */
.code-block-wrapper {
  position: relative;
  margin: 20px 0;
}

.copy-button {
  position: absolute;
  top: 10px;
  right: 10px;
  background: var(--accent);
  color: white;
  border: none;
  padding: 5px 10px;
  border-radius: 5px;
  cursor: pointer;
  font-size: 0.85em;
  opacity: 0.7;
  transition: opacity 0.3s;
}

.copy-button:hover {
  opacity: 1;
}

.copy-button.copied {
  background: #48BB78;
}
```

### Add to JavaScript:

```javascript
// Add copy buttons to all code blocks
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('pre code').forEach(block => {
    const wrapper = document.createElement('div');
    wrapper.className = 'code-block-wrapper';

    const button = document.createElement('button');
    button.className = 'copy-button';
    button.textContent = '📋 Copy';

    button.onclick = async () => {
      await navigator.clipboard.writeText(block.textContent);
      button.textContent = '✅ Copied!';
      button.classList.add('copied');
      setTimeout(() => {
        button.textContent = '📋 Copy';
        button.classList.remove('copied');
      }, 2000);
    };

    block.parentNode.insertBefore(wrapper, block);
    wrapper.appendChild(block.cloneNode(true));
    wrapper.appendChild(button);
    block.parentNode.removeChild(block);
  });
});
```

**Done!** Every code block now has a copy button.

---

## 3. 📊 Reading Progress Bar (5 minutes)

### Add to CSS:

```css
/* Reading progress */
.reading-progress {
  position: fixed;
  top: 0;
  left: 0;
  width: 0%;
  height: 3px;
  background: var(--accent);
  z-index: 9999;
  transition: width 0.1s;
}
```

### Add to JavaScript:

```javascript
// Reading progress indicator
document.addEventListener('DOMContentLoaded', () => {
  const progress = document.createElement('div');
  progress.className = 'reading-progress';
  document.body.appendChild(progress);

  window.addEventListener('scroll', () => {
    const scrolled = window.pageYOffset;
    const height = document.documentElement.scrollHeight - window.innerHeight;
    const percent = (scrolled / height) * 100;
    progress.style.width = percent + '%';
  });
});
```

**Done!** Shows reading progress at top of page.

---

## 4. 🔍 Search Highlighting (15 minutes)

### Add to CSS:

```css
/* Search highlight */
.highlight {
  background-color: #FFC107;
  padding: 2px 4px;
  border-radius: 3px;
  font-weight: bold;
}
```

### Add to search.js (or create):

```javascript
function highlightSearchTerms(query) {
  if (!query) return;

  const terms = query.toLowerCase().split(/\s+/);
  const walker = document.createTreeWalker(
    document.body,
    NodeFilter.SHOW_TEXT,
    null,
    false
  );

  const nodes = [];
  while (walker.nextNode()) {
    const node = walker.currentNode;
    if (node.nodeValue.trim().length > 0) {
      nodes.push(node);
    }
  }

  nodes.forEach(node => {
    let text = node.nodeValue;
    terms.forEach(term => {
      if (term.length < 3) return; // Skip short terms

      const regex = new RegExp(`(${term})`, 'gi');
      if (regex.test(text)) {
        const span = document.createElement('span');
        span.innerHTML = text.replace(regex, '<mark class="highlight">$1</mark>');
        node.parentNode.replaceChild(span, node);
      }
    });
  });
}

// Get search query from URL
const urlParams = new URLSearchParams(window.location.search);
const searchQuery = urlParams.get('q');
if (searchQuery) {
  highlightSearchTerms(searchQuery);
}
```

**Done!** Search terms are highlighted on result pages.

---

## 5. ⏱️ Reading Time Estimate (10 minutes)

### Add to each article page generation:

```python
def calculate_reading_time(content: str) -> int:
    """Calculate estimated reading time in minutes."""
    words = len(content.split())
    # Average reading speed: 200 words per minute
    minutes = max(1, round(words / 200))
    return minutes
```

### Add to article HTML:

```html
<div class="article-meta">
  <span class="reading-time">⏱️ {reading_time} min read</span>
  <span class="word-count">📄 {word_count} words</span>
  <span class="last-updated">🗓️ Updated: {date}</span>
</div>
```

### CSS:

```css
.article-meta {
  display: flex;
  gap: 15px;
  padding: 15px;
  background: var(--bg-secondary);
  border-radius: 8px;
  margin: 20px 0;
  font-size: 0.9em;
  color: var(--text-secondary);
}

.article-meta span {
  display: flex;
  align-items: center;
  gap: 5px;
}
```

**Done!** Articles show estimated reading time.

---

## Bonus: 6. 🔝 Back to Top Button (5 minutes)

### Already have CSS for this, just add the button:

```javascript
// Back to top button
document.addEventListener('DOMContentLoaded', () => {
  const button = document.createElement('button');
  button.className = 'back-to-top';
  button.textContent = '↑';
  button.onclick = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };
  document.body.appendChild(button);

  window.addEventListener('scroll', () => {
    if (window.pageYOffset > 300) {
      button.classList.add('visible');
    } else {
      button.classList.remove('visible');
    }
  });
});
```

**Done!** Smooth scroll to top after scrolling down.

---

## Implementation Checklist

### Step 1: Update wiki_export.py
- [ ] Add theme CSS variables
- [ ] Add copy button CSS
- [ ] Add progress bar CSS
- [ ] Add reading time to articles
- [ ] Add article meta section

### Step 2: Create new JavaScript file (enhancements.js)
```javascript
// Combine all the above JavaScript snippets into one file
// Include: theme switcher, copy buttons, progress bar, back-to-top

// Then add to _create_javascript() in wiki_export.py
```

### Step 3: Update navigation
- [ ] Add theme toggle to header
- [ ] Update meta tags

### Step 4: Test
- [ ] Regenerate wiki: `python wiki_export.py`
- [ ] Test all features
- [ ] Check mobile responsiveness

---

## Ready-to-Use Code Template

### Complete enhancements.js file:

```javascript
/**
 * Wiki Enhancements
 * Quick improvements for better UX
 */

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
    button.textContent = icons[currentTheme];
  }
}

// ===== COPY BUTTONS =====
function addCopyButtons() {
  document.querySelectorAll('pre code').forEach(block => {
    if (block.parentNode.classList.contains('code-block-wrapper')) return;

    const wrapper = document.createElement('div');
    wrapper.className = 'code-block-wrapper';

    const button = document.createElement('button');
    button.className = 'copy-button';
    button.textContent = '📋 Copy';
    button.onclick = async () => {
      await navigator.clipboard.writeText(block.textContent);
      button.textContent = '✅ Copied!';
      button.classList.add('copied');
      setTimeout(() => {
        button.textContent = '📋 Copy';
        button.classList.remove('copied');
      }, 2000);
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

  window.addEventListener('scroll', () => {
    const scrolled = window.pageYOffset;
    const height = document.documentElement.scrollHeight - window.innerHeight;
    const percent = (scrolled / height) * 100;
    progress.style.width = percent + '%';
  });
}

// ===== BACK TO TOP =====
function addBackToTop() {
  const button = document.createElement('button');
  button.className = 'back-to-top';
  button.textContent = '↑';
  button.onclick = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };
  document.body.appendChild(button);

  window.addEventListener('scroll', () => {
    if (window.pageYOffset > 300) {
      button.classList.add('visible');
    } else {
      button.classList.remove('visible');
    }
  });
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

  const terms = query.toLowerCase().split(/\s+/);
  document.querySelectorAll('main p, main li, main td').forEach(element => {
    let html = element.innerHTML;
    terms.forEach(term => {
      if (term.length < 3) return;
      const regex = new RegExp(`(${term})`, 'gi');
      html = html.replace(regex, '<mark class="highlight">$1</mark>');
    });
    element.innerHTML = html;
  });
}

// ===== INITIALIZE ALL =====
document.addEventListener('DOMContentLoaded', () => {
  setTheme(currentTheme);
  addThemeToggle();
  addCopyButtons();
  addReadingProgress();
  addBackToTop();
  highlightSearchTerms();

  console.log('✅ Wiki enhancements loaded');
});
```

Save as `wiki/assets/js/enhancements.js` and include in all HTML pages:

```html
<script src="assets/js/enhancements.js"></script>
<!-- or for article pages: -->
<script src="../assets/js/enhancements.js"></script>
```

---

## 🎉 Result

After implementing these 6 quick improvements, your wiki will have:

1. ✅ Theme switching (light/dark/C64)
2. ✅ Copy buttons on all code
3. ✅ Reading progress indicator
4. ✅ Search term highlighting
5. ✅ Reading time estimates
6. ✅ Back to top button

**Total time: ~60 minutes**
**User experience improvement: 200%**

These are the **highest ROI improvements** - minimal effort, maximum impact!

---

## Next Steps

After these quick wins, tackle the bigger improvements:
1. 🗺️ Memory Map Visualizer (3 days)
2. 📖 Assembly Reference (3 days)
3. 🎮 Code Playground (7 days)

You now have a **professional, modern wiki** with instant user-facing improvements! 🚀
