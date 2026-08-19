// Chunks browser with search and pagination

let allChunks = [];
let filteredChunks = [];
let currentPage = 1;
const chunksPerPage = 50;

async function loadChunks() {
    try {
        const response = await fetch('assets/data/chunks.json');
        allChunks = await response.json();
        filteredChunks = allChunks;
        updateStats();
        displayChunks();
        setupEventListeners();
    } catch (error) {
        console.error('Error loading chunks:', error);
    }
}

function setupEventListeners() {
    // Search
    document.getElementById('chunk-search').addEventListener('input', (e) => {
        const query = e.target.value.toLowerCase();
        if (query) {
            filteredChunks = allChunks.filter(chunk =>
                chunk.full_content.toLowerCase().includes(query) ||
                chunk.doc_title.toLowerCase().includes(query)
            );
        } else {
            filteredChunks = allChunks;
        }
        currentPage = 1;
        updateStats();
        displayChunks();
    });

    // Back to top
    const backToTop = document.getElementById('back-to-top');
    window.addEventListener('scroll', () => {
        if (window.pageYOffset > 300) {
            backToTop.classList.add('visible');
        } else {
            backToTop.classList.remove('visible');
        }
    });
    backToTop.onclick = () => window.scrollTo({ top: 0, behavior: 'smooth' });
}

function updateStats() {
    const stats = document.getElementById('chunk-stats');
    const totalChunks = filteredChunks.length;
    const uniqueDocs = new Set(filteredChunks.map(c => c.doc_id)).size;

    stats.innerHTML = `
        Showing <strong>${totalChunks.toLocaleString()}</strong> chunks from
        <strong>${uniqueDocs}</strong> documents
    `;
}

function displayChunks() {
    const container = document.getElementById('chunks-list');
    container.innerHTML = '';

    const startIdx = (currentPage - 1) * chunksPerPage;
    const endIdx = Math.min(startIdx + chunksPerPage, filteredChunks.length);
    const pageChunks = filteredChunks.slice(startIdx, endIdx);

    if (pageChunks.length === 0) {
        container.innerHTML = '<p style="text-align: center; padding: 40px; color: #718096;">No chunks found.</p>';
        return;
    }

    for (const chunk of pageChunks) {
        const item = createChunkItem(chunk);
        container.appendChild(item);
    }

    displayPagination();
}

function createChunkItem(chunk) {
    const item = document.createElement('div');
    item.className = 'chunk-item';

    const pageInfo = chunk.page ? ` • Page ${chunk.page}` : '';

    item.innerHTML = `
        <div class="chunk-header">
            <a href="docs/${chunk.doc_filename}" class="chunk-doc-link">
                📄 ${escapeHtml(chunk.doc_title)}
            </a>
            <span class="chunk-meta">${chunk.file_type}${pageInfo} • ${chunk.content_length} chars</span>
        </div>
        <div class="chunk-content">${escapeHtml(chunk.content)}</div>
    `;

    return item;
}

function displayPagination() {
    const pagination = document.getElementById('pagination');
    const totalPages = Math.ceil(filteredChunks.length / chunksPerPage);

    if (totalPages <= 1) {
        pagination.innerHTML = '';
        return;
    }

    let html = '<div class="pagination-controls">';

    // Previous
    if (currentPage > 1) {
        html += `<button onclick="goToPage(${currentPage - 1})">← Previous</button>`;
    }

    // Page numbers
    const startPage = Math.max(1, currentPage - 2);
    const endPage = Math.min(totalPages, currentPage + 2);

    if (startPage > 1) {
        html += `<button onclick="goToPage(1)">1</button>`;
        if (startPage > 2) html += '<span>...</span>';
    }

    for (let i = startPage; i <= endPage; i++) {
        const active = i === currentPage ? ' class="active"' : '';
        html += `<button${active} onclick="goToPage(${i})">${i}</button>`;
    }

    if (endPage < totalPages) {
        if (endPage < totalPages - 1) html += '<span>...</span>';
        html += `<button onclick="goToPage(${totalPages})">${totalPages}</button>`;
    }

    // Next
    if (currentPage < totalPages) {
        html += `<button onclick="goToPage(${currentPage + 1})">Next →</button>`;
    }

    html += '</div>';
    pagination.innerHTML = html;
}

function goToPage(page) {
    currentPage = page;
    displayChunks();
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

loadChunks();
