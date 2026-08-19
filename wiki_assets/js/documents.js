// Documents browser with filtering and sorting

let allDocuments = [];
let filteredDocuments = [];
let currentTypeFilter = 'all';
let currentSort = 'title';

async function loadDocuments() {
    try {
        const response = await fetch('assets/data/documents.json');
        allDocuments = await response.json();
        filteredDocuments = allDocuments;
        displayDocuments();
        setupEventListeners();
    } catch (error) {
        console.error('Error loading documents:', error);
    }
}

function setupEventListeners() {
    // Search
    document.getElementById('doc-search').addEventListener('input', (e) => {
        filterAndDisplay();
    });

    // Type filters
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            currentTypeFilter = e.target.dataset.type;
            filterAndDisplay();
        });
    });

    // Sort
    document.getElementById('sort-select').addEventListener('change', (e) => {
        currentSort = e.target.value;
        displayDocuments();
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

function filterAndDisplay() {
    const searchQuery = document.getElementById('doc-search').value.toLowerCase();

    filteredDocuments = allDocuments.filter(doc => {
        // Type filter
        if (currentTypeFilter !== 'all' && doc.file_type.toLowerCase() !== currentTypeFilter) {
            return false;
        }

        // Search filter
        if (searchQuery) {
            return doc.title.toLowerCase().includes(searchQuery) ||
                   doc.tags.some(tag => tag.toLowerCase().includes(searchQuery));
        }

        return true;
    });

    displayDocuments();
}

function displayDocuments() {
    // Sort
    const sorted = [...filteredDocuments].sort((a, b) => {
        switch (currentSort) {
            case 'title':
                return a.title.localeCompare(b.title);
            case 'title-desc':
                return b.title.localeCompare(a.title);
            case 'chunks':
                return b.total_chunks - a.total_chunks;
            case 'chunks-asc':
                return a.total_chunks - b.total_chunks;
            case 'date':
                return new Date(b.indexed_at) - new Date(a.indexed_at);
            default:
                return 0;
        }
    });

    const grid = document.getElementById('documents-grid');
    grid.innerHTML = '';

    if (sorted.length === 0) {
        grid.innerHTML = '<p style="text-align: center; padding: 40px; color: #718096;">No documents found.</p>';
        return;
    }

    for (const doc of sorted) {
        const card = createDocCard(doc);
        grid.appendChild(card);
    }
}

function createDocCard(doc) {
    const card = document.createElement('div');
    card.className = 'doc-card';

    const safeFilename = doc.id.replace(/[^\w\-]/g, '_') + '.html';
    const isPDF = doc.file_type.toLowerCase() === 'pdf';
    const tags = doc.tags.map(t => `<span class="tag">${escapeHtml(t)}</span>`).join('');

    // Determine if source file viewing is available
    const hasSourceFile = doc.file_path_in_wiki && doc.file_path_in_wiki.length > 0;

    card.innerHTML = `
        <h3><a href="docs/${safeFilename}">${escapeHtml(doc.title)}</a></h3>
        <div class="doc-card-meta">
            ${doc.file_type} • ${doc.total_chunks} chunks
            ${doc.total_pages ? ` • ${doc.total_pages} pages` : ''}
        </div>
        <div class="doc-tags">${tags}</div>
        <div class="doc-actions">
            ${isPDF && doc.pdf_available ? `<a href="viewer.html?file=pdfs/${doc.id.replace(/[^\w\-]/g, '_')}.pdf&name=${escapeHtml(doc.filename)}&type=pdf" class="view-btn view-pdf-btn">📄 View PDF</a>` : ''}
            ${hasSourceFile ? `<a href="viewer.html?file=${doc.file_path_in_wiki}&name=${escapeHtml(doc.filename)}&type=${doc.file_type}" class="view-btn view-source-btn">📁 View Source</a>` : ''}
        </div>
    `;

    return card;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

loadDocuments();
