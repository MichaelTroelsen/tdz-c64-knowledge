// Main JavaScript for TDZ C64 Knowledge Base Wiki

// Load navigation data and populate categories
async function loadNavigation() {
    try {
        const response = await fetch('assets/data/navigation.json');
        const nav = await response.json();

        // Populate category list
        const categoryList = document.getElementById('category-list');
        if (categoryList) {
            categoryList.innerHTML = '';

            // Tags
            for (const tag of nav.all_tags) {
                const count = nav.by_tags[tag].length;
                const card = document.createElement('div');
                card.className = 'category-card';
                card.innerHTML = `<h3>${escapeHtml(tag)}</h3><p>${count} documents</p>`;
                card.onclick = () => window.location.href = `#tag-${tag}`;
                categoryList.appendChild(card);
            }
        }
    } catch (error) {
        console.error('Error loading navigation:', error);
    }
}

// Load and display documents
async function loadDocuments() {
    try {
        const response = await fetch('assets/data/documents.json');
        const documents = await response.json();

        const docList = document.getElementById('doc-list');
        if (docList) {
            docList.innerHTML = '';

            // Show first 20 documents
            for (const doc of documents.slice(0, 20)) {
                const card = createDocCard(doc);
                docList.appendChild(card);
            }
        }
    } catch (error) {
        console.error('Error loading documents:', error);
    }
}

// Create document card element
function createDocCard(doc) {
    const card = document.createElement('div');
    card.className = 'doc-card';

    const safeFilename = doc.id.replace(/[^\w\-]/g, '_') + '.html';
    const tags = doc.tags.map(t => `<span class="tag">${escapeHtml(t)}</span>`).join('');

    card.innerHTML = `
        <h3><a href="docs/${safeFilename}">${escapeHtml(doc.title)}</a></h3>
        <div class="doc-card-meta">
            ${doc.file_type} • ${doc.total_chunks} chunks
        </div>
        <div class="doc-tags">${tags}</div>
    `;

    return card;
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Initialize on page load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        loadNavigation();
        loadDocuments();
        generateTagCloud();
        setupTagCloudControls();
    });
} else {
    loadNavigation();
    loadDocuments();
    generateTagCloud();
    setupTagCloudControls();
}
