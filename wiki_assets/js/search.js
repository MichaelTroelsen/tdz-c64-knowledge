// Search functionality using Fuse.js

let searchIndex = [];
let fuse = null;

// Load search index
async function loadSearchIndex() {
    try {
        const response = await fetch('assets/data/search.json');
        searchIndex = (await response.json()).items;

        // Initialize Fuse.js
        fuse = new Fuse(searchIndex, {
            keys: ['title', 'description', 'tags'],
            threshold: 0.3,
            includeScore: true,
            minMatchCharLength: 2,
            ignoreLocation: true
        });

        console.log('Search index loaded:', searchIndex.length, 'documents');
    } catch (error) {
        console.error('Error loading search index:', error);
    }
}

// Wire up every search box present on the page. The nav bar's box
// (#nav-search / #nav-search-results) is present on every page; index.html
// additionally has its own larger homepage box (#search-input /
// #search-results). Both share the same underlying Fuse index.
const SEARCH_BOX_IDS = [
    ['nav-search', 'nav-search-results'],
    ['search-input', 'search-results']
];

for (const [inputId, resultsId] of SEARCH_BOX_IDS) {
    const input = document.getElementById(inputId);
    const results = document.getElementById(resultsId);
    if (!input || !results) continue;

    input.addEventListener('input', (event) => handleSearch(event, results));

    document.addEventListener('click', (e) => {
        if (!input.contains(e.target) && !results.contains(e.target)) {
            results.classList.remove('active');
        }
    });
}

function handleSearch(event, searchResults) {
    const query = event.target.value.trim();

    if (query.length < 2) {
        searchResults.classList.remove('active');
        searchResults.innerHTML = '';
        return;
    }

    if (!fuse) {
        searchResults.innerHTML = '<div class="search-result">Loading search index...</div>';
        searchResults.classList.add('active');
        return;
    }

    // Perform search
    const results = fuse.search(query, { limit: 10 });

    if (results.length === 0) {
        searchResults.innerHTML = '<div class="search-result">No results found</div>';
        searchResults.classList.add('active');
        return;
    }

    // Display results
    searchResults.innerHTML = '';
    for (const result of results) {
        const doc = result.item;
        const resultDiv = createSearchResult(doc);
        searchResults.appendChild(resultDiv);
    }
    searchResults.classList.add('active');
}

function createSearchResult(doc) {
    const div = document.createElement('div');
    div.className = 'search-result';

    div.innerHTML = `
        <div class="search-result-title">${escapeHtml(doc.title)}</div>
        <div class="search-result-preview">${escapeHtml(doc.description || '')}</div>
        <div class="search-result-meta">${escapeHtml(doc.category || doc.type)}</div>
    `;

    div.onclick = () => {
        window.location.href = doc.url;
    };

    return div;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ===== TAG CLOUD =====
async function generateTagCloud() {
    const tagCloudElement = document.getElementById('tag-cloud');
    if (!tagCloudElement) return;

    try {
        // Load navigation data (contains tags)
        const response = await fetch('assets/data/navigation.json');
        const nav = await response.json();

        // Get tags with document counts
        const tagData = Object.keys(nav.by_tags).map(tag => ({
            tag: tag,
            count: nav.by_tags[tag].length
        }));

        // Sort by count descending
        tagData.sort((a, b) => b.count - a.count);

        // Calculate size classes
        const maxCount = Math.max(...tagData.map(t => t.count));
        const minCount = Math.min(...tagData.map(t => t.count));

        // Function to determine size class
        function getSizeClass(count) {
            const ratio = (count - minCount) / (maxCount - minCount);
            if (ratio > 0.8) return 'size-xxl';
            if (ratio > 0.6) return 'size-xl';
            if (ratio > 0.4) return 'size-lg';
            if (ratio > 0.2) return 'size-md';
            if (ratio > 0.1) return 'size-sm';
            return 'size-xs';
        }

        // Generate HTML
        const html = tagData.map((item, index) => {
            const sizeClass = getSizeClass(item.count);
            return `
                <span class="tag-cloud-item ${sizeClass}"
                      data-tag="${escapeHtml(item.tag)}"
                      data-count="${item.count}"
                      style="animation: fadeIn 0.3s ease-out ${index * 0.02}s both">
                    ${escapeHtml(item.tag)}
                    <span class="tag-count">${item.count}</span>
                </span>
            `;
        }).join('');

        tagCloudElement.innerHTML = html;

        // Add click handlers
        document.querySelectorAll('.tag-cloud-item').forEach(item => {
            item.onclick = () => {
                const tag = item.getAttribute('data-tag');
                // Navigate to documents page with tag filter
                window.location.href = `documents.html#tag-${tag}`;
            };
        });

        // Add stats
        const statsContainer = document.getElementById('tag-cloud-stats');
        if (statsContainer) {
            const totalTags = tagData.length;
            const totalDocs = tagData.reduce((sum, t) => sum + t.count, 0);
            const avgDocs = Math.round(totalDocs / totalTags);

            statsContainer.innerHTML = `
                <span>📊 ${totalTags} tags</span>
                <span>📚 ${totalDocs} total document-tag mappings</span>
                <span>📈 ${avgDocs} avg docs per tag</span>
            `;
        }

    } catch (error) {
        console.error('Failed to generate tag cloud:', error);
        tagCloudElement.innerHTML = '<div class="tag-cloud-empty">Unable to load tag cloud</div>';
    }
}

function setupTagCloudControls() {
    const controls = document.querySelectorAll('.tag-cloud-control-btn');
    const tagCloud = document.getElementById('tag-cloud');
    if (!tagCloud) return;

    controls.forEach(btn => {
        btn.onclick = async () => {
            // Update active state
            controls.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            const sortBy = btn.getAttribute('data-sort');

            // Re-fetch and re-sort
            try {
                const response = await fetch('assets/data/navigation.json');
                const nav = await response.json();

                let tagData = Object.keys(nav.by_tags).map(tag => ({
                    tag: tag,
                    count: nav.by_tags[tag].length
                }));

                // Sort based on selected option
                if (sortBy === 'alpha') {
                    tagData.sort((a, b) => a.tag.localeCompare(b.tag));
                } else if (sortBy === 'count') {
                    tagData.sort((a, b) => b.count - a.count);
                } else if (sortBy === 'random') {
                    tagData = tagData.sort(() => Math.random() - 0.5);
                }

                // Calculate size classes
                const maxCount = Math.max(...tagData.map(t => t.count));
                const minCount = Math.min(...tagData.map(t => t.count));

                function getSizeClass(count) {
                    const ratio = (count - minCount) / (maxCount - minCount);
                    if (ratio > 0.8) return 'size-xxl';
                    if (ratio > 0.6) return 'size-xl';
                    if (ratio > 0.4) return 'size-lg';
                    if (ratio > 0.2) return 'size-md';
                    if (ratio > 0.1) return 'size-sm';
                    return 'size-xs';
                }

                // Re-render
                const html = tagData.map((item, index) => {
                    const sizeClass = getSizeClass(item.count);
                    return `
                        <span class="tag-cloud-item ${sizeClass}"
                              data-tag="${escapeHtml(item.tag)}"
                              data-count="${item.count}"
                              style="animation: fadeIn 0.3s ease-out ${index * 0.02}s both">
                            ${escapeHtml(item.tag)}
                            <span class="tag-count">${item.count}</span>
                        </span>
                    `;
                }).join('');

                tagCloud.innerHTML = html;

                // Re-add click handlers
                document.querySelectorAll('.tag-cloud-item').forEach(item => {
                    item.onclick = () => {
                        const tag = item.getAttribute('data-tag');
                        window.location.href = `documents.html#tag-${tag}`;
                    };
                });

            } catch (error) {
                console.error('Failed to re-sort tag cloud:', error);
            }
        };
    });
}

// Load search index on page load
loadSearchIndex();
