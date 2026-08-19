// Enhanced entities browser with clickable entities and navigation

let entitiesData = {};
let currentFilter = '';
let currentTypeFilter = 'all';

async function loadEntities() {
    try {
        const response = await fetch('assets/data/entities.json');
        entitiesData = await response.json();
        initializeUI();
        displayEntities(entitiesData);
    } catch (error) {
        console.error('Error loading entities:', error);
    }
}

function initializeUI() {
    // Create type navigation buttons
    const typeNav = document.getElementById('entity-type-nav');
    typeNav.innerHTML = '';

    const allBtn = createTypeButton('all', 'All Types', true);
    typeNav.appendChild(allBtn);

    for (const entityType of Object.keys(entitiesData)) {
        const count = entitiesData[entityType].length;
        const btn = createTypeButton(entityType, `${entityType} (${count})`);
        typeNav.appendChild(btn);
    }

    // Update stats
    updateStats();

    // Setup modal close handlers
    setupModal();

    // Setup back to top button
    setupBackToTop();
}

function createTypeButton(type, label, active = false) {
    const btn = document.createElement('button');
    btn.className = 'entity-type-btn' + (active ? ' active' : '');
    btn.textContent = label;
    btn.onclick = () => filterByType(type);
    return btn;
}

function filterByType(type) {
    currentTypeFilter = type;

    // Update button states
    document.querySelectorAll('.entity-type-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.classList.add('active');

    // Filter and display
    const filtered = type === 'all' ? entitiesData : { [type]: entitiesData[type] };
    displayEntities(filtered, currentFilter);
}

function updateStats() {
    const stats = document.getElementById('entity-stats');
    const totalEntities = Object.values(entitiesData).reduce((sum, arr) => sum + arr.length, 0);
    const totalTypes = Object.keys(entitiesData).length;
    const totalDocs = new Set(
        Object.values(entitiesData).flatMap(entities =>
            entities.flatMap(e => e.documents.map(d => d.id))
        )
    ).size;

    stats.innerHTML = `
        <strong>${totalEntities}</strong> entities across
        <strong>${totalTypes}</strong> types, referenced in
        <strong>${totalDocs}</strong> documents
    `;
}

function displayEntities(data, searchQuery = '') {
    const container = document.getElementById('entities-container');
    if (!container) return;

    container.innerHTML = '';

    for (const [entityType, entities] of Object.entries(data)) {
        // Filter by search query
        let filteredEntities = entities;
        if (searchQuery) {
            filteredEntities = entities.filter(e =>
                e.text.toLowerCase().includes(searchQuery.toLowerCase())
            );
        }

        if (filteredEntities.length === 0) continue;

        const section = document.createElement('div');
        section.className = 'entity-type-section';
        section.id = `type-${entityType}`;

        // Collapsible header
        const header = document.createElement('div');
        header.className = 'entity-type-header';
        header.innerHTML = `
            <h3>${entityType} (${filteredEntities.length})</h3>
            <span class="collapse-icon">▼</span>
        `;
        header.onclick = () => toggleSection(section);
        section.appendChild(header);

        // Entity list
        const list = document.createElement('div');
        list.className = 'entity-list';

        for (const entity of filteredEntities) {
            const item = document.createElement('div');
            item.className = 'entity-item';
            item.innerHTML = `
                <span class="entity-name">${escapeHtml(entity.text)}</span>
                <span class="entity-count">${entity.doc_count} docs</span>
            `;
            item.onclick = () => showEntityDetails(entity, entityType);
            list.appendChild(item);
        }

        section.appendChild(list);
        container.appendChild(section);
    }

    if (container.children.length === 0) {
        container.innerHTML = '<p style="text-align: center; color: #718096; padding: 40px;">No entities found matching your search.</p>';
    }
}

function toggleSection(section) {
    section.classList.toggle('collapsed');
}

function showEntityDetails(entity, entityType) {
    const modal = document.getElementById('entity-modal');
    const modalTitle = document.getElementById('modal-title');
    const modalBody = document.getElementById('modal-body');

    modalTitle.textContent = `${entity.text} (${entityType})`;

    modalBody.innerHTML = `
        <div class="modal-section">
            <h3>Overview</h3>
            <p>
                <strong>Type:</strong> ${entityType}<br>
                <strong>Documents:</strong> ${entity.doc_count}<br>
                <strong>Confidence:</strong> ${(entity.confidence * 100).toFixed(0)}%
            </p>
        </div>

        <div class="modal-section">
            <h3>Related Documents (${entity.documents.length})</h3>
            ${entity.documents.map(doc => `
                <a href="docs/${doc.filename}" class="document-link">
                    📄 ${escapeHtml(doc.title)}
                </a>
            `).join('')}
        </div>
    `;

    modal.classList.add('active');
}

function setupModal() {
    const modal = document.getElementById('entity-modal');
    const closeBtn = document.querySelector('.modal-close');

    closeBtn.onclick = () => {
        modal.classList.remove('active');
    };

    window.onclick = (event) => {
        if (event.target === modal) {
            modal.classList.remove('active');
        }
    };

    // Close on Escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            modal.classList.remove('active');
        }
    });
}

function setupBackToTop() {
    const backToTop = document.getElementById('back-to-top');

    window.addEventListener('scroll', () => {
        if (window.pageYOffset > 300) {
            backToTop.classList.add('visible');
        } else {
            backToTop.classList.remove('visible');
        }
    });

    backToTop.onclick = () => {
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    };
}

// Filter entities by search
const filterInput = document.getElementById('entity-filter');
if (filterInput) {
    filterInput.addEventListener('input', (e) => {
        currentFilter = e.target.value.trim();

        const filtered = currentTypeFilter === 'all'
            ? entitiesData
            : { [currentTypeFilter]: entitiesData[currentTypeFilter] };

        displayEntities(filtered, currentFilter);
    });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Initialize
loadEntities();
