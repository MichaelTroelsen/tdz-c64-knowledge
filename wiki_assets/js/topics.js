// Topics and clusters browser

async function loadTopicsAndClusters() {
    try {
        const [topicsResp, clustersResp] = await Promise.all([
            fetch('assets/data/topics.json'),
            fetch('assets/data/clusters.json')
        ]);

        const topics = await topicsResp.json();
        const clusters = await clustersResp.json();

        displayTopics(topics);
        displayClusters(clusters);
    } catch (error) {
        console.error('Error loading topics/clusters:', error);
    }
}

function displayTopics(topicsData) {
    const container = document.getElementById('topics-container');
    if (!container) return;

    container.innerHTML = '';

    for (const [modelType, topics] of Object.entries(topicsData)) {
        const section = document.createElement('div');
        section.innerHTML = `<h3>${modelType.toUpperCase()} (${topics.length} topics)</h3>`;

        const grid = document.createElement('div');
        grid.className = 'doc-grid';

        for (const topic of topics) {
            const card = document.createElement('div');
            card.className = 'doc-card';
            card.innerHTML = `
                <h3>Topic ${topic.number}</h3>
                <div class="doc-card-meta">${escapeHtml(topic.words)}</div>
                ${topic.coherence ? `<div class="doc-tags"><span class="tag">Coherence: ${topic.coherence}</span></div>` : ''}
            `;
            grid.appendChild(card);
        }

        section.appendChild(grid);
        container.appendChild(section);
    }
}

function displayClusters(clustersData) {
    const container = document.getElementById('clusters-container');
    if (!container) return;

    container.innerHTML = '';

    for (const [algorithm, clusters] of Object.entries(clustersData)) {
        const section = document.createElement('div');
        section.innerHTML = `<h3>${algorithm.toUpperCase()} (${clusters.length} clusters)</h3>`;

        const grid = document.createElement('div');
        grid.className = 'doc-grid';

        for (const cluster of clusters) {
            const card = document.createElement('div');
            card.className = 'doc-card';

            // Create document list
            let docsList = '';
            if (cluster.documents && cluster.documents.length > 0) {
                const displayDocs = cluster.documents.slice(0, 10); // Show first 10
                docsList = '<div style="margin-top: 12px;"><ul style="list-style: none; padding: 0; font-size: 0.9em;">';
                for (const doc of displayDocs) {
                    const safeFilename = doc.id.replace(/[^\w\-]/g, '_') + '.html';
                    docsList += `<li style="margin: 4px 0;"><a href="docs/${safeFilename}" style="color: var(--accent-color); text-decoration: none;">${escapeHtml(doc.title)}</a></li>`;
                }
                if (cluster.documents.length > 10) {
                    docsList += `<li style="margin: 8px 0; font-style: italic; color: var(--text-muted);">...and ${cluster.documents.length - 10} more</li>`;
                }
                docsList += '</ul></div>';
            }

            card.innerHTML = `
                <h3>Cluster ${cluster.number}</h3>
                <div class="doc-card-meta">${cluster.doc_count} documents</div>
                ${docsList}
            `;
            grid.appendChild(card);
        }

        section.appendChild(grid);
        container.appendChild(section);
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

loadTopicsAndClusters();
