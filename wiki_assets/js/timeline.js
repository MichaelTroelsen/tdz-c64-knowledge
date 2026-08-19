// Timeline browser

async function loadTimeline() {
    try {
        const response = await fetch('assets/data/events.json');
        const events = await response.json();
        displayTimeline(events);
    } catch (error) {
        console.error('Error loading timeline:', error);
    }
}

function displayTimeline(events) {
    const container = document.getElementById('timeline-container');
    if (!container) return;

    container.innerHTML = '';

    if (events.length === 0) {
        container.innerHTML = '<p>No timeline events found.</p>';
        return;
    }

    const timeline = document.createElement('div');
    timeline.className = 'timeline';

    for (const event of events) {
        const eventDiv = document.createElement('div');
        eventDiv.className = 'timeline-event';

        eventDiv.innerHTML = `
            <div class="timeline-year">${event.year || 'Unknown'}</div>
            <div class="timeline-title">${escapeHtml(event.title)}</div>
            <div class="timeline-desc">${escapeHtml(event.description || '')}</div>
            <div class="timeline-meta">
                ${event.type} • Confidence: ${(event.confidence * 100).toFixed(0)}%
            </div>
        `;

        timeline.appendChild(eventDiv);
    }

    container.appendChild(timeline);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

loadTimeline();
