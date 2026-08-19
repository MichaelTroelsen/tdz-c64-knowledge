// PDF.js viewer integration

let pdfDoc = null;
let pageNum = 1;
let pageRendering = false;
let pageNumPending = null;
let scale = 1.5;
const canvas = document.getElementById('pdf-canvas');
const ctx = canvas.getContext('2d');

// Get PDF filename from URL parameter
const urlParams = new URLSearchParams(window.location.search);
const pdfFile = urlParams.get('file');

if (!pdfFile) {
    document.getElementById('pdf-title').textContent = 'No PDF specified';
    document.getElementById('pdf-container').innerHTML = '<p style="text-align: center; padding: 40px;">No PDF file specified. Please select a document to view.</p>';
} else {
    loadPDF(`pdfs/${pdfFile}`);
}

function loadPDF(url) {
    // Configure PDF.js worker
    pdfjsLib.GlobalWorkerOptions.workerSrc = 'lib/pdf.worker.min.js';

    // Load the PDF
    pdfjsLib.getDocument(url).promise.then(function(pdf) {
        pdfDoc = pdf;
        document.getElementById('page-info').textContent = `Page ${pageNum} of ${pdfDoc.numPages}`;

        // Set download link
        document.getElementById('download-link').href = url;
        document.getElementById('download-link').download = pdfFile;

        // Initial render
        renderPage(pageNum);
    }).catch(function(error) {
        console.error('Error loading PDF:', error);
        document.getElementById('pdf-container').innerHTML = `
            <p style="text-align: center; padding: 40px; color: #e53e3e;">
                Error loading PDF: ${error.message}<br>
                File: ${url}
            </p>
        `;
    });
}

function renderPage(num) {
    pageRendering = true;

    pdfDoc.getPage(num).then(function(page) {
        const viewport = page.getViewport({ scale: scale });
        canvas.height = viewport.height;
        canvas.width = viewport.width;

        const renderContext = {
            canvasContext: ctx,
            viewport: viewport
        };

        const renderTask = page.render(renderContext);

        renderTask.promise.then(function() {
            pageRendering = false;
            if (pageNumPending !== null) {
                renderPage(pageNumPending);
                pageNumPending = null;
            }
        });
    });

    document.getElementById('page-info').textContent = `Page ${num} of ${pdfDoc.numPages}`;
    updateButtons();
}

function queueRenderPage(num) {
    if (pageRendering) {
        pageNumPending = num;
    } else {
        renderPage(num);
    }
}

function onPrevPage() {
    if (pageNum <= 1) return;
    pageNum--;
    queueRenderPage(pageNum);
}

function onNextPage() {
    if (pageNum >= pdfDoc.numPages) return;
    pageNum++;
    queueRenderPage(pageNum);
}

function onZoomIn() {
    scale += 0.25;
    queueRenderPage(pageNum);
}

function onZoomOut() {
    if (scale <= 0.5) return;
    scale -= 0.25;
    queueRenderPage(pageNum);
}

function updateButtons() {
    document.getElementById('prev-page').disabled = (pageNum <= 1);
    document.getElementById('next-page').disabled = (pageNum >= pdfDoc.numPages);
}

// Event listeners
document.getElementById('prev-page').addEventListener('click', onPrevPage);
document.getElementById('next-page').addEventListener('click', onNextPage);
document.getElementById('zoom-in').addEventListener('click', onZoomIn);
document.getElementById('zoom-out').addEventListener('click', onZoomOut);

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    if (e.key === 'ArrowLeft') onPrevPage();
    if (e.key === 'ArrowRight') onNextPage();
    if (e.key === '+' || e.key === '=') onZoomIn();
    if (e.key === '-' || e.key === '_') onZoomOut();
});
