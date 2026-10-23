/**
 * ==========================================================
 *  KANJI RECOGNITION - Frontend Script
 * ==========================================================
 *  Canvas drawing, API communication, result rendering.
 * ==========================================================
 */

// ==================== DOM ELEMENTS ====================
const canvas = document.getElementById('drawCanvas');
const ctx = canvas.getContext('2d');
const canvasWrapper = document.getElementById('canvasWrapper');

const btnClear = document.getElementById('btnClear');
const btnPredict = document.getElementById('btnPredict');
const brushSizeInput = document.getElementById('brushSize');
const brushSizeLabel = document.getElementById('brushSizeLabel');
const themeToggle = document.getElementById('themeToggle');

const resultsPlaceholder = document.getElementById('resultsPlaceholder');
const loadingOverlay = document.getElementById('loadingOverlay');
const resultsContainer = document.getElementById('resultsContainer');

// CNN elements
const cnnTop1Kanji = document.getElementById('cnnTop1Kanji');
const cnnTop1Confidence = document.getElementById('cnnTop1Confidence');
const cnnConfidenceFill = document.getElementById('cnnConfidenceFill');
const cnnTop3List = document.getElementById('cnnTop3List');
const cnnError = document.getElementById('cnnError');

// ResNet elements
const resnetTop1Kanji = document.getElementById('resnetTop1Kanji');
const resnetTop1Confidence = document.getElementById('resnetTop1Confidence');
const resnetConfidenceFill = document.getElementById('resnetConfidenceFill');
const resnetTop3List = document.getElementById('resnetTop3List');
const resnetError = document.getElementById('resnetError');

// ==================== STATE ====================
let isDrawing = false;
let lastX = 0;
let lastY = 0;
let brushSize = 8;
let hasDrawn = false;

// API endpoint
const API_URL = window.location.origin + '/predict';

// ==================== CANVAS SETUP ====================
function initCanvas() {
    // Match training dataset: black background, white strokes
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Brush settings (white ink)
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = brushSize;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.imageSmoothingEnabled = true;
}

initCanvas();

// ==================== DRAWING FUNCTIONS ====================

/**
 * Lấy tọa độ trên canvas từ event (hỗ trợ mouse & touch).
 */
function getCanvasCoords(e) {
    const rect = canvas.getBoundingClientRect();

    // Scale factor (canvas có thể bị CSS resize)
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    let clientX, clientY;

    if (e.touches && e.touches.length > 0) {
        clientX = e.touches[0].clientX;
        clientY = e.touches[0].clientY;
    } else {
        clientX = e.clientX;
        clientY = e.clientY;
    }

    return {
        x: (clientX - rect.left) * scaleX,
        y: (clientY - rect.top) * scaleY
    };
}

function startDrawing(e) {
    e.preventDefault();
    isDrawing = true;
    hasDrawn = true;
    canvasWrapper.classList.add('drawing');

    const coords = getCanvasCoords(e);
    lastX = coords.x;
    lastY = coords.y;

    // Vẽ chấm tại điểm bắt đầu (white ink)
    ctx.beginPath();
    ctx.arc(lastX, lastY, brushSize / 2, 0, Math.PI * 2);
    ctx.fillStyle = '#ffffff';
    ctx.fill();
}

function draw(e) {
    if (!isDrawing) return;
    e.preventDefault();

    const coords = getCanvasCoords(e);

    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(coords.x, coords.y);
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = brushSize;
    ctx.stroke();

    lastX = coords.x;
    lastY = coords.y;
}

function stopDrawing(e) {
    if (e) e.preventDefault();
    isDrawing = false;
    canvasWrapper.classList.remove('drawing');
}

// --- Mouse Events ---
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseleave', stopDrawing);

// --- Touch Events ---
canvas.addEventListener('touchstart', startDrawing, { passive: false });
canvas.addEventListener('touchmove', draw, { passive: false });
canvas.addEventListener('touchend', stopDrawing, { passive: false });
canvas.addEventListener('touchcancel', stopDrawing, { passive: false });

// ==================== CLEAR CANVAS ====================
function clearCanvas() {
    // Reset to training polarity: black background
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    hasDrawn = false;

    // Reset kết quả
    resultsPlaceholder.style.display = 'flex';
    resultsContainer.style.display = 'none';
    loadingOverlay.style.display = 'none';
    cnnError.style.display = 'none';
    resnetError.style.display = 'none';
}

btnClear.addEventListener('click', clearCanvas);

// ==================== BRUSH SIZE ====================
brushSizeInput.addEventListener('input', (e) => {
    brushSize = parseInt(e.target.value);
    brushSizeLabel.textContent = brushSize;
    ctx.lineWidth = brushSize;
});

// ==================== THEME TOGGLE ====================
function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('kanji-theme', theme);
}

function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme');
    setTheme(current === 'dark' ? 'light' : 'dark');
}

// Load saved theme
const savedTheme = localStorage.getItem('kanji-theme') || 'dark';
setTheme(savedTheme);

themeToggle.addEventListener('click', toggleTheme);

// ==================== PREDICT ====================

/**
 * Gửi ảnh canvas tới API và hiển thị kết quả.
 */
async function predict() {
    if (!hasDrawn) {
        // Highlight canvas border
        canvasWrapper.style.borderColor = 'var(--accent-danger)';
        setTimeout(() => {
            canvasWrapper.style.borderColor = '';
        }, 1000);
        return;
    }

    // Hiển thị loading
    resultsPlaceholder.style.display = 'none';
    loadingOverlay.style.display = 'flex';
    resultsContainer.style.display = 'none';
    btnPredict.disabled = true;

    try {
        // Lấy ảnh từ canvas dưới dạng base64
        const imageData = canvas.toDataURL('image/png');

        // Gọi API
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `HTTP ${response.status}`);
        }

        const result = await response.json();

        // Kiểm tra lỗi chung
        if (result.error) {
            throw new Error(result.error);
        }

        // Render kết quả
        renderResults(result);

    } catch (error) {
        console.error('Prediction error:', error);

        // Hiển thị lỗi trên cả 2 card
        loadingOverlay.style.display = 'none';
        resultsContainer.style.display = 'flex';

        renderModelResult('cnn', {
            error: error.message,
            top1: '?',
            confidence: 0,
            top3: []
        });
        renderModelResult('resnet', {
            error: error.message,
            top1: '?',
            confidence: 0,
            top3: []
        });
    } finally {
        btnPredict.disabled = false;
    }
}

btnPredict.addEventListener('click', predict);

// ==================== RENDER RESULTS ====================

function renderResults(result) {
    loadingOverlay.style.display = 'none';
    resultsContainer.style.display = 'flex';

    // Trigger re-animation
    resultsContainer.querySelectorAll('.result-card').forEach(card => {
        card.style.animation = 'none';
        // Force reflow
        void card.offsetHeight;
        card.style.animation = '';
    });

    renderModelResult('cnn', result.cnn);
    renderModelResult('resnet', result.resnet);
}

function renderModelResult(modelType, data) {
    const isResnet = modelType === 'resnet';
    const top1KanjiEl = isResnet ? resnetTop1Kanji : cnnTop1Kanji;
    const top1ConfEl = isResnet ? resnetTop1Confidence : cnnTop1Confidence;
    const confFillEl = isResnet ? resnetConfidenceFill : cnnConfidenceFill;
    const top3ListEl = isResnet ? resnetTop3List : cnnTop3List;
    const errorEl = isResnet ? resnetError : cnnError;

    // Handle error
    if (data.error) {
        errorEl.textContent = data.error;
        errorEl.style.display = 'block';
    } else {
        errorEl.style.display = 'none';
    }

    // Top 1
    top1KanjiEl.textContent = data.top1 || '?';
    const confPercent = Math.round((data.confidence || 0) * 100);
    top1ConfEl.textContent = confPercent + '%';

    // Animate confidence bar
    setTimeout(() => {
        confFillEl.style.width = confPercent + '%';
    }, 100);

    // Top 3
    top3ListEl.innerHTML = '';
    if (data.top3 && data.top3.length > 0) {
        data.top3.forEach((item, index) => {
            const prob = Math.round(item.prob * 100);
            const isTop1 = index === 0;

            const div = document.createElement('div');
            div.className = 'top3-item' + (isTop1 ? ' highlight' : '');

            div.innerHTML = `
                <span class="top3-rank">#${index + 1}</span>
                <span class="top3-kanji">${escapeHtml(item.label)}</span>
                <div class="top3-bar-wrapper">
                    <div class="top3-bar">
                        <div class="top3-bar-fill" style="width: 0%"></div>
                    </div>
                    <span class="top3-prob">${prob}%</span>
                </div>
            `;

            top3ListEl.appendChild(div);

            // Animate bar fill
            setTimeout(() => {
                const fill = div.querySelector('.top3-bar-fill');
                if (fill) fill.style.width = prob + '%';
            }, 200 + index * 150);
        });
    }
}

/**
 * Escape HTML để tránh XSS.
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ==================== KEYBOARD SHORTCUTS ====================
document.addEventListener('keydown', (e) => {
    // Ctrl + Enter → Predict
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        predict();
    }
    // Escape → Clear
    if (e.key === 'Escape') {
        clearCanvas();
    }
});

// ==================== INIT ====================
console.log('Kanji Recognition app initialized.');
console.log('Shortcuts: Ctrl+Enter = Predict, Escape = Clear');
