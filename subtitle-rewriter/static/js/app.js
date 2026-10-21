const videoInput = document.getElementById('video-input');
const srtInput = document.getElementById('srt-input');
const videoBox = document.getElementById('video-box');
const srtBox = document.getElementById('srt-box');
const videoPlaceholder = document.getElementById('video-placeholder');
const srtPlaceholder = document.getElementById('srt-placeholder');
const videoSelected = document.getElementById('video-selected');
const srtSelected = document.getElementById('srt-selected');
const analyzeBtn = document.getElementById('analyze-btn');
const progress = document.getElementById('progress');
const progressText = document.getElementById('progress-text');

let videoFile = null;
let srtFile = null;
const apiKeyInput = document.getElementById('api-key-input');

// Click to open file picker
videoBox.addEventListener('click', () => videoInput.click());
srtBox.addEventListener('click', () => srtInput.click());

// Handle file selection
videoInput.addEventListener('change', (e) => {
    videoFile = e.target.files[0] || null;
    if (videoFile) {
        document.getElementById('video-name').textContent = videoFile.name;
        document.getElementById('video-size').textContent = formatSize(videoFile.size);
        videoPlaceholder.hidden = true;
        videoSelected.hidden = false;
        videoBox.classList.add('has-file');
    } else {
        videoPlaceholder.hidden = false;
        videoSelected.hidden = true;
        videoBox.classList.remove('has-file');
    }
    updateBtn();
});

srtInput.addEventListener('change', (e) => {
    srtFile = e.target.files[0] || null;
    if (srtFile) {
        document.getElementById('srt-name').textContent = srtFile.name;
        srtPlaceholder.hidden = true;
        srtSelected.hidden = false;
        srtBox.classList.add('has-file');
    } else {
        srtPlaceholder.hidden = false;
        srtSelected.hidden = true;
        srtBox.classList.remove('has-file');
    }
    updateBtn();
});

function updateBtn() {
    const hasKey = apiKeyInput.value.trim().length > 0;
    analyzeBtn.disabled = !(videoFile && srtFile && hasKey);
}

apiKeyInput.addEventListener('input', updateBtn);

function formatSize(bytes) {
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(0) + ' KB';
    if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    return (bytes / (1024 * 1024 * 1024)).toFixed(2) + ' GB';
}

// Submit
analyzeBtn.addEventListener('click', async () => {
    if (!videoFile || !srtFile) return;

    const formData = new FormData();
    formData.append('video', videoFile);
    formData.append('srt', srtFile);
    formData.append('api_key', apiKeyInput.value.trim());
    formData.append('custom_prompt', document.getElementById('prompt-input').value.trim());

    analyzeBtn.disabled = true;
    progress.hidden = false;
    progressText.textContent = 'Đang upload...';

    try {
        const res = await fetch('/api/jobs', { method: 'POST', body: formData });
        const data = await res.json();

        if (!res.ok) {
            alert('Lỗi: ' + (data.error || 'Không xác định'));
            analyzeBtn.disabled = false;
            progress.hidden = true;
            return;
        }

        // Start polling
        progressText.textContent = 'Đang xử lý (upload + Gemini)...';
        pollStatus(data.id);
    } catch (err) {
        alert('Lỗi mạng: ' + err.message);
        analyzeBtn.disabled = false;
        progress.hidden = true;
    }
});

function pollStatus(jobId) {
    const interval = setInterval(async () => {
        try {
            const res = await fetch(`/api/jobs/${jobId}/status`);
            const data = await res.json();

            if (data.status === 'uploading') {
                progressText.textContent = 'Đang upload video lên Gemini...';
            } else if (data.status === 'processing') {
                progressText.textContent = 'Đang phân tích và viết lại phụ đề...';
            } else if (data.status === 'done') {
                clearInterval(interval);
                progress.hidden = true;
                location.reload();
            } else if (data.status === 'failed') {
                clearInterval(interval);
                progress.hidden = true;
                alert('Xử lý thất bại: ' + data.error);
                location.reload();
            }
        } catch (e) {
            // Network error, keep polling
        }
    }, 3000);
}
