let sessionId = null;
let uploadedModels = [];
let uploadedImages = [];
let currentResults = null;
let currentImageIndex = 0;
let imageList = [];

// Tab switching
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        
        btn.classList.add('active');
        document.getElementById(btn.dataset.tab + '-tab').classList.add('active');
    });
});

// Model upload
const modelUploadArea = document.getElementById('model-upload-area');
const modelInput = document.getElementById('model-input');

setupDragAndDrop(modelUploadArea, modelInput, handleModelUpload);

function handleModelUpload(files) {
    if (!files || files.length === 0) {
        alert('请选择要上传的文件');
        return;
    }
    
    if (!sessionId) {
        sessionId = generateSessionId();
    }
    
    const formData = new FormData();
    formData.append('session_id', sessionId);
    
    Array.from(files).forEach(file => {
        formData.append('files[]', file);
    });
    
    const uploadArea = document.getElementById('model-upload-area');
    const progressContainer = document.getElementById('model-progress');
    const progressFill = document.getElementById('model-progress-fill');
    const progressText = document.getElementById('model-progress-text');
    
    uploadArea.style.opacity = '0.6';
    uploadArea.style.pointerEvents = 'none';
    progressContainer.style.display = 'flex';
    progressFill.style.width = '0%';
    progressText.textContent = '0%';
    
    const xhr = new XMLHttpRequest();
    
    xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable) {
            const percentComplete = (e.loaded / e.total) * 100;
            progressFill.style.width = percentComplete + '%';
            progressText.textContent = Math.round(percentComplete) + '%';
        }
    });
    
    xhr.addEventListener('load', () => {
        uploadArea.style.opacity = '1';
        uploadArea.style.pointerEvents = 'auto';
        progressContainer.style.display = 'none';
        
        if (xhr.status === 200) {
            try {
                const data = JSON.parse(xhr.responseText);
                if (data.error) {
                    alert('上传失败: ' + data.error);
                    return;
                }
                
                sessionId = data.session_id || sessionId;
                uploadedModels = data.models || [];
                updateModelList();
                checkPredictButton();
            } catch (e) {
                console.error('Error parsing response:', e);
                alert('上传失败: 响应解析错误');
            }
        } else {
            try {
                const data = JSON.parse(xhr.responseText);
                alert('上传失败: ' + (data.error || `HTTP ${xhr.status}`));
            } catch (e) {
                alert('上传失败: HTTP ' + xhr.status);
            }
        }
    });
    
    xhr.addEventListener('error', () => {
        uploadArea.style.opacity = '1';
        uploadArea.style.pointerEvents = 'auto';
        progressContainer.style.display = 'none';
        alert('上传失败: 网络错误');
    });
    
    xhr.addEventListener('abort', () => {
        uploadArea.style.opacity = '1';
        uploadArea.style.pointerEvents = 'auto';
        progressContainer.style.display = 'none';
        alert('上传已取消');
    });
    
    xhr.open('POST', '/api/upload/models');
    xhr.send(formData);
}

// Data upload
const dataUploadArea = document.getElementById('data-upload-area');
const dataInput = document.getElementById('data-input');

setupDragAndDrop(dataUploadArea, dataInput, handleDataUpload);

function handleDataUpload(files) {
    if (!files || files.length === 0) {
        alert('请选择要上传的文件');
        return;
    }
    
    if (!sessionId) {
        alert('请先上传模型文件');
        return;
    }
    
    const formData = new FormData();
    formData.append('session_id', sessionId);
    
    Array.from(files).forEach(file => {
        formData.append('files[]', file);
    });
    
    const uploadArea = document.getElementById('data-upload-area');
    const progressContainer = document.getElementById('data-progress');
    const progressFill = document.getElementById('data-progress-fill');
    const progressText = document.getElementById('data-progress-text');
    
    uploadArea.style.opacity = '0.6';
    uploadArea.style.pointerEvents = 'none';
    progressContainer.style.display = 'flex';
    progressFill.style.width = '0%';
    progressText.textContent = '0%';
    
    const xhr = new XMLHttpRequest();
    
    xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable) {
            const percentComplete = (e.loaded / e.total) * 100;
            progressFill.style.width = percentComplete + '%';
            progressText.textContent = Math.round(percentComplete) + '%';
        }
    });
    
    xhr.addEventListener('load', () => {
        uploadArea.style.opacity = '1';
        uploadArea.style.pointerEvents = 'auto';
        progressContainer.style.display = 'none';
        
        if (xhr.status === 200) {
            try {
                const data = JSON.parse(xhr.responseText);
                if (data.error) {
                    alert('上传失败: ' + data.error);
                    return;
                }
                
                uploadedImages = data.image_paths || [];
                updateDataList();
                checkPredictButton();
            } catch (e) {
                console.error('Error parsing response:', e);
                alert('上传失败: 响应解析错误');
            }
        } else {
            try {
                const data = JSON.parse(xhr.responseText);
                alert('上传失败: ' + (data.error || `HTTP ${xhr.status}`));
            } catch (e) {
                alert('上传失败: HTTP ' + xhr.status);
            }
        }
    });
    
    xhr.addEventListener('error', () => {
        uploadArea.style.opacity = '1';
        uploadArea.style.pointerEvents = 'auto';
        progressContainer.style.display = 'none';
        alert('上传失败: 网络错误');
    });
    
    xhr.addEventListener('abort', () => {
        uploadArea.style.opacity = '1';
        uploadArea.style.pointerEvents = 'auto';
        progressContainer.style.display = 'none';
        alert('上传已取消');
    });
    
    xhr.open('POST', '/api/upload/data');
    xhr.send(formData);
}

// Threshold sync
const thresholdSlider = document.getElementById('threshold-slider');
const thresholdInput = document.getElementById('threshold-input');

thresholdSlider.addEventListener('input', (e) => {
    thresholdInput.value = e.target.value;
});

thresholdInput.addEventListener('input', (e) => {
    let value = parseFloat(e.target.value);
    if (value < 0) value = 0;
    if (value > 1) value = 1;
    thresholdSlider.value = value;
    thresholdInput.value = value;
});

// Predict button
document.getElementById('predict-btn').addEventListener('click', runPrediction);

function runPrediction() {
    if (uploadedModels.length === 0 || uploadedImages.length === 0) {
        alert('请先上传模型和数据文件');
        return;
    }
    
    const btn = document.getElementById('predict-btn');
    btn.disabled = true;
    btn.textContent = '预测中...';
    
    const threshold = parseFloat(thresholdInput.value);
    const maxSize = parseInt(document.getElementById('max-size-input').value);
    
    fetch('/api/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            session_id: sessionId,
            model_paths: uploadedModels.map(m => m.path),
            image_paths: uploadedImages,
            threshold: threshold,
            max_size: maxSize
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert('预测失败: ' + data.error);
            return;
        }
        
        currentResults = data;
        imageList = Object.keys(data.predictions);
        displayResults(data);
        
        btn.disabled = false;
        btn.textContent = '开始预测';
    })
    .catch(error => {
        console.error('Error:', error);
        alert('预测失败: ' + error.message);
        btn.disabled = false;
        btn.textContent = '开始预测';
    });
}

function displayResults(data) {
    const resultsSection = document.getElementById('results-section');
    const resultsGrid = document.getElementById('results-grid');
    
    resultsSection.style.display = 'block';
    resultsGrid.innerHTML = '';
    
    // Create legend
    const legend = {};
    data.model_infos.forEach((model, idx) => {
        if (currentResults && currentResults.predictions) {
            const firstImg = Object.values(currentResults.predictions)[0];
            if (firstImg && firstImg.predictions[idx]) {
                legend[model.name] = firstImg.predictions[idx].color;
            }
        }
    });
    
    Object.entries(data.predictions).forEach(([imgName, pred]) => {
        const item = document.createElement('div');
        item.className = 'result-item';
        item.onclick = () => openImageViewer(imgName);
        
        item.innerHTML = `
            <img src="${pred.image}" alt="${imgName}">
            <div class="info">${imgName}</div>
        `;
        
        resultsGrid.appendChild(item);
    });
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// Image viewer
const modal = document.getElementById('image-viewer');
const viewerImage = document.getElementById('viewer-image');
const viewerFilename = document.getElementById('viewer-filename');
const viewerLegend = document.getElementById('viewer-legend');
const closeBtn = document.querySelector('.close');
const prevBtn = document.getElementById('prev-btn');
const nextBtn = document.getElementById('next-btn');

closeBtn.onclick = () => modal.style.display = 'none';
window.onclick = (e) => {
    if (e.target == modal) modal.style.display = 'none';
};

prevBtn.onclick = () => navigateImage(-1);
nextBtn.onclick = () => navigateImage(1);

document.addEventListener('keydown', (e) => {
    if (modal.style.display === 'block') {
        if (e.key === 'ArrowLeft') navigateImage(-1);
        if (e.key === 'ArrowRight') navigateImage(1);
        if (e.key === 'Escape') modal.style.display = 'none';
    }
});

function openImageViewer(imgName) {
    if (!currentResults || !currentResults.predictions[imgName]) return;
    
    currentImageIndex = imageList.indexOf(imgName);
    updateViewer();
    modal.style.display = 'block';
}

function navigateImage(direction) {
    currentImageIndex += direction;
    if (currentImageIndex < 0) currentImageIndex = imageList.length - 1;
    if (currentImageIndex >= imageList.length) currentImageIndex = 0;
    updateViewer();
}

function updateViewer() {
    const imgName = imageList[currentImageIndex];
    const pred = currentResults.predictions[imgName];
    
    viewerImage.src = pred.image;
    viewerFilename.textContent = imgName;
    
    // Update legend
    viewerLegend.innerHTML = '';
    pred.predictions.forEach(predGroup => {
        const legendItem = document.createElement('div');
        legendItem.className = 'legend-item';
        legendItem.innerHTML = `
            <div class="legend-color" style="background-color: rgb(${predGroup.color.join(',')})"></div>
            <span>${predGroup.model_name}</span>
        `;
        viewerLegend.appendChild(legendItem);
    });
}

// Helper functions
function setupDragAndDrop(area, input, handler) {
    area.addEventListener('click', () => input.click());
    
    area.addEventListener('dragover', (e) => {
        e.preventDefault();
        area.classList.add('drag-over');
    });
    
    area.addEventListener('dragleave', () => {
        area.classList.remove('drag-over');
    });
    
    area.addEventListener('drop', (e) => {
        e.preventDefault();
        area.classList.remove('drag-over');
        handler(e.dataTransfer.files);
    });
    
    input.addEventListener('change', (e) => {
        handler(e.target.files);
    });
}

function updateModelList() {
    const list = document.getElementById('model-list');
    list.innerHTML = '';
    
    uploadedModels.forEach((model, idx) => {
        const item = document.createElement('div');
        item.className = 'file-item';
        item.innerHTML = `
            <span>${model.name}</span>
            <span style="color: #999; font-size: 12px;">(${model.type}${model.sub_type ? ' - ' + model.sub_type : ''})</span>
        `;
        list.appendChild(item);
    });
}

function updateDataList() {
    const list = document.getElementById('data-list');
    list.innerHTML = '';
    
    uploadedImages.forEach((imgPath, idx) => {
        const imgName = imgPath.split(/[/\\]/).pop();
        const item = document.createElement('div');
        item.className = 'file-item';
        item.innerHTML = `
            <span>${imgName}</span>
        `;
        list.appendChild(item);
    });
}

function checkPredictButton() {
    const btn = document.getElementById('predict-btn');
    btn.disabled = uploadedModels.length === 0 || uploadedImages.length === 0;
}

function generateSessionId() {
    return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

