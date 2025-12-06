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
                // Accumulate models instead of replacing
                const newModels = data.models || [];
                // Merge new models with existing ones, avoiding duplicates
                const existingPaths = new Set(uploadedModels.map(m => m.path));
                newModels.forEach(model => {
                    if (!existingPaths.has(model.path)) {
                        uploadedModels.push(model);
                    }
                });
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
                
                // Accumulate images instead of replacing
                const newImages = data.image_paths || [];
                // Merge new images with existing ones, avoiding duplicates
                const existingImages = new Set(uploadedImages);
                newImages.forEach(img => {
                    if (!existingImages.has(img)) {
                        uploadedImages.push(img);
                    }
                });
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
    
    // Prepare model configs with individual thresholds and max sizes
    const modelConfigs = uploadedModels.map(model => ({
        path: model.path,
        threshold: model.threshold || 0.5,
        max_size: model.maxSize || 1536
    }));
    
    fetch('/api/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            session_id: sessionId,
            model_configs: modelConfigs,
            image_paths: uploadedImages
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
    
    // Bind clear results button
    bindClearResultsButton();
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function openImageViewer(imgName) {
    if (!currentResults || !currentResults.result_id) return;
    
    // Navigate to viewer page
    window.location.href = `/viewer/${currentResults.result_id}`;
}

function clearResults() {
    if (!currentResults) return;
    
    if (confirm('确定要清空所有预测结果吗？此操作不可恢复。')) {
        const resultsSection = document.getElementById('results-section');
        const resultsGrid = document.getElementById('results-grid');
        
        resultsGrid.innerHTML = '';
        resultsSection.style.display = 'none';
        currentResults = null;
        imageList = [];
    }
}

// Bind clear results button (button may be dynamically created)
function bindClearResultsButton() {
    const clearBtn = document.getElementById('clear-results-btn');
    if (clearBtn && !clearBtn.dataset.bound) {
        clearBtn.addEventListener('click', clearResults);
        clearBtn.dataset.bound = 'true';
    }
}

// Helper functions
function setupDragAndDrop(area, input, handler) {
    // Prevent multiple event bindings
    if (area.dataset.dragSetup === 'true') {
        return;
    }
    area.dataset.dragSetup = 'true';
    
    // Click on area to trigger file input (but not if clicking directly on input)
    area.addEventListener('click', (e) => {
        // Don't trigger if clicking directly on the input element
        if (e.target === input || input.contains(e.target)) {
            return;
        }
        e.preventDefault();
        e.stopPropagation();
        input.click();
    });
    
    // Prevent input from triggering area click
    input.addEventListener('click', (e) => {
        e.stopPropagation();
    });
    
    area.addEventListener('dragover', (e) => {
        e.preventDefault();
        area.classList.add('drag-over');
    });
    
    area.addEventListener('dragleave', () => {
        area.classList.remove('drag-over');
    });
    
    area.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        area.classList.remove('drag-over');
        handler(e.dataTransfer.files);
    });
    
    input.addEventListener('change', (e) => {
        e.stopPropagation();
        if (e.target.files && e.target.files.length > 0) {
            handler(e.target.files);
            // Reset input to allow selecting the same file again
            e.target.value = '';
        }
    });
}

function updateModelList() {
    const list = document.getElementById('model-list');
    list.innerHTML = '';
    
    uploadedModels.forEach((model, idx) => {
        const item = document.createElement('div');
        item.className = 'model-item';
        item.dataset.modelIdx = idx;
        
        // Initialize model config if not exists
        if (!model.threshold) model.threshold = 0.5;
        if (!model.maxSize) model.maxSize = 1536;
        
        item.innerHTML = `
            <div class="model-item-header">
                <span class="model-name">${model.name}</span>
                <span class="model-type">(${model.type}${model.sub_type ? ' - ' + model.sub_type : ''})</span>
                <button class="remove-model-btn" onclick="removeModel(${idx})" title="移除模型">×</button>
            </div>
            <div class="model-item-config">
                <div class="config-row">
                    <label>置信度阈值:</label>
                    <input type="range" class="model-threshold-slider" min="0" max="1" step="0.01" 
                           value="${model.threshold}" oninput="updateModelThreshold(${idx}, this.value)">
                    <input type="number" class="model-threshold-input" min="0" max="1" step="0.01" 
                           value="${model.threshold}" oninput="updateModelThreshold(${idx}, this.value)">
                </div>
                <div class="config-row">
                    <label>最大尺寸:</label>
                    <input type="number" class="model-maxsize-input" min="512" max="4096" step="64" 
                           value="${model.maxSize}" oninput="updateModelMaxSize(${idx}, this.value)">
                </div>
            </div>
        `;
        list.appendChild(item);
    });
}

function updateModelThreshold(idx, value) {
    if (uploadedModels[idx]) {
        uploadedModels[idx].threshold = parseFloat(value);
        // Sync slider and input
        const item = document.querySelector(`[data-model-idx="${idx}"]`);
        if (item) {
            const slider = item.querySelector('.model-threshold-slider');
            const input = item.querySelector('.model-threshold-input');
            if (slider) slider.value = value;
            if (input) input.value = value;
        }
    }
}

function updateModelMaxSize(idx, value) {
    if (uploadedModels[idx]) {
        uploadedModels[idx].maxSize = parseInt(value);
    }
}

function removeModel(idx) {
    if (confirm('确定要移除此模型吗？')) {
        uploadedModels.splice(idx, 1);
        updateModelList();
        checkPredictButton();
    }
}

function updateDataList() {
    const list = document.getElementById('data-list');
    list.innerHTML = '';
    
    const count = uploadedImages.length;
    if (count > 0) {
        const item = document.createElement('div');
        item.className = 'file-item';
        item.innerHTML = `
            <span>已上传 ${count} 张图片</span>
        `;
        list.appendChild(item);
    }
}

function checkPredictButton() {
    const btn = document.getElementById('predict-btn');
    btn.disabled = uploadedModels.length === 0 || uploadedImages.length === 0;
}

function generateSessionId() {
    return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

