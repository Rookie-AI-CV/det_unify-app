#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DetUnify Studio - 多模型检测统一工作平台
Flask Web Application for Unified Model Prediction and Evaluation
"""

import os
import zipfile
import shutil
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, send_from_directory, Response, send_file
from werkzeug.utils import secure_filename
import uuid
from loguru import logger
from collections import defaultdict
import threading
import time

# Import prediction utilities
import sys
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
from src.predict.predict import detect_model_type

# Import README generator from same directory
from readme_generator import generate_readme

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB max file size
app.config['UPLOAD_FOLDER'] = Path(__file__).parent / 'static' / 'uploads'
app.config['RESULTS_FOLDER'] = Path(__file__).parent / 'static' / 'results'
app.config['EXPORT_CACHE_FOLDER'] = Path(__file__).parent / 'static' / 'export_cache'

# 确保导出缓存目录存在
app.config['EXPORT_CACHE_FOLDER'].mkdir(parents=True, exist_ok=True)

# 导出任务进度存储（task_id -> progress_info）
export_tasks = {}
export_tasks_lock = threading.Lock()

# Global error handler to ensure JSON responses
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}", exc_info=True)
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {e}", exc_info=True)
    return jsonify({'error': f'Server error: {str(e)}'}), 500

# Create directories
app.config['UPLOAD_FOLDER'].mkdir(parents=True, exist_ok=True)
app.config['RESULTS_FOLDER'].mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {'zip', 'pt', 'pth', 'ckpt'}
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def cleanup_old_results(max_age_hours=3):
    """清理3小时前的预测结果数据"""
    try:
        results_folder = app.config['RESULTS_FOLDER']
        if not results_folder.exists():
            return
        
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        deleted_count = 0
        deleted_size = 0
        
        for result_dir in results_folder.iterdir():
            if not result_dir.is_dir():
                continue
            
            try:
                # 检查目录的最后修改时间
                mtime = datetime.fromtimestamp(result_dir.stat().st_mtime)
                
                # 如果是结果目录，检查 results.json 文件的修改时间
                results_file = result_dir / 'results.json'
                if results_file.exists():
                    file_mtime = datetime.fromtimestamp(results_file.stat().st_mtime)
                    # 使用文件的修改时间，因为这代表预测完成的时间
                    if file_mtime < cutoff_time:
                        # 计算目录大小
                        dir_size = sum(f.stat().st_size for f in result_dir.rglob('*') if f.is_file())
                        # 删除目录
                        shutil.rmtree(result_dir)
                        deleted_count += 1
                        deleted_size += dir_size
                        logger.info(f"已清理过期预测结果: {result_dir.name} (创建时间: {file_mtime.strftime('%Y-%m-%d %H:%M:%S')})")
                elif mtime < cutoff_time:
                    # 如果没有 results.json，使用目录修改时间
                    dir_size = sum(f.stat().st_size for f in result_dir.rglob('*') if f.is_file())
                    shutil.rmtree(result_dir)
                    deleted_count += 1
                    deleted_size += dir_size
                    logger.info(f"已清理过期预测结果目录: {result_dir.name} (修改时间: {mtime.strftime('%Y-%m-%d %H:%M:%S')})")
            except Exception as e:
                logger.warning(f"清理预测结果目录 {result_dir} 时出错: {e}")
                continue
        
        if deleted_count > 0:
            size_mb = deleted_size / (1024 * 1024)
            logger.info(f"预测结果清理完成: 删除了 {deleted_count} 个目录，释放空间 {size_mb:.2f} MB")
        
    except Exception as e:
        logger.error(f"清理预测结果时出错: {e}")


def cleanup_old_uploads(max_age_hours=3):
    """清理3小时前的上传数据"""
    try:
        upload_folder = app.config['UPLOAD_FOLDER']
        if not upload_folder.exists():
            return
        
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        deleted_count = 0
        deleted_size = 0
        
        for session_dir in upload_folder.iterdir():
            if not session_dir.is_dir():
                continue
            
            try:
                # 检查目录的最后修改时间
                mtime = datetime.fromtimestamp(session_dir.stat().st_mtime)
                
                if mtime < cutoff_time:
                    # 计算目录大小
                    dir_size = sum(f.stat().st_size for f in session_dir.rglob('*') if f.is_file())
                    # 删除目录
                    shutil.rmtree(session_dir)
                    deleted_count += 1
                    deleted_size += dir_size
                    logger.info(f"已清理过期上传数据: {session_dir.name} (修改时间: {mtime.strftime('%Y-%m-%d %H:%M:%S')})")
            except Exception as e:
                logger.warning(f"清理上传数据目录 {session_dir} 时出错: {e}")
                continue
        
        if deleted_count > 0:
            size_mb = deleted_size / (1024 * 1024)
            logger.info(f"上传数据清理完成: 删除了 {deleted_count} 个目录，释放空间 {size_mb:.2f} MB")
        
    except Exception as e:
        logger.error(f"清理上传数据时出错: {e}")


def cleanup_old_cache(max_age_hours=3):
    """清理所有过期缓存（预测结果和上传数据）"""
    logger.info(f"开始清理 {max_age_hours} 小时前的缓存数据...")
    cleanup_old_results(max_age_hours)
    cleanup_old_uploads(max_age_hours)
    logger.info("缓存清理完成")


def allowed_image_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


def extract_zip(zip_path, extract_to):
    """Extract zip file and return list of extracted files"""
    extracted_files = []
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        for file_info in zip_ref.namelist():
            extracted_files.append(os.path.join(extract_to, file_info))
    return extracted_files


def find_model_files(directory):
    """Find all model files (.pt, .pth, .ckpt) in directory"""
    model_files = []
    seen_paths = set()
    for ext in ['*.pt', '*.pth', '*.ckpt']:
        found = list(Path(directory).glob(ext))
        found.extend(Path(directory).rglob(ext))
        for f in found:
            # Convert to absolute path and normalize to avoid duplicates
            abs_path = str(f.resolve())
            if abs_path not in seen_paths:
                seen_paths.add(abs_path)
                model_files.append(abs_path)
    return model_files


def find_image_files(directory):
    """Find all image files in directory"""
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(Path(directory).glob(ext))
        image_files.extend(Path(directory).rglob(ext))
    return sorted([str(f) for f in image_files])


def load_coco_json(json_path):
    """Load COCO format JSON file"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading COCO JSON: {e}")
        return None


def draw_rectangle_with_style(img, pt1, pt2, color, thickness, style='solid'):
    """Draw rectangle with different line styles (solid, dashed, dotted, dashdot)"""
    x1, y1 = pt1
    x2, y2 = pt2
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    if style == 'solid':
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    elif style == 'dashed':
        # Draw dashed line by drawing short segments
        gap = 10
        dash = 15
        # Top
        x = x1
        while x < x2:
            cv2.line(img, (x, y1), (min(x + dash, x2), y1), color, thickness)
            x += dash + gap
        # Bottom
        x = x1
        while x < x2:
            cv2.line(img, (x, y2), (min(x + dash, x2), y2), color, thickness)
            x += dash + gap
        # Left
        y = y1
        while y < y2:
            cv2.line(img, (x1, y), (x1, min(y + dash, y2)), color, thickness)
            y += dash + gap
        # Right
        y = y1
        while y < y2:
            cv2.line(img, (x2, y), (x2, min(y + dash, y2)), color, thickness)
            y += dash + gap
    elif style == 'dotted':
        # Draw dotted line
        gap = 8
        # Top and bottom
        for x in range(x1, x2, gap * 2):
            cv2.circle(img, (x, y1), thickness, color, -1)
            cv2.circle(img, (x, y2), thickness, color, -1)
        # Left and right
        for y in range(y1, y2, gap * 2):
            cv2.circle(img, (x1, y), thickness, color, -1)
            cv2.circle(img, (x2, y), thickness, color, -1)
    elif style == 'dashdot':
        # Draw dash-dot pattern
        gap = 8
        dash = 12
        # Top
        x = x1
        while x < x2:
            cv2.line(img, (x, y1), (min(x + dash, x2), y1), color, thickness)
            x += dash + gap
            if x < x2:
                cv2.circle(img, (x, y1), thickness, color, -1)
                x += gap
        # Bottom
        x = x1
        while x < x2:
            cv2.line(img, (x, y2), (min(x + dash, x2), y2), color, thickness)
            x += dash + gap
            if x < x2:
                cv2.circle(img, (x, y2), thickness, color, -1)
                x += gap
        # Left
        y = y1
        while y < y2:
            cv2.line(img, (x1, y), (x1, min(y + dash, y2)), color, thickness)
            y += dash + gap
            if y < y2:
                cv2.circle(img, (x1, y), thickness, color, -1)
                y += gap
        # Right
        y = y1
        while y < y2:
            cv2.line(img, (x2, y), (x2, min(y + dash, y2)), color, thickness)
            y += dash + gap
            if y < y2:
                cv2.circle(img, (x2, y), thickness, color, -1)
                y += gap


def get_model_short_name(model_name, max_length=20):
    """Get shortened model name showing only last few characters"""
    if len(model_name) <= max_length:
        return model_name
    return '...' + model_name[-max_length:]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/upload/models', methods=['POST'])
def upload_models():
    """Handle model file uploads"""
    try:
        if 'files[]' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files[]')
        if not files or all(f.filename == '' for f in files):
            return jsonify({'error': 'No valid files provided'}), 400
        
        session_id = request.form.get('session_id')
        if not session_id:
            session_id = str(uuid.uuid4())
        
        session_dir = app.config['UPLOAD_FOLDER'] / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        model_files = []
        
        for file in files:
            if file.filename == '':
                continue
            
            filename = secure_filename(file.filename)
            if not filename:
                continue
            
            if not allowed_file(filename):
                logger.warning(f"File {filename} is not allowed")
                continue
            
            try:
                filepath = session_dir / filename
                file.save(str(filepath))
                logger.info(f"Saved file: {filepath}")
                
                if filename.endswith('.zip'):
                    extract_dir = session_dir / f"{filename}_extracted"
                    extract_dir.mkdir(exist_ok=True)
                    extract_zip(str(filepath), str(extract_dir))
                    found_models = find_model_files(str(extract_dir))
                    model_files.extend(found_models)
                elif filename.endswith(('.pt', '.pth', '.ckpt')):
                    model_files.append(str(filepath))
            except Exception as e:
                logger.error(f"Error processing file {filename}: {e}")
                continue
        
        if not model_files:
            return jsonify({'error': 'No valid model files found'}), 400
        
        models_info = []
        seen_paths = set()
        for model_file in model_files:
            # Normalize path to avoid duplicates
            normalized_path = str(Path(model_file).resolve())
            if normalized_path in seen_paths:
                continue
            seen_paths.add(normalized_path)
            
            try:
                model_type, hq_sub_type = detect_model_type(normalized_path)
                if model_type:
                    models_info.append({
                        'path': normalized_path,
                        'name': os.path.basename(model_file),
                        'type': model_type,
                        'sub_type': hq_sub_type
                    })
            except Exception as e:
                logger.error(f"Error detecting model type for {model_file}: {e}")
                continue
        
        if not models_info:
            return jsonify({'error': 'No valid models detected'}), 400
        
        return jsonify({
            'session_id': session_id,
            'models': models_info
        })
    except Exception as e:
        logger.error(f"Error in upload_models: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload/data', methods=['POST'])
def upload_data():
    """Handle data file uploads"""
    try:
        if 'files[]' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files[]')
        if not files or all(f.filename == '' for f in files):
            return jsonify({'error': 'No valid files provided'}), 400
        
        session_id = request.form.get('session_id')
        if not session_id:
            return jsonify({'error': 'No session_id provided'}), 400
        
        session_dir = app.config['UPLOAD_FOLDER'] / session_id
        data_dir = session_dir / 'data'
        data_dir.mkdir(parents=True, exist_ok=True)
        
        image_files = []
        
        for file in files:
            if file.filename == '':
                continue
            
            filename = secure_filename(file.filename)
            if not filename:
                continue
            
            try:
                filepath = data_dir / filename
                file.save(str(filepath))
                logger.info(f"Saved file: {filepath}")
                
                if filename.endswith('.zip'):
                    extract_dir = data_dir / f"{filename}_extracted"
                    extract_dir.mkdir(exist_ok=True)
                    extract_zip(str(filepath), str(extract_dir))
                    found_images = find_image_files(str(extract_dir))
                    image_files.extend(found_images)
                elif allowed_image_file(filename):
                    image_files.append(str(filepath))
            except Exception as e:
                logger.error(f"Error processing file {filename}: {e}")
                continue
        
        if not image_files:
            return jsonify({'error': 'No valid image files found'}), 400
        
        response = jsonify({
            'session_id': session_id,
            'images': [os.path.basename(img) for img in image_files],
            'image_paths': image_files
        })
        response.headers['Content-Type'] = 'application/json'
        return response
    except Exception as e:
        logger.error(f"Error in upload_data: {e}", exc_info=True)
        response = jsonify({'error': str(e)})
        response.headers['Content-Type'] = 'application/json'
        return response, 500


@app.route('/api/upload/models/path', methods=['POST'])
def upload_models_by_path():
    """Handle model file imports from local paths"""
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json(silent=True)
        if not data:
            return jsonify({'error': 'Invalid JSON data'}), 400
        
        paths_str = data.get('paths', '')
        if not paths_str or not paths_str.strip():
            return jsonify({'error': 'No paths provided'}), 400
        
        session_id = data.get('session_id')
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Split paths by semicolon
        paths = [p.strip() for p in paths_str.split(';') if p.strip()]
        if not paths:
            return jsonify({'error': 'No valid paths provided'}), 400
        
        models_info = []
        seen_paths = set()
        
        for path_str in paths:
            try:
                path_obj = Path(path_str).resolve()
                
                # Check if path exists
                if not path_obj.exists():
                    logger.warning(f"Path does not exist: {path_str}")
                    continue
                
                # Handle directory: find all model files in it
                if path_obj.is_dir():
                    found_models = find_model_files(str(path_obj))
                    for model_file in found_models:
                        normalized_path = str(Path(model_file).resolve())
                        if normalized_path in seen_paths:
                            continue
                        seen_paths.add(normalized_path)
                        
                        try:
                            model_type, hq_sub_type = detect_model_type(normalized_path)
                            if model_type:
                                models_info.append({
                                    'path': normalized_path,
                                    'name': os.path.basename(model_file),
                                    'type': model_type,
                                    'sub_type': hq_sub_type
                                })
                        except Exception as e:
                            logger.error(f"Error detecting model type for {model_file}: {e}")
                            continue
                
                # Handle file: check if it's a model file
                elif path_obj.is_file():
                    if not allowed_file(path_obj.name):
                        logger.warning(f"File is not a model file: {path_str}")
                        continue
                    
                    normalized_path = str(path_obj.resolve())
                    if normalized_path in seen_paths:
                        continue
                    seen_paths.add(normalized_path)
                    
                    try:
                        model_type, hq_sub_type = detect_model_type(normalized_path)
                        if model_type:
                            models_info.append({
                                'path': normalized_path,
                                'name': path_obj.name,
                                'type': model_type,
                                'sub_type': hq_sub_type
                            })
                    except Exception as e:
                        logger.error(f"Error detecting model type for {path_str}: {e}")
                        continue
                else:
                    logger.warning(f"Path is neither file nor directory: {path_str}")
                    continue
                    
            except Exception as e:
                logger.error(f"Error processing path {path_str}: {e}")
                continue
        
        if not models_info:
            return jsonify({'error': 'No valid model files found in provided paths'}), 400
        
        return jsonify({
            'session_id': session_id,
            'models': models_info
        })
    except Exception as e:
        logger.error(f"Error in upload_models_by_path: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload/data/path', methods=['POST'])
def upload_data_by_path():
    """Handle data file imports from local directory path"""
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json(silent=True)
        if not data:
            return jsonify({'error': 'Invalid JSON data'}), 400
        
        dir_path = data.get('path', '').strip()
        if not dir_path:
            return jsonify({'error': 'No directory path provided'}), 400
        
        session_id = data.get('session_id')
        if not session_id:
            return jsonify({'error': 'No session_id provided'}), 400
        
        # Check if path exists and is a directory
        path_obj = Path(dir_path).resolve()
        if not path_obj.exists():
            return jsonify({'error': f'Directory does not exist: {dir_path}'}), 400
        
        if not path_obj.is_dir():
            return jsonify({'error': f'Path is not a directory: {dir_path}'}), 400
        
        # Find all image files in the directory
        image_files = find_image_files(str(path_obj))
        
        if not image_files:
            return jsonify({'error': 'No valid image files found in directory'}), 400
        
        response = jsonify({
            'session_id': session_id,
            'images': [os.path.basename(img) for img in image_files],
            'image_paths': image_files
        })
        response.headers['Content-Type'] = 'application/json'
        return response
    except Exception as e:
        logger.error(f"Error in upload_data_by_path: {e}", exc_info=True)
        response = jsonify({'error': str(e)})
        response.headers['Content-Type'] = 'application/json'
        return response, 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """Run prediction using predict.py script"""
    try:
        # 清理3小时前的缓存数据
        cleanup_old_cache(max_age_hours=3)
        
        # Ensure request has JSON content type
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        data = request.get_json(silent=True)
        if not data:
            return jsonify({'error': 'No JSON data provided or invalid JSON'}), 400
        
        session_id = data.get('session_id')
        model_configs = data.get('model_configs', [])  # List of {path, threshold, max_size}
        image_paths = data.get('image_paths', [])
        
        # Support legacy format (backward compatibility)
        if not model_configs and data.get('model_paths'):
            model_paths = data.get('model_paths', [])
            threshold = float(data.get('threshold', 0.5))
            max_size = int(data.get('max_size', 1536))
            model_configs = [{'path': p, 'threshold': threshold, 'max_size': max_size} for p in model_paths]
        
        if not session_id or not model_configs or not image_paths:
            return jsonify({'error': 'Missing required parameters'}), 400
        
        result_id = str(uuid.uuid4())
        result_dir = app.config['RESULTS_FOLDER'] / result_id
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # Create image list file for batch prediction
        image_list_file = result_dir / 'image_list.txt'
        with open(image_list_file, 'w', encoding='utf-8') as f:
            for img_path in image_paths:
                f.write(f"{img_path}\n")
        
        # Detect model types and collect model infos
        model_infos = []
        for model_config in model_configs:
            model_path = model_config['path']
            try:
                model_type, hq_sub_type = detect_model_type(model_path)
                if model_type:
                    model_infos.append({
                        'path': model_path,
                        'name': os.path.basename(model_path),
                        'type': model_type,
                        'sub_type': hq_sub_type,
                        'threshold': model_config.get('threshold', 0.5),
                        'max_size': model_config.get('max_size', 1536)
                    })
            except Exception as e:
                logger.error(f"Error detecting model type for {model_path}: {e}")
                continue
        
        if not model_infos:
            return jsonify({'error': 'No valid models detected'}), 400
        
        # Generate line styles for each model (different styles for different models)
        line_styles = ['solid', 'dashed', 'dotted', 'dashdot']
        model_styles = {}
        for i in range(len(model_configs)):
            model_styles[i] = line_styles[i % len(line_styles)]
        
        # Generate colors for each label (consistent across models)
        label_colors = {}
        np.random.seed(42)  # Fixed seed for consistent colors
        
        # Run prediction for each model and combine results
        all_predictions = {}  # image_name -> list of model predictions
        coco_results = {}  # model_name -> coco_data
        
        predict_script = ROOT_DIR / 'src' / 'predict' / 'predict.py'
        
        for model_idx, model_config in enumerate(model_configs):
            model_path = model_config['path']
            model_info = next((m for m in model_infos if m['path'] == model_path), None)
            if not model_info:
                continue
            
            threshold = model_info['threshold']
            max_size = model_info['max_size']
            
            # Create output directory for this model
            model_output_dir = result_dir / f"model_{model_idx}"
            model_output_dir.mkdir(exist_ok=True)
            coco_json_path = model_output_dir / '_annotations.coco.json'
            
            # Build command
            cmd = [
                sys.executable,
                str(predict_script),
                '--checkpoint', model_path,
                '--image-list', str(image_list_file),
                '--threshold', str(threshold),
                '--output', str(model_output_dir),
            ]
            
            if model_info['type'] == 'hq_det' and model_info['sub_type']:
                cmd.extend(['--model-type', 'hq_det', '--hq-model-type', model_info['sub_type']])
                cmd.extend(['--max-size', str(max_size)])
            elif model_info['type'] == 'dino':
                cmd.extend(['--model-type', 'dino'])
            
            # Run prediction script with real-time output
            try:
                import subprocess
                logger.info(f"开始执行预测: {' '.join(cmd)}")
                logger.info(f"模型 {model_idx + 1}/{len(model_configs)}: {model_info['name']}")
                
                # 使用 Popen 实时打印输出
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,  # 将 stderr 合并到 stdout
                    text=True,
                    cwd=str(ROOT_DIR)
                )
                
                # 实时读取并打印输出
                stderr_lines = []
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        line = output.rstrip('\n\r')
                        if line:
                            print(line, flush=True)  # 实时打印到控制台
                            # 如果是错误信息，也记录下来
                            if any(keyword in line.lower() for keyword in ['error', 'warning', 'failed', 'exception']):
                                stderr_lines.append(line)
                
                # 确保进程完成
                process.wait()
                
                if process.returncode != 0:
                    error_msg = '\n'.join(stderr_lines) if stderr_lines else f"预测失败，返回码: {process.returncode}"
                    logger.error(f"Prediction failed for model {model_path}: {error_msg}")
                    continue
                
                # Load COCO JSON
                coco_data = load_coco_json(coco_json_path)
                if not coco_data:
                    logger.warning(f"No COCO JSON found for model {model_path}")
                    continue
                
                # Get short name for display
                short_name = get_model_short_name(model_info['name'])
                
                coco_results[model_info['name']] = {
                    'coco_data': coco_data,
                    'short_name': short_name,
                    'line_style': model_styles[model_idx],
                    'model_info': model_info,
                    'model_idx': model_idx
                }
                
            except Exception as e:
                logger.error(f"Error running prediction for model {model_path}: {e}")
                continue
        
        if not coco_results:
            return jsonify({'error': 'No predictions generated'}), 400
        
        # Collect all labels first to assign consistent colors
        all_labels = set()
        first_coco = next(iter(coco_results.values()))['coco_data']
        for model_result in coco_results.values():
            coco_data = model_result['coco_data']
            for ann in coco_data.get('annotations', []):
                category_id = ann['category_id']
                category = next((cat for cat in coco_data.get('categories', []) 
                               if cat['id'] == category_id), None)
                if category:
                    all_labels.add(category['name'])
        
        # Generate colors for all labels at once (consistent across all images)
        import colorsys
        sorted_labels = sorted(all_labels)
        for idx, label in enumerate(sorted_labels):
            hue = idx * 137.508  # Golden angle for better color distribution
            h = (hue % 360) / 360.0
            s = 0.7
            v = 0.9
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            label_colors[label] = tuple(map(int, (r * 255, g * 255, b * 255)))
        
        # Combine results from all models and generate visualizations
        predictions = {}
        
        # Build image map from first COCO file
        image_map = {img['id']: img for img in first_coco.get('images', [])}
        
        for img_info in first_coco.get('images', []):
            img_name = img_info['file_name']
            img_id = img_info['id']
            
            # Find original image path
            original_img_path = None
            for img_path in image_paths:
                if os.path.basename(img_path) == img_name:
                    original_img_path = img_path
                    break
            
            if not original_img_path:
                continue
            
            # Read original image
            img = cv2.imread(original_img_path)
            if img is None:
                continue
            
            # Collect predictions from all models
            img_predictions = []
            
            for model_name, model_result in coco_results.items():
                coco_data = model_result['coco_data']
                line_style = model_result['line_style']
                short_name = model_result['short_name']
                model_info = model_result['model_info']
                model_idx = model_result['model_idx']
                
                # Find annotations for this image
                annotations = [ann for ann in coco_data.get('annotations', []) 
                             if ann['image_id'] == img_id]
                
                if annotations:
                    bboxes = []
                    for ann in annotations:
                        # COCO format: [x, y, w, h] -> convert to [x1, y1, x2, y2]
                        x, y, w, h = ann['bbox']
                        category_id = ann['category_id']
                        score = ann.get('score', 1.0)
                        
                        # Find category name
                        category = next((cat for cat in coco_data.get('categories', []) 
                                       if cat['id'] == category_id), None)
                        label = category['name'] if category else 'object'
                        
                        # Get color for label (already assigned above)
                        color = label_colors.get(label, (128, 128, 128))  # Default gray if not found
                        
                        bboxes.append({
                            'bbox': [x, y, x + w, y + h],
                            'label': label,
                            'score': score,
                            'color': color
                        })
                    
                    if bboxes:
                        img_predictions.append({
                            'model_name': short_name,
                            'full_name': model_name,
                            'model_idx': model_idx,
                            'line_style': line_style,
                            'bboxes': bboxes
                        })
            
            # Draw predictions (only bounding boxes, no text on image) or save original image
            vis_img = img.copy()
            if img_predictions:
                # Draw bounding boxes if there are predictions
                for pred_group in img_predictions:
                    line_style = pred_group['line_style']
                    for bbox_info in pred_group['bboxes']:
                        bbox = bbox_info['bbox']
                        color_rgb = bbox_info['color']  # Color based on label (RGB format)
                        # Convert RGB to BGR for OpenCV
                        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
                        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                        draw_rectangle_with_style(vis_img, (x1, y1), (x2, y2), color_bgr, 2, style=line_style)
            
            # Save image (with or without predictions)
            output_path = result_dir / img_name
            cv2.imwrite(str(output_path), vis_img)
            
            # Always add image to predictions, even if no predictions were made
            predictions[img_name] = {
                'image': f'/static/results/{result_id}/{img_name}',
                'predictions': img_predictions,  # Empty list if no predictions
                'notes': '',  # Initialize notes field
                'statuses': [],  # Initialize statuses field (list of 'false_positive', 'missed', 'low_confidence')
                'false_positive_labels': [],  # Initialize false positive labels (list of label names for false positive bboxes)
                'missed_labels': [],  # Initialize missed labels (list of label names for missed detections)
                'low_confidence_labels': []  # Initialize low confidence labels (list of label names for low confidence detections)
            }
        
        if not predictions:
            return jsonify({'error': 'No predictions generated'}), 400
        
        # Prepare model infos with display names and line styles
        display_model_infos = []
        style_map = {
            'solid': '实线',
            'dashed': '虚线',
            'dotted': '点线',
            'dashdot': '点划线'
        }
        for i, model_info in enumerate(model_infos):
            display_model_infos.append({
                **model_info,
                'short_name': get_model_short_name(model_info['name']),
                'line_style': model_styles.get(i, 'solid'),
                'style_desc': style_map.get(model_styles.get(i, 'solid'), '实线')
            })
        
        # 尝试读取类别列表（从第一个模型的输出目录）
        all_class_names = None
        if model_infos:
            first_model_dir = result_dir / 'model_0'
            class_names_file = first_model_dir / '_class_names.json'
            if class_names_file.exists():
                try:
                    with open(class_names_file, 'r', encoding='utf-8') as f:
                        class_names_data = json.load(f)
                        all_class_names = class_names_data.get('class_names', [])
                        logger.info(f"加载类别列表: {len(all_class_names)} 个类别")
                except Exception as e:
                    logger.warning(f"无法读取类别列表文件: {e}")
        
        # Save results to JSON file for later retrieval
        results_data = {
            'result_id': result_id,
            'predictions': predictions,
            'model_infos': display_model_infos,
            'label_colors': {k: list(v) for k, v in label_colors.items()},  # Convert tuple to list for JSON
            'all_class_names': all_class_names if all_class_names else []  # 所有类别名称列表
        }
        results_file = result_dir / 'results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
        
        response = jsonify(results_data)
        response.headers['Content-Type'] = 'application/json'
        return response
    except Exception as e:
        logger.error(f"Error in predict endpoint: {e}", exc_info=True)
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)


@app.route('/viewer/<result_id>')
def viewer(result_id):
    """Image viewer page"""
    return render_template('viewer.html', result_id=result_id)


@app.route('/api/results/<result_id>')
def get_results(result_id):
    """Get prediction results by result_id"""
    try:
        result_dir = app.config['RESULTS_FOLDER'] / result_id
        if not result_dir.exists():
            return jsonify({'error': 'Result not found'}), 404
        
        # Try to find results JSON file
        results_file = result_dir / 'results.json'
        if results_file.exists():
            with open(results_file, 'r', encoding='utf-8') as f:
                return jsonify(json.load(f))
        
        # If no JSON file, scan directory for images
        image_files = list(result_dir.glob('*.jpg')) + list(result_dir.glob('*.png'))
        if not image_files:
            return jsonify({'error': 'No results found'}), 404
        
        # Return basic info
        predictions = {}
        for img_file in image_files:
            img_name = img_file.name
            predictions[img_name] = {
                'image': f'/static/results/{result_id}/{img_name}',
                'predictions': []
            }
        
        return jsonify({
            'result_id': result_id,
            'predictions': predictions
        })
    except Exception as e:
        logger.error(f"Error getting results: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/results/<result_id>/notes', methods=['POST'])
def save_image_notes(result_id):
    """Save notes and status for an image"""
    try:
        result_dir = app.config['RESULTS_FOLDER'] / result_id
        if not result_dir.exists():
            return jsonify({'error': 'Result not found'}), 404
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        image_name = data.get('image')
        notes = data.get('notes', '')
        statuses = data.get('statuses', [])  # List of 'false_positive', 'missed', 'low_confidence'
        false_positive_labels = data.get('false_positive_labels', [])  # List of label names for false positive bboxes
        missed_labels = data.get('missed_labels', [])  # List of label names for missed detections
        low_confidence_labels = data.get('low_confidence_labels', [])  # List of label names for low confidence detections
        
        # Load existing results.json
        results_file = result_dir / 'results.json'
        if not results_file.exists():
            return jsonify({'error': 'Results file not found'}), 404
        
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                results_data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error when loading results.json: {e}")
            return jsonify({'error': f'Invalid JSON in results file: {str(e)}'}), 500
        
        # Update notes and statuses for the image
        if image_name in results_data.get('predictions', {}):
            # Ensure notes is a string and handle None
            notes_str = str(notes) if notes is not None else ''
            results_data['predictions'][image_name]['notes'] = notes_str
            
            # Update statuses (list)
            if isinstance(statuses, list):
                results_data['predictions'][image_name]['statuses'] = statuses
            else:
                # Backward compatibility: if status is provided as single value
                old_status = data.get('status')
                if old_status:
                    results_data['predictions'][image_name]['statuses'] = [old_status] if old_status else []
                else:
                    results_data['predictions'][image_name]['statuses'] = []
            
            # Update false_positive_labels (list)
            if isinstance(false_positive_labels, list):
                results_data['predictions'][image_name]['false_positive_labels'] = false_positive_labels
            else:
                results_data['predictions'][image_name]['false_positive_labels'] = []
            
            # Update missed_labels (list)
            if isinstance(missed_labels, list):
                results_data['predictions'][image_name]['missed_labels'] = missed_labels
            else:
                results_data['predictions'][image_name]['missed_labels'] = []
            
            # Update low_confidence_labels (list)
            if isinstance(low_confidence_labels, list):
                results_data['predictions'][image_name]['low_confidence_labels'] = low_confidence_labels
            else:
                results_data['predictions'][image_name]['low_confidence_labels'] = []
            
            # Backward compatibility: maintain old 'status' field for now
            if results_data['predictions'][image_name].get('statuses'):
                results_data['predictions'][image_name]['status'] = results_data['predictions'][image_name]['statuses'][0]
            else:
                results_data['predictions'][image_name]['status'] = None
            
            # Save back to file with error handling
            try:
                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump(results_data, f, ensure_ascii=False, indent=2)
            except (TypeError, ValueError) as e:
                logger.error(f"Error saving JSON: {e}")
                logger.error(f"Problematic data: image={image_name}, notes={notes_str}, status={status}")
                return jsonify({'error': f'Error saving data: {str(e)}'}), 500
            
            return jsonify({'success': True})
        else:
            return jsonify({'error': 'Image not found'}), 404
            
    except Exception as e:
        logger.error(f"Error saving notes: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500




def _export_results_with_progress(result_id, description, export_originals, export_models, progress_callback=None):
    """执行导出任务，支持进度回调
    
    Args:
        result_id: 结果ID
        description: 说明信息
        export_originals: 是否导出原图
        export_models: 是否导出模型文件
        progress_callback: 进度回调函数 callback(stage, detail, progress)
    
    Returns:
        zip_buffer: BytesIO对象，包含ZIP文件数据
    """
    result_dir = app.config['RESULTS_FOLDER'] / result_id
    if not result_dir.exists():
        raise FileNotFoundError('Result not found')
    
    # Load results.json to get prediction data
    results_file = result_dir / 'results.json'
    if not results_file.exists():
        raise FileNotFoundError('Results file not found')
    
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in results.json: {e}")
        raise ValueError(f'Invalid JSON in results file: {str(e)}')
    
    if progress_callback:
        progress_callback('正在读取结果数据...', '加载预测结果和模型信息', 5)
    
    # Create temporary ZIP file
    import tempfile
    import io
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Find all model output directories and get model names
        # Sort by numeric index in directory name (model_0, model_1, ...)
        def get_model_index(path):
            try:
                return int(path.name.split('_')[1])
            except (ValueError, IndexError):
                return 999
        
        model_dirs = sorted(result_dir.glob('model_*'), key=get_model_index)
        model_names_map = {}  # model_dir -> model_name
        
        # Get model names from results.json
        model_infos = results_data.get('model_infos', [])
        for idx, model_info in enumerate(model_infos):
            model_name = model_info.get('name', f'模型_{idx+1}')
            # 确保 model_name 不是 None，并且是字符串类型
            if model_name is None:
                model_name = f'模型_{idx+1}'
            else:
                model_name = str(model_name)
            
            # Use full name as directory name, sanitize for filesystem
            try:
                safe_model_name = secure_filename(model_name)
                if not safe_model_name:
                    safe_model_name = f'模型_{idx+1}'
            except (TypeError, AttributeError) as e:
                logger.warning(f"secure_filename failed for model name '{model_name}': {e}")
                safe_model_name = f'模型_{idx+1}'
            
            # Match model directory by index (model_0 corresponds to index 0)
            if idx < len(model_dirs):
                model_names_map[model_dirs[idx]] = safe_model_name
        
        if progress_callback:
            progress_callback('正在打包模型文件...', '正在添加模型COCO文件到压缩包', 15)
        
        # 1. Export each model's COCO file to its own directory
        total_models = len(model_dirs)
        for idx, model_dir in enumerate(model_dirs):
            coco_json_path = model_dir / '_annotations.coco.json'
            if not coco_json_path.exists():
                continue
            
            # Get model name for this directory
            model_name = model_names_map.get(model_dir, model_dir.name)
            
            # Read COCO JSON
            coco_data = load_coco_json(coco_json_path)
            if not coco_data:
                continue
            
            # Write to ZIP: model_name/_annotations.coco.json
            coco_json_str = json.dumps(coco_data, ensure_ascii=False, indent=2)
            zip_path = f"{model_name}/_annotations.coco.json"
            zip_file.writestr(zip_path, coco_json_str.encode('utf-8'))
            
            if progress_callback:
                progress = 15 + int((idx + 1) / total_models * 10) if total_models > 0 else 20
                progress_callback('正在打包模型文件...', f'已处理 {idx + 1}/{total_models} 个模型', progress)
        
        # Export model files if requested
        if export_models:
            if progress_callback:
                progress_callback('正在打包模型文件...', '正在添加模型权重文件', 25)
            model_infos_list = results_data.get('model_infos', [])
            for idx, model_dir in enumerate(model_dirs):
                if idx < len(model_infos_list):
                    model_info = model_infos_list[idx]
                    model_file_path = Path(model_info.get('path', ''))
                    if model_file_path.exists() and model_file_path.is_file():
                        # Get model name for this directory
                        model_name = model_names_map.get(model_dir, model_dir.name)
                        # Copy model file to model directory in ZIP
                        model_file_name = model_file_path.name
                        zip_path = f"{model_name}/{model_file_name}"
                        try:
                            zip_file.write(model_file_path, zip_path)
                        except Exception as e:
                            logger.warning(f"Failed to export model file {model_file_path}: {e}")
        
        if progress_callback:
            progress_callback('正在打包图片文件...', '正在添加标注图片到压缩包', 30)
        
        # 2. Add all visualized images to 标注图/ directory
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            for img_file in result_dir.glob(ext):
                if img_file.is_file() and not any(
                    str(img_file).startswith(str(model_dir))
                    for model_dir in model_dirs
                ):
                    image_files.append(img_file)
        
        total_images = len(image_files)
        for idx, img_file in enumerate(sorted(image_files)):
            zip_path = f"标注图/{img_file.name}"
            zip_file.write(img_file, zip_path)
            if progress_callback and total_images > 0:
                progress = 30 + int((idx + 1) / total_images * 20) if total_images > 0 else 50
                progress_callback('正在打包图片文件...', f'已处理 {idx + 1}/{total_images} 张图片', progress)
        
        # 3. Export all original images to 原图/ directory
        if export_originals:
            if progress_callback:
                progress_callback('正在查找原图文件...', '正在搜索所有原图文件路径', 50)
            
            predictions = results_data.get('predictions', {})
            
            # 收集所有预测结果中的图片名称
            all_image_names = set(predictions.keys())
            
            # Find original image paths (check upload folder by session)
            # Try to find original images from upload folder
            upload_folder = app.config['UPLOAD_FOLDER']
            original_images_map = {}  # image_name -> original_path
            
            # Search for original images in upload folder
            for session_dir in upload_folder.glob('*'):
                if session_dir.is_dir():
                    data_dir = session_dir / 'data'
                    if data_dir.exists():
                        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                            for orig_img in data_dir.rglob(ext):
                                if orig_img.is_file():
                                    img_name = orig_img.name
                                    if img_name not in original_images_map:
                                        original_images_map[img_name] = orig_img
                    
                    # Also check extracted directories
                    for extract_dir in session_dir.glob('*_extracted'):
                        if extract_dir.is_dir():
                            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                                for orig_img in extract_dir.rglob(ext):
                                    if orig_img.is_file():
                                        img_name = orig_img.name
                                        if img_name not in original_images_map:
                                            original_images_map[img_name] = orig_img
            
            # Export all original images that have predictions
            if progress_callback:
                progress_callback('正在打包原图文件...', '正在添加所有原图到压缩包', 60)
            
            # 只导出有预测结果的原图
            images_to_export = [img_name for img_name in all_image_names if img_name in original_images_map]
            total_originals = len(images_to_export)
            
            for idx, img_name in enumerate(sorted(images_to_export)):
                if img_name in original_images_map:
                    orig_path = original_images_map[img_name]
                    zip_path = f"原图/{img_name}"
                    zip_file.write(orig_path, zip_path)
                    if progress_callback and total_originals > 0:
                        progress = 60 + int((idx + 1) / total_originals * 30) if total_originals > 0 else 90
                        progress_callback('正在打包原图文件...', f'已处理 {idx + 1}/{total_originals} 张原图', progress)
        
        if progress_callback:
            progress_callback('正在生成分类信息...', '正在生成误检/漏检/低置信度分类JSON', 82)
        
        # 4. Generate classification JSON file
        predictions = results_data.get('predictions', {})
        classification_data = {
            'false_positive': [],  # 误检图片列表，每个包含filename和labels
            'missed': [],  # 漏检图片列表，每个包含filename和labels
            'low_confidence': []  # 低置信度图片列表，每个包含filename和labels
        }
        
        for img_name, img_data in predictions.items():
            # 支持新的多状态格式（statuses数组）或旧的单一状态格式（向后兼容）
            statuses = img_data.get('statuses', [])
            if not statuses:
                old_status = img_data.get('status')
                if old_status:
                    statuses = [old_status]
            
            # 收集误检信息
            if 'false_positive' in statuses:
                false_positive_labels = img_data.get('false_positive_labels', [])
                classification_data['false_positive'].append({
                    'filename': img_name,
                    'labels': false_positive_labels
                })
            
            # 收集漏检信息
            if 'missed' in statuses:
                missed_labels = img_data.get('missed_labels', [])
                classification_data['missed'].append({
                    'filename': img_name,
                    'labels': missed_labels
                })
            
            # 收集低置信度信息
            if 'low_confidence' in statuses:
                low_confidence_labels = img_data.get('low_confidence_labels', [])
                classification_data['low_confidence'].append({
                    'filename': img_name,
                    'labels': low_confidence_labels
                })
        
        # 写入分类JSON文件
        classification_json = json.dumps(classification_data, ensure_ascii=False, indent=2)
        zip_file.writestr('classification.json', classification_json.encode('utf-8'))
        
        if progress_callback:
            progress_callback('正在生成报告...', '正在生成README.html报告文件', 85)
        
        # 5. Generate and add README file
        readme_content = generate_readme(results_data, model_dirs, model_names_map, description=description, result_id=result_id)
        zip_file.writestr('README.html', readme_content.encode('utf-8'))
        
        if progress_callback:
            progress_callback('正在添加拆分脚本...', '正在添加Python拆分脚本', 90)
        
        # 6. Add Python script for splitting images
        split_script = '''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图片分类拆分脚本
根据classification.json文件，将标注图和原图分别拆分到误检、漏检、低置信度目录
"""
import json
import shutil
from pathlib import Path

def main():
    # 获取脚本所在目录（导出根目录）
    script_dir = Path(__file__).parent.absolute()
    
    # 读取分类信息
    classification_file = script_dir / 'classification.json'
    if not classification_file.exists():
        print(f"错误: 找不到分类文件 {classification_file}")
        input("按回车键退出...")
        return
    
    with open(classification_file, 'r', encoding='utf-8') as f:
        classification_data = json.load(f)
    
    # 定义源目录和目标目录
    annotated_dir = script_dir / '标注图'
    original_dir = script_dir / '原图'
    
    output_dirs = {
        'false_positive': {
            'annotated': script_dir / '误检' / '标注图',
            'original': script_dir / '误检' / '原图'
        },
        'missed': {
            'annotated': script_dir / '漏检' / '标注图',
            'original': script_dir / '漏检' / '原图'
        },
        'low_confidence': {
            'annotated': script_dir / '低置信度' / '标注图',
            'original': script_dir / '低置信度' / '原图'
        }
    }
    
    # 创建输出目录
    for category, dirs in output_dirs.items():
        dirs['annotated'].mkdir(parents=True, exist_ok=True)
        dirs['original'].mkdir(parents=True, exist_ok=True)
    
    # 统计信息
    stats = {
        'false_positive': {'annotated': 0, 'original': 0},
        'missed': {'annotated': 0, 'original': 0},
        'low_confidence': {'annotated': 0, 'original': 0}
    }
    
    # 处理每个分类
    for category, images in classification_data.items():
        if category not in output_dirs:
            continue
        
        print(f"\\n处理 {category} 图片...")
        target_dirs = output_dirs[category]
        
        for img_info in images:
            filename = img_info['filename']
            labels = img_info.get('labels', [])
            
            # 复制标注图
            annotated_src = annotated_dir / filename
            if annotated_src.exists():
                annotated_dst = target_dirs['annotated'] / filename
                shutil.copy2(annotated_src, annotated_dst)
                stats[category]['annotated'] += 1
                print(f"  已复制标注图: {filename} (标签: {', '.join(labels) if labels else '无'})")
            else:
                print(f"  警告: 找不到标注图 {filename}")
            
            # 复制原图
            original_src = original_dir / filename
            if original_src.exists():
                original_dst = target_dirs['original'] / filename
                shutil.copy2(original_src, original_dst)
                stats[category]['original'] += 1
            else:
                print(f"  警告: 找不到原图 {filename}")
    
    # 打印统计信息
    print("\\n" + "="*50)
    print("拆分完成！统计信息：")
    print("="*50)
    for category, count in stats.items():
        category_name = {'false_positive': '误检', 'missed': '漏检', 'low_confidence': '低置信度'}.get(category, category)
        print(f"{category_name}:")
        print(f"  标注图: {count['annotated']} 张")
        print(f"  原图: {count['original']} 张")
        print()
    
    print("所有图片已拆分到对应目录：")
    print("  - 误检/标注图/ 和 误检/原图/")
    print("  - 漏检/标注图/ 和 漏检/原图/")
    print("  - 低置信度/标注图/ 和 低置信度/原图/")
    print("\\n按回车键退出...")
    input()

if __name__ == '__main__':
    main()
'''
        zip_file.writestr('split_images.pyw', split_script.encode('utf-8'))
        
        if progress_callback:
            progress_callback('正在完成打包...', '正在完成ZIP文件压缩', 95)
        
        zip_buffer.seek(0)
        if progress_callback:
            progress_callback('导出完成！', 'ZIP文件已生成', 100)
        
        return zip_buffer


@app.route('/api/export/<result_id>')
def export_results(result_id):
    """Export prediction results as ZIP file with real-time progress via SSE"""
    try:
        # 获取查询参数：说明信息、导出选项
        description = request.args.get('description', '').strip()
        export_originals = request.args.get('export_originals', '1') == '1'  # 默认导出原图
        export_models = request.args.get('export_models', '0') == '1'  # 默认不导出模型文件
        
        # 用于存储进度的队列
        progress_queue = []
        progress_lock = threading.Lock()
        
        def progress_callback(stage, detail, progress):
            """进度回调函数，将进度存储到队列"""
            with progress_lock:
                progress_queue.append({
                    'stage': stage,
                    'detail': detail,
                    'progress': progress
                })
        
        def generate():
            """生成器函数，用于SSE流式响应"""
            try:
                # 在后台线程执行导出
                zip_buffer = None
                export_error = None
                
                def export_thread():
                    nonlocal zip_buffer, export_error
                    try:
                        zip_buffer = _export_results_with_progress(
                            result_id, description, export_originals, export_models,
                            progress_callback=progress_callback
                        )
                    except Exception as e:
                        export_error = e
                        logger.error(f"Export error: {e}", exc_info=True)
                
                # 启动导出线程
                thread = threading.Thread(target=export_thread)
                thread.daemon = True
                thread.start()
                
                # 持续发送进度，直到导出完成
                last_progress = -1
                while thread.is_alive() or len(progress_queue) > 0:
                    # 发送队列中的进度
                    with progress_lock:
                        while len(progress_queue) > 0:
                            progress_info = progress_queue.pop(0)
                            data = json.dumps(progress_info, ensure_ascii=False)
                            yield f"data: {data}\n\n"
                            last_progress = progress_info.get('progress', last_progress)
                    
                    time.sleep(0.1)  # 避免CPU占用过高
                
                # 等待线程完全完成（增加超时时间）
                thread.join(timeout=300)  # 最多等待5分钟
                
                if thread.is_alive():
                    # 线程超时，返回错误
                    error_data = {
                        'stage': '导出失败',
                        'detail': '导出超时，请重试',
                        'progress': 0,
                        'error': '导出超时'
                    }
                    yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
                    return
                
                # 检查是否有错误
                if export_error:
                    error_data = {
                        'stage': '导出失败',
                        'detail': f'错误: {str(export_error)}',
                        'progress': 0,
                        'error': str(export_error)
                    }
                    yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
                    return
                
                # 检查zip_buffer是否生成
                if not zip_buffer:
                    error_data = {
                        'stage': '导出失败',
                        'detail': 'ZIP文件生成失败',
                        'progress': 0,
                        'error': 'ZIP文件生成失败'
                    }
                    yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
                    return
                
                # 将ZIP文件保存到临时文件，返回文件路径
                try:
                    import tempfile
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.zip', dir=str(app.config['EXPORT_CACHE_FOLDER']))
                    zip_buffer.seek(0)
                    zip_data = zip_buffer.read()
                    if len(zip_data) == 0:
                        temp_file.close()
                        error_data = {
                            'stage': '导出失败',
                            'detail': 'ZIP文件为空',
                            'progress': 0,
                            'error': 'ZIP文件为空'
                        }
                        yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
                        return
                    
                    temp_file.write(zip_data)
                    temp_file.close()
                    
                    # 生成任务ID
                    task_id = str(uuid.uuid4())
                    with export_tasks_lock:
                        export_tasks[task_id] = {
                            'file_path': temp_file.name,
                            'filename': f'predictions_{result_id}.zip',
                            'created_at': datetime.now()
                        }
                    
                    # 发送完成信号和文件下载信息（合并为一个消息，确保前端能同时收到）
                    complete_data = {
                        'stage': '导出完成！',
                        'detail': 'ZIP文件已生成，正在下载',
                        'progress': 100,
                        'complete': True,
                        'ready': True,
                        'task_id': task_id
                    }
                    yield f"data: {json.dumps(complete_data, ensure_ascii=False)}\n\n"
                    
                except Exception as save_error:
                    logger.error(f"Error saving ZIP file: {save_error}", exc_info=True)
                    error_data = {
                        'stage': '导出失败',
                        'detail': f'保存ZIP文件失败: {str(save_error)}',
                        'progress': 0,
                        'error': f'保存ZIP文件失败: {str(save_error)}'
                    }
                    yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
                    return
                
            except Exception as e:
                logger.error(f"Error in export generation: {e}", exc_info=True)
                error_data = {
                    'stage': '导出失败',
                    'detail': f'错误: {str(e)}',
                    'progress': 0,
                    'error': str(e)
                }
                yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
        
        return Response(generate(), mimetype='text/event-stream', headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        })
        
    except Exception as e:
        logger.error(f"Error exporting results: {e}", exc_info=True)
        return jsonify({'error': f'Export failed: {str(e)}'}), 500


@app.route('/api/export/download/<task_id>')
def download_export(task_id):
    """下载导出完成的ZIP文件"""
    try:
        with export_tasks_lock:
            if task_id not in export_tasks:
                return jsonify({'error': 'Task not found'}), 404
            
            task_info = export_tasks[task_id]
            file_path = Path(task_info['file_path'])
            filename = task_info['filename']
            
            if not file_path.exists():
                return jsonify({'error': 'Export file not found'}), 404
            
            # 发送文件
            return send_file(
                str(file_path),
                mimetype='application/zip',
                as_attachment=True,
                download_name=filename
            )
    except Exception as e:
        logger.error(f"Error downloading export: {e}", exc_info=True)
        return jsonify({'error': f'Download failed: {str(e)}'}), 500


if __name__ == '__main__':
    # Disable Flask's default error handlers for API routes
    app.config['PROPAGATE_EXCEPTIONS'] = True
    app.run(host='0.0.0.0', port=6006, debug=True)

