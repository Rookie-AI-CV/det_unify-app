#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Flask Web Application for Model Prediction and Evaluation
"""

import os
import zipfile
import shutil
import json
import cv2
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory, Response, send_file
from werkzeug.utils import secure_filename
import uuid
from loguru import logger
from collections import defaultdict

# Import prediction utilities
import sys
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
from src.predict.predict import detect_model_type

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB max file size
app.config['UPLOAD_FOLDER'] = Path(__file__).parent / 'static' / 'uploads'
app.config['RESULTS_FOLDER'] = Path(__file__).parent / 'static' / 'results'

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
            
            # Run prediction script
            try:
                import subprocess
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT_DIR))
                if result.returncode != 0:
                    logger.error(f"Prediction failed for model {model_path}: {result.stderr}")
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
            
            if img_predictions:
                # Draw predictions (only bounding boxes, no text on image)
                vis_img = img.copy()
                for pred_group in img_predictions:
                    line_style = pred_group['line_style']
                    for bbox_info in pred_group['bboxes']:
                        bbox = bbox_info['bbox']
                        color_rgb = bbox_info['color']  # Color based on label (RGB format)
                        # Convert RGB to BGR for OpenCV
                        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
                        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                        draw_rectangle_with_style(vis_img, (x1, y1), (x2, y2), color_bgr, 2, style=line_style)
                
                output_path = result_dir / img_name
                cv2.imwrite(str(output_path), vis_img)
                
                predictions[img_name] = {
                    'image': f'/static/results/{result_id}/{img_name}',
                    'predictions': img_predictions,
                    'notes': '',  # Initialize notes field
                    'status': None  # Initialize status field (None, 'false_positive', or 'missed')
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
        
        # Save results to JSON file for later retrieval
        results_data = {
            'result_id': result_id,
            'predictions': predictions,
            'model_infos': display_model_infos,
            'label_colors': {k: list(v) for k, v in label_colors.items()}  # Convert tuple to list for JSON
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
        status = data.get('status')  # 'false_positive', 'missed', or None
        
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
        
        # Update notes and status for the image
        if image_name in results_data.get('predictions', {}):
            # Ensure notes is a string and handle None
            notes_str = str(notes) if notes is not None else ''
            results_data['predictions'][image_name]['notes'] = notes_str
            results_data['predictions'][image_name]['status'] = status
            
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


def generate_readme(results_data, model_dirs, model_names_map):
    """Generate a beautiful HTML README file with model info and statistics"""
    from datetime import datetime
    import html as html_lib  # Import html module with alias to avoid conflict
    
    model_infos = results_data.get('model_infos', [])
    label_colors = results_data.get('label_colors', {})
    predictions = results_data.get('predictions', {})
    
    # Statistics: Count predictions per model per label
    stats = {}  # {model_name: {label: count}}
    total_images = len(predictions)
    total_predictions = 0
    
    # Initialize stats for all models
    for model_info in model_infos:
        model_name = model_info.get('name', 'Unknown')
        stats[model_name] = {}
    
    # Count predictions from all images
    for img_name, img_data in predictions.items():
        for pred_group in img_data.get('predictions', []):
            model_name = pred_group.get('full_name', pred_group.get('model_name', 'Unknown'))
            if model_name not in stats:
                stats[model_name] = {}
            
            for bbox in pred_group.get('bboxes', []):
                label = bbox.get('label', 'unknown')
                stats[model_name][label] = stats[model_name].get(label, 0) + 1
                total_predictions += 1
    
    # Collect all unique labels
    all_labels = set()
    for model_stats in stats.values():
        all_labels.update(model_stats.keys())
    all_labels = sorted(all_labels)
    
    # Style descriptions
    style_map = {
        'solid': '实线',
        'dashed': '虚线',
        'dotted': '点线',
        'dashdot': '点划线'
    }
    
    # Generate HTML
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>预测结果说明文档</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Microsoft YaHei', '微软雅黑', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 700;
        }}
        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        .content {{
            padding: 40px;
        }}
        .section {{
            margin-bottom: 40px;
        }}
        .section h2 {{
            color: #667eea;
            font-size: 1.8em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }}
        .section h3 {{
            color: #764ba2;
            font-size: 1.4em;
            margin: 30px 0 15px 0;
        }}
        .model-card {{
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 8px;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .model-card:hover {{
            transform: translateX(5px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}
        .model-name {{
            font-size: 1.3em;
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
        }}
        .model-info {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .info-item {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .info-label {{
            font-weight: 600;
            color: #666;
            min-width: 80px;
        }}
        .info-value {{
            color: #333;
        }}
        .line-style-demo {{
            display: inline-block;
            width: 60px;
            height: 3px;
            margin: 0 8px;
            vertical-align: middle;
        }}
        .line-style-demo.solid {{
            border-bottom: 3px solid #333;
            border-bottom-style: solid;
        }}
        .line-style-demo.dashed {{
            border-bottom: 3px solid #333;
            border-bottom-style: dashed;
        }}
        .line-style-demo.dotted {{
            border-bottom: 3px solid #333;
            border-bottom-style: dotted;
        }}
        .line-style-demo.dashdot {{
            border-bottom: 3px solid #333;
            border-bottom-style: dashdot;
        }}
        .label-color-box {{
            display: inline-block;
            width: 24px;
            height: 24px;
            border-radius: 4px;
            border: 2px solid #ddd;
            vertical-align: middle;
            margin-right: 8px;
        }}
        .stats-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .stats-table th {{
            background: #667eea;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        .stats-table td {{
            padding: 12px 15px;
            border-bottom: 1px solid #eee;
        }}
        .stats-table tr:hover {{
            background: #f8f9fa;
        }}
        .stats-table tr:last-child td {{
            border-bottom: none;
        }}
        .number-badge {{
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 4px 10px;
            border-radius: 12px;
            font-weight: 600;
            font-size: 0.9em;
        }}
        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .summary-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }}
        .summary-card h3 {{
            font-size: 0.9em;
            opacity: 0.9;
            margin-bottom: 10px;
            color: white;
        }}
        .summary-card .value {{
            font-size: 2.5em;
            font-weight: bold;
        }}
        .footer {{
            background: #f8f9fa;
            padding: 20px 40px;
            text-align: center;
            color: #666;
            border-top: 1px solid #eee;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 预测结果说明文档</h1>
            <p>生成时间: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}</p>
        </div>
        
        <div class="content">
            <!-- Summary Section -->
            <div class="section">
                <h2>📈 总体统计</h2>
                <div class="summary-cards">
                    <div class="summary-card">
                        <h3>图片总数</h3>
                        <div class="value">{total_images}</div>
                    </div>
                    <div class="summary-card">
                        <h3>检测框总数</h3>
                        <div class="value">{total_predictions}</div>
                    </div>
                    <div class="summary-card">
                        <h3>模型数量</h3>
                        <div class="value">{len(model_infos)}</div>
                    </div>
                    <div class="summary-card">
                        <h3>标签类别</h3>
                        <div class="value">{len(all_labels)}</div>
                    </div>
                </div>
            </div>
            
            <!-- Model Information Section -->
            <div class="section">
                <h2>🤖 模型信息</h2>
"""
    
    # Add model cards
    for model_info in model_infos:
        model_name = model_info.get('name', 'Unknown')
        model_type = model_info.get('type', 'Unknown')
        sub_type = model_info.get('sub_type', '')
        line_style = model_info.get('line_style', 'solid')
        style_desc = model_info.get('style_desc', style_map.get(line_style, '实线'))
        
        model_total = sum(stats.get(model_name, {}).values())
        
        html += f"""
                <div class="model-card">
                    <div class="model-name">🔹 {model_name}</div>
                    <div class="model-info">
                        <div class="info-item">
                            <span class="info-label">模型类型:</span>
                            <span class="info-value">{model_type}{' - ' + sub_type if sub_type else ''}</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">线条风格:</span>
                            <span class="info-value">
                                <span class="line-style-demo {line_style}"></span>
                                {style_desc}
                            </span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">检测总数:</span>
                            <span class="info-value"><span class="number-badge">{model_total}</span></span>
                        </div>
                    </div>
                </div>
"""
    
    html += """
            </div>
            
            <!-- Label Colors Section -->
            <div class="section">
                <h2>🎨 标签颜色说明</h2>
                <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px; margin-top: 20px;">
"""
    
    # Add label colors
    for label in all_labels:
        color = label_colors.get(label, [128, 128, 128])
        html += f"""
                    <div style="display: flex; align-items: center; padding: 10px; background: #f8f9fa; border-radius: 6px;">
                        <span class="label-color-box" style="background-color: rgb({color[0]}, {color[1]}, {color[2]});"></span>
                        <span style="font-weight: 500;">{label}</span>
                    </div>
"""
    
    html += """
                </div>
            </div>
            
            <!-- Statistics Section -->
            <div class="section">
                <h2>📊 详细统计</h2>
                <h3>各模型标签检测数量统计</h3>
                <table class="stats-table">
                    <thead>
                        <tr>
                            <th>标签</th>
"""
    
    # Add model columns
    for model_info in model_infos:
        model_name = model_info.get('name', 'Unknown')
        short_name = model_info.get('short_name', model_name)
        html += f'                            <th>{short_name}</th>\n'
    
    html += """                            <th>总计</th>
                        </tr>
                    </thead>
                    <tbody>
"""
    
    # Add statistics rows
    for label in all_labels:
        html += f'                        <tr>\n                            <td style="font-weight: 600;">{label}</td>\n'
        label_total = 0
        for model_info in model_infos:
            model_name = model_info.get('name', 'Unknown')
            count = stats.get(model_name, {}).get(label, 0)
            label_total += count
            html += f'                            <td style="text-align: center;"><span class="number-badge">{count}</span></td>\n'
        html += f'                            <td style="text-align: center; font-weight: bold;"><span class="number-badge">{label_total}</span></td>\n                        </tr>\n'
    
    html += """                    </tbody>
                </table>
            </div>
            
            <!-- Image Notes and Status Section -->
            <div class="section">
                <h2>📝 图片备注与状态</h2>
"""
    
    # Collect images with notes or status
    images_with_info = []
    false_positive_count = 0
    missed_count = 0
    
    for img_name, img_data in predictions.items():
        # Safely get notes and status, handle None and non-string values
        notes = img_data.get('notes', '')
        if notes is None:
            notes = ''
        else:
            notes = str(notes).strip()
        
        status = img_data.get('status')
        if notes or status:
            images_with_info.append({
                'name': str(img_name),
                'notes': notes,
                'status': status
            })
            if status == 'false_positive':
                false_positive_count += 1
            elif status == 'missed':
                missed_count += 1
    
    if images_with_info:
        html += f"""
                <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; margin-top: 20px;">
                    <h3>标注统计</h3>
                    <div style="display: flex; gap: 20px; margin-bottom: 20px;">
                        <div style="background: #e74c3c; color: white; padding: 10px 20px; border-radius: 6px;">
                            <strong>误检图片:</strong> {false_positive_count} 张
                        </div>
                        <div style="background: #f39c12; color: white; padding: 10px 20px; border-radius: 6px;">
                            <strong>漏检图片:</strong> {missed_count} 张
                        </div>
                    </div>
                    <table class="stats-table">
                        <thead>
                            <tr>
                                <th>图片名称</th>
                                <th>状态</th>
                                <th>备注</th>
                            </tr>
                        </thead>
                        <tbody>
"""
        for img_info in sorted(images_with_info, key=lambda x: x['name']):
            status_text = ''
            status_class = ''
            if img_info['status'] == 'false_positive':
                status_text = '<span style="background: #e74c3c; color: white; padding: 4px 10px; border-radius: 12px; font-size: 0.9em;">误检</span>'
            elif img_info['status'] == 'missed':
                status_text = '<span style="background: #f39c12; color: white; padding: 4px 10px; border-radius: 12px; font-size: 0.9em;">漏检</span>'
            else:
                status_text = '<span style="color: #999;">-</span>'
            
            notes_raw = img_info.get('notes', '') or ''
            if notes_raw:
                # Escape HTML special characters in notes
                notes_display = html_lib.escape(str(notes_raw))
            else:
                notes_display = '<span style="color: #999; font-style: italic;">无备注</span>'
            
            html += f"""
                            <tr>
                                <td style="font-weight: 500;">{html_lib.escape(str(img_info.get('name', '')))}</td>
                                <td>{status_text}</td>
                                <td style="max-width: 400px; word-wrap: break-word;">{notes_display}</td>
                            </tr>
"""
        html += """
                        </tbody>
                    </table>
                </div>
"""
    else:
        html += """
                <p style="color: #999; font-style: italic;">暂无图片备注或状态标记</p>
"""
    
    html += """
            </div>
            
            <!-- Directory Structure Section -->
            <div class="section">
                <h2>📁 目录结构说明</h2>
                <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; font-family: 'Courier New', monospace; line-height: 1.8;">
                    <div style="margin-bottom: 10px;"><strong>predictions.zip</strong></div>
"""
    
    for model_info in model_infos:
        model_name = model_info.get('name', 'Unknown')
        safe_name = secure_filename(model_name) or f'模型_{model_infos.index(model_info) + 1}'
        html += f'                    <div style="margin-left: 20px;">├── <strong>{safe_name}/</strong></div>\n'
        html += f'                    <div style="margin-left: 40px;">└── _annotations.coco.json</div>\n'
    
    html += """                    <div style="margin-left: 20px;">├── <strong>images/</strong></div>
                    <div style="margin-left: 40px;">├── *.jpg (标注后的可视化图片)</div>
                    <div style="margin-left: 40px;">└── *.png (标注后的可视化图片)</div>
"""
    if false_positive_count > 0:
        html += """                    <div style="margin-left: 20px;">├── <strong>false_positives/</strong></div>
                    <div style="margin-left: 40px;">└── 误检图片原图</div>
"""
    if missed_count > 0:
        html += """                    <div style="margin-left: 20px;">└── <strong>missed/</strong></div>
                    <div style="margin-left: 40px;">└── 漏检图片原图</div>
"""
    html += """                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>Generated by Detection Prediction System | 本报告由目标检测预测系统自动生成</p>
        </div>
    </div>
</body>
</html>
"""
    
    return html


@app.route('/api/export/<result_id>')
def export_results(result_id):
    """Export prediction results as ZIP file (COCO JSON + visualized images)"""
    try:
        result_dir = app.config['RESULTS_FOLDER'] / result_id
        if not result_dir.exists():
            return jsonify({'error': 'Result not found'}), 404
        
        # Load results.json to get prediction data
        results_file = result_dir / 'results.json'
        if not results_file.exists():
            return jsonify({'error': 'Results file not found'}), 404
        
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                results_data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in results.json: {e}")
            logger.error(f"File path: {results_file}")
            # Try to read raw content for debugging
            try:
                with open(results_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    logger.error(f"File size: {len(content)} bytes")
                    if len(content) > 0:
                        logger.error(f"First 500 chars: {content[:500]}")
                        if len(content) > 500:
                            logger.error(f"Last 500 chars: {content[-500:]}")
            except Exception as debug_e:
                logger.error(f"Error reading file for debug: {debug_e}")
            return jsonify({'error': f'Invalid JSON in results file: {str(e)}'}), 500
        
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
                # Use full name as directory name, sanitize for filesystem
                safe_model_name = secure_filename(model_name)
                if not safe_model_name:
                    safe_model_name = f'模型_{idx+1}'
                
                # Match model directory by index (model_0 corresponds to index 0)
                if idx < len(model_dirs):
                    model_names_map[model_dirs[idx]] = safe_model_name
            
            # 1. Export each model's COCO file to its own directory
            for model_dir in model_dirs:
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
            
            # 2. Add all visualized images to images/ directory
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                for img_file in result_dir.glob(ext):
                    if img_file.is_file() and not any(
                        str(img_file).startswith(str(model_dir))
                        for model_dir in model_dirs
                    ):
                        image_files.append(img_file)
            
            for img_file in sorted(image_files):
                zip_path = f"images/{img_file.name}"
                zip_file.write(img_file, zip_path)
            
            # 3. Save original images for false positives and missed detections
            predictions = results_data.get('predictions', {})
            false_positive_images = []
            missed_images = []
            
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
            
            # Categorize images by status
            for img_name, img_data in predictions.items():
                status = img_data.get('status')
                if status == 'false_positive':
                    false_positive_images.append(img_name)
                elif status == 'missed':
                    missed_images.append(img_name)
            
            # Add false positive original images
            for img_name in false_positive_images:
                if img_name in original_images_map:
                    orig_path = original_images_map[img_name]
                    zip_path = f"false_positives/{img_name}"
                    zip_file.write(orig_path, zip_path)
            
            # Add missed detection original images
            for img_name in missed_images:
                if img_name in original_images_map:
                    orig_path = original_images_map[img_name]
                    zip_path = f"missed/{img_name}"
                    zip_file.write(orig_path, zip_path)
            
            # 4. Generate and add README file
            readme_content = generate_readme(results_data, model_dirs, model_names_map)
            zip_file.writestr('README.html', readme_content.encode('utf-8'))
        
        # Prepare response
        zip_buffer.seek(0)
        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'predictions_{result_id}.zip'
        )
        
    except Exception as e:
        logger.error(f"Error exporting results: {e}", exc_info=True)
        return jsonify({'error': f'Export failed: {str(e)}'}), 500


if __name__ == '__main__':
    # Disable Flask's default error handlers for API routes
    app.config['PROPAGATE_EXCEPTIONS'] = True
    app.run(host='0.0.0.0', port=6006, debug=True)

