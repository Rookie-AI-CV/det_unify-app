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
from flask import Flask, render_template, request, jsonify, send_from_directory, Response
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
                        color = bbox_info['color']  # Color based on label
                        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                        draw_rectangle_with_style(vis_img, (x1, y1), (x2, y2), color, 2, style=line_style)
                
                output_path = result_dir / img_name
                cv2.imwrite(str(output_path), vis_img)
                
                predictions[img_name] = {
                    'image': f'/static/results/{result_id}/{img_name}',
                    'predictions': img_predictions
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


if __name__ == '__main__':
    # Disable Flask's default error handlers for API routes
    app.config['PROPAGATE_EXCEPTIONS'] = True
    app.run(host='0.0.0.0', port=6006, debug=True)

