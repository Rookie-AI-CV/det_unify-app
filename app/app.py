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
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import uuid
from loguru import logger

# Import prediction utilities
import sys
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
from src.predict.predict import detect_model_type

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB max file size
app.config['UPLOAD_FOLDER'] = Path(__file__).parent / 'static' / 'uploads'
app.config['RESULTS_FOLDER'] = Path(__file__).parent / 'static' / 'results'

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
    for ext in ['*.pt', '*.pth', '*.ckpt']:
        model_files.extend(Path(directory).glob(ext))
        model_files.extend(Path(directory).rglob(ext))
    return [str(f) for f in model_files]


def find_image_files(directory):
    """Find all image files in directory"""
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(Path(directory).glob(ext))
        image_files.extend(Path(directory).rglob(ext))
    return sorted([str(f) for f in image_files])


def load_model_wrapper(checkpoint_path):
    """Load model and return model object and type info"""
    model_type, hq_sub_type = detect_model_type(checkpoint_path)
    if model_type is None:
        return None, None, None, "无法识别模型类型"
    
    try:
        if model_type == 'dino':
            from ml_backend.model import ModelInfo, MODE
            from ml_backend.predict.algos.det02 import DET02Predictor
            model_info = ModelInfo(model_id="dino", model_type="dino", 
                                  checkpoint_path=checkpoint_path, mode=MODE.PREDICT)
            predictor = DET02Predictor(model_info)
            predictor.load_model()
            return predictor, 'dino', None, None
        else:
            from src.predict.predict_hq_det import load_hq_model
            model = load_hq_model(checkpoint_path, hq_sub_type, device='cuda:0')
            return model, 'hq_det', hq_sub_type, None
    except Exception as e:
        return None, None, None, str(e)


def predict_with_model(model, model_type, hq_sub_type, image_path, threshold, max_size=1536):
    """Predict with loaded model"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None, "无法读取图片"
        
        if model_type == 'dino':
            result = model.predict(image_path, threshold)
            return result, None
        else:
            import inspect
            sig = inspect.signature(model.predict)
            params = list(sig.parameters.keys())
            
            predict_kwargs = {}
            if 'bgr' in params:
                predict_kwargs['bgr'] = True
            if 'confidence' in params or 'conf' in params:
                predict_kwargs['confidence' if 'confidence' in params else 'conf'] = threshold
            if 'max_size' in params or 'imgsz' in params:
                predict_kwargs['max_size' if 'max_size' in params else 'imgsz'] = max_size
            
            predict_results = model.predict([img], **predict_kwargs)
            result = predict_results[0] if isinstance(predict_results, list) else predict_results
            return result, None
    except Exception as e:
        return None, str(e)


def convert_result_to_bboxes(result, model_type):
    """Convert prediction result to bboxes format"""
    bboxes = []
    
    if model_type == 'dino':
        if hasattr(result, 'predictions') and result.predictions:
            for pred in result.predictions:
                for point in pred.points:
                    bboxes.append({
                        'bbox': [point.x, point.y, point.x + point.w, point.y + point.h],
                        'label': pred.name,
                        'score': pred.confidence
                    })
    else:
        if hasattr(result, 'bboxes') and result.bboxes:
            labels = getattr(result, 'labels', None)
            scores = getattr(result, 'scores', None)
            for i, bbox in enumerate(result.bboxes):
                if len(bbox) >= 4:
                    label = str(labels[i]) if labels and i < len(labels) and labels[i] else "object"
                    score = float(scores[i]) if scores and i < len(scores) else 1.0
                    bboxes.append({
                        'bbox': [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                        'label': label,
                        'score': score
                    })
    
    return bboxes


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
        for model_file in model_files:
            try:
                model_type, hq_sub_type = detect_model_type(model_file)
                if model_type:
                    models_info.append({
                        'path': model_file,
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
        
        return jsonify({
            'session_id': session_id,
            'images': [os.path.basename(img) for img in image_files],
            'image_paths': image_files
        })
    except Exception as e:
        logger.error(f"Error in upload_data: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """Run prediction"""
    data = request.json
    session_id = data.get('session_id')
    model_paths = data.get('model_paths', [])
    image_paths = data.get('image_paths', [])
    threshold = float(data.get('threshold', 0.5))
    max_size = int(data.get('max_size', 1536))
    
    if not session_id or not model_paths or not image_paths:
        return jsonify({'error': 'Missing required parameters'}), 400
    
    result_id = str(uuid.uuid4())
    result_dir = app.config['RESULTS_FOLDER'] / result_id
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all models
    loaded_models = []
    model_infos = []
    for model_path in model_paths:
        model, model_type, hq_sub_type, error = load_model_wrapper(model_path)
        if model:
            loaded_models.append(model)
            model_infos.append({
                'path': model_path,
                'name': os.path.basename(model_path),
                'type': model_type,
                'sub_type': hq_sub_type
            })
        else:
            logger.warning(f"Failed to load model {model_path}: {error}")
    
    if not loaded_models:
        return jsonify({'error': 'No models loaded successfully'}), 400
    
    # Generate colors for each model
    colors = []
    np.random.seed(42)
    for i in range(len(loaded_models)):
        colors.append(tuple(map(int, np.random.randint(0, 255, 3))))
    
    # Predict all images
    predictions = {}
    
    for img_idx, image_path in enumerate(image_paths):
        img_name = os.path.basename(image_path)
        img = cv2.imread(image_path)
        if img is None:
            continue
        
        img_predictions = []
        
        for model_idx, (model, model_info) in enumerate(zip(loaded_models, model_infos)):
            result, error = predict_with_model(
                model, model_info['type'], model_info['sub_type'],
                image_path, threshold, max_size
            )
            
            if result:
                bboxes = convert_result_to_bboxes(result, model_info['type'])
                img_predictions.append({
                    'model_name': model_info['name'],
                    'model_idx': model_idx,
                    'color': colors[model_idx],
                    'bboxes': bboxes
                })
        
        if img_predictions:
            # Draw predictions
            vis_img = img.copy()
            for pred_group in img_predictions:
                color = pred_group['color']
                for bbox_info in pred_group['bboxes']:
                    bbox = bbox_info['bbox']
                    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
                    
                    label = f"{pred_group['model_name']}: {bbox_info['label']} {bbox_info['score']:.2f}"
                    cv2.putText(vis_img, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            output_path = result_dir / img_name
            cv2.imwrite(str(output_path), vis_img)
            
            predictions[img_name] = {
                'image': f'/static/results/{result_id}/{img_name}',
                'predictions': img_predictions
            }
    
    return jsonify({
        'result_id': result_id,
        'predictions': predictions,
        'model_infos': model_infos
    })


@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6006, debug=True)

