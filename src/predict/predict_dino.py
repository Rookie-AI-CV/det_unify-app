#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DINO batch prediction script
Output: COCO format JSON file
"""

import argparse
import json
import os
import sys
import re
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ml_backend.predict.algos.det02 import DET02Predictor
from ml_backend.model import ModelInfo, MODE
from loguru import logger


def get_images(image_path=None, image_dir=None, image_list=None):
    """Get image file list"""
    if image_path:
        return [image_path] if os.path.isfile(image_path) else []
    elif image_dir:
        exts = {'.jpg', '.jpeg', '.png'}
        return [str(p) for p in Path(image_dir).rglob('*') if p.suffix.lower() in exts]
    elif image_list:
        return [line.strip() for line in open(image_list) if line.strip() and os.path.isfile(line.strip())]
    return []


def get_size(img_path):
    """Get image width and height"""
    img = cv2.imread(img_path)
    return (img.shape[1], img.shape[0]) if img is not None else (0, 0)


def contains_chinese(text):
    """Check if text contains Chinese characters"""
    pattern = re.compile(r'[\u4e00-\u9fa5]')
    return bool(pattern.search(text))


def find_chinese_font():
    """Find a Chinese font file"""
    font_paths = [
        "/opt/product/ml_backend/simsun.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "/System/Library/Fonts/PingFang.ttc",  # macOS
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            return font_path
    
    # Try to use default font (may not support Chinese)
    try:
        return ImageFont.load_default()
    except:
        return None


def draw_text_chinese(img, text, position, font_size, color=(255, 255, 255)):
    """Draw text on image using PIL (supports Chinese)"""
    # Convert BGR to RGB for PIL
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    
    # Get font
    font_path_or_obj = find_chinese_font()
    font = None
    
    try:
        if isinstance(font_path_or_obj, str):
            # It's a font file path
            font = ImageFont.truetype(font_path_or_obj, font_size, encoding="utf-8")
        elif isinstance(font_path_or_obj, ImageFont.FreeTypeFont):
            # It's already a font object, but we need to resize it
            # Try to reload with new size if it's from a file
            font = font_path_or_obj
        else:
            # Use default font
            font = ImageFont.load_default()
    except Exception as e:
        logger.warning(f"Failed to load font, using default: {e}")
        try:
            font = ImageFont.load_default()
        except:
            # Last resort: use a basic font
            font = None
    
    # Draw text
    if font:
        draw.text(position, text, fill=color, font=font)
    else:
        draw.text(position, text, fill=color)
    
    # Convert back to BGR for OpenCV
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def resize_image_max_size(img, max_size=1536):
    """Resize image keeping aspect ratio, max width or height is max_size"""
    h, w = img.shape[:2]
    if max(h, w) <= max_size:
        return img
    
    if h > w:
        new_h = max_size
        new_w = int(w * max_size / h)
    else:
        new_w = max_size
        new_h = int(h * max_size / w)
    
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def draw_detections(image, result, max_size=1536):
    """Draw bounding boxes with labels and confidence scores on image"""
    img = image.copy()
    orig_h, orig_w = img.shape[:2]
    
    # Resize image if needed
    scale = 1.0
    if max(orig_h, orig_w) > max_size:
        img = resize_image_max_size(img, max_size)
        new_h, new_w = img.shape[:2]
        scale = min(new_w / orig_w, new_h / orig_h)
    else:
        new_h, new_w = orig_h, orig_w
    
    # Calculate font size based on resized image size
    font_scale = min(new_h, new_w) * 0.0008
    thickness = max(1, int(min(new_h, new_w) * 0.001))
    box_thickness = max(2, int(min(new_h, new_w) * 0.002))
    
    if not hasattr(result, 'predictions') or not result.predictions:
        return img
    
    # Generate colors for different classes
    np.random.seed(42)
    colors = {}
    
    for pred in result.predictions:
        class_name = pred.name
        if class_name not in colors:
            colors[class_name] = tuple(map(int, np.random.randint(0, 255, 3)))
        
        color = colors[class_name]
        
        for point in pred.points:
            # Scale coordinates if image was resized
            x1 = int(point.x * scale)
            y1 = int(point.y * scale)
            x2 = int((point.x + point.w) * scale)
            y2 = int((point.y + point.h) * scale)
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, box_thickness)
            
            # Prepare label text
            label = f"{class_name} {pred.confidence:.2f}"
            
            # Check if text contains Chinese
            has_chinese = contains_chinese(label)
            
            if has_chinese:
                # Use PIL for Chinese text
                # Estimate text size (approximate)
                font_size = int(min(new_h, new_w) * 0.02)  # Adjust based on image size
                font_size = max(12, min(font_size, 40))  # Limit font size
                
                # Create a temporary PIL image to measure text size
                temp_img = Image.new('RGB', (100, 100))
                temp_draw = ImageDraw.Draw(temp_img)
                font_path_or_obj = find_chinese_font()
                font = None
                
                try:
                    if isinstance(font_path_or_obj, str):
                        font = ImageFont.truetype(font_path_or_obj, font_size, encoding="utf-8")
                    else:
                        font = ImageFont.load_default()
                except Exception:
                    font = ImageFont.load_default()
                
                if font is None:
                    font = ImageFont.load_default()
                
                bbox = temp_draw.textbbox((0, 0), label, font=font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
                baseline = int(th * 0.2)
                
                # Draw label background
                cv2.rectangle(
                    img, 
                    (x1, y1 - th - baseline - 5), 
                    (x1 + tw + 5, y1), 
                    color, 
                    -1
                )
                
                # Draw text using PIL (returns modified image)
                img = draw_text_chinese(
                    img, 
                    label, 
                    (x1, y1 - th - baseline - 3), 
                    font_size, 
                    (255, 255, 255)
                )
            else:
                # Use OpenCV for non-Chinese text (faster)
                (tw, th), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                )
                
                # Draw label background
                cv2.rectangle(
                    img, 
                    (x1, y1 - th - baseline - 5), 
                    (x1 + tw, y1), 
                    color, 
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    img,
                    label,
                    (x1, y1 - baseline - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )
    
    return img


def to_coco(results, output_file):
    """Convert results to COCO format JSON"""
    coco = {"images": [], "annotations": [], "categories": []}
    cat_map, cat_id, ann_id = {}, 1, 1
    
    for img_idx, result in enumerate(results):
        # 使用image_data字段存储图片路径
        img_path = getattr(result, 'image_data', '')
        if not img_path or not isinstance(img_path, str):
            logger.warning(f"跳过结果 {img_idx}: 缺少有效的图片路径")
            continue
        w, h = get_size(img_path)
        if w == 0 or h == 0:
            logger.warning(f"跳过图片 {img_path}: 无法读取尺寸")
            continue
        
        img_id = img_idx + 1
        coco["images"].append({"id": img_id, "file_name": os.path.basename(img_path), "width": w, "height": h})
        
        if hasattr(result, 'predictions') and result.predictions:
            logger.debug(f"图片 {img_path} 有 {len(result.predictions)} 个预测结果")
            for pred in result.predictions:
                if pred.name not in cat_map:
                    cat_map[pred.name] = cat_id
                    coco["categories"].append({"id": cat_id, "name": pred.name, "supercategory": "none"})
                    cat_id += 1
                
                for point in pred.points:
                    coco["annotations"].append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cat_map[pred.name],
                        "bbox": [point.x, point.y, point.w, point.h],
                        "area": point.w * point.h,
                        "iscrowd": 0,
                        "score": pred.confidence
                    })
                    ann_id += 1
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved to {output_file}: {len(coco['images'])} images, {len(coco['annotations'])} annotations")
    
    # 如果结果为空，输出调试信息
    if len(coco['images']) == 0:
        logger.warning("COCO JSON为空！可能的原因：")
        logger.warning(f"  1. 输入results数量: {len(results)}")
        for idx, result in enumerate(results):
            has_data = hasattr(result, 'image_data') and result.image_data
            has_pred = hasattr(result, 'predictions') and result.predictions
            logger.warning(f"  结果 {idx}: image_data={has_data}, predictions={has_pred}")
            if has_pred:
                logger.warning(f"    预测数量: {len(result.predictions)}")


def main():
    parser = argparse.ArgumentParser(description='DINO batch prediction')
    parser.add_argument('--checkpoint', required=True, help='Model checkpoint path')
    parser.add_argument('--image', default=None, help='Single image path')
    parser.add_argument('--image-dir', default=None, help='Image directory')
    parser.add_argument('--image-list', default=None, help='Image list file')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--batch-size', type=int, default=6, help='Batch size')
    parser.add_argument('--output', required=True, help='Output COCO JSON file or directory')
    args = parser.parse_args()
    
    # Get images
    images = get_images(args.image, args.image_dir, args.image_list)
    if not images:
        logger.error("No images found")
        sys.exit(1)
    
    # Load model
    model_info = ModelInfo(model_id="dino", model_type="dino", checkpoint_path=args.checkpoint, mode=MODE.PREDICT)
    predictor = DET02Predictor(model_info)
    predictor.load_model()
    
    # Predict
    results = []
    if len(images) == 1:
        result = predictor.predict(images[0], args.threshold)
        result.image_data = images[0]  # 使用image_data字段存储图片路径
        results.append(result)
    else:
        batches = [images[i:i+args.batch_size] for i in range(0, len(images), args.batch_size)]
        for batch in batches:
            batch_result = predictor.batch_predict(batch, args.threshold)
            for i, result in enumerate(batch_result.results):
                print(result)
                result.image_data = batch[i]  # 使用image_data字段存储图片路径
                results.append(result)
    
    # Determine output mode: file or directory
    output_path = Path(args.output)
    is_directory = output_path.is_dir() or (not output_path.suffix and not output_path.exists())
    
    if is_directory:
        # Output directory mode
        output_dir = output_path
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create preds directory for visualization images
        preds_dir = output_dir / "preds"
        preds_dir.mkdir(exist_ok=True)
        
        # Save COCO JSON
        coco_json_path = output_dir / "_annotations.coco.json"
        to_coco(results, str(coco_json_path))
        
        # Save visualization images
        logger.info(f"Saving visualization images to {preds_dir}")
        for result in results:
            img_path = getattr(result, 'image_data', '')
            if not img_path or not isinstance(img_path, str) or not os.path.isfile(img_path):
                continue
            
            # Read original image
            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"无法读取图片: {img_path}")
                continue
            
            # Draw detections
            vis_img = draw_detections(img, result, max_size=1536)
            
            # Save visualization image
            img_name = os.path.basename(img_path)
            output_img_path = preds_dir / img_name
            cv2.imwrite(str(output_img_path), vis_img)
            logger.debug(f"保存可视化图片: {output_img_path}")
        
        logger.info(f"输出完成: COCO JSON -> {coco_json_path}, 可视化图片 -> {preds_dir}")
    else:
        # Output file mode (original behavior)
        to_coco(results, args.output)
    
    predictor.release()


if __name__ == '__main__':
    main()
