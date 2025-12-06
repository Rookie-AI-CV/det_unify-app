#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HQ-Det models batch prediction script
Output: COCO format JSON file
"""

import argparse
import json
import os
import sys
import re
import warnings
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from loguru import logger
import torch

# 抑制 torch.load 的 FutureWarning
warnings.filterwarnings('ignore', category=FutureWarning, message='.*torch.load.*weights_only.*')

# 配置日志级别，减少 DEBUG 输出
logger.remove()  # 移除默认处理器
logger.add(sys.stderr, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")


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


def get_font(size):
    """Get Chinese font with fallback"""
    font_paths = [
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/usr/share/fonts/truetype/arphic/uming.ttc',
        '/opt/product/ml_backend/simsun.ttc',
        '/System/Library/Fonts/PingFang.ttc',
    ]
    
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except:
                continue
    
    return ImageFont.load_default()


def draw_text_chinese(img, text, position, font_size, color=(255, 255, 255)):
    """Draw text on image using PIL (supports Chinese)"""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    font = get_font(font_size)
    draw.text(position, text, fill=color, font=font)
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


def draw_detections(image, result, max_size=1536, class_names=None):
    """Draw bounding boxes with labels and confidence scores on image"""
    img = image.copy()
    orig_h, orig_w = img.shape[:2]
    
    scale = 1.0
    if max(orig_h, orig_w) > max_size:
        img = resize_image_max_size(img, max_size)
        new_h, new_w = img.shape[:2]
        scale = min(new_w / orig_w, new_h / orig_h)
    else:
        new_h, new_w = orig_h, orig_w
    
    font_scale = min(new_h, new_w) * 0.0008
    thickness = max(1, int(min(new_h, new_w) * 0.001))
    box_thickness = max(2, int(min(new_h, new_w) * 0.002))
    
    if not hasattr(result, 'bboxes') or result.bboxes is None or len(result.bboxes) == 0:
        return img
    
    bboxes = result.bboxes
    labels = getattr(result, 'cls', None)
    scores = getattr(result, 'scores', None)
    
    np.random.seed(42)
    colors = {}
    
    for i, bbox in enumerate(bboxes):
        if len(bbox) >= 4:
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            
            # 获取类别名称：如果有类别名称映射，使用映射；否则使用类别ID
            if labels is not None and i < len(labels) and labels[i] is not None:
                label_id = int(labels[i])
                if class_names is not None and label_id < len(class_names) and class_names[label_id]:
                    class_name = class_names[label_id]
                else:
                    class_name = str(label_id)
            else:
                class_name = "object"
            confidence = float(scores[i]) if scores is not None and i < len(scores) else 1.0
            
            if class_name not in colors:
                colors[class_name] = tuple(map(int, np.random.randint(0, 255, 3)))
            color = colors[class_name]
            
            x1 = int(x1 * scale)
            y1 = int(y1 * scale)
            x2 = int(x2 * scale)
            y2 = int(y2 * scale)
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, box_thickness)
            
            label = f"{class_name} {confidence:.2f}"
            has_chinese = contains_chinese(label)
            
            if has_chinese:
                font_size = max(12, min(int(min(new_h, new_w) * 0.02), 40))
                temp_img = Image.new('RGB', (100, 100))
                temp_draw = ImageDraw.Draw(temp_img)
                font = get_font(font_size)
                bbox_text = temp_draw.textbbox((0, 0), label, font=font)
                tw = bbox_text[2] - bbox_text[0]
                th = bbox_text[3] - bbox_text[1]
                baseline = int(th * 0.2)
                
                cv2.rectangle(img, (x1, y1 - th - baseline - 5), (x1 + tw + 5, y1), color, -1)
                img = draw_text_chinese(img, label, (x1, y1 - th - baseline - 3), font_size, (255, 255, 255))
            else:
                (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                cv2.rectangle(img, (x1, y1 - th - baseline - 5), (x1 + tw, y1), color, -1)
                cv2.putText(img, label, (x1, y1 - baseline - 3), cv2.FONT_HERSHEY_SIMPLEX,
                           font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    return img


def load_hq_model(checkpoint_path, model_type, device='cuda:0'):
    logger.info(f"加载 {model_type.upper()} 模型: {checkpoint_path}")
    
    model_map = {
        'dino': ('hq_det.models.dino.hq_dino', 'HQDINO'),
        'rtdetr': ('hq_det.models.rtdetr.hq_rtdetr', 'HQRTDETR'),
        'rtmdet': ('hq_det.models.rtmdet', 'HQRTMDET'),
        'yolo': ('hq_det.models.yolo', 'HQYOLO'),
        'lwdetr': ('hq_det.models.lwdetr.hq_lwdetr', 'HQLWDETR'),
        'rfdetr': ('hq_det.models.rfdetr.hq_rfdetr', 'HQRFDETR'),
        'codetr': ('hq_det.models.codetr.hq_codetr', 'HQCoDetr'),
    }
    
    if model_type not in model_map:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    module_path, class_name = model_map[model_type]
    module = __import__(module_path, fromlist=[class_name])
    model_class = getattr(module, class_name)
    
    model = model_class(model=checkpoint_path)
    model.eval()
    
    # 确保模型移动到指定设备
    device_str = str(device)
    if hasattr(model, 'to'):
        model.to(device)
        # 确保模型的 device 属性也被设置（某些模型需要，如 RTDETR）
        if hasattr(model, 'device'):
            if isinstance(device, str):
                model.device = torch.device(device)
            else:
                model.device = device
    
    # 验证模型是否在 GPU 上
    if device_str.startswith('cuda') and torch.cuda.is_available():
        # 检查模型参数是否在 GPU 上
        if hasattr(model, 'model'):
            first_param = next(model.model.parameters()) if hasattr(model.model, 'parameters') else None
        else:
            first_param = next(model.parameters()) if hasattr(model, 'parameters') else None
        
        if first_param is not None:
            actual_device = str(first_param.device)
            if actual_device.startswith('cuda'):
                logger.info(f"✓ 模型已成功加载到 GPU: {actual_device}")
            else:
                logger.warning(f"⚠ 警告: 模型在 {actual_device} 上，而不是 GPU")
    else:
        logger.info(f"使用设备: {device_str}")
    
    return model


def to_coco(results, output_file, class_names=None):
    """Convert results to COCO format JSON"""
    coco = {"images": [], "annotations": [], "categories": []}
    cat_map, cat_id, ann_id = {}, 1, 1
    
    for img_idx, result_data in enumerate(results):
        result, img_path = result_data
        if not img_path or not isinstance(img_path, str):
            logger.warning(f"跳过结果 {img_idx}: 缺少有效的图片路径")
            continue
        w, h = get_size(img_path)
        if w == 0 or h == 0:
            logger.warning(f"跳过图片 {img_path}: 无法读取尺寸")
            continue
        
        img_id = img_idx + 1
        coco["images"].append({"id": int(img_id), "file_name": os.path.basename(img_path), "width": int(w), "height": int(h)})
        
        if hasattr(result, 'bboxes') and result.bboxes is not None and len(result.bboxes) > 0:
            bboxes = result.bboxes
            labels = getattr(result, 'cls', None)
            scores = getattr(result, 'scores', None)
            
            for i, bbox in enumerate(bboxes):
                if len(bbox) < 4:
                    continue
                
                x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                x = float(min(x1, x2))
                y = float(min(y1, y2))
                w = float(abs(x2 - x1))
                h = float(abs(y2 - y1))
                
                # 获取类别名称：如果有类别名称映射，使用映射；否则使用类别ID
                if labels is not None and i < len(labels) and labels[i] is not None:
                    label_id = int(labels[i])
                    if class_names is not None and label_id < len(class_names) and class_names[label_id]:
                        class_name = class_names[label_id]
                    else:
                        class_name = str(label_id)
                else:
                    class_name = "object"
                score = float(scores[i]) if scores is not None and i < len(scores) else 1.0
                
                if class_name not in cat_map:
                    cat_map[class_name] = cat_id
                    coco["categories"].append({"id": int(cat_id), "name": class_name, "supercategory": "none"})
                    cat_id += 1
                
                coco["annotations"].append({
                    "id": int(ann_id),
                    "image_id": int(img_id),
                    "category_id": int(cat_map[class_name]),
                    "bbox": [x, y, w, h],
                    "area": float(w * h),
                    "iscrowd": 0,
                    "score": score
                })
                ann_id += 1
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved to {output_file}: {len(coco['images'])} images, {len(coco['annotations'])} annotations")
    
    if len(coco['images']) == 0:
        logger.warning("COCO JSON为空！可能的原因：")
        logger.warning(f"  1. 输入results数量: {len(results)}")
        for idx, result_data in enumerate(results):
            result, img_path = result_data
            has_bboxes = hasattr(result, 'bboxes') and result.bboxes is not None and len(result.bboxes) > 0
            logger.warning(f"  结果 {idx}: image_path={img_path}, bboxes={has_bboxes}")
            if has_bboxes:
                logger.warning(f"    检测框数量: {len(result.bboxes)}")


def main():
    parser = argparse.ArgumentParser(
        description='HQ-Det models batch prediction'
    )
    parser.add_argument('--checkpoint', required=True, help='Model checkpoint path')
    parser.add_argument('--image', default=None, help='Single image path')
    parser.add_argument('--image-dir', default=None, help='Image directory')
    parser.add_argument('--image-list', default=None, help='Image list file')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--output', required=True, help='Output COCO JSON file or directory')
    parser.add_argument('--max-size', type=int, default=1536, help='Max image size for prediction')
    parser.add_argument('--model-type', required=True,
                       choices=['dino', 'rtdetr', 'rtmdet', 'yolo', 'lwdetr', 'rfdetr', 'codetr'],
                       help='Model type (required)')
    parser.add_argument('--device', default='cuda:0', help='Device to use (default: cuda:0)')
    args = parser.parse_args()
    
    images = get_images(args.image, args.image_dir, args.image_list)
    if not images:
        logger.error("No images found")
        sys.exit(1)
    
    try:
        model = load_hq_model(args.checkpoint, args.model_type, args.device)
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        sys.exit(1)

    # 获取类别名称映射
    class_names = None
    if hasattr(model, 'get_class_names'):
        try:
            class_names = model.get_class_names()
            logger.info(f"加载类别名称: {len(class_names)} 个类别")
        except Exception as e:
            logger.warning(f"无法获取类别名称: {e}")
    elif hasattr(model, 'id2names'):
        try:
            id2names = model.id2names
            if isinstance(id2names, dict):
                max_id = max(id2names.keys()) if id2names else -1
                class_names = [''] * (max_id + 1)
                for k, v in id2names.items():
                    class_names[k] = v
                logger.info(f"从 id2names 加载类别名称: {len(class_names)} 个类别")
        except Exception as e:
            logger.warning(f"无法从 id2names 获取类别名称: {e}")

    results = []
    from tqdm import tqdm
    
    for img_idx, img_path in enumerate(tqdm(images, desc="Predicting")):
        img = cv2.imread(img_path)
        if img is None:
            logger.warning(f"无法读取图片: {img_path}")
            continue
        
        try:
            if hasattr(model, 'predict'):
                import inspect
                sig = inspect.signature(model.predict)
                params = list(sig.parameters.keys())
                
                predict_kwargs = {}
                if 'bgr' in params:
                    predict_kwargs['bgr'] = True
                if 'confidence' in params or 'conf' in params:
                    predict_kwargs['confidence' if 'confidence' in params else 'conf'] = args.threshold
                if 'max_size' in params or 'imgsz' in params:
                    predict_kwargs['max_size' if 'max_size' in params else 'imgsz'] = args.max_size
                
                predict_results = model.predict([img], **predict_kwargs)
                result = predict_results[0] if isinstance(predict_results, list) else predict_results
            else:
                logger.error("模型没有 predict 方法，无法进行预测")
                sys.exit(1)
        except Exception as e:
            logger.error(f"预测失败 {img_path}: {e}")
            continue
        
        results.append((result, img_path))
    
    output_path = Path(args.output)
    is_directory = output_path.is_dir() or (not output_path.suffix and not output_path.exists())
    
    if is_directory:
        output_dir = output_path
        output_dir.mkdir(parents=True, exist_ok=True)
        preds_dir = output_dir / "preds"
        preds_dir.mkdir(exist_ok=True)
        
        coco_json_path = output_dir / "_annotations.coco.json"
        to_coco(results, str(coco_json_path), class_names=class_names)
        
        logger.info(f"Saving visualization images to {preds_dir}")
        from tqdm import tqdm
        for result, img_path in tqdm(results, desc="Visualizing"):
            if not img_path or not isinstance(img_path, str) or not os.path.isfile(img_path):
                continue
            
            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"无法读取图片: {img_path}")
                continue
            
            vis_img = draw_detections(img, result, max_size=args.max_size, class_names=class_names)
            img_name = os.path.basename(img_path)
            output_img_path = preds_dir / img_name
            cv2.imwrite(str(output_img_path), vis_img)
            # logger.debug(f"保存可视化图片: {output_img_path}")  # 已禁用 DEBUG 输出
        
        logger.info(f"输出完成: COCO JSON -> {coco_json_path}, 可视化图片 -> {preds_dir}")
    else:
        to_coco(results, args.output, class_names=class_names)


if __name__ == '__main__':
    main()