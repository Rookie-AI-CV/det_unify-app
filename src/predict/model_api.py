#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""可复用的「常驻模型」预测 API。

与原 CLI（predict.py 经 subprocess 一次性写 COCO）互补：本模块只 load_model 一次，
之后逐图 predict_one，便于 DetForge-Studio 增量写库与断点续跑。

输出统一为平台 ext 同形：
    {"width": W, "height": H, "predictions": [
        {"name": str, "confidence": float, "type": "rect",
         "points": [{"x": x, "y": y, "w": w, "h": h}]}
    ]}

重型依赖（torch / cv2 / ml_backend / hq_det）全部在 load_model / predict_one 内惰性导入，
因此仅 import 本模块不会触发这些依赖，单测与 Flask 启动不受影响。
"""
from __future__ import annotations

import os

try:
    from .predict import detect_model_type, detect_hq_model_type  # noqa: F401
    from .hq_model_registry import HQ_MODEL_KEYS
except ImportError:  # 作为顶层脚本导入时
    from predict import detect_model_type, detect_hq_model_type  # noqa: F401
    from hq_model_registry import HQ_MODEL_KEYS


def _image_size(img_path):
    import cv2
    img = cv2.imread(img_path)
    if img is None:
        return 0, 0
    return int(img.shape[1]), int(img.shape[0])


class LoadedModel:
    """已加载到设备的模型句柄；线程内复用，按图调用 predict_one。"""

    def __init__(self, framework, sub_type, model, class_names=None,
                 threshold=0.5, max_size=1536, device='cuda:0', checkpoint_path=''):
        self.framework = framework
        self.sub_type = sub_type
        self.model = model
        self.class_names = class_names
        self.threshold = float(threshold)
        self.max_size = int(max_size)
        self.device = device
        self.checkpoint_path = checkpoint_path

    # ── 预测 ────────────────────────────────────────────────────
    def predict_one(self, img_path, threshold=None):
        thr = self.threshold if threshold is None else float(threshold)
        if not img_path or not os.path.isfile(img_path):
            raise FileNotFoundError(f'图片不存在: {img_path}')
        if self.framework == 'hq_det':
            return self._predict_hq(img_path, thr)
        return self._predict_dino(img_path, thr)

    def _predict_hq(self, img_path, threshold):
        import inspect
        import cv2
        import numpy as np

        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f'无法读取图片: {img_path}')
        h, w = img.shape[:2]
        img = np.ascontiguousarray(img, dtype=np.uint8)

        predict_kwargs = {}
        if hasattr(self.model, 'predict'):
            params = list(inspect.signature(self.model.predict).parameters.keys())
            if 'bgr' in params:
                predict_kwargs['bgr'] = True
            if 'confidence' in params:
                predict_kwargs['confidence'] = threshold
            elif 'conf' in params:
                predict_kwargs['conf'] = threshold
            if 'max_size' in params:
                predict_kwargs['max_size'] = self.max_size
            elif 'imgsz' in params:
                predict_kwargs['imgsz'] = self.max_size
        out = self.model.predict([img], **predict_kwargs)
        result = out[0] if isinstance(out, (list, tuple)) else out

        predictions = []
        bboxes = getattr(result, 'bboxes', None)
        labels = getattr(result, 'cls', None)
        scores = getattr(result, 'scores', None)
        if bboxes is not None:
            for i, bbox in enumerate(bboxes):
                if bbox is None or len(bbox) < 4:
                    continue
                x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                conf = float(scores[i]) if scores is not None and i < len(scores) else 1.0
                if conf < threshold:
                    continue
                name = self._class_name(labels, i)
                predictions.append(self._make_pred(name, conf, x1, y1, x2 - x1, y2 - y1))
        return {'width': w, 'height': h, 'predictions': predictions}

    def _predict_dino(self, img_path, threshold):
        result = self.model.predict(img_path, threshold)
        w, h = _image_size(img_path)
        predictions = []
        for pred in (getattr(result, 'predictions', None) or []):
            name = getattr(pred, 'name', '') or 'object'
            conf = float(getattr(pred, 'confidence', 0) or 0)
            for point in (getattr(pred, 'points', None) or []):
                predictions.append(self._make_pred(
                    name, conf,
                    float(getattr(point, 'x', 0)), float(getattr(point, 'y', 0)),
                    float(getattr(point, 'w', 0)), float(getattr(point, 'h', 0)),
                ))
        return {'width': w, 'height': h, 'predictions': predictions}

    def _class_name(self, labels, i):
        if labels is not None and i < len(labels) and labels[i] is not None:
            label_id = int(labels[i])
            if self.class_names and label_id < len(self.class_names) and self.class_names[label_id]:
                return self.class_names[label_id]
            return str(label_id)
        return 'object'

    @staticmethod
    def _make_pred(name, conf, x, y, w, h):
        return {
            'name': name,
            'confidence': round(float(conf), 6),
            'type': 'rect',
            'points': [{
                'x': round(float(x), 2), 'y': round(float(y), 2),
                'w': round(float(w), 2), 'h': round(float(h), 2),
            }],
        }

    def release(self):
        rel = getattr(self.model, 'release', None)
        if callable(rel):
            try:
                rel()
            except Exception:
                pass


def load_model(checkpoint_path, framework=None, sub_type=None,
               device='cuda:0', max_size=1536, threshold=0.5):
    """加载一次模型，返回 LoadedModel。framework 未指定时自动检测。"""
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'模型文件不存在: {checkpoint_path}')

    if not framework:
        framework, detected_sub = detect_model_type(checkpoint_path)
        if framework is None:
            raise ValueError('无法识别模型类型，请显式指定 framework=dino|hq_det')
        sub_type = sub_type or detected_sub

    if framework == 'hq_det':
        if not sub_type:
            sub_type = detect_hq_model_type(checkpoint_path)
            if not sub_type:
                raise ValueError(f'无法识别 HQ-Det 子类型，可选: {HQ_MODEL_KEYS}')
        model, class_names = _load_hq(checkpoint_path, sub_type, device)
        return LoadedModel('hq_det', sub_type, model, class_names,
                           threshold, max_size, device, checkpoint_path)

    if framework == 'dino':
        model, class_names = _load_dino(checkpoint_path)
        return LoadedModel('dino', None, model, class_names,
                           threshold, max_size, device, checkpoint_path)

    raise ValueError(f'不支持的 framework: {framework}')


def _load_hq(checkpoint_path, sub_type, device):
    try:
        from .predict_hq_det import load_hq_model
    except ImportError:
        from predict_hq_det import load_hq_model
    model = load_hq_model(checkpoint_path, sub_type, device)
    class_names = None
    if hasattr(model, 'get_class_names'):
        try:
            class_names = model.get_class_names()
        except Exception:
            class_names = None
    if class_names is None and hasattr(model, 'id2names'):
        try:
            id2names = model.id2names
            if isinstance(id2names, dict) and id2names:
                max_id = max(id2names.keys())
                class_names = [''] * (max_id + 1)
                for k, v in id2names.items():
                    class_names[k] = v
        except Exception:
            class_names = None
    return model, class_names


def _load_dino(checkpoint_path):
    import contextlib
    import io
    from ml_backend.predict.algos.det02 import DET02Predictor
    from ml_backend.model import ModelInfo, MODE
    with contextlib.redirect_stdout(io.StringIO()):
        model_info = ModelInfo(model_id='dino', model_type='dino',
                               checkpoint_path=checkpoint_path, mode=MODE.PREDICT)
        predictor = DET02Predictor(model_info)
        predictor.load_model()
    class_names = None
    try:
        if hasattr(predictor, 'infer') and hasattr(predictor.infer, 'id2name'):
            id2name = predictor.infer.id2name
            if isinstance(id2name, dict) and id2name:
                max_id = max(id2name.keys())
                class_names = [''] * (max_id + 1)
                for k, v in id2name.items():
                    class_names[k] = v
    except Exception:
        class_names = None
    return predictor, class_names
