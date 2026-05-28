"""不确定性预测脚本：每个检测框输出 top N 置信度分数。

用法: python run_predict_uncertainty.py <model_path> <input_path> <output_path> [--top_n 5] [--confidence 0.3] [--max_per_img 300]
      可选：--coco_annotations <path> 从 COCO 标注中取图列表；--max_samples N 只预测 N 张（采样）

关键修复：
- 每个框只输出一个框，附带 top_n 个类别及其置信度（非展开为多个框）
- 置信度过滤、NMS、max_per_img 限制框数量
- 保存 bbox 为 COCO 格式 [x, y, w, h]，JSON 输出名为 _annotations.coco.json
"""
import sys
import os
import argparse
import json
import random
from tqdm import tqdm
import cv2
import torch
import numpy as np
from typing import List
import torchvision.ops
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hq_det.common import PredictionResult
from hq_det.models.dino.hq_dino import HQDINO
from hq_det import torch_utils


class PredictionResultUncertainty(PredictionResult):
    """扩展 PredictionResult，每个框带有 top N 类别置信度。"""
    topn_probs: np.ndarray = None   # (N_boxes, top_n) 每个框的 top N 概率
    topn_classes: np.ndarray = None  # (N_boxes, top_n) 每个框的 top N 类别 ID


class HQDINOUncertainty(HQDINO):
    """HQDINO 不确定性扩展：每个检测框输出 top N 置信度分数，并严格控制框数量。"""

    def postprocess(self, forward_result, batch_data, confidence=0.0, top_n=5,
                    max_per_img=300, nms_threshold=0.7):
        """后处理：每个框输出 top N 置信度，限制框数量。

        关键修复：
        1. 置信度过滤：仅保留 max_score > confidence 的框
        2. NMS：去除重复框
        3. max_per_img：每张图最多 max_per_img 个框
        4. 每个框附带 top_n 个类别及其置信度（非展开为多个框）
        """
        head_inputs_dict = forward_result['head_inputs_dict']
        hidden_states = head_inputs_dict['hidden_states']
        references = head_inputs_dict['references']

        outs = self.model.bbox_head(hidden_states, references)
        all_cls_scores = outs[0]   # (num_layers, bs, num_queries, num_classes)
        all_bbox_preds = outs[1]   # (num_layers, bs, num_queries, 4)

        num_classes = all_cls_scores.shape[-1]
        top_n = min(top_n, num_classes)

        cls_scores = all_cls_scores[-1]   # (bs, num_queries, num_classes)
        bbox_preds = all_bbox_preds[-1]   # (bs, num_queries, 4)

        probs = cls_scores.sigmoid()  # (bs, num_queries, num_classes)
        max_scores, max_indices = probs.max(dim=-1)  # (bs, num_queries)

        results = []
        batch_img_metas = [ds.metainfo for ds in batch_data['data_samples']]

        for batch_idx in range(cls_scores.shape[0]):
            img_h, img_w = batch_img_metas[batch_idx]['img_shape'][:2]
            scale_fct = bbox_preds.new_tensor([img_w, img_h, img_w, img_h])

            valid_mask = max_scores[batch_idx] > confidence
            valid_indices = valid_mask.nonzero(as_tuple=True)[0]

            if len(valid_indices) == 0:
                record = PredictionResultUncertainty()
                record.bboxes = np.zeros((0, 4), dtype=np.float32)
                record.scores = np.zeros((0,), dtype=np.float32)
                record.cls = np.zeros((0,), dtype=np.int32)
                record.topn_probs = np.zeros((0, top_n), dtype=np.float32)
                record.topn_classes = np.zeros((0, top_n), dtype=np.int32)
                results.append(record)
                continue

            probs_i = probs[batch_idx][valid_indices]  # (N_valid, num_classes)
            bbox_i = bbox_preds[batch_idx][valid_indices]  # (N_valid, 4)

            topn_probs, topn_indices = torch.topk(probs_i, top_n, dim=-1)
            max_indices_i = max_indices[batch_idx][valid_indices]
            max_scores_i = max_scores[batch_idx][valid_indices]

            bbox_xyxy = bbox_cxcywh_to_xyxy(bbox_i)
            bbox_xyxy = bbox_xyxy * scale_fct.unsqueeze(0)

            keep = torchvision.ops.batched_nms(
                bbox_xyxy,
                max_scores_i,
                max_indices_i,
                nms_threshold
            ).cpu().numpy()

            keep = keep[:max_per_img]

            record = PredictionResultUncertainty()
            record.bboxes = bbox_xyxy[keep].cpu().numpy()
            record.scores = max_scores_i[keep].cpu().numpy()
            record.cls = max_indices_i[keep].cpu().numpy().astype(np.int32)
            record.topn_probs = topn_probs[keep].cpu().numpy()
            record.topn_classes = topn_indices[keep].cpu().numpy().astype(np.int32)
            results.append(record)

        return results

    def predict_with_uncertainty(self, imgs, bgr=False, confidence=0.0, max_size=-1,
                                 top_n=5, max_per_img=300, nms_threshold=0.7) -> List[PredictionResultUncertainty]:
        """预测并返回每个框的 top N 置信度。"""
        if not bgr:
            imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in imgs]

        img_scales = np.ones((len(imgs),))
        if max_size > 0:
            for i in range(len(imgs)):
                max_hw = max(imgs[i].shape[0], imgs[i].shape[1])
                if max_hw > max_size:
                    rate = max_size / max_hw
                    imgs[i] = cv2.resize(
                        imgs[i],
                        (int(imgs[i].shape[1] * rate), int(imgs[i].shape[0] * rate))
                    )
                    img_scales[i] = rate

        device = self.device
        with torch.no_grad():
            batch_data = self.imgs_to_batch(imgs)
            batch_data = torch_utils.batch_to_device(batch_data, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=False):
                forward_result = self.forward(batch_data)
                preds = self.postprocess(
                    forward_result, batch_data,
                    confidence=confidence,
                    top_n=top_n,
                    max_per_img=max_per_img,
                    nms_threshold=nms_threshold
                )

        for i in range(len(preds)):
            preds[i].bboxes = preds[i].bboxes / img_scales[i]

        return preds


def _xyxy_to_xywh(bbox_xyxy):
    """[x1,y1,x2,y2] -> [x,y,w,h] COCO 格式。"""
    x1, y1, x2, y2 = bbox_xyxy
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def _load_image_list_from_input_dir(input_dir):
    """从目录列出所有图片路径。"""
    return [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(('.jpg', '.png', '.jpeg'))
    ]


def _load_image_list_from_coco(coco_path, image_dir):
    """从 COCO 标注文件读取图片列表，路径为 image_dir + file_name。"""
    with open(coco_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)
    images = coco.get('images', [])
    out = []
    for img in images:
        fn = img.get('file_name') or img.get('filename')
        if fn:
            out.append(os.path.join(image_dir, fn))
    return out


def main():
    parser = argparse.ArgumentParser(description='DINO 不确定性预测：每个框输出 top N 置信度')
    parser.add_argument('model', help='模型权重路径')
    parser.add_argument('input', help='输入图片目录')
    parser.add_argument('output', help='输出目录')
    parser.add_argument('--top_n', type=int, default=5, help='每个框输出的 top N 置信度数量')
    parser.add_argument('--confidence', type=float, default=0.3, help='置信度阈值')
    parser.add_argument('--max_per_img', type=int, default=300, help='每张图最大框数量')
    parser.add_argument('--nms_threshold', type=float, default=0.7, help='NMS IoU 阈值')
    parser.add_argument('--max_size', type=int, default=1536, help='图片最大边长')
    parser.add_argument('--save_json', action='store_true', help='是否保存 JSON 结果')
    parser.add_argument('--coco_annotations', type=str, default=None,
                        help='COCO 标注 JSON 路径；指定后从该文件中取图片列表，图片路径为 input + file_name')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='仅对指定张数进行预测（从图片列表中随机采样）；不指定则处理全部')
    parser.add_argument('--seed', type=int, default=42, help='--max_samples 采样时的随机种子')
    parser.add_argument('--vis', '--draw_boxes', dest='vis', action='store_true',
                        help='是否输出带检测框的可视化图片到 output 目录')
    args = parser.parse_args()

    model = HQDINOUncertainty(model=args.model)
    model.eval()
    model.to("cuda:0")

    if args.coco_annotations and os.path.isfile(args.coco_annotations):
        filenames = _load_image_list_from_coco(args.coco_annotations, args.input)
    else:
        filenames = _load_image_list_from_input_dir(args.input)

    if args.max_samples is not None and args.max_samples > 0 and len(filenames) > args.max_samples:
        random.seed(args.seed)
        filenames = random.sample(filenames, args.max_samples)
        print(f'已从列表中采样 {args.max_samples} 张进行预测')

    os.makedirs(args.output, exist_ok=True)
    # COCO 格式：images, annotations, categories；每条 annotation 上新增 topn_* 等字段
    coco_images = []
    coco_annotations = []
    next_img_id = 1
    next_ann_id = 1

    for filename in tqdm(filenames, desc='Predicting'):
        img = cv2.imread(filename)
        if img is None:
            print(f'跳过无法读取的图片: {filename}')
            continue

        results = model.predict_with_uncertainty(
            [img],
            bgr=True,
            confidence=args.confidence,
            max_size=args.max_size,
            top_n=args.top_n,
            max_per_img=args.max_per_img,
            nms_threshold=args.nms_threshold,
        )

        result = results[0]
        class_names = model.get_class_names()

        # 可选：输出带框图的可视化
        if args.vis:
            vis_img = img.copy()
            for i, bbox in enumerate(result.bboxes):
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cls_id = int(result.cls[i])
                score = float(result.scores[i])
                top_cls = result.topn_classes[i]
                top_prob = result.topn_probs[i]
                top_str = f'{class_names[cls_id]}:{score:.2f}'
                if args.top_n >= 2:
                    top_str += '\n' + '\n'.join(
                        f'{class_names[c]}:{p:.2f}' for c, p in zip(top_cls[1:], top_prob[1:])
                    )
                y_pos = y1 - 5
                for line in top_str.split('\n')[:min(args.top_n, 4)]:
                    cv2.putText(vis_img, line, (x1, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    y_pos += 15

            out_path = os.path.join(args.output, os.path.basename(filename))
            cv2.imwrite(out_path, vis_img)

        # 保存为 COCO 格式 JSON，每条 annotation 上新增 topn_* 等字段
        if args.save_json:
            h, w = img.shape[:2]
            img_id = next_img_id
            next_img_id += 1
            coco_images.append({
                'id': img_id,
                'file_name': os.path.basename(filename),
                'width': int(w),
                'height': int(h),
            })
            for i in range(len(result.bboxes)):
                bbox_xywh = _xyxy_to_xywh(result.bboxes[i])
                x, y, bw, bh = bbox_xywh
                area = float(bw * bh)
                ann = {
                    'id': next_ann_id,
                    'image_id': img_id,
                    'category_id': int(result.cls[i]),
                    'bbox': bbox_xywh,
                    'area': area,
                    'iscrowd': 0,
                    'score': float(result.scores[i]),
                    'topn_classes': result.topn_classes[i].tolist(),
                    'topn_probs': result.topn_probs[i].tolist(),
                    'topn_names': [class_names[c] for c in result.topn_classes[i]],
                }
                coco_annotations.append(ann)
                next_ann_id += 1

    if args.save_json:
        class_names = model.get_class_names()
        coco_categories = [{'id': i, 'name': class_names[i]} for i in range(len(class_names))]
        coco_output = {
            'images': coco_images,
            'annotations': coco_annotations,
            'categories': coco_categories,
        }
        json_path = os.path.join(args.output, '_annotations.coco.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(coco_output, f, ensure_ascii=False, indent=2)
        print(f'JSON 已保存至: {json_path}')

    print(f'完成，共处理 {len(filenames)} 张图片，输出至 {args.output}')


if __name__ == '__main__':
    main()
