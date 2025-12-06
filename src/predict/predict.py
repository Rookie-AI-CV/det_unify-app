#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一预测入口脚本
自动识别模型类型并调用相应的预测脚本
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
from loguru import logger


def try_load_dino(checkpoint_path):
    """尝试加载 DINO (ml_backend) 模型，返回 (成功标志, 错误信息)"""
    try:
        from ml_backend.model import ModelInfo, MODE
        from ml_backend.predict.algos.det02 import DET02Predictor
        model_info = ModelInfo(model_id="dino", model_type="dino", 
                              checkpoint_path=checkpoint_path, mode=MODE.PREDICT)
        # 真正尝试加载模型，验证模型文件格式
        predictor = DET02Predictor(model_info)
        predictor.load_model()
        del predictor
        return True, None
    except ImportError:
        return False, "ml_backend 模块不可用"
    except Exception as e:
        # 如果加载失败，说明不是 DINO 格式的模型
        return False, str(e)


def try_load_hq_model(checkpoint_path, model_type):
    """尝试加载 HQ-Det 模型，返回 (成功标志, 错误信息)"""
    try:
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
            return False, f"未知模型类型: {model_type}"
        
        module_path, class_name = model_map[model_type]
        module = __import__(module_path, fromlist=[class_name])
        model_class = getattr(module, class_name)
        model = model_class(model=checkpoint_path)
        del model
        return True, None
    except Exception as e:
        return False, str(e)


def detect_hq_model_type(checkpoint_path):
    """检测 HQ-Det 模型的子类型，返回子类型名称"""
    if not os.path.exists(checkpoint_path):
        logger.error(f"模型文件不存在: {checkpoint_path}")
        return None
    
    # 准备进度条
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False
    
    # HQ-Det 模型子类型列表
    hq_model_types = ['dino', 'rtdetr', 'rtmdet', 'yolo', 'lwdetr', 'rfdetr', 'codetr']
    
    iterator = tqdm(hq_model_types, desc="检测 HQ-Det 模型类型", leave=False) if use_tqdm else hq_model_types
    for sub_type in iterator:
        success, error = try_load_hq_model(checkpoint_path, sub_type)
        if success:
            if use_tqdm:
                iterator.set_description(f"检测到: {sub_type.upper()}")
            logger.info(f"检测到 HQ-Det 模型类型: {sub_type.upper()}")
            return sub_type
    
    logger.error("无法识别 HQ-Det 模型子类型，请使用 --hq-model-type 手动指定")
    return None


def detect_model_type(checkpoint_path):
    """检测模型类型，返回 ('dino'|'hq_det', hq_model_type)"""
    if not os.path.exists(checkpoint_path):
        logger.error(f"模型文件不存在: {checkpoint_path}")
        return None, None
    
    # 准备进度条
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False
    
    # 尝试列表：先尝试 DINO (ml_backend)，再尝试所有 HQ-Det 模型
    attempts = [
        ('dino', 'ml_backend'),
        ('hq_det', 'dino'),
        ('hq_det', 'rtdetr'),
        ('hq_det', 'rtmdet'),
        ('hq_det', 'yolo'),
        ('hq_det', 'lwdetr'),
        ('hq_det', 'rfdetr'),
        ('hq_det', 'codetr'),
    ]
    
    iterator = tqdm(attempts, desc="检测模型类型", leave=False) if use_tqdm else attempts
    for model_type, sub_type in iterator:
        if model_type == 'dino':
            success, error = try_load_dino(checkpoint_path)
            if success:
                if use_tqdm:
                    iterator.set_description(f"检测到: DINO (ml_backend)")
                logger.info("检测到模型类型: DINO (ml_backend)")
                return 'dino', None
        else:  # hq_det
            success, error = try_load_hq_model(checkpoint_path, sub_type)
            if success:
                if use_tqdm:
                    iterator.set_description(f"检测到: {sub_type.upper()}")
                logger.info(f"检测到模型类型: {sub_type.upper()} (HQ-Det)")
                return 'hq_det', sub_type
    
    logger.error("无法识别模型类型，请使用 --model-type 手动指定")
    return None, None


def main():
    parser = argparse.ArgumentParser(
        description='统一预测入口 - 自动识别模型类型并执行预测',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 自动检测模型类型并预测（推荐）
  python predict.py --checkpoint model.pth --image test.jpg --output results/
  python predict.py --checkpoint model.pth --image-dir images/ --output results/
  
  # 手动指定模型类型（可选）
  python predict.py --checkpoint model.pth --model-type hq_det --image-dir images/ --output results/
  python predict.py --checkpoint model.pth --model-type dino --image test.jpg --output results/
  
  # 指定阈值和最大尺寸
  python predict.py --checkpoint model.pth --image-dir images/ --threshold 0.3 --max-size 2048 --output results/
        """
    )
    
    parser.add_argument('--checkpoint', required=True, help='模型文件路径')
    parser.add_argument('--image', default=None, help='单张图片路径')
    parser.add_argument('--image-dir', default=None, help='图片目录路径')
    parser.add_argument('--image-list', default=None, help='图片列表文件路径')
    parser.add_argument('--threshold', type=float, default=0.5, help='置信度阈值 (默认: 0.5)')
    parser.add_argument('--batch-size', type=int, default=6, help='批次大小 (仅DINO模型，默认: 6)')
    parser.add_argument('--max-size', type=int, default=1536, help='最大图片尺寸 (仅HQ-Det模型，默认: 1536)')
    parser.add_argument('--device', default='cuda:0', help='设备 (默认: cuda:0，使用GPU)')
    parser.add_argument('--output', required=True, help='输出路径 (COCO JSON文件或目录)')
    parser.add_argument('--model-type', choices=['dino', 'hq_det'], default=None,
                       help='手动指定模型类型（dino 或 hq_det），未指定时自动检测所有类型')
    parser.add_argument('--hq-model-type', 
                       choices=['dino', 'rtdetr', 'rtmdet', 'yolo', 'lwdetr', 'rfdetr', 'codetr'],
                       default=None, help='手动指定 HQ-Det 模型子类型，未指定时自动检测')
    
    args = parser.parse_args()
    
    # 确定模型类型（默认自动检测）
    hq_model_type = None
    if args.model_type is None:
        # 自动检测所有模型类型
        model_type, hq_model_type = detect_model_type(args.checkpoint)
        if model_type is None:
            logger.error("无法识别模型类型，请使用 --model-type 手动指定")
            sys.exit(1)
        logger.info(f"自动检测到模型类型: {model_type}" + 
                   (f" ({hq_model_type})" if hq_model_type else ""))
    else:
        # 使用指定的模型类型
        model_type = args.model_type
        if model_type == 'hq_det':
            hq_model_type = args.hq_model_type
            # 如果未指定 hq_model_type，则自动检测
            if hq_model_type is None:
                hq_model_type = detect_hq_model_type(args.checkpoint)
                if hq_model_type is None:
                    logger.error("无法识别 HQ-Det 模型子类型，请使用 --hq-model-type 手动指定")
                    sys.exit(1)
        logger.info(f"使用指定的模型类型: {model_type}" + 
                   (f" ({hq_model_type})" if hq_model_type else ""))
    
    # 获取当前脚本目录
    script_dir = Path(__file__).parent.absolute()
    
    # 根据模型类型调用相应的脚本
    if model_type == 'dino':
        script_path = script_dir / "predict_dino.py"
        cmd = [
            sys.executable,
            str(script_path),
            "--checkpoint", args.checkpoint,
            "--threshold", str(args.threshold),
            "--batch-size", str(args.batch_size),
            "--output", args.output,
        ]
    elif model_type == 'hq_det':
        script_path = script_dir / "predict_hq_det.py"
        cmd = [
            sys.executable,
            str(script_path),
            "--checkpoint", args.checkpoint,
            "--model-type", hq_model_type,
            "--threshold", str(args.threshold),
            "--max-size", str(args.max_size),
            "--device", args.device,
            "--output", args.output,
        ]
    else:
        logger.error(f"未知的模型类型: {model_type}")
        sys.exit(1)
    
    # 添加图片输入参数（三选一）
    if args.image:
        cmd.extend(["--image", args.image])
    elif args.image_dir:
        cmd.extend(["--image-dir", args.image_dir])
    elif args.image_list:
        cmd.extend(["--image-list", args.image_list])
    else:
        logger.error("必须指定 --image, --image-dir 或 --image-list 之一")
        sys.exit(1)
    
    # 执行预测脚本
    logger.info(f"执行预测脚本: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True)
        logger.info("预测完成")
        sys.exit(result.returncode)
    except subprocess.CalledProcessError as e:
        logger.error(f"预测脚本执行失败: {e}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        logger.warning("用户中断预测")
        sys.exit(1)
    except Exception as e:
        logger.error(f"执行出错: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

