#!/usr/bin/env python3
"""DetUnify 预测 worker：由 DetForge Studio 以外部 Python 进程调用。

一次加载模型，批量预测多张图，结果以 JSON 输出到 stdout（便于与主程序环境隔离）。

用法:
  python predict_job_worker.py --payload-file job.json
  echo '{...}' | python predict_job_worker.py

payload 示例见 DetForge studio/forge/predict_runtime.py 文档字符串。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback


def _predict_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, '..', 'src', 'predict'))


def _load_payload(args):
    if args.payload_file:
        with open(args.payload_file, encoding='utf-8') as f:
            return json.load(f)
    if not sys.stdin.isatty():
        return json.load(sys.stdin)
    raise SystemExit('缺少 --payload-file 或 stdin JSON')


def main():
    parser = argparse.ArgumentParser(description='DetUnify batch predict worker (subprocess)')
    parser.add_argument('--payload-file', help='JSON 任务描述文件')
    args = parser.parse_args()

    payload = _load_payload(args)

    if payload.get('mode') == 'health_check':
        predict_dir = _predict_dir()
        if predict_dir not in sys.path:
            sys.path.insert(0, predict_dir)
        import model_api  # noqa: WPS433
        model_cfg = payload.get('model') or {}
        checkpoint = model_cfg.get('checkpoint_path')
        if not checkpoint:
            raise SystemExit('health_check 需要 model.checkpoint_path')
        device = payload.get('device') or 'cuda:0'
        threshold = float(payload.get('threshold', 0.5))
        max_size = int(payload.get('max_size', 1536))
        framework = model_cfg.get('framework')
        sub_type = model_cfg.get('sub_type')
        info = {'ok': False, 'framework': framework, 'device': device}
        try:
            loaded = model_api.load_model(
                checkpoint, framework=framework, sub_type=sub_type,
                device=device, max_size=max_size, threshold=threshold,
            )
            try:
                loaded.release()
            except Exception:
                pass
            info['ok'] = True
        except Exception as e:  # noqa: BLE001
            info['error'] = str(e)
        print(json.dumps({'success': True, **info}, ensure_ascii=False))
        return

    model_cfg = payload.get('model') or {}
    checkpoint = model_cfg.get('checkpoint_path') or payload.get('checkpoint_path')
    if not checkpoint:
        raise SystemExit('payload.model.checkpoint_path 必填')

    predict_dir = _predict_dir()
    if predict_dir not in sys.path:
        sys.path.insert(0, predict_dir)

    import model_api  # noqa: WPS433

    device = payload.get('device') or 'cuda:0'
    threshold = float(payload.get('threshold', 0.5))
    max_size = int(payload.get('max_size', 1536))
    framework = model_cfg.get('framework') or payload.get('framework')
    sub_type = model_cfg.get('sub_type') or payload.get('sub_type')

    images = payload.get('images') or []
    if not images:
        raise SystemExit('payload.images 不能为空')

    loaded = model_api.load_model(
        checkpoint,
        framework=framework,
        sub_type=sub_type,
        device=device,
        max_size=max_size,
        threshold=threshold,
    )

    results = []
    try:
        for item in images:
            img_path = item if isinstance(item, str) else item.get('path')
            thr = threshold if isinstance(item, str) else float(item.get('threshold', threshold))
            try:
                out = loaded.predict_one(img_path, threshold=thr)
                results.append({
                    'path': img_path,
                    'ok': True,
                    'width': out.get('width'),
                    'height': out.get('height'),
                    'predictions': out.get('predictions') or [],
                })
            except Exception as e:  # noqa: BLE001
                results.append({'path': img_path, 'ok': False, 'error': str(e)})
    finally:
        try:
            loaded.release()
        except Exception:
            pass

    print(json.dumps({'success': True, 'results': results}, ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        print(json.dumps({'success': False, 'error': str(e), 'trace': traceback.format_exc()}, ensure_ascii=False))
        sys.exit(1)
