#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
README 生成器
生成预测结果的 HTML 报告
"""

import html
import urllib.parse
from datetime import datetime
from pathlib import Path
from werkzeug.utils import secure_filename


def generate_readme(results_data, model_dirs=None, model_names_map=None, description='', result_id=None):
    """生成精美的 HTML README 报告
    
    Args:
        results_data: 预测结果数据
        model_dirs: 模型目录列表
        model_names_map: 模型名称映射
        description: 说明信息（可选）
        result_id: 结果ID（用于访问图片）
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        if results_data is None:
            return '<html><body><h1>错误</h1><p>结果数据为空</p></body></html>'
        
        model_infos = results_data.get('model_infos', []) or []
        label_colors = results_data.get('label_colors', {}) or {}
        predictions = results_data.get('predictions', {}) or {}
    except (AttributeError, TypeError) as e:
        # 如果 results_data 本身有问题，返回错误信息
        error_msg = f'无法读取结果数据: {str(e)}'
        logger.error(error_msg)
        return f'<html><body><h1>错误</h1><p>{error_msg}</p></body></html>'
    
    # 统计数据
    stats = {}
    total_images = len(predictions)
    total_predictions = 0
    
    # 初始化统计
    for model_info in model_infos:
        model_name = model_info.get('name', 'Unknown')
        # 确保 model_name 不是 None
        if model_name is None:
            model_name = 'Unknown'
        else:
            model_name = str(model_name)
        stats[model_name] = {}
    
    # 统计检测结果
    for img_name, img_data in predictions.items():
        for pred_group in img_data.get('predictions', []):
            model_name = pred_group.get('full_name', pred_group.get('model_name', 'Unknown'))
            # 确保 model_name 不是 None
            if model_name is None:
                model_name = 'Unknown'
            else:
                model_name = str(model_name)
            if model_name not in stats:
                stats[model_name] = {}
            
            for bbox in pred_group.get('bboxes', []):
                label = bbox.get('label', 'unknown')
                stats[model_name][label] = stats[model_name].get(label, 0) + 1
                total_predictions += 1
    
    # 收集所有标签
    all_labels = set()
    for model_stats in stats.values():
        all_labels.update(model_stats.keys())
    all_labels = sorted(all_labels)
    
    # 线条样式映射
    style_map = {
        'solid': '实线',
        'dashed': '虚线',
        'dotted': '点线',
        'dashdot': '点划线'
    }
    
    # 收集所有图片信息（支持多状态），包括没有备注或状态的图片，以便筛选
    images_with_info = []
    false_positive_count = 0
    missed_count = 0
    low_confidence_count = 0
    
    for img_name, img_data in predictions.items():
        notes = img_data.get('notes', '')
        if notes is None:
            notes = ''
        else:
            notes = str(notes).strip()
        
        # 支持新的多状态格式（statuses数组）或旧的单一状态格式（向后兼容）
        statuses = img_data.get('statuses', [])
        if not statuses:
            # 向后兼容：如果有旧的status字段，转换为statuses数组
            old_status = img_data.get('status')
            if old_status:
                statuses = [old_status]
        
        # 检查是否有低置信度（最大置信度 < 0.5）
        max_score = 0.0
        for pred_group in img_data.get('predictions', []):
            for bbox in pred_group.get('bboxes', []):
                score = bbox.get('score', 0.0)
                if score > max_score:
                    max_score = score
        
        # 如果最大置信度 < 0.5，自动添加 low_confidence 状态
        if max_score < 0.5 and max_score > 0:
            if 'low_confidence' not in statuses:
                statuses.append('low_confidence')
        
        # 收集所有图片（包括没有备注或状态的），以便筛选
        images_with_info.append({
            'name': str(img_name),
            'notes': notes,
            'statuses': statuses,
            'max_score': max_score
        })
        
        # 统计各种状态的图片数量
        if 'false_positive' in statuses:
            false_positive_count += 1
        if 'missed' in statuses:
            missed_count += 1
        if 'low_confidence' in statuses:
            low_confidence_count += 1
    
    # 生成 HTML
    try:
        html_content = _generate_html(
            model_infos=model_infos,
            label_colors=label_colors,
            predictions=predictions,
            stats=stats,
            total_images=total_images,
            total_predictions=total_predictions,
            all_labels=all_labels,
            style_map=style_map,
            images_with_info=images_with_info,
            false_positive_count=false_positive_count,
            missed_count=missed_count,
            low_confidence_count=low_confidence_count,
            description=description
        )
        return html_content
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        logger.error(f"生成 README 时出错: {e}\n{error_detail}")
        return f'<html><body><h1>生成报告时出错</h1><p>错误: {html.escape(str(e))}</p></body></html>'


def _generate_html(model_infos, label_colors, predictions, stats, total_images,
                   total_predictions, all_labels, style_map, images_with_info,
                   false_positive_count, missed_count, low_confidence_count, description='', result_id=None):
    """生成 HTML 内容"""
    
    # HTML 头部和样式
    result_id_meta = f'    <meta name="result-id" content="{html.escape(str(result_id)) if result_id else ""}">' if result_id else ''
    html_parts = [
        '<!DOCTYPE html>',
        '<html lang="zh-CN">',
        '<head>',
        '    <meta charset="UTF-8">',
        '    <meta name="viewport" content="width=device-width, initial-scale=1.0">',
        result_id_meta,
        '    <title>DetUnify Studio - 预测结果报告</title>',
        '',
        _get_styles(),
        '</head>',
        '<body>',
        '    <div class="page-wrapper">',
        '        <div class="document">',
        '            <header class="document-header">',
        '                <h1 class="document-title">DetUnify Studio - 预测结果报告</h1>',
        f'                <p class="document-meta">生成时间: {datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")}</p>',
    ]
    
    # 如果有说明信息，添加到头部
    if description:
        description_escaped = html.escape(description)
        html_parts.append('                <div class="document-description">')
        html_parts.append(f'                    <p>{description_escaped.replace(chr(10), "<br>")}</p>')
        html_parts.append('                </div>')
    
    html_parts.extend([
        '            </header>',
        '            <main class="document-content">',
    ])
    
    # 总体统计
    html_parts.append(_generate_summary_section(total_images, total_predictions, len(model_infos), len(all_labels)))
    
    # 模型信息
    html_parts.append(_generate_models_section(model_infos, stats, style_map))
    
    # 标签颜色
    html_parts.append(_generate_labels_section(all_labels, label_colors))
    
    # 详细统计
    html_parts.append(_generate_statistics_section(model_infos, all_labels, stats))
    
    # 图片备注与状态 - 全屏图片浏览器
    html_parts.append(_generate_images_section(images_with_info, false_positive_count, missed_count, low_confidence_count, predictions, all_labels))
    
    # 目录结构
    html_parts.append(_generate_directory_section(model_infos, false_positive_count, missed_count))
    
    # 结尾
    html_parts.extend([
        '            </main>',
        '            <footer class="document-footer">',
        '                <p>Detection Prediction System</p>',
        '            </footer>',
        '        </div>',
        '    </div>',
        '</body>',
        '</html>'
    ])
    
    return '\n'.join(html_parts)


def _get_styles():
    """返回 CSS 样式"""
    return '''    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "微软雅黑", Arial, sans-serif;
            line-height: 1.7;
            color: #2c3e50;
            background: #f5f7fa;
            padding: 24px 16px;
            -webkit-font-smoothing: antialiased;
        }
        
        .page-wrapper {
            max-width: 1080px;
            margin: 0 auto;
        }
        
        .document {
            background: #ffffff;
            border-radius: 8px;
            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
            overflow: hidden;
        }
        
        .document-header {
            background: linear-gradient(180deg, #34495e 0%, #2c3e50 100%);
            color: #ffffff;
            padding: 48px 40px;
            text-align: center;
            position: relative;
        }
        
        .document-header::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        }
        
        .document-title {
            font-size: 32px;
            font-weight: 600;
            margin-bottom: 8px;
            letter-spacing: -0.5px;
        }
        
        .document-meta {
            font-size: 14px;
            opacity: 0.85;
            font-weight: 300;
        }
        
        .document-description {
            margin-top: 20px;
            padding: 16px 20px;
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            border-radius: 4px;
            font-size: 14px;
            line-height: 1.6;
            color: #495057;
        }
        
        .document-description p {
            margin: 0;
        }
        
        .document-content {
            padding: 48px 40px;
        }
        
        .document-footer {
            background: #f8f9fa;
            padding: 24px 40px;
            text-align: center;
            color: #6c757d;
            font-size: 13px;
            border-top: 1px solid #e9ecef;
        }
        
        .section {
            margin-bottom: 48px;
        }
        
        .section:last-child {
            margin-bottom: 0;
        }
        
        .section-title {
            font-size: 24px;
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 24px;
            padding-bottom: 12px;
            border-bottom: 2px solid #e9ecef;
        }
        
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 32px;
        }
        
        .summary-card {
            background: #2c3e50;
            color: white;
            padding: 28px 24px;
            border-radius: 6px;
            text-align: center;
            position: relative;
            overflow: hidden;
        }
        
        .summary-card::before {
            content: '';
            position: absolute;
            top: -50%;
            right: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
            pointer-events: none;
        }
        
        .summary-card h3 {
            font-size: 13px;
            font-weight: 500;
            opacity: 0.9;
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .summary-card .value {
            font-size: 36px;
            font-weight: 700;
            line-height: 1;
        }
        
        .model-list {
            display: grid;
            gap: 16px;
        }
        
        .model-item {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 6px;
            padding: 20px;
            transition: all 0.2s ease;
        }
        
        .model-item:hover {
            border-color: #667eea;
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.15);
        }
        
        .model-name {
            font-size: 18px;
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 16px;
        }
        
        .model-details {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            font-size: 14px;
        }
        
        .model-detail-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .model-detail-label {
            color: #6c757d;
            font-weight: 500;
            min-width: 70px;
        }
        
        .model-detail-value {
            color: #2c3e50;
        }
        
        .line-demo {
            display: inline-block;
            width: 50px;
            height: 2px;
            margin: 0 6px;
            vertical-align: middle;
        }
        
        .line-demo.solid {
            border-bottom: 2px solid #2c3e50;
        }
        
        .line-demo.dashed {
            border-bottom: 2px dashed #2c3e50;
        }
        
        .line-demo.dotted {
            border-bottom: 2px dotted #2c3e50;
        }
        
        .line-demo.dashdot {
            border-bottom: 2px dashed #2c3e50;
            border-top: 1px solid #2c3e50;
        }
        
        .badge {
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
        }
        
        .labels-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
            gap: 12px;
            margin-top: 20px;
        }
        
        .label-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px 14px;
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 4px;
        }
        
        .label-color {
            width: 20px;
            height: 20px;
            border-radius: 3px;
            border: 1px solid #ddd;
            flex-shrink: 0;
        }
        
        .label-name {
            font-size: 14px;
            color: #2c3e50;
            font-weight: 500;
        }
        
        .stats-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            background: white;
            border: 1px solid #e9ecef;
            border-radius: 6px;
            overflow: hidden;
        }
        
        .stats-table thead {
            background: #2c3e50;
            color: white;
        }
        
        .stats-table th {
            padding: 14px 16px;
            text-align: left;
            font-weight: 600;
            font-size: 14px;
        }
        
        .stats-table td {
            padding: 12px 16px;
            border-bottom: 1px solid #f1f3f5;
            font-size: 14px;
        }
        
        .stats-table tbody tr:hover {
            background: #f8f9fa;
        }
        
        .stats-table tbody tr:last-child td {
            border-bottom: none;
        }
        
        .stats-table td:not(:first-child) {
            text-align: center;
        }
        
        .stats-table td.image-cell {
            text-align: left;
        }
        
        .images-section {
            background: #f8f9fa;
            padding: 24px;
            border-radius: 6px;
            border: 1px solid #e9ecef;
        }
        
        .status-badges {
            display: flex;
            gap: 16px;
            margin-bottom: 24px;
            flex-wrap: wrap;
        }
        
        .status-badge {
            padding: 10px 18px;
            border-radius: 4px;
            font-size: 14px;
            font-weight: 500;
        }
        
        .status-badge.error {
            background: #fee;
            color: #c33;
            border: 1px solid #fcc;
        }
        
        .status-badge.warning {
            background: #fff4e6;
            color: #d97706;
            border: 1px solid #ffd89b;
        }
        
        .image-link {
            color: #667eea;
            text-decoration: none;
            font-weight: 500;
            transition: color 0.2s;
        }
        
        .image-link:hover {
            color: #764ba2;
            text-decoration: underline;
        }
        
        .image-preview {
            max-width: 300px;
            max-height: 200px;
            border-radius: 4px;
            border: 1px solid #e9ecef;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            display: block;
            margin: 4px 0;
        }
        
        .image-preview:hover {
            transform: scale(1.02);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }
        
        .image-cell {
            vertical-align: top;
            padding: 16px !important;
        }
        
        .image-name {
            display: block;
            margin-top: 8px;
            font-size: 12px;
            color: #667eea;
            text-decoration: none;
        }
        
        .image-name:hover {
            text-decoration: underline;
        }
        
        .image-browser-container {
            background: #f8f9fa;
            padding: 24px;
            border-radius: 6px;
            border: 1px solid #e9ecef;
        }
        
        .browser-filters {
            display: flex;
            gap: 24px;
            margin-bottom: 24px;
            padding: 16px;
            background: white;
            border-radius: 4px;
            flex-wrap: wrap;
            align-items: flex-start;
        }
        
        .filter-group {
            display: flex;
            flex-direction: column;
            gap: 8px;
            min-width: 150px;
        }
        
        .filter-group label {
            font-size: 13px;
            color: #495057;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        
        .filter-group > label:first-child {
            font-weight: 600;
            margin-bottom: 4px;
        }
        
        .filter-group input[type="checkbox"] {
            margin: 0;
        }
        
        .filter-group select {
            padding: 6px;
            border: 1px solid #e9ecef;
            border-radius: 4px;
            font-size: 13px;
            min-width: 150px;
        }
        
        .filter-actions {
            display: flex;
            align-items: flex-end;
        }
        
        .filter-actions button {
            padding: 8px 16px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
        }
        
        .filter-actions button:hover {
            background: #764ba2;
        }
        
        
        .image-browser {
            background: white;
            border-radius: 6px;
            overflow: hidden;
            min-height: 600px;
        }
        
        .browser-header {
            padding: 16px 24px;
            background: #2c3e50;
            color: white;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .image-info {
            font-size: 14px;
            font-weight: 500;
        }
        
        .image-name-header {
            font-size: 14px;
            opacity: 0.9;
        }
        
        .toggle-image-btn {
            padding: 6px 12px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 500;
            transition: background 0.2s;
        }
        
        .toggle-image-btn:hover {
            background: #764ba2;
        }
        
        .browser-viewport {
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 500px;
            background: #000;
            padding: 40px 80px;
        }
        
        .browser-content-wrapper {
            display: flex;
            flex: 1;
            align-items: stretch;
            gap: 20px;
            max-width: 100%;
            min-height: 500px;
        }
        
        .image-wrapper {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            min-width: 0;
            position: relative;
            overflow: hidden;
        }
        
        .image-wrapper img {
            max-width: 100%;
            max-height: 70vh;
            object-fit: contain;
            border-radius: 4px;
            transition: transform 0.1s;
            transform-origin: center center;
            cursor: grab;
            user-select: none;
            -webkit-user-select: none;
            -moz-user-select: none;
            -ms-user-select: none;
        }
        
        .image-wrapper img:active {
            cursor: grabbing;
        }
        
        .nav-btn {
            position: absolute;
            top: 50%;
            transform: translateY(-50%);
            width: 50px;
            height: 50px;
            background: rgba(255, 255, 255, 0.9);
            border: none;
            border-radius: 50%;
            font-size: 32px;
            color: #2c3e50;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s;
            z-index: 10;
        }
        
        .nav-btn:hover {
            background: white;
            transform: translateY(-50%) scale(1.1);
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        
        .prev-btn {
            left: 20px;
        }
        
        .next-btn {
            right: 180px;  /* 为右侧检测框信息栏留出空间（160px宽度+20px边距） */
        }
        
        @media (max-width: 1200px) {
            .next-btn {
                right: 20px;  /* 小屏幕上恢复原位置 */
            }
            .browser-content-wrapper {
                flex-direction: column;
            }
            .image-info-sidebar {
                width: 100%;
                min-width: unset;
                border-right: none;
                border-bottom: 1px solid #e9ecef;
                max-height: 200px;
            }
            .bbox-list-sidebar {
                width: 100%;
                min-width: unset;
                border-left: none;
                border-top: 1px solid #e9ecef;
                max-height: 300px;
            }
        }
        
        
        .header-buttons {
            display: flex;
            gap: 8px;
            align-items: center;
        }
        
        .toggle-image-btn,
        .fullscreen-btn {
            padding: 6px 12px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 500;
            transition: background 0.2s;
        }
        
        .toggle-image-btn:hover,
        .fullscreen-btn:hover {
            background: #764ba2;
        }
        
        .fullscreen-btn {
            background: #28a745;
        }
        
        .fullscreen-btn:hover {
            background: #218838;
        }
        
        .image-browser.fullscreen-mode {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            z-index: 9999;
            border-radius: 0;
            height: 100vh;
            display: flex;
            flex-direction: column;
            background: #1a1a1a;
        }
        
        .image-browser.fullscreen-mode .browser-header {
            position: relative;
            z-index: 10000;
            flex-shrink: 0;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
        }
        
        .image-browser.fullscreen-mode .image-name-header {
            color: #000000;
            font-weight: 600;
        }
        
        .image-browser.fullscreen-mode .browser-viewport {
            flex: 1;
            min-height: 0;
            overflow: hidden;
            padding: 0;
            margin-top: 0;
            background: #001f3f;
        }
        
        .image-browser.fullscreen-mode .nav-btn {
            display: none !important;
        }
        
        .image-browser.fullscreen-mode .browser-content-wrapper {
            height: calc(100vh - 80px); /* 减去导航栏高度 */
            margin-top: 0;
            padding-top: 0;
        }
        
        .image-browser.fullscreen-mode .image-wrapper {
            position: relative;
            overflow: auto;
        }
        
        .image-browser.fullscreen-mode .image-wrapper img {
            max-height: calc(100vh - 80px);
            cursor: grab;
            transition: transform 0.1s;
        }
        
        .image-browser.fullscreen-mode .image-wrapper img:active {
            cursor: grabbing;
        }
        
        .image-browser.fullscreen-mode .image-info-sidebar,
        .image-browser.fullscreen-mode .bbox-list-sidebar {
            max-height: calc(100vh - 80px); /* 减去导航栏高度 */
            overflow-y: auto;
        }
        
        .bbox-list-sidebar {
            width: 160px;
            min-width: 160px;
            background: #f8f9fa;
            border-left: 1px solid #e9ecef;
            padding: 8px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
        }
        
        .bbox-list-title {
            font-size: 11px;
            font-weight: 600;
            color: #495057;
            margin-bottom: 8px;
            padding-bottom: 4px;
            border-bottom: 1px solid #667eea;
        }
        
        .bbox-items {
            display: flex;
            flex-direction: column;
            gap: 4px;
            flex: 1;
        }
        
        .bbox-item {
            display: flex;
            align-items: center;
            gap: 4px;
            padding: 4px 6px;
            background: white;
            border-radius: 3px;
            border: 1px solid #e9ecef;
            font-size: 10px;
            transition: all 0.2s;
        }
        
        .bbox-item:hover {
            border-color: #667eea;
            box-shadow: 0 2px 4px rgba(102, 126, 234, 0.1);
        }
        
        .bbox-line-preview {
            display: inline-block;
            width: 30px;
            height: 2px;
            border-width: 1.5px;
            flex-shrink: 0;
        }
        
        .bbox-label {
            font-weight: 500;
            color: #2c3e50;
            flex-shrink: 0;
            font-size: 10px;
        }
        
        .bbox-score {
            color: #6c757d;
            font-size: 9px;
            flex-shrink: 0;
        }
        
        
        .status-badge-inline {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 10px;
            font-size: 11px;
            font-weight: 600;
            margin-right: 6px;
        }
        
        .status-badge-inline.error {
            background: #fee;
            color: #c33;
            border: 1px solid #fcc;
        }
        
        .status-badge-inline.warning {
            background: #fff4e6;
            color: #e67e22;
            border: 1px solid #ffd4a3;
        }
        
        .status-badge-inline.low-confidence {
            background: #e3f2fd;
            color: #1976d2;
            border: 1px solid #90caf9;
        }
        
        .status-normal {
            color: #999;
            margin-right: 6px;
        }
        
        .image-status {
            margin-bottom: 12px;
        }
        
        .image-notes {
            margin-bottom: 12px;
        }
        
        .notes-content {
            padding: 12px;
            background: white;
            border-left: 3px solid #667eea;
            border-radius: 4px;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        
        .image-details {
            display: flex;
            gap: 24px;
            font-size: 13px;
            color: #6c757d;
        }
        
        .detail-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .directory-tree {
            background: #f8f9fa;
            padding: 24px;
            border-radius: 6px;
            border: 1px solid #e9ecef;
            font-family: "SF Mono", "Monaco", "Consolas", monospace;
            font-size: 13px;
            line-height: 1.8;
            color: #495057;
        }
        
        .tree-line {
            margin-left: 20px;
        }
        
        .tree-file {
            margin-left: 40px;
        }
        
        @media (max-width: 768px) {
            .document-content {
                padding: 32px 24px;
            }
            
            .document-header {
                padding: 32px 24px;
            }
            
            .document-title {
                font-size: 24px;
            }
            
            .summary-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>'''


def _generate_summary_section(total_images, total_predictions, model_count, label_count):
    """生成总体统计部分"""
    return f'''
                <section class="section">
                    <h2 class="section-title">总体统计</h2>
                    <div class="summary-grid">
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
                            <div class="value">{model_count}</div>
                        </div>
                        <div class="summary-card">
                            <h3>标签类别</h3>
                            <div class="value">{label_count}</div>
                        </div>
                    </div>
                </section>'''


def _generate_models_section(model_infos, stats, style_map):
    """生成模型信息部分"""
    html_parts = [
        '                <section class="section">',
        '                    <h2 class="section-title">模型信息</h2>',
        '                    <div class="model-list">'
    ]
    
    for model_info in model_infos:
        # 安全获取模型名称，处理 None 值
        raw_model_name = model_info.get('name', 'Unknown')
        if raw_model_name is None:
            raw_model_name = 'Unknown'
        model_name = html.escape(str(raw_model_name))
        
        raw_model_type = model_info.get('type', 'Unknown')
        if raw_model_type is None:
            raw_model_type = 'Unknown'
        model_type = html.escape(str(raw_model_type))
        
        raw_sub_type = model_info.get('sub_type', '')
        sub_type = html.escape(str(raw_sub_type)) if raw_sub_type else ''
        
        line_style = model_info.get('line_style', 'solid')
        style_desc = style_map.get(line_style, '实线')
        model_total = sum(stats.get(str(raw_model_name), {}).values())
        
        html_parts.append(f'''
                        <div class="model-item">
                            <div class="model-name">{model_name}</div>
                            <div class="model-details">
                                <div class="model-detail-item">
                                    <span class="model-detail-label">类型:</span>
                                    <span class="model-detail-value">{model_type}{" - " + sub_type if sub_type else ""}</span>
                                </div>
                                <div class="model-detail-item">
                                    <span class="model-detail-label">线条:</span>
                                    <span class="model-detail-value">
                                        <span class="line-demo {line_style}"></span>
                                        {style_desc}
                                    </span>
                                </div>
                                <div class="model-detail-item">
                                    <span class="model-detail-label">检测数:</span>
                                    <span class="model-detail-value"><span class="badge">{model_total}</span></span>
                                </div>
                            </div>
                        </div>''')
    
    html_parts.extend([
        '                    </div>',
        '                </section>'
    ])
    
    return '\n'.join(html_parts)


def _generate_labels_section(all_labels, label_colors):
    """生成标签颜色部分"""
    html_parts = [
        '                <section class="section">',
        '                    <h2 class="section-title">标签颜色说明</h2>',
        '                    <div class="labels-grid">'
    ]
    
    for label in all_labels:
        color = label_colors.get(label, [128, 128, 128])
        label_escaped = html.escape(label)
        html_parts.append(f'''
                        <div class="label-item">
                            <span class="label-color" style="background-color: rgb({color[0]}, {color[1]}, {color[2]});"></span>
                            <span class="label-name">{label_escaped}</span>
                        </div>''')
    
    html_parts.extend([
        '                    </div>',
        '                </section>'
    ])
    
    return '\n'.join(html_parts)


def _generate_statistics_section(model_infos, all_labels, stats):
    """生成详细统计部分"""
    html_parts = [
        '                <section class="section">',
        '                    <h2 class="section-title">详细统计</h2>',
        '                    <table class="stats-table">',
        '                        <thead>',
        '                            <tr>',
        '                                <th>标签</th>'
    ]
    
    for model_info in model_infos:
        # 安全获取短名称，处理 None 值
        short_name_raw = model_info.get('short_name') or model_info.get('name', 'Unknown')
        if short_name_raw is None:
            short_name_raw = 'Unknown'
        short_name = html.escape(str(short_name_raw))
        html_parts.append(f'                                <th>{short_name}</th>')
    
    html_parts.extend([
        '                                <th>总计</th>',
        '                            </tr>',
        '                        </thead>',
        '                        <tbody>'
    ])
    
    for label in all_labels:
        label_escaped = html.escape(label)
        html_parts.append(f'                            <tr>')
        html_parts.append(f'                                <td style="font-weight: 600;">{label_escaped}</td>')
        
        label_total = 0
        for model_info in model_infos:
            # 安全获取模型名称，处理 None 值
            raw_model_name = model_info.get('name', 'Unknown')
            if raw_model_name is None:
                raw_model_name = 'Unknown'
            model_name = str(raw_model_name)
            count = stats.get(model_name, {}).get(label, 0)
            label_total += count
            html_parts.append(f'                                <td><span class="badge">{count}</span></td>')
        
        html_parts.append(f'                                <td style="font-weight: 600;"><span class="badge">{label_total}</span></td>')
        html_parts.append('                            </tr>')
    
    html_parts.extend([
        '                        </tbody>',
        '                    </table>',
        '                </section>'
    ])
    
    return '\n'.join(html_parts)


def _generate_images_section(images_with_info, false_positive_count, missed_count, low_confidence_count, predictions, all_labels):
    """生成全屏图片浏览器部分（支持多状态）"""
    # 收集所有图片的完整信息
    all_images_data = []
    for img_info in sorted(images_with_info, key=lambda x: x['name']):
        img_name = str(img_info.get('name', ''))
        
        # 支持新的多状态格式（statuses数组）或旧的单一状态格式（向后兼容）
        statuses = img_info.get('statuses', [])
        if not statuses:
            old_status = img_info.get('status')
            if old_status:
                statuses = [old_status]
        
        notes = img_info.get('notes', '') or ''
        max_score = img_info.get('max_score', 0.0)
        
        # 获取图片的预测数据以获取标签信息
        img_pred_data = predictions.get(img_name, {})
        labels_in_image = set()
        
        # 收集所有检测框信息（包括颜色、线条风格、类别、置信度）
        all_bboxes = []
        for pred_group in img_pred_data.get('predictions', []):
            line_style = pred_group.get('line_style', 'solid')
            model_name = pred_group.get('model_name', '')
            for bbox in pred_group.get('bboxes', []):
                label = bbox.get('label', 'unknown')
                labels_in_image.add(label)
                
                # 收集检测框详细信息
                bbox_info = {
                    'label': label,
                    'score': bbox.get('score', 0.0),
                    'color': bbox.get('color', [128, 128, 128]),  # RGB color
                    'line_style': line_style,
                    'model_name': model_name
                }
                all_bboxes.append(bbox_info)
        
        # 获取类别标签信息
        false_positive_labels = img_pred_data.get('false_positive_labels', [])
        missed_labels = img_pred_data.get('missed_labels', [])
        low_confidence_labels = img_pred_data.get('low_confidence_labels', [])
        
        # 如果最大置信度 < 0.5，自动添加 low_confidence 状态
        if max_score < 0.5 and max_score > 0:
            if 'low_confidence' not in statuses:
                statuses.append('low_confidence')
        
        # 确定图片路径（统一使用标注图目录）
        img_name_escaped = html.escape(img_name)
        img_src = f'标注图/{img_name_escaped}'
        
        # 原图路径
        original_src = f'原图/{img_name_escaped}'
        
        all_images_data.append({
            'name': img_name,
            'name_escaped': img_name_escaped,
            'src': img_src,  # 标注图路径
            'original_src': original_src,  # 原图路径
            'statuses': statuses,  # 多状态数组
            'notes': notes,
            'max_score': max_score,
            'labels': list(labels_in_image),
            'bboxes': all_bboxes,  # 所有检测框信息
            'false_positive_labels': false_positive_labels,  # 误检标签
            'missed_labels': missed_labels,  # 漏检标签
            'low_confidence_labels': low_confidence_labels  # 低置信度标签
        })
    
    if not all_images_data:
        return '''
                <section class="section">
                    <h2 class="section-title">图片备注与状态</h2>
                    <div class="images-section">
                        <p style="color: #999; font-style: italic;">暂无图片备注或状态标记</p>
                    </div>
                </section>'''
    
    # 将图片数据编码为 JSON
    import json as json_module
    images_json = json_module.dumps(all_images_data, ensure_ascii=False)
    
    html_parts = [
        '                <section class="section">',
        '                    <h2 class="section-title">图片备注与状态</h2>',
        '                    <div class="image-browser-container">',
        '                        <div class="browser-filters" id="browser-filters">',
        '                            <div class="filter-group">',
        '                                <label>状态筛选:</label>',
        '                                <label><input type="checkbox" value="false_positive" checked> 误检</label>',
        '                                <label><input type="checkbox" value="missed" checked> 漏检</label>',
        '                                <label><input type="checkbox" value="low_confidence" checked> 低置信度</label>',
        '                                <label><input type="checkbox" value="normal" checked> 正常</label>',
        '                            </div>',
        '                            <div class="filter-group">',
        '                                <label>标签筛选:</label>',
        '                                <select id="filter-label" multiple size="3">',
        '                                    <option value="">全部</option>'
    ]
    
    # 添加标签选项
    for label in sorted(all_labels):
        label_escaped = html.escape(label)
        html_parts.append(f'                                    <option value="{label_escaped}">{label_escaped}</option>')
    
    html_parts.extend([
        '                                </select>',
        '                            </div>',
        '                            <div class="filter-actions">',
        '                                <button onclick="resetFilters()">重置筛选</button>',
        '                            </div>',
        '                        </div>',
        '                        <div class="image-browser" id="image-browser">',
        '                            <div class="browser-header">',
        '                                <span id="current-image-info" class="image-info">1 / ' + str(len(all_images_data)) + '</span>',
        '                                <span id="current-image-name" class="image-name-header"></span>',
        '                                <div class="header-buttons">',
        '                                    <button id="toggle-image-btn" class="toggle-image-btn" onclick="toggleImageType()" title="切换图片类型 (T键)">切换到原图</button>',
        '                                    <button id="fullscreen-btn" class="fullscreen-btn" onclick="toggleFullscreen()" title="全屏查看 (F键)">全屏</button>',
        '                                </div>',
        '                            </div>',
        '                            <div class="browser-viewport">',
        '                                <button class="nav-btn prev-btn" onclick="navigateImage(-1)" title="上一张 (← 或 A)">‹</button>',
        '                                <div class="browser-content-wrapper">',
        '                                    <div id="image-info-sidebar" class="image-info-sidebar"></div>',
        '                                    <div class="image-wrapper">',
        '                                        <img id="browser-image" src="" alt="" />',
        '                                    </div>',
        '                                    <div id="bbox-list" class="bbox-list-sidebar"></div>',
        '                                </div>',
        '                                <button class="nav-btn next-btn" onclick="navigateImage(1)" title="下一张 (→ 或 D)">›</button>',
        '                            </div>',
        '                        </div>',
        '                    </div>',
        '                </section>',
        '                <script>',
        '                    const imagesData = ' + images_json + ';',
        '                    let currentIndex = 0;',
        '                    let filteredImages = [...imagesData];',
        '                    let showOriginal = false;  // 是否显示原图',
        '',
        '                    function updateBrowser() {',
        '                        if (filteredImages.length === 0) {',
        '                            document.getElementById("browser-image").src = "";',
        '                            document.getElementById("current-image-info").textContent = "0 / 0";',
        '                            document.getElementById("current-image-name").textContent = "无图片";',
        '                            document.getElementById("image-info-sidebar").innerHTML = "";',
        '                            document.getElementById("bbox-list").innerHTML = "";',
        '                            return;',
        '                        }',
        '                        ',
        '                        const img = filteredImages[currentIndex];',
        '                        // 根据切换状态选择图片路径',
        '                        const imgSrc = showOriginal && img.original_src ? encodeURI(img.original_src) : encodeURI(img.src);',
        '                        ',
        '                        const browserImage = document.getElementById("browser-image");',
        '                        browserImage.src = imgSrc;',
        '                        document.getElementById("current-image-info").textContent = `${currentIndex + 1} / ${filteredImages.length}`;',
        '                        document.getElementById("current-image-name").textContent = img.name;',
        '                        ',
        '                        // 图片加载完成后重置缩放和位置',
        '                        browserImage.onload = function() {',
        '                            if (typeof resetImagePosition === "function") {',
        '                                resetImagePosition();',
        '                            }',
        '                        };',
        '                        ',
        '                        // 更新切换按钮',
        '                        const toggleBtn = document.getElementById("toggle-image-btn");',
        '                        if (toggleBtn) {',
        '                            toggleBtn.textContent = showOriginal ? "切换到标注图" : "切换到原图";',
        '                        }',
        '                        ',
        '                        // 更新信息行（单行显示所有信息）',
        '                        const infoLineDiv = document.getElementById("image-info-sidebar");',
        '                        const statuses = img.statuses || [];',
        '                        let infoParts = [];',
        '                        ',
        '                        // 状态',
        '                        if (statuses.includes("false_positive")) {',
        '                            infoParts.push(\'<span class="status-badge-inline error">误检</span>\');',
        '                        }',
        '                        if (statuses.includes("missed")) {',
        '                            infoParts.push(\'<span class="status-badge-inline warning">漏检</span>\');',
        '                        }',
        '                        if (statuses.includes("low_confidence")) {',
        '                            infoParts.push(\'<span class="status-badge-inline low-confidence">低置信度</span>\');',
        '                        }',
        '                        if (infoParts.length === 0) {',
        '                            infoParts.push(\'<span class="status-normal">正常</span>\');',
        '                        }',
        '                        ',
        '                        // 类别标签信息',
        '                        const labelsInfo = [];',
        '                        if (img.false_positive_labels && img.false_positive_labels.length > 0) {',
        '                            labelsInfo.push(`<span style="color: #e74c3c;">误检: ${img.false_positive_labels.join(\', \')}</span>`);',
        '                        }',
        '                        if (img.missed_labels && img.missed_labels.length > 0) {',
        '                            labelsInfo.push(`<span style="color: #f39c12;">漏检: ${img.missed_labels.join(\', \')}</span>`);',
        '                        }',
        '                        if (img.low_confidence_labels && img.low_confidence_labels.length > 0) {',
        '                            labelsInfo.push(`<span style="color: #1976d2;">低置信度: ${img.low_confidence_labels.join(\', \')}</span>`);',
        '                        }',
        '                        ',
        '                        // 备注',
        '                        const notesText = img.notes ? escapeHtml(img.notes) : \'<span style="color: #999; font-style: italic;">无备注</span>\';',
        '                        ',
        '                        // 组装信息（垂直布局，移除最大置信度和标签）',
        '                        let infoHtml = [];',
        '                        ',
        '                        // 状态',
        '                        if (infoParts.length > 0) {',
        '                            infoHtml.push(\'<div style="margin-bottom: 8px;">\' + infoParts.join(\' \') + \'</div>\');',
        '                        }',
        '                        ',
        '                        // 类别标签信息',
        '                        if (labelsInfo.length > 0) {',
        '                            infoHtml.push(\'<div style="margin-bottom: 8px;">\' + labelsInfo.join(\'<br>\') + \'</div>\');',
        '                        }',
        '                        ',
        '                        // 备注',
        '                        infoHtml.push(\'<div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #e9ecef;">\' + notesText + \'</div>\');',
        '                        ',
        '                        infoLineDiv.innerHTML = infoHtml.join(\'\');',
        '                        ',
        '                        // 更新检测框信息列表',
        '                        updateBboxList(img);',
        '                    }',
        '                    ',
        '                    function updateBboxList(img) {',
        '                        const bboxListDiv = document.getElementById("bbox-list");',
        '                        if (!bboxListDiv) return;',
        '                        ',
        '                        const bboxes = img.bboxes || [];',
        '                        if (bboxes.length === 0) {',
        '                            bboxListDiv.innerHTML = "";',
        '                            return;',
        '                        }',
        '                        ',
        '                        // 样式映射',
        '                        const styleMap = {',
        '                            "solid": "实线",',
        '                            "dashed": "虚线",',
        '                            "dotted": "点线",',
        '                            "dashdot": "点划线"',
        '                        };',
        '                        ',
        '                        let bboxHtml = \'<div class="bbox-list-title">检测框信息:</div>\';',
        '                        bboxHtml += \'<div class="bbox-items">\';',
        '                        ',
        '                        bboxes.forEach((bbox, idx) => {',
        '                            const color = bbox.color || [128, 128, 128];',
        '                            const colorStr = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;',
        '                            const lineStyle = bbox.line_style || "solid";',
        '                            const styleDesc = styleMap[lineStyle] || "实线";',
        '                            const label = escapeHtml(bbox.label || "unknown");',
        '                            const score = bbox.score ? (bbox.score * 100).toFixed(1) : "0.0";',
        '                            ',
        '                            bboxHtml += `<div class="bbox-item">`;',
        '                            bboxHtml += `<span class="bbox-line-preview" style="border-color: ${colorStr}; border-style: ${lineStyle};"></span>`;',
        '                            bboxHtml += `<span class="bbox-label">${label}</span>`;',
        '                            bboxHtml += `<span class="bbox-score">(${score}%)</span>`;',
        '                            bboxHtml += `</div>`;',
        '                        });',
        '                        ',
        '                        bboxHtml += \'</div>\';',
        '                        bboxListDiv.innerHTML = bboxHtml;',
        '                    }',
        '                    ',
        '                    function toggleImageType() {',
        '                        showOriginal = !showOriginal;',
        '                        updateBrowser();',
        '                    }',
        '                    ',
        '                    function toggleFullscreen() {',
        '                        const browser = document.getElementById("image-browser");',
        '                        if (!browser) return;',
        '                        ',
        '                        if (browser.classList.contains("fullscreen-mode")) {',
        '                            // 退出全屏',
        '                            browser.classList.remove("fullscreen-mode");',
        '                            document.body.style.overflow = "";',
        '                            const btn = document.getElementById("fullscreen-btn");',
        '                            if (btn) btn.textContent = "全屏";',
        '                            // 退出全屏时重置缩放和位置',
        '                            resetImagePosition();',
        '                        } else {',
        '                            // 进入全屏',
        '                            browser.classList.add("fullscreen-mode");',
        '                            document.body.style.overflow = "hidden";',
        '                            const btn = document.getElementById("fullscreen-btn");',
        '                            if (btn) btn.textContent = "退出全屏";',
        '                            // 进入全屏时重置缩放和位置',
        '                            resetImagePosition();',
        '                        }',
        '                    }',
        '',
        '                    function navigateImage(direction) {',
        '                        if (filteredImages.length === 0) return;',
        '                        currentIndex += direction;',
        '                        if (currentIndex < 0) currentIndex = filteredImages.length - 1;',
        '                        if (currentIndex >= filteredImages.length) currentIndex = 0;',
        '                        // 切换图片时重置缩放和位置',
        '                        if (typeof resetImagePosition === "function") {',
        '                            resetImagePosition();',
        '                        }',
        '                        updateBrowser();',
        '                    }',
        '',
        '                    function applyFilters() {',
        '                        const statusFilters = Array.from(document.querySelectorAll("#browser-filters input[type=\'checkbox\']:checked")).map(cb => cb.value);',
        '                        const labelFilters = Array.from(document.getElementById("filter-label").selectedOptions).map(opt => opt.value).filter(v => v);',
        '                        ',
        '                        filteredImages = imagesData.filter(img => {',
        '                            // 状态筛选（支持多状态）',
        '                            const imgStatuses = img.statuses || [];',
        '                            if (imgStatuses.length === 0) {',
        '                                // 如果没有状态，视为正常',
        '                                if (!statusFilters.includes("normal")) return false;',
        '                            } else {',
        '                                // 检查图片的任意状态是否在筛选列表中',
        '                                const hasSelectedStatus = imgStatuses.some(s => statusFilters.includes(s));',
        '                                if (!hasSelectedStatus) return false;',
        '                            }',
        '                            ',
        '                            // 标签筛选',
        '                            if (labelFilters.length > 0 && !labelFilters.some(label => img.labels.includes(label))) return false;',
        '                            ',
        '                            return true;',
        '                        });',
        '                        ',
        '                        currentIndex = 0;',
        '                        updateBrowser();',
        '                    }',
        '',
        '                    function resetFilters() {',
        '                        document.querySelectorAll("#browser-filters input[type=\'checkbox\']").forEach(cb => cb.checked = true);',
        '                        const labelSelect = document.getElementById("filter-label");',
        '                        for (let i = 0; i < labelSelect.options.length; i++) {',
        '                            labelSelect.options[i].selected = false;',
        '                        }',
        '                        if (labelSelect.options.length > 0) labelSelect.options[0].selected = true;',
        '                        applyFilters();',
        '                    }',
        '',
        '                    function escapeHtml(text) {',
        '                        const div = document.createElement("div");',
        '                        div.textContent = text;',
        '                        return div.innerHTML;',
        '                    }',
        '                    ',
        '                    // 筛选导出功能已移除',
        '                    // 绑定筛选事件',
        '                    document.addEventListener("DOMContentLoaded", function() {',
        '                        document.querySelectorAll("#browser-filters input, #browser-filters select").forEach(el => {',
        '                            el.addEventListener("change", applyFilters);',
        '                        });',
        '                        ',
        '                        // 图片拖动和缩放相关变量',
        '                        let zoomLevel = 1;',
        '                        let isDragging = false;',
        '                        let dragStart = { x: 0, y: 0 };',
        '                        let imageOffset = { x: 0, y: 0 };',
        '                        ',
        '                        // 重置图片位置和缩放',
        '                        function resetImagePosition() {',
        '                            zoomLevel = 1;',
        '                            imageOffset.x = 0;',
        '                            imageOffset.y = 0;',
        '                            updateImageTransform();',
        '                        }',
        '                        ',
        '                        // 更新图片变换',
        '                        function updateImageTransform() {',
        '                            const img = document.getElementById("browser-image");',
        '                            if (img) {',
        '                                img.style.transform = `translate(${imageOffset.x}px, ${imageOffset.y}px) scale(${zoomLevel})`;',
        '                            }',
        '                        }',
        '                        ',
        '                        // 图片拖动功能',
        '                        const browserImage = document.getElementById("browser-image");',
        '                        const imageWrapper = document.querySelector(".image-wrapper");',
        '                        ',
        '                        if (browserImage) {',
        '                            browserImage.addEventListener("mousedown", function(e) {',
        '                                if (e.button === 0) { // 左键',
        '                                    isDragging = true;',
        '                                    dragStart.x = e.clientX - imageOffset.x;',
        '                                    dragStart.y = e.clientY - imageOffset.y;',
        '                                    browserImage.style.cursor = "grabbing";',
        '                                    browserImage.style.transition = "none";',
        '                                    e.preventDefault();',
        '                                }',
        '                            });',
        '                        }',
        '                        ',
        '                        document.addEventListener("mousemove", function(e) {',
        '                            if (isDragging) {',
        '                                imageOffset.x = e.clientX - dragStart.x;',
        '                                imageOffset.y = e.clientY - dragStart.y;',
        '                                updateImageTransform();',
        '                                e.preventDefault();',
        '                            }',
        '                        });',
        '                        ',
        '                        document.addEventListener("mouseup", function(e) {',
        '                            if (e.button === 0 && isDragging) {',
        '                                isDragging = false;',
        '                                const img = document.getElementById("browser-image");',
        '                                if (img) {',
        '                                    img.style.cursor = "grab";',
        '                                    img.style.transition = "transform 0.1s";',
        '                                }',
        '                            }',
        '                        });',
        '                        ',
        '                        // 鼠标滚轮缩放',
        '                        if (imageWrapper) {',
        '                            imageWrapper.addEventListener("wheel", function(e) {',
        '                                if (e.ctrlKey || e.metaKey) {',
        '                                    e.preventDefault();',
        '                                    const delta = e.deltaY > 0 ? -0.1 : 0.1;',
        '                                    zoomLevel = Math.max(0.5, Math.min(5, zoomLevel + delta));',
        '                                    updateImageTransform();',
        '                                }',
        '                            });',
        '                        }',
        '                        ',
        '                        // 键盘导航（全屏模式下仍可使用键盘切换）',
        '                        document.addEventListener("keydown", function(e) {',
        '                            if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT" || e.target.tagName === "TEXTAREA") return;',
        '                            ',
        '                            const browser = document.querySelector(".image-browser");',
        '                            const isFullscreen = browser && browser.classList.contains("fullscreen-mode");',
        '                            ',
        '                            // 左右切换（全屏和非全屏模式下都支持）',
        '                            if (e.key === "ArrowLeft" || e.key === "a" || e.key === "A") {',
        '                                e.preventDefault();',
        '                                navigateImage(-1);',
        '                            } else if (e.key === "ArrowRight" || e.key === "d" || e.key === "D") {',
        '                                e.preventDefault();',
        '                                navigateImage(1);',
        '                            }',
        '                            ',
        '                            // 全屏模式下支持缩放',
        '                            if (isFullscreen) {',
        '                                if (e.key === "+" || e.key === "=") {',
        '                                    e.preventDefault();',
        '                                    zoomLevel = Math.min(zoomLevel + 0.2, 5);',
        '                                    updateImageTransform();',
        '                                } else if (e.key === "-") {',
        '                                    e.preventDefault();',
        '                                    zoomLevel = Math.max(zoomLevel - 0.2, 0.5);',
        '                                    updateImageTransform();',
        '                                } else if (e.key === "0") {',
        '                                    e.preventDefault();',
        '                                    resetImagePosition();',
        '                                }',
        '                            }',
        '                            ',
        '                            if (e.key === "t" || e.key === "T") {',
        '                                e.preventDefault();',
        '                                toggleImageType();',
        '                            } else if (e.key === "f" || e.key === "F") {',
        '                                e.preventDefault();',
        '                                toggleFullscreen();',
        '                            }',
        '                        });',
        '                        ',
        '                        updateBrowser();',
        '                    });',
        '                </script>'
    ])
    
    return '\n'.join(html_parts)


def _generate_directory_section(model_infos, false_positive_count, missed_count):
    """生成目录结构部分"""
    html_parts = [
        '                <section class="section">',
        '                    <h2 class="section-title">目录结构说明</h2>',
        '                    <div class="directory-tree">',
        '                        <div><strong>predictions.zip</strong></div>'
    ]
    
    for idx, model_info in enumerate(model_infos):
        model_name = model_info.get('name', 'Unknown')
        # 确保 model_name 不是 None，并且是字符串类型
        if model_name is None:
            model_name = f'模型_{idx + 1}'
        else:
            model_name = str(model_name)
        
        # 安全处理文件名，如果返回空或 None，使用默认名称
        try:
            safe_name = secure_filename(model_name)
            if not safe_name:
                safe_name = f'模型_{idx + 1}'
        except (TypeError, AttributeError):
            safe_name = f'模型_{idx + 1}'
        
        safe_name_escaped = html.escape(safe_name)
        html_parts.append(f'                        <div class="tree-line">├── <strong>{safe_name_escaped}/</strong></div>')
        html_parts.append('                        <div class="tree-file">└── _annotations.coco.json</div>')
    
    html_parts.extend([
        '                        <div class="tree-line">├── <strong>标注图/</strong></div>',
        '                        <div class="tree-file">├── *.jpg (标注后的可视化图片)</div>',
        '                        <div class="tree-file">└── *.png (标注后的可视化图片)</div>',
        '                        <div class="tree-line">└── <strong>原图/</strong></div>',
        '                        <div class="tree-file">└── 所有原始图片</div>'
    ])
    
    html_parts.extend([
        '                    </div>',
        '                </section>'
    ])
    
    return '\n'.join(html_parts)