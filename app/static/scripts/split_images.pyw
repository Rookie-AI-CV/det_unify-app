#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图片分类拆分脚本
根据classification.json文件，将标注图和原图分别拆分到误检、漏检、低置信度目录
"""
import json
import shutil
from pathlib import Path

def main():
    # 获取脚本所在目录（导出根目录）
    script_dir = Path(__file__).parent.absolute()
    
    # 读取分类信息
    classification_file = script_dir / 'classification.json'
    if not classification_file.exists():
        print(f"错误: 找不到分类文件 {classification_file}")
        input("按回车键退出...")
        return
    
    with open(classification_file, 'r', encoding='utf-8') as f:
        classification_data = json.load(f)
    
    # 定义源目录和目标目录
    annotated_dir = script_dir / '标注图'
    original_dir = script_dir / '原图'
    
    output_dirs = {
        'false_positive': {
            'annotated': script_dir / '误检' / '标注图',
            'original': script_dir / '误检' / '原图'
        },
        'missed': {
            'annotated': script_dir / '漏检' / '标注图',
            'original': script_dir / '漏检' / '原图'
        },
        'low_confidence': {
            'annotated': script_dir / '低置信度' / '标注图',
            'original': script_dir / '低置信度' / '原图'
        }
    }
    
    # 创建输出目录
    for category, dirs in output_dirs.items():
        dirs['annotated'].mkdir(parents=True, exist_ok=True)
        dirs['original'].mkdir(parents=True, exist_ok=True)
    
    # 统计信息
    stats = {
        'false_positive': {'annotated': 0, 'original': 0},
        'missed': {'annotated': 0, 'original': 0},
        'low_confidence': {'annotated': 0, 'original': 0}
    }
    
    # 处理每个分类
    for category, images in classification_data.items():
        if category not in output_dirs:
            continue
        
        print(f"\n处理 {category} 图片...")
        target_dirs = output_dirs[category]
        
        for img_info in images:
            filename = img_info['filename']
            labels = img_info.get('labels', [])
            
            # 复制标注图
            annotated_src = annotated_dir / filename
            if annotated_src.exists():
                annotated_dst = target_dirs['annotated'] / filename
                shutil.copy2(annotated_src, annotated_dst)
                stats[category]['annotated'] += 1
                print(f"  已复制标注图: {filename} (标签: {', '.join(labels) if labels else '无'})")
            else:
                print(f"  警告: 找不到标注图 {filename}")
            
            # 复制原图
            original_src = original_dir / filename
            if original_src.exists():
                original_dst = target_dirs['original'] / filename
                shutil.copy2(original_src, original_dst)
                stats[category]['original'] += 1
            else:
                print(f"  警告: 找不到原图 {filename}")
    
    # 打印统计信息
    print("\n" + "="*50)
    print("拆分完成！统计信息：")
    print("="*50)
    for category, count in stats.items():
        category_name = {'false_positive': '误检', 'missed': '漏检', 'low_confidence': '低置信度'}.get(category, category)
        print(f"{category_name}:")
        print(f"  标注图: {count['annotated']} 张")
        print(f"  原图: {count['original']} 张")
        print()
    
    print("所有图片已拆分到对应目录：")
    print("  - 误检/标注图/ 和 误检/原图/")
    print("  - 漏检/标注图/ 和 漏检/原图/")
    print("  - 低置信度/标注图/ 和 低置信度/原图/")
    print("\n按回车键退出...")
    input()

if __name__ == '__main__':
    main()

