# DetUnify Studio

**多模型检测统一工作平台** - 一个统一的 Web 应用程序，用于多种检测模型的预测、可视化和结果管理。

## 📋 项目简介

DetUnify Studio 是一个基于 Flask 的 Web 应用平台，提供统一的接口来运行、管理和评估多种目标检测模型。支持自动模型类型检测、批量预测、结果可视化、标注管理和导出功能。

## ✨ 主要功能

### 🎯 模型预测
- **多模型支持**：支持同时加载多个模型进行批量预测
- **自动模型识别**：自动检测模型类型，无需手动配置
- **批量处理**：支持单张图片、图片目录或图片列表文件
- **实时进度显示**：预测过程中显示实时进度和状态

### 🔍 结果查看与管理
- **可视化查看器**：交互式图片查看器，支持缩放、拖拽、全屏
- **检测框标注**：显示检测框、类别、置信度等信息
- **多状态标注**：支持误检、漏检、低置信度等多种状态标注
- **类别标注**：为误检、漏检等标注具体类别信息
- **备注功能**：为每张图片添加备注信息
- **快捷键操作**：支持键盘快捷键快速操作

### 📊 结果导出
- **完整导出**：导出标注图片、原图、模型文件
- **分类整理**：自动生成分类 JSON 和 Python 脚本，方便后续整理
- **HTML 报告**：生成详细的 HTML 格式预测报告
- **进度显示**：导出过程中显示详细进度信息

### 🎨 用户体验
- **现代化 UI**：简洁美观的用户界面
- **响应式设计**：适配不同屏幕尺寸
- **实时反馈**：操作即时反馈，状态清晰可见

## 🚀 支持的模型类型

### DINO 模型
- 基于 `ml_backend` 的 DINO 检测模型
- 支持批量预测和单张预测

### HQ-Det 模型系列
支持以下 HQ-Det 子类型：
- **DINO** (`dino`)
- **RT-DETR** (`rtdetr`)
- **RTMDet** (`rtmdet`)
- **YOLO** (`yolo`)
- **LW-DETR** (`lwdetr`)
- **RF-DETR** (`rfdetr`)
- **Co-DETR** (`codetr`)

## 📦 安装

### 环境要求
- Python 3.8+
- CUDA（推荐，用于 GPU 加速）
- PyTorch
- Flask

### 安装步骤

1. **克隆项目**
```bash
git clone <repository-url>
cd det_unify-app
```

2. **安装依赖**
```bash
# 安装 Python 依赖
pip install flask werkzeug loguru opencv-python numpy pathlib

# 安装模型相关依赖（根据使用的模型类型）
# DINO 模型需要 ml_backend
# HQ-Det 模型需要相应的 hq_det 包
```

3. **配置环境**
```bash
# 确保模型文件和相关依赖已正确安装
# 检查 CUDA 是否可用（如果使用 GPU）
python -c "import torch; print(torch.cuda.is_available())"
```

## 🎮 使用方法

### 启动应用

```bash
cd app
python app.py
```

应用将在 `http://localhost:6006` 启动（默认端口）。

### Web 界面使用

#### 1. 上传模型
- 拖拽模型文件（`.pt`, `.pth`, `.ckpt`）或 ZIP 压缩包到上传区域
- 或通过本地路径导入模型文件
- 支持同时上传多个模型

#### 2. 上传数据
- 拖拽图片文件或 ZIP 压缩包
- 或通过本地路径导入图片目录

#### 3. 开始预测
- 设置预测参数（阈值、最大尺寸等）
- 点击"开始预测"按钮
- 等待预测完成

#### 4. 查看结果
- 点击结果卡片进入查看器
- 使用快捷键或按钮进行导航
- 标注误检、漏检、低置信度等状态
- 添加备注信息

#### 5. 导出结果
- 在结果查看器中点击"导出"按钮
- 选择导出选项（原图、标注图、模型文件等）
- 等待导出完成并下载 ZIP 文件

### 命令行使用

项目也提供了命令行预测接口：

```bash
# 自动检测模型类型并预测
python src/predict/predict.py --checkpoint model.pth --image test.jpg --output results/

# 指定图片目录
python src/predict/predict.py --checkpoint model.pth --image-dir images/ --output results/

# 手动指定模型类型
python src/predict/predict.py --checkpoint model.pth --model-type hq_det --hq-model-type rtdetr --image-dir images/ --output results/

# 指定参数
python src/predict/predict.py --checkpoint model.pth --image-dir images/ --threshold 0.3 --max-size 2048 --output results/
```

## ⌨️ 快捷键

在结果查看器中支持以下快捷键：

- `←` / `A` - 上一张图片
- `→` / `D` - 下一张图片
- `F` - 标记为误检
- `M` - 标记为漏检
- `L` - 标记为低置信度
- `T` - 切换原图/标注图
- `F11` / `F` - 全屏模式
- `ESC` - 退出全屏

## 📁 项目结构

```
det_unify-app/
├── app/                    # Flask 应用主目录
│   ├── app.py             # 主应用文件
│   ├── readme_generator.py # HTML 报告生成器
│   ├── templates/         # HTML 模板
│   │   ├── index.html     # 主页面
│   │   └── viewer.html    # 结果查看器
│   └── static/            # 静态文件
│       ├── css/           # 样式文件
│       ├── js/            # JavaScript 文件
│       ├── uploads/       # 上传文件目录
│       └── results/       # 预测结果目录
├── src/                    # 源代码目录
│   └── predict/           # 预测相关代码
│       ├── predict.py     # 统一预测入口
│       ├── predict_hq_det.py  # HQ-Det 模型预测
│       └── predict_dino.py    # DINO 模型预测
└── README.md              # 本文件
```

## 🔧 配置说明

### 应用配置

在 `app/app.py` 中可以配置：

- `MAX_CONTENT_LENGTH`：最大上传文件大小（默认 2GB）
- `UPLOAD_FOLDER`：上传文件存储目录
- `RESULTS_FOLDER`：结果文件存储目录
- `EXPORT_CACHE_FOLDER`：导出缓存目录

### 预测参数

- `--threshold`：置信度阈值（默认 0.5）
- `--max-size`：最大图片尺寸（HQ-Det 模型，默认 1536）
- `--batch-size`：批次大小（DINO 模型，默认 6）
- `--device`：计算设备（默认 cuda:0）

## 🎯 功能特性详解

### 类别管理
- 从模型文件中自动提取所有类别名称
- 漏检标注时显示完整类别列表，支持选择所有可能的类别
- 支持多类别选择

### 状态标注
- **误检（False Positive）**：标记错误检测的框
- **漏检（Missed）**：标记应该检测到但未检测到的类别
- **低置信度（Low Confidence）**：标记置信度过低的检测框

### 导出功能
- 导出标注图片和原图
- 生成分类 JSON 文件（包含误检、漏检、低置信度的图片和类别信息）
- 提供 Python 脚本（`split_images.pyw`）用于自动分类整理
- 生成详细的 HTML 报告

## 🐛 故障排除

### 常见问题

1. **模型加载失败**
   - 检查模型文件路径是否正确
   - 确认模型类型是否支持
   - 查看日志中的详细错误信息

2. **GPU 不可用**
   - 检查 CUDA 是否正确安装
   - 确认 PyTorch 是否支持 CUDA
   - 可以设置 `--device cpu` 使用 CPU

3. **预测失败**
   - 检查图片格式是否支持
   - 确认模型和图片路径正确
   - 查看详细错误日志

4. **导出失败**
   - 检查磁盘空间是否充足
   - 确认文件权限正确
   - 查看导出进度日志

## 📝 开发说明

### 添加新模型类型

1. 在 `src/predict/` 目录下创建新的预测脚本
2. 在 `src/predict/predict.py` 中添加模型检测逻辑
3. 实现模型的加载和预测接口
4. 确保输出格式符合 COCO JSON 标准

### 扩展功能

项目采用模块化设计，便于扩展：
- `app/app.py`：主应用逻辑
- `app/readme_generator.py`：报告生成
- `src/predict/`：预测模块
- `app/templates/`：前端界面

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📧 联系方式

**作者**: Rookie  
**邮箱**: RookieEmail@163.com


---

**DetUnify Studio** - 让模型预测和管理更简单 🚀
