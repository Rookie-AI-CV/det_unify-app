# DetUnify Studio

本地多模型检测预测与结果整理工具。Web 端上传权重与图片即可跑批；支持多模型同图对比、误检/漏检/低置信度标注，以及导出 ZIP 与 HTML 报告。

详细操作步骤（含界面截图）见飞书文档：[预测工具用户使用手册](https://zcnce50wan15.feishu.cn/wiki/L1nFwAeQXiTdsmkfC5bcoAlDnJc)。

## 能做什么

- 自动识别 **DINO（ml_backend）** 与 **HQ-Det** 系列权重，无需手写模型类型（也可命令行手动指定）。
- 一次任务可挂多个 `.pth` / `.pt`，结果按模型分目录，查看器里可叠加对比。
- 对预测结果打标：误检、漏检、低置信度，并写备注。
- 导出：标注图、原图（可选）、COCO JSON、分类用 JSON、`split_images.pyw`、HTML 报告。

## 支持的模型

| 框架 | CLI 类型 | 说明 |
|------|----------|------|
| ml_backend | `dino` | 公司内部 DINO 检测流程 |
| hq_det | `hq_det` + `--hq-model-type` | 见下表 |

HQ-Det 子类型（`--hq-model-type`）：`dino`、`dino2`、`rtdetr`、`rtmdet`、`yolo`、`lwdetr`、`rfdetr`、`codetr`。

新增 HQ-Det 子类型时，只需在 `src/predict/hq_model_registry.py` 的 `HQ_MODELS` 中增加一行，无需改 `predict.py` / `predict_hq_det.py`。

## 安装

### 1. 获取代码

作为 [online_data_tool](https://github.com/algo-boost/online_data_tool) 子模块时：

```bash
git submodule update --init --recursive tools/DetUnify-Studio
cd tools/DetUnify-Studio
```

单独使用本仓库：

```bash
git clone https://github.com/algo-boost/DetUnify-Studio.git
cd DetUnify-Studio
```

### 2. Web 界面（必需）

```bash
cd app
pip install flask werkzeug loguru opencv-python numpy pillow tqdm
python app.py
```

浏览器打开 **http://localhost:6006**（端口在 `app/app.py` 中配置）。

也可执行 `app/run.sh`（等价于在 `app` 目录下运行 `python app.py`）。

### 3. 模型推理依赖（按实际权重安装）

| 权重类型 | 需要安装的包 |
|----------|----------------|
| DINO | `ml_backend` 及对应训练环境 |
| HQ-Det 各子类型 | `hq_det`；另按子模型可能需要 `mmdet`、`ultralytics`、`supervision` 等 |

`hq_det` 若不在默认 Python 路径下，可设置：

```bash
export PYTHONPATH="/path/to/hq_det:${PYTHONPATH}"
```

检查 GPU（可选）：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

未安装某框架时，Web 仍可启动，但加载该类型权重会失败；日志里会有具体 import 错误。

## 使用

### Web

1. 导入模型：拖拽或填写本地路径（支持 `.pt` / `.pth` / `.ckpt` 及 zip）。
2. 导入图片：目录或 zip。
3. 设置阈值、`max_size`（HQ-Det）等，开始预测。
4. 在结果列表进入查看器，标注并导出。

界面说明见上文飞书手册。

### 命令行

统一入口 `src/predict/predict.py`，输出为 COCO JSON（目录模式会额外生成 `preds/` 可视化图）。

```bash
# 自动识别模型类型
python src/predict/predict.py \
  --checkpoint /path/to/model.pth \
  --image-dir /path/to/images \
  --output /path/to/out

# 单张图
python src/predict/predict.py \
  --checkpoint model.pth \
  --image test.jpg \
  --output results/

# 指定 HQ-Det 子类型
python src/predict/predict.py \
  --checkpoint model.pth \
  --model-type hq_det \
  --hq-model-type rtdetr \
  --image-dir images/ \
  --output results/

# 调参
python src/predict/predict.py \
  --checkpoint model.pth \
  --image-dir images/ \
  --threshold 0.3 \
  --max-size 2048 \
  --device cuda:0 \
  --output results/
```

直接调用子脚本（一般不必）：

```bash
python src/predict/predict_hq_det.py --checkpoint model.pth --model-type rtdetr --image-dir images/ --output results/
python src/predict/predict_dino.py --checkpoint model.pth --image-dir images/ --output results/
```

## 查看器快捷键

| 按键 | 作用 |
|------|------|
| `←` / `A` | 上一张 |
| `→` / `D` | 下一张 |
| `F` | 误检（再按取消） |
| `M` | 漏检 |
| `L` | 低置信度 |
| `+` / `-` | 缩放 |
| `0` | 重置缩放与位置 |
| `Ctrl` / `Cmd` + 滚轮 | 缩放 |

## 目录结构

```
DetUnify-Studio/
├── app/
│   ├── app.py                 # Flask 主程序
│   ├── readme_generator.py    # 导出 HTML 报告
│   ├── run.sh
│   ├── templates/             # index / viewer
│   └── static/
│       ├── scripts/split_images.pyw   # 导出包内按状态分目录
│       ├── uploads/
│       └── results/
└── src/predict/
    ├── predict.py             # 统一入口（类型检测 + 调子脚本）
    ├── predict_hq_det.py
    ├── predict_dino.py
    ├── hq_model_registry.py   # HQ-Det 模型注册表
    └── test_hq_model_registry.py
```

## 导出包说明

导出 ZIP 中常见内容：

- 各模型的 `_annotations.coco.json`
- `preds/` 标注可视化图
- `原图/`（勾选导出原图时）
- `README.md` / HTML 报告
- `split_images.pyw`：在导出目录双击运行，按误检、漏检、低置信度分子目录整理图片

## 常见问题

**模型加载失败**  
确认权重路径正确；HQ-Det 是否已安装且 `PYTHONPATH` 包含 `hq_det`；子类型不对时可加 `--hq-model-type` 手动指定。

**只能用 CPU**  
命令行加 `--device cpu`；或检查 CUDA 与 PyTorch 是否匹配。

**预测无框 / COCO 为空**  
调高或调低 `--threshold`；HQ-Det 可尝试调整 `--max-size`。

**导出很慢或失败**  
检查磁盘空间；`app.py` 中 `MAX_CONTENT_LENGTH` 默认 2GB，超大包需分批。

## 开发

- 注册新 HQ-Det 类型：编辑 `src/predict/hq_model_registry.py`。
- 跑注册表相关单元测试：`python -m unittest src.predict.test_hq_model_registry -v`（在项目根目录执行）。

## 联系

Rookie — RookieEmail@163.com
