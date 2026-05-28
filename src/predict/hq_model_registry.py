#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HQ-Det 模型注册表（单一数据源）

新增子模型：在 HQ_MODELS 末尾追加一条 HQModelEntry 即可，无需修改 predict.py / predict_hq_det.py。

示例::
    HQModelEntry("deformable_detr", "hq_det.models.deformable_detr.hq_deformable", "HQDeformableDETR"),
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class HQModelEntry:
    """HQ-Det 子模型注册项。"""

    key: str
    module_path: str
    class_name: str


# 列表顺序 = 自动检测时的尝试顺序（常见类型可放前面以加快检测）
HQ_MODELS: Tuple[HQModelEntry, ...] = (
    HQModelEntry("dino", "hq_det.models.dino.hq_dino", "HQDINO"),
    HQModelEntry("dino2", "hq_det.models.dino2.hq_dino", "HQDINO"),
    HQModelEntry("rtdetr", "hq_det.models.rtdetr.hq_rtdetr", "HQRTDETR"),
    HQModelEntry("rtmdet", "hq_det.models.rtmdet", "HQRTMDET"),
    HQModelEntry("yolo", "hq_det.models.yolo", "HQYOLO"),
    HQModelEntry("lwdetr", "hq_det.models.lwdetr.hq_lwdetr", "HQLWDETR"),
    HQModelEntry("rfdetr", "hq_det.models.rfdetr.hq_rfdetr", "HQRFDETR"),
    HQModelEntry("codetr", "hq_det.models.codetr.hq_codetr", "HQCoDetr"),
)

HQ_MODEL_BY_KEY = {entry.key: entry for entry in HQ_MODELS}
HQ_MODEL_KEYS: List[str] = [entry.key for entry in HQ_MODELS]


def _import_model_class(entry: HQModelEntry):
    module = import_module(entry.module_path)
    return getattr(module, entry.class_name)


def instantiate_hq_model(checkpoint_path: str, model_type: str):
    """按注册表加载 HQ-Det 模型实例（未 eval / 未 to device）。"""
    entry = HQ_MODEL_BY_KEY.get(model_type)
    if entry is None:
        supported = ", ".join(HQ_MODEL_KEYS)
        raise ValueError(f"不支持的 HQ-Det 模型类型: {model_type}. 可选: {supported}")
    model_class = _import_model_class(entry)
    return model_class(model=checkpoint_path)


def try_load_hq_model(checkpoint_path: str, model_type: str) -> Tuple[bool, Optional[str]]:
    """尝试加载模型以验证 checkpoint 是否匹配该类型。返回 (成功, 错误信息)。"""
    try:
        model = instantiate_hq_model(checkpoint_path, model_type)
        del model
        return True, None
    except Exception as e:
        return False, str(e)
