#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""HQ 模型注册表与 predict 入口的单元测试（不依赖真实 checkpoint）。"""

import argparse
import importlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

PREDICT_DIR = Path(__file__).resolve().parent
ROOT_DIR = PREDICT_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


class TestHQModelRegistry(unittest.TestCase):
    def test_registry_keys_unique(self):
        from src.predict.hq_model_registry import HQ_MODELS, HQ_MODEL_BY_KEY, HQ_MODEL_KEYS

        keys = [e.key for e in HQ_MODELS]
        self.assertEqual(len(keys), len(set(keys)), f"重复 key: {keys}")
        self.assertEqual(HQ_MODEL_KEYS, keys)
        self.assertEqual(set(HQ_MODEL_BY_KEY), set(keys))

    def test_unknown_model_type_raises(self):
        from src.predict.hq_model_registry import instantiate_hq_model

        with self.assertRaises(ValueError) as ctx:
            instantiate_hq_model("/fake/model.pth", "not_a_model")
        self.assertIn("not_a_model", str(ctx.exception))
        self.assertIn("dino", str(ctx.exception))

    @patch("src.predict.hq_model_registry._import_model_class")
    def test_instantiate_hq_model(self, mock_import):
        from src.predict.hq_model_registry import instantiate_hq_model

        mock_cls = MagicMock(return_value=MagicMock())
        mock_import.return_value = mock_cls

        model = instantiate_hq_model("/fake/rtdetr.pth", "rtdetr")
        mock_import.assert_called_once()
        mock_cls.assert_called_once_with(model="/fake/rtdetr.pth")
        self.assertIsNotNone(model)

    @patch("src.predict.hq_model_registry.instantiate_hq_model")
    def test_try_load_hq_model_success(self, mock_instantiate):
        from src.predict.hq_model_registry import try_load_hq_model

        mock_instantiate.return_value = MagicMock()
        ok, err = try_load_hq_model("/fake.pth", "yolo")
        self.assertTrue(ok)
        self.assertIsNone(err)

    @patch("src.predict.hq_model_registry.instantiate_hq_model")
    def test_try_load_hq_model_failure(self, mock_instantiate):
        from src.predict.hq_model_registry import try_load_hq_model

        mock_instantiate.side_effect = RuntimeError("bad checkpoint")
        ok, err = try_load_hq_model("/fake.pth", "yolo")
        self.assertFalse(ok)
        self.assertIn("bad checkpoint", err)


class TestPredictImports(unittest.TestCase):
    def test_package_import_predict(self):
        from src.predict.predict import detect_model_type, detect_hq_model_type
        from src.predict.hq_model_registry import HQ_MODEL_KEYS

        self.assertTrue(callable(detect_model_type))
        self.assertTrue(callable(detect_hq_model_type))
        self.assertEqual(len(HQ_MODEL_KEYS), 8)

    def test_script_import_predict_hq_det(self):
        """模拟 python path/to/predict_hq_det.py 运行（脚本目录在 sys.path）。"""
        import subprocess

        r = subprocess.run(
            [sys.executable, str(PREDICT_DIR / "predict_hq_det.py"), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("rtdetr", r.stdout)
        self.assertIn("codetr", r.stdout)


class TestDetectModelType(unittest.TestCase):
    def test_missing_checkpoint(self):
        from src.predict.predict import detect_model_type, detect_hq_model_type

        self.assertIsNone(detect_model_type("/nonexistent/model.pth")[0])
        self.assertIsNone(detect_hq_model_type("/nonexistent/model.pth"))

    @patch("src.predict.predict.try_load_dino")
    @patch("src.predict.predict.try_load_hq_model")
    def test_detect_dino_first(self, mock_hq, mock_dino):
        from src.predict.predict import detect_model_type

        with tempfile.NamedTemporaryFile(suffix=".pth") as f:
            mock_dino.return_value = (True, None)
            mock_hq.return_value = (False, "skip")

            model_type, sub = detect_model_type(f.name)
            self.assertEqual(model_type, "dino")
            self.assertIsNone(sub)
            mock_dino.assert_called_once()
            mock_hq.assert_not_called()

    @patch("src.predict.predict.try_load_dino")
    @patch("src.predict.predict.try_load_hq_model")
    def test_detect_hq_rtdetr(self, mock_hq, mock_dino):
        from src.predict.predict import detect_model_type

        def hq_side_effect(path, model_type):
            return model_type == "rtdetr", None

        mock_dino.return_value = (False, "not dino")
        mock_hq.side_effect = hq_side_effect

        with tempfile.NamedTemporaryFile(suffix=".pth") as f:
            model_type, sub = detect_model_type(f.name)
            self.assertEqual(model_type, "hq_det")
            self.assertEqual(sub, "rtdetr")

    @patch("src.predict.predict.try_load_hq_model")
    def test_detect_hq_model_type_only(self, mock_hq):
        from src.predict.predict import detect_hq_model_type

        mock_hq.side_effect = lambda path, t: (t == "codetr", None)

        with tempfile.NamedTemporaryFile(suffix=".pth") as f:
            self.assertEqual(detect_hq_model_type(f.name), "codetr")


class TestHQDetOptionalIntegration(unittest.TestCase):
    """hq_det 已安装时，校验注册表中的模块路径可导入。"""

    def test_registry_modules_importable(self):
        try:
            import hq_det  # noqa: F401
        except ImportError:
            self.skipTest("hq_det 未安装，跳过模块导入测试")

        from src.predict.hq_model_registry import HQ_MODELS, _import_model_class

        for entry in HQ_MODELS:
            with self.subTest(model=entry.key):
                cls = _import_model_class(entry)
                self.assertTrue(callable(cls), f"{entry.key} 的类不可调用")


class TestArgparseChoices(unittest.TestCase):
    def test_hq_model_type_choices_match_registry(self):
        from src.predict.hq_model_registry import HQ_MODEL_KEYS
        from src.predict import predict as predict_mod

        parser = argparse.ArgumentParser()
        predict_mod.main.__globals__["argparse"].ArgumentParser = argparse.ArgumentParser
        # 从 predict.py 源码解析 --hq-model-type choices
        import inspect

        src = inspect.getsource(predict_mod.main)
        self.assertIn("choices=HQ_MODEL_KEYS", src)

        from src.predict import predict_hq_det as hq_mod

        hq_src = inspect.getsource(hq_mod.main)
        self.assertIn("choices=HQ_MODEL_KEYS", hq_src)
        self.assertEqual(len(HQ_MODEL_KEYS), 8)


class TestLoadHqModel(unittest.TestCase):
    @patch("src.predict.predict_hq_det.instantiate_hq_model")
    def test_load_hq_model_eval_and_device(self, mock_instantiate):
        import torch
        from src.predict.predict_hq_det import load_hq_model

        mock_model = MagicMock()
        mock_model.model.parameters.return_value = iter([torch.tensor([1.0])])
        mock_instantiate.return_value = mock_model

        with patch("torch.cuda.is_available", return_value=False):
            load_hq_model("/fake.pth", "rtdetr", device="cpu")

        mock_instantiate.assert_called_once_with("/fake.pth", "rtdetr")
        mock_model.eval.assert_called_once()
        mock_model.to.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
