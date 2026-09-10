"""验证 LOL 测试入口的结构预测图导出行为。"""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from test_lol import infer_one, save_grayscale


class FakePredictedModel(nn.Module):
    """用可控输出模拟带结构预测头的生成网络。"""

    def __init__(self, with_predictor=True):
        """初始化是否具备 predicted 结构头的测试模型。"""
        super().__init__()
        self.structure_predictor = nn.Identity() if with_predictor else None

    def forward(self, image, return_aux=False):
        """返回原图与由水平坐标构造的单通道结构图。"""
        if not return_aux:
            return image
        width = image.shape[-1]
        structure = torch.linspace(
            0.0, 1.0, width, device=image.device,
            dtype=image.dtype).view(1, 1, 1, width)
        structure = structure.expand(
            image.shape[0], 1, image.shape[-2], width)
        return image, {'structure': structure}


class TestStructurePredictionExport(unittest.TestCase):
    """检查原尺寸裁剪、固定值域转换和错误配置拒绝。"""

    def setUp(self):
        """构造一张不能被 8 整除的 RGB 图像。"""
        self.device = torch.device('cpu')
        self.image = np.zeros((5, 7, 3), dtype=np.uint8)

    def test_infer_one_exports_original_size_structure(self):
        """验证补边前向后的结构图精确裁回原尺寸。"""
        restored, structure = infer_one(
            FakePredictedModel(), self.image, self.device, factor=8,
            return_structure=True)

        self.assertEqual(restored.shape, self.image.shape)
        self.assertEqual(structure.shape, self.image.shape[:2])
        self.assertEqual(structure.dtype, np.uint8)
        self.assertEqual(int(structure[0, 0]), 0)
        self.assertEqual(int(structure[0, -1]), 219)

    def test_infer_one_keeps_previous_return_when_export_is_disabled(self):
        """验证默认推理仍只返回 RGB 增强图。"""
        restored = infer_one(
            FakePredictedModel(), self.image, self.device, factor=8)

        self.assertIsInstance(restored, np.ndarray)
        self.assertEqual(restored.shape, self.image.shape)

    def test_fixed_structure_model_is_rejected(self):
        """验证 fixed 结构配置不会被误标为预测头输出。"""
        with self.assertRaisesRegex(ValueError, 'structure_source=predicted'):
            infer_one(
                FakePredictedModel(with_predictor=False), self.image,
                self.device, factor=8, return_structure=True)

    def test_save_grayscale_preserves_uint8_values(self):
        """验证预测图以单通道 PNG 原值保存。"""
        image = np.array([[0, 64], [128, 255]], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / 'nested' / 'structure.png'
            save_grayscale(output_path, image)
            loaded = cv2.imread(str(output_path), cv2.IMREAD_UNCHANGED)

        np.testing.assert_array_equal(loaded, image)


if __name__ == '__main__':
    unittest.main()
