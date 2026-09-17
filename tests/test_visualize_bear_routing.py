"""验证单图 BEAR 路由行为可视化的提取与保存。"""

import json
import math
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from visualize_bear_routing import (infer_routing_behavior,
                                    resolve_input_images,
                                    save_routing_visualization,
                                    unit_map_to_grayscale,
                                    visualize_input_set)


class FakeBEARModel(nn.Module):
    """用可控路由网格模拟 predicted+BEPR 的 BEAR-BioIR。"""

    def __init__(self, routing_mode='bepr', structure_source='predicted'):
        """初始化可切换路由与结构来源的测试模型。"""
        super().__init__()
        self.routing_mode = routing_mode
        self.structure_source = structure_source
        self.structure_predictor = (
            nn.Identity() if structure_source == 'predicted' else None)
        self.pad_multiple = 4
        self.route_patch = 2

    def forward(self, image, return_aux=False):
        """返回原图和覆盖内部补边尺寸的三尺度可控路由量。"""
        if not return_aux:
            return image
        height, width = image.shape[-2:]
        padded_height = math.ceil(height / self.pad_multiple) * self.pad_multiple
        padded_width = math.ceil(width / self.pad_multiple) * self.pad_multiple
        grid_height = padded_height // self.route_patch
        grid_width = padded_width // self.route_patch
        grid = torch.linspace(
            0.0, 1.0, grid_height * grid_width,
            device=image.device, dtype=image.dtype).view(
                1, 1, grid_height, grid_width)
        structure = torch.linspace(
            0.0, 1.0, padded_width,
            device=image.device, dtype=image.dtype).view(
                1, 1, 1, padded_width)
        structure = structure.expand(1, 1, padded_height, padded_width)
        return image, {
            'budget': image.new_tensor([[0.4]]),
            'structure': structure,
            'scopes': [grid, grid * 0.8, grid * 0.6],
            'reliabilities': [1.0 - grid, 1.0 - grid * 0.8,
                              1.0 - grid * 0.6],
        }


class TestBEARRoutingVisualization(unittest.TestCase):
    """检查尺寸对齐、固定标尺、错误配置拒绝与文件落盘。"""

    def setUp(self):
        """构造不能被内部补边倍数整除的 RGB 输入。"""
        self.device = torch.device('cpu')
        self.low = np.zeros((5, 7, 3), dtype=np.uint8)

    def test_inference_exports_three_aligned_route_scales(self):
        """验证三尺度 A/R 最近邻展开后均裁回原图尺寸。"""
        result = infer_routing_behavior(
            FakeBEARModel(), self.low, self.device)

        self.assertEqual(result['restored'].shape, self.low.shape)
        self.assertEqual(result['structure'].shape, self.low.shape[:2])
        self.assertEqual(result['padded_size'], (8, 8))
        self.assertEqual(result['route_grids']['A_1'].shape, (4, 4))
        self.assertEqual(tuple(result['dense_maps']),
                         ('A_1', 'A_2', 'A_3', 'R_1', 'R_2', 'R_3'))
        for value in result['dense_maps'].values():
            self.assertEqual(value.shape, self.low.shape[:2])
            self.assertGreaterEqual(float(value.min()), 0.0)
            self.assertLessEqual(float(value.max()), 1.0)
        # route_patch=2，因此同一区域内两个横向像素必须保持同值。
        self.assertEqual(
            float(result['dense_maps']['A_1'][0, 0]),
            float(result['dense_maps']['A_1'][0, 1]))
        self.assertNotEqual(
            float(result['dense_maps']['A_1'][0, 1]),
            float(result['dense_maps']['A_1'][0, 2]))

    def test_non_predictive_configurations_are_rejected(self):
        """验证常量路由和固定结构不会被误标成训练后的预测图。"""
        with self.assertRaisesRegex(ValueError, 'routing_mode=bepr'):
            infer_routing_behavior(
                FakeBEARModel(routing_mode='constant'),
                self.low, self.device)
        with self.assertRaisesRegex(ValueError, 'structure_source=predicted'):
            infer_routing_behavior(
                FakeBEARModel(structure_source='fixed'),
                self.low, self.device)

    def test_fixed_unit_scale_does_not_stretch_constant_map(self):
        """验证保存值使用固定 [0,1] 标尺而非逐图 min-max 拉伸。"""
        value = np.full((2, 3), 0.25, dtype=np.float32)
        image = unit_map_to_grayscale(value)

        self.assertEqual(image.dtype, np.uint8)
        self.assertTrue(np.all(image == 64))

    def test_save_writes_overview_raw_arrays_and_metadata(self):
        """验证论文观察图、原始网格和复现实验元数据完整落盘。"""
        result = infer_routing_behavior(
            FakeBEARModel(), self.low, self.device)
        with tempfile.TemporaryDirectory() as directory:
            output_directory = Path(directory) / 'routing'
            save_routing_visualization(
                output_directory, self.low, result,
                Path('low.png'), Path('option.yml'), Path('best_G.pth'))
            expected_files = [
                'input_low.png', 'enhanced.png', 'S_hat.png',
                'A_1.png', 'A_2.png', 'A_3.png',
                'R_1.png', 'R_2.png', 'R_3.png',
                'routing_overview.png', 'routing_arrays.npz',
                'routing_statistics.csv', 'metadata.json',
            ]
            for filename in expected_files:
                self.assertTrue((output_directory / filename).is_file())
            overview = cv2.imread(
                str(output_directory / 'routing_overview.png'),
                cv2.IMREAD_COLOR)
            with (output_directory / 'metadata.json').open(
                    'r', encoding='utf-8') as file:
                metadata = json.load(file)
            with np.load(output_directory / 'routing_arrays.npz') as arrays:
                array_names = set(arrays.files)

        self.assertEqual(overview.shape[1], self.low.shape[1] * 3)
        self.assertEqual(metadata['map_range'], [0.0, 1.0])
        self.assertFalse(metadata['per_image_normalization'])
        self.assertIn('A_1_grid', array_names)
        self.assertIn('S_hat', array_names)

    def test_yaml_validation_directory_exports_all_images(self):
        """验证省略 --image 时递归处理 YAML 指定的完整测试集。"""
        with tempfile.TemporaryDirectory() as directory:
            temporary_root = Path(directory)
            input_root = temporary_root / 'eval15' / 'low'
            first_path = input_root / 'scene_a' / '1.png'
            second_path = input_root / 'scene_b' / '1.png'
            first_path.parent.mkdir(parents=True)
            second_path.parent.mkdir(parents=True)
            cv2.imwrite(str(first_path), np.zeros((5, 7, 3), dtype=np.uint8))
            cv2.imwrite(str(second_path), np.full(
                (5, 7, 3), 64, dtype=np.uint8))
            opt = {
                'datasets': {
                    'val': {
                        'dataroot_lq': str(input_root),
                    },
                },
            }
            resolved_root, image_entries = resolve_input_images(opt)
            output_root = temporary_root / 'routing_output'
            rows = visualize_input_set(
                FakeBEARModel(), image_entries, output_root, self.device,
                Path('option.yml'), Path('best_G.pth'))

            self.assertEqual(resolved_root, input_root)
            self.assertEqual(len(rows), 2)
            self.assertTrue(
                (output_root / 'scene_a' / '1' /
                 'routing_overview.png').is_file())
            self.assertTrue(
                (output_root / 'scene_b' / '1' /
                 'routing_overview.png').is_file())
            summary_path = output_root / 'routing_dataset_statistics.csv'
            with summary_path.open('r', encoding='utf-8') as file:
                summary_lines = file.read().splitlines()

        self.assertEqual(len(summary_lines), 3)
        self.assertIn('A_1_mean', summary_lines[0])

    def test_empty_validation_directory_is_rejected(self):
        """验证空测试目录会明确报错而不是静默生成空统计表。"""
        with tempfile.TemporaryDirectory() as directory:
            opt = {
                'datasets': {
                    'val': {
                        'dataroot_lq': directory,
                    },
                },
            }
            with self.assertRaisesRegex(RuntimeError, 'No test images'):
                resolve_input_images(opt)


if __name__ == '__main__':
    unittest.main()
