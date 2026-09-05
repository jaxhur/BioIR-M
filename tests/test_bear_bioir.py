"""验证 BEAR-BioIR v2 的路由、稠密结构与反向传播闭环。"""

import unittest
from collections import OrderedDict

import torch
import torch.nn.functional as F

from basicsr.models.archs.BEAR_BioIR_arch import (
    BEARBioIR, DenseStructurePredictor, FixedCIConvW,
    StructureChannelRefinement)
from basicsr.models.bear_bioir_model import BEARBioIRModel


class FakeTensorBoardWriter:
    """记录 ``add_image`` 调用，避免单元测试依赖真实事件文件。"""

    def __init__(self):
        self.images = []

    def add_image(self, tag, image, global_step, dataformats):
        """保存一次图片记录调用的标签、形状与迭代信息。"""
        self.images.append((tag, image.clone(), global_step, dataformats))


class TestBEARBioIR(unittest.TestCase):
    """在 CPU 合成 Tensor 上检查方案 3 v2 的关键几何与梯度连接。"""

    def test_route_outputs_and_backward(self):
        """验证三尺度路由、结构目标和 12 个 SARI 的反向路径。"""
        torch.manual_seed(0)
        model = BEARBioIR(dim=8, attention_heads=4)
        low = torch.rand(1, 3, 64, 64, requires_grad=True)
        gt = torch.rand(1, 3, 64, 64)
        restored, aux = model(low, return_aux=True)
        targets = model.build_routing_targets(low.detach(), gt)

        self.assertEqual(restored.shape, low.shape)
        self.assertEqual(aux['budget'].shape, (1, 1))
        self.assertEqual(aux['structure'].shape, (1, 1, 64, 64))
        self.assertEqual(targets['budget'].shape, (1, 1))
        self.assertEqual(targets['reliability'].shape, (1, 1, 4, 4))
        self.assertEqual(targets['structure'].shape, (1, 1, 64, 64))
        self.assertEqual(len(aux['scopes']), 3)
        self.assertEqual(len(aux['reliabilities']), 3)
        self.assertIsNone(
            model.encoder_level1[0].interaction.structure_refinement)
        self.assertTrue(all(
            block.interaction.structure_refinement is not None
            for block in model.refinement))
        for scope, reliability in zip(aux['scopes'], aux['reliabilities']):
            self.assertEqual(scope.shape, (1, 1, 4, 4))
            self.assertEqual(reliability.shape, (1, 1, 4, 4))

        loss = F.l1_loss(restored, gt)
        loss = loss + F.smooth_l1_loss(aux['budget'], targets['budget'])
        loss = loss + sum(
            F.smooth_l1_loss(reliability, targets['reliability'])
            for reliability in aux['reliabilities'])
        loss = loss + F.smooth_l1_loss(
            aux['structure'], targets['structure'])
        loss.backward()
        self.assertIsNotNone(model.output.weight.grad)
        self.assertTrue(torch.isfinite(model.output.weight.grad).all())
        structure_grad = model.structure_predictor.input_projection[0].weight.grad
        self.assertIsNotNone(structure_grad)
        self.assertTrue(torch.isfinite(structure_grad).all())

    def test_fixed_ciconv_has_no_trainable_path(self):
        """验证 CIConv-W 核固定、形状正确且不会向输入回传梯度。"""
        ciconv = FixedCIConvW()
        image = torch.rand(2, 3, 31, 47, requires_grad=True)
        prior = ciconv(image)
        self.assertEqual(ciconv.kernels.shape, (3, 1, 15, 15))
        self.assertEqual(sum(parameter.numel()
                             for parameter in ciconv.parameters()), 0)
        self.assertEqual(prior.shape, (2, 1, 31, 47))
        self.assertFalse(prior.requires_grad)
        self.assertTrue(torch.isfinite(prior).all())

    def test_soft_sobel_target_and_structure_refinement(self):
        """验证 GT 软结构目标及 refinement 通道交互的梯度连接。"""
        predictor = DenseStructurePredictor(channels=8, target_scale=0.1)
        flat_gt = torch.zeros(1, 3, 16, 16)
        edge_gt = flat_gt.clone()
        edge_gt[:, :, :, 8:] = 1.0
        flat_target = predictor.build_target(flat_gt)
        edge_target = predictor.build_target(edge_gt)
        self.assertEqual(edge_target.shape, (1, 1, 16, 16))
        self.assertLess(flat_target.max().item(), 1e-5)
        self.assertGreater(edge_target.max().item(), 0.9)

        refiner = StructureChannelRefinement(dim=8)
        detail = torch.rand(1, 8, 16, 16, requires_grad=True)
        structure = torch.rand(1, 1, 16, 16, requires_grad=True)
        refined = refiner(detail, structure)
        self.assertEqual(refined.shape, detail.shape)
        refined.mean().backward()
        self.assertIsNotNone(detail.grad)
        self.assertIsNotNone(structure.grad)
        self.assertTrue(torch.isfinite(structure.grad).all())

    def test_non_multiple_input_is_cropped_back(self):
        """验证完整图像的 64 倍数入口补边不会改变输出有效区域尺寸。"""
        model = BEARBioIR(
            dim=8, num_blocks=[0, 0, 0], num_refinement_blocks=0,
            attention_heads=4).eval()
        low = torch.rand(1, 3, 65, 67)
        with torch.no_grad():
            restored = model(low)
        self.assertEqual(restored.shape, low.shape)

    def test_tensorboard_validation_visuals(self):
        """验证 TensorBoard 记录五类图像并裁回原始空间尺寸。"""
        wrapper = object.__new__(BEARBioIRModel)
        wrapper.opt = {
            'val': {
                'val_freq': 1000,
                'tensorboard_images': {
                    'enabled': True,
                    'interval': 1000,
                    'max_samples': 1,
                    'include_structure_target': True,
                },
            },
        }
        wrapper.net_g = BEARBioIR(
            dim=8, num_blocks=[0, 0, 0], num_refinement_blocks=0,
            attention_heads=4)
        wrapper.lq = torch.rand(1, 3, 65, 67)
        wrapper.gt = torch.rand(1, 3, 65, 67)
        wrapper._collect_tensorboard_visuals = True
        wrapper.test()

        self.assertEqual(wrapper.output.shape, wrapper.lq.shape)
        self.assertEqual(
            wrapper.structure_prediction.shape, (1, 1, 65, 67))
        visuals = OrderedDict([
            ('lq', wrapper.lq.detach().cpu()),
            ('result', wrapper.output.detach().cpu()),
            ('gt', wrapper.gt.detach().cpu()),
        ])
        writer = FakeTensorBoardWriter()
        wrapper._log_validation_images_to_tensorboard(
            writer, 1000, 0, 'LOLv1-test', '1', visuals)

        expected_tags = [
            'validation/LOLv1-test/1/01_input_lq',
            'validation/LOLv1-test/1/02_prediction',
            'validation/LOLv1-test/1/03_ground_truth',
            'validation/LOLv1-test/1/04_structure_prediction',
            'validation/LOLv1-test/1/05_structure_target',
        ]
        self.assertEqual(
            [record[0] for record in writer.images], expected_tags)
        self.assertTrue(all(record[2] == 1000 for record in writer.images))
        self.assertTrue(all(record[3] == 'CHW' for record in writer.images))
        self.assertEqual(writer.images[3][1].shape, (1, 65, 67))
        self.assertEqual(writer.images[4][1].shape, (1, 65, 67))

        wrapper._log_validation_images_to_tensorboard(
            writer, 1500, 0, 'LOLv1-test', '1', visuals)
        wrapper._log_validation_images_to_tensorboard(
            writer, 2000, 1, 'LOLv1-test', '2', visuals)
        self.assertEqual(len(writer.images), 5)


if __name__ == '__main__':
    unittest.main()
