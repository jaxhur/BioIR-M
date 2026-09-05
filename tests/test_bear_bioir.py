"""验证 BEAR-BioIR v2 的路由、稠密结构与反向传播闭环。"""

import unittest

import torch
import torch.nn.functional as F

from basicsr.models.archs.BEAR_BioIR_arch import (
    BEARBioIR, DenseStructurePredictor, FixedCIConvW,
    StructureChannelRefinement)


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


if __name__ == '__main__':
    unittest.main()
