"""验证 BEAR-BioIR 的路由形状、补边和反向传播最小闭环。"""

import unittest

import torch
import torch.nn.functional as F

from basicsr.models.archs.BEAR_BioIR_arch import BEARBioIR


class TestBEARBioIR(unittest.TestCase):
    """在 CPU 合成 Tensor 上检查方案 3 的关键几何与梯度连接。"""

    def test_route_outputs_and_backward(self):
        """验证三尺度路由图形状、代理目标和 12 个 SARI 的反向路径。"""
        torch.manual_seed(0)
        model = BEARBioIR(dim=8, attention_heads=4)
        low = torch.rand(1, 3, 64, 64, requires_grad=True)
        gt = torch.rand(1, 3, 64, 64)
        restored, aux = model(low, return_aux=True)
        targets = model.build_routing_targets(low.detach(), gt)

        self.assertEqual(restored.shape, low.shape)
        self.assertEqual(aux['budget'].shape, (1, 1))
        self.assertEqual(targets['budget'].shape, (1, 1))
        self.assertEqual(targets['reliability'].shape, (1, 1, 4, 4))
        self.assertEqual(len(aux['scopes']), 3)
        self.assertEqual(len(aux['reliabilities']), 3)
        for scope, reliability in zip(aux['scopes'], aux['reliabilities']):
            self.assertEqual(scope.shape, (1, 1, 4, 4))
            self.assertEqual(reliability.shape, (1, 1, 4, 4))

        loss = F.l1_loss(restored, gt)
        loss = loss + F.smooth_l1_loss(aux['budget'], targets['budget'])
        loss = loss + sum(
            F.smooth_l1_loss(reliability, targets['reliability'])
            for reliability in aux['reliabilities'])
        loss.backward()
        self.assertIsNotNone(model.output.weight.grad)
        self.assertTrue(torch.isfinite(model.output.weight.grad).all())

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
