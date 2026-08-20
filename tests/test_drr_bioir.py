"""DRR-BioIR v0 的 CPU 结构与数值单元测试。"""

import unittest

import torch

from basicsr.models.archs.drr_bioir_arch import (DRRBioIR,
                                                  aggregate_detail_gates,
                                                  build_drr_targets,
                                                  build_gaussian_smoothed_absolute_demand_target,
                                                  build_relative_demand_target,
                                                  calculate_luminance,
                                                  gaussian_smooth_luminance)


class DRRBioIRTest(unittest.TestCase):
    """验证补边、先验、ADRI 与可选 RA-GDFN 的最小可运行行为。"""

    @staticmethod
    def _build_network(use_ra_gdfn=False):
        """构建轻量测试网络，减少 CPU 测试耗时。"""
        return DRRBioIR(dim=8,
                         num_blocks=(1, 1, 1),
                         num_refinement_blocks=1,
                         ffn_expansion_factor=2,
                         prior_hidden_channels=4,
                         use_ra_gdfn=use_ra_gdfn)

    def test_default_forward_crops_to_original_shape(self):
        """非 32 倍数输入在内部补边后仍应返回原始空间尺寸。"""
        network = self._build_network().eval()
        image = torch.rand(1, 3, 65, 97)
        with torch.no_grad():
            restored = network(image)
        self.assertEqual(restored.shape, image.shape)
        self.assertTrue(torch.isfinite(restored).all())

    def test_auxiliary_maps_match_input_and_stay_bounded(self):
        """训练辅助输出必须与原图对齐，且 Sigmoid 后位于合法范围。"""
        network = self._build_network().eval()
        image = torch.rand(1, 3, 64, 96)
        with torch.no_grad():
            restored, auxiliary = network(image, return_aux=True)
        self.assertEqual(restored.shape, image.shape)
        self.assertEqual(auxiliary['demand'].shape, (1, 1, 64, 96))
        self.assertEqual(auxiliary['reliability'].shape, (1, 1, 64, 96))
        for prior in auxiliary.values():
            self.assertGreaterEqual(float(prior.min()), 0.0)
            self.assertLessEqual(float(prior.max()), 1.0)

    def test_shared_detail_gate_grid_has_expected_token_count(self):
        """A/R 的 16×16 聚合应生成三尺度共享的行优先 token 网格。"""
        demand = torch.rand(2, 1, 96, 128)
        reliability = torch.rand_like(demand)
        demand_tokens, reliability_tokens = aggregate_detail_gates(
            demand, reliability)
        self.assertEqual(demand_tokens.shape, (2, 48, 1))
        self.assertEqual(reliability_tokens.shape, (2, 48, 1))
        self.assertTrue(torch.all((demand_tokens >= 0.0) & (demand_tokens <= 1.0)))
        self.assertTrue(
            torch.all((reliability_tokens >= 0.0) & (reliability_tokens <= 1.0)))

    def test_targets_are_finite_and_bounded(self):
        """配对监督生成的 A*、R*、E_g 必须数值稳定且位于零一范围。"""
        low_quality = torch.rand(2, 3, 64, 96)
        ground_truth = torch.rand_like(low_quality)
        targets = build_drr_targets(low_quality, ground_truth)
        for target in targets.values():
            self.assertEqual(target.shape, (2, 1, 64, 96))
            self.assertTrue(torch.isfinite(target).all())
            self.assertTrue(torch.all((target >= 0.0) & (target <= 1.0)))

    def test_relative_demand_formula_is_preserved(self):
        """新增 A* 模式后，原相对亮度缺口公式必须保持不变。"""
        luminance_lq = torch.tensor([[[[0.1, 0.4, 0.8]]]])
        luminance_gt = torch.tensor([[[[0.5, 0.2, 0.8]]]])
        actual = build_relative_demand_target(luminance_lq,
                                              luminance_gt,
                                              demand_epsilon=0.05)
        expected = torch.tensor([[[[0.4 / 0.55, 0.0, 0.0]]]])
        torch.testing.assert_close(actual, expected)

    def test_gaussian_smoothed_absolute_demand_matches_definition(self):
        """新版 A* 必须严格等于低通亮度正缺口除以固定 tau_A。"""
        low_quality = torch.rand(1, 3, 64, 96)
        ground_truth = torch.rand_like(low_quality)
        luminance_lq = calculate_luminance(low_quality)
        luminance_gt = calculate_luminance(ground_truth)
        expected = (
            (gaussian_smooth_luminance(luminance_gt, 15, 3.0) -
             gaussian_smooth_luminance(luminance_lq, 15, 3.0)).clamp_min(0.0) /
            0.5).clamp(0.0, 1.0)
        actual = build_gaussian_smoothed_absolute_demand_target(
            luminance_lq,
            luminance_gt,
            kernel_size=15,
            sigma=3.0,
            demand_tau=0.5)
        torch.testing.assert_close(actual, expected)

        targets = build_drr_targets(
            low_quality,
            ground_truth,
            demand_target_type='gaussian_smoothed_absolute_gap',
            demand_gaussian_kernel_size=15,
            demand_gaussian_sigma=3.0,
            demand_tau=0.5)
        torch.testing.assert_close(targets['demand'], expected)

    def test_plain_and_ra_gdfn_support_backward(self):
        """普通 GDFN 与可选 RA-GDFN 均应完成可微前向与反向传播。"""
        for use_ra_gdfn in (False, True):
            network = self._build_network(use_ra_gdfn=use_ra_gdfn)
            image = torch.rand(1, 3, 64, 64)
            restored, auxiliary = network(image, return_aux=True)
            loss = restored.mean() + auxiliary['demand'].mean() + auxiliary[
                'reliability'].mean()
            loss.backward()
            self.assertTrue(
                any(parameter.grad is not None for parameter in network.parameters()))


if __name__ == '__main__':
    unittest.main()
