"""验证两版 DRR 目标公式、区域聚合和统计口径。"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIRECTORY = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from drr_target_visualization_common import (  # noqa: E402
    ReliabilityParameters,
    average_pool_map,
    build_reliability_target,
    gaussian_smooth_luminance,
    map_statistics,
    topk_pool_map,
)
from visualize_drr_targets_original import (  # noqa: E402
    build_original_demand_target,
)
from visualize_drr_targets_revised import (  # noqa: E402
    build_revised_demand_target,
)


class DRRTargetVisualizationTest(unittest.TestCase):
    """覆盖新旧 A*、原 R*、池化尺寸与分布统计。"""

    def test_original_demand_matches_documented_formula(self):
        """原版函数必须逐元素等于文档中的相对亮度缺口公式。"""

        luminance_lq = torch.tensor([[[[0.1, 0.4, 0.8]]]])
        luminance_gt = torch.tensor([[[[0.5, 0.2, 0.8]]]])
        actual = build_original_demand_target(
            luminance_lq, luminance_gt, demand_epsilon=0.05)
        expected = torch.tensor([[[[0.4 / 0.55, 0.0, 0.0]]]])
        torch.testing.assert_close(actual, expected)

    def test_gaussian_smoothing_preserves_constant_luminance(self):
        """归一化固定低通不能改变常量亮度场。"""

        luminance = torch.full((1, 1, 31, 47), 0.37)
        smoothed = gaussian_smooth_luminance(
            luminance, kernel_size=15, sigma=3.0)
        torch.testing.assert_close(
            smoothed,
            luminance,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_revised_demand_keeps_more_spatial_range(self):
        """乘性曝光样例中，新版绝对缺口应避免旧版接近常数。"""

        horizontal = torch.linspace(0.05, 1.0, 96).view(1, 1, 1, 96)
        luminance_gt = horizontal.expand(1, 1, 64, 96).contiguous()
        luminance_lq = 0.20 * luminance_gt
        original = build_original_demand_target(
            luminance_lq, luminance_gt, demand_epsilon=0.05)
        revised = build_revised_demand_target(
            luminance_lq,
            luminance_gt,
            kernel_size=15,
            sigma=3.0,
            tau_a=0.65,
        )
        self.assertGreater(
            float(revised.std(unbiased=False)),
            2.0 * float(original.std(unbiased=False)),
        )
        self.assertTrue(torch.all((revised >= 0.0) & (revised <= 1.0)))

    def test_reliability_target_is_finite_and_bounded(self):
        """原 R* 在平坦区和结构区均应保持有限且位于零一范围。"""

        luminance_gt = torch.zeros(1, 1, 64, 96)
        luminance_gt[:, :, 16:48, 24:72] = 0.8
        luminance_lq = 0.25 * luminance_gt
        reliability, edge_presence = build_reliability_target(
            luminance_lq,
            luminance_gt,
            ReliabilityParameters(),
        )
        for target in (reliability, edge_presence):
            self.assertTrue(torch.isfinite(target).all())
            self.assertTrue(torch.all((target >= 0.0) & (target <= 1.0)))
        self.assertGreater(float(edge_presence.max()), 0.0)

    def test_region_aggregation_matches_lol_v1_native_grid(self):
        """400×600 整图按既定裁边口径应得到 25×37 区域网格。"""

        target = torch.rand(1, 1, 400, 600)
        averaged = average_pool_map(target, pool_size=16)
        topk = topk_pool_map(target, pool_size=16, topk=16)
        self.assertEqual(averaged.shape, (1, 1, 25, 37))
        self.assertEqual(topk.shape, (1, 1, 25, 37))
        self.assertTrue(torch.all(topk >= averaged))

    def test_map_statistics_uses_population_std_and_robust_span(self):
        """逐图统计应使用总体标准差与 q95-q05 鲁棒动态范围。"""

        values = np.arange(100, dtype=np.float32).reshape(10, 10) / 99.0
        statistics = map_statistics(values)
        self.assertAlmostEqual(statistics["mean"], 0.5, places=6)
        self.assertAlmostEqual(
            statistics["spatial_std"],
            float(values.astype(np.float64).std(ddof=0)),
            places=6,
        )
        self.assertAlmostEqual(
            statistics["q95_minus_q05"],
            float(np.quantile(values, 0.95) - np.quantile(values, 0.05)),
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
