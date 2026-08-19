"""使用原方案公式批量生成 LOL 数据集的 ``A*`` 与 ``R*``。

原版 ``A*``：

``A* = clamp(max(Y_gt - Y_lq, 0) / (Y_gt + epsilon_A), 0, 1)``

``R*`` 严格复用 DRR-BioIR v0 中的 3×3 Gaussian、log-Sobel、结构存在度、
梯度幅值一致性与方向一致性定义。

示例：

``python scripts/visualize_drr_targets_original.py --datasets LOL-v1 --split train``

``python scripts/visualize_drr_targets_original.py --datasets all --split test --max-visualizations 0``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from drr_target_visualization_common import (DATASET_SPECS,
                                             DemandBuilder,
                                             add_common_arguments,
                                             run_selected_datasets)


def build_original_demand_target(
    luminance_lq: torch.Tensor,
    luminance_gt: torch.Tensor,
    demand_epsilon: float,
) -> torch.Tensor:
    """按原方案的相对正向曝光缺口构造 ``A*``。

    Args:
        luminance_lq: LQ 亮度，形状 ``B×1×H×W``。
        luminance_gt: GT 亮度，形状同 LQ。
        demand_epsilon: 分母稳定常数，原方案默认 0.05。

    Returns:
        范围为 ``[0,1]`` 的原版 ``A*``。
    """

    positive_gap = (luminance_gt - luminance_lq).clamp_min(0.0)
    demand = positive_gap / (luminance_gt + demand_epsilon)
    return demand.clamp(0.0, 1.0)


def parse_args() -> argparse.Namespace:
    """解析原公式可视化脚本参数。

    Returns:
        包含数据、输出、统计与公式参数的命令行对象。
    """

    parser = argparse.ArgumentParser(
        description="按原 DRR 定义输出 LOL-v1/LOL-v2 的 A*、R* 与分布统计。")
    add_common_arguments(
        parser,
        default_output_root=Path(
            "analysis_artifacts/drr_target_visualization/original"),
    )
    parser.add_argument(
        "--demand-epsilon",
        type=float,
        default=0.05,
        help="原 A* 公式中 Y_gt 分母的稳定常数，默认 0.05。",
    )
    args = parser.parse_args()
    if args.demand_epsilon <= 0.0:
        raise ValueError("--demand-epsilon 必须大于 0。")
    return args


def main() -> None:
    """为用户选择的数据集执行原版 A*/R* 可视化和统计。"""

    args = parse_args()
    builders: dict[str, DemandBuilder] = {}
    parameters: dict[str, dict[str, float | str]] = {}
    for spec in DATASET_SPECS:
        # 默认参数通过闭包固定，避免循环变量或后续配置变化影响已注册 builder。
        epsilon = float(args.demand_epsilon)

        def builder(
            luminance_lq: torch.Tensor,
            luminance_gt: torch.Tensor,
            dataset_name: str,
            demand_epsilon: float = epsilon,
        ) -> torch.Tensor:
            """转发到原公式；dataset_name 仅用于统一两版入口签名。"""

            del dataset_name
            return build_original_demand_target(
                luminance_lq, luminance_gt, demand_epsilon)

        builders[spec.name] = builder
        parameters[spec.name] = {
            "definition": "relu(Y_gt-Y_lq)/(Y_gt+demand_epsilon)",
            "demand_epsilon": epsilon,
        }
    run_selected_datasets(
        args=args,
        formula_name="original_relative_gap",
        demand_builders=builders,
        formula_parameters=parameters,
    )


if __name__ == "__main__":
    main()
