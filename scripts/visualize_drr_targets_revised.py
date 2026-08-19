"""使用低频绝对亮度缺口批量生成 LOL 数据集的新版 ``A*`` 与原版 ``R*``。

新版 ``A*``：

1. 用固定 Gaussian 低通分别平滑 LQ/GT 亮度；
2. 计算正向低频绝对亮度缺口 ``D_A``；
3. 仅在对应训练集上统计 ``D_A>0`` 的指定分位数 ``tau_A``；
4. 使用 ``A* = clamp(D_A / tau_A, 0, 1)``。

``R*`` 与原版脚本完全相同，便于只观察 ``A*`` 定义变化。

示例：

``python scripts/visualize_drr_targets_revised.py --datasets LOL-v1 --split train``

``python scripts/visualize_drr_targets_revised.py --datasets all --split test --max-visualizations 0``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from drr_target_visualization_common import (DATASET_SPECS,
                                             DemandBuilder,
                                             add_common_arguments,
                                             build_pairs,
                                             estimate_positive_quantile,
                                             gaussian_low_pass,
                                             resolve_device,
                                             run_selected_datasets,
                                             selected_specs)


def build_revised_demand_target(
    luminance_lq: torch.Tensor,
    luminance_gt: torch.Tensor,
    kernel_size: int,
    sigma: float,
    tau_a: float,
) -> torch.Tensor:
    """按低频绝对亮度缺口与固定训练集尺度构造新版 ``A*``。

    Args:
        luminance_lq: LQ 亮度，形状 ``B×1×H×W``。
        luminance_gt: GT 亮度，形状同 LQ。
        kernel_size: 固定 Gaussian 低通核边长。
        sigma: Gaussian 标准差，单位为像素。
        tau_a: 从训练集正缺口统计并冻结的归一化尺度。

    Returns:
        范围为 ``[0,1]`` 的新版 ``A*``。
    """

    smoothed_lq = gaussian_low_pass(luminance_lq, kernel_size, sigma)
    smoothed_gt = gaussian_low_pass(luminance_gt, kernel_size, sigma)
    absolute_deficit = (smoothed_gt - smoothed_lq).clamp_min(0.0)
    return (absolute_deficit / tau_a).clamp(0.0, 1.0)


def parse_args() -> argparse.Namespace:
    """解析新版公式可视化脚本参数。

    Returns:
        包含数据、输出、统计、低通和 tau_A 参数的命令行对象。
    """

    parser = argparse.ArgumentParser(
        description="按低频绝对亮度缺口输出 LOL-v1/LOL-v2 的 A*、R* 与统计。")
    add_common_arguments(
        parser,
        default_output_root=Path(
            "analysis_artifacts/drr_target_visualization/revised"),
    )
    parser.add_argument(
        "--low-pass-kernel-size",
        type=int,
        default=15,
        help="固定 Gaussian 低通核边长，必须为正奇数，默认 15。",
    )
    parser.add_argument(
        "--low-pass-sigma",
        type=float,
        default=3.0,
        help="固定 Gaussian 核标准差，单位为像素，默认 3.0。",
    )
    parser.add_argument(
        "--tau-quantile",
        type=float,
        default=0.90,
        help="训练集正低频缺口用于 tau_A 的分位数，默认 0.90。",
    )
    parser.add_argument(
        "--tau-histogram-bins",
        type=int,
        default=65536,
        help="流式估计 tau_A 的 [0,1] 直方图精度，默认 65536 bins。",
    )
    parser.add_argument(
        "--tau-a",
        type=float,
        default=None,
        help="手动覆盖 tau_A；默认对每套数据使用其完整训练集自动统计。",
    )
    parser.add_argument(
        "--max-tau-images",
        type=int,
        default=0,
        help="仅调试时限制 tau_A 训练图数量；0 表示完整训练集。",
    )
    args = parser.parse_args()
    if args.low_pass_kernel_size <= 0 or args.low_pass_kernel_size % 2 == 0:
        raise ValueError("--low-pass-kernel-size 必须为正奇数。")
    if args.low_pass_sigma <= 0.0:
        raise ValueError("--low-pass-sigma 必须大于 0。")
    if not 0.0 < args.tau_quantile < 1.0:
        raise ValueError("--tau-quantile 必须位于 (0,1)。")
    if args.tau_histogram_bins < 256:
        raise ValueError("--tau-histogram-bins 不能小于 256。")
    if args.tau_a is not None and args.tau_a <= 0.0:
        raise ValueError("--tau-a 必须大于 0。")
    if args.max_tau_images < 0:
        raise ValueError("--max-tau-images 不能为负数。")
    return args


def main() -> None:
    """先从训练集确定 tau_A，再执行新版 A*/R* 可视化和统计。"""

    args = parse_args()
    data_root = args.data_root.resolve()
    device = resolve_device(args.device)
    builders: dict[str, DemandBuilder] = {}
    parameters: dict[str, dict[str, float | int | str]] = {}
    selected_names = {spec.name for spec in selected_specs(args.datasets)}

    for spec in DATASET_SPECS:
        if spec.name not in selected_names:
            continue
        if args.tau_a is None:
            # tau_A 始终来自训练 split，即使本次可视化的是 test split。
            tau_pairs = build_pairs(
                data_root,
                spec,
                split="train",
                max_images=args.max_tau_images,
            )
            print(
                f"[tau_A][{spec.name}] 使用 {len(tau_pairs)} 对训练图统计 "
                f"q={args.tau_quantile:.4f}。")
            tau_a, positive_count = estimate_positive_quantile(
                pairs=tau_pairs,
                device=device,
                kernel_size=args.low_pass_kernel_size,
                sigma=args.low_pass_sigma,
                quantile=args.tau_quantile,
                histogram_bins=args.tau_histogram_bins,
            )
            tau_source = "estimated_from_training_split"
        else:
            tau_a = float(args.tau_a)
            positive_count = 0
            tau_source = "manual_override"
        print(f"[tau_A][{spec.name}] tau_A={tau_a:.8f} ({tau_source})")

        kernel_size = int(args.low_pass_kernel_size)
        sigma = float(args.low_pass_sigma)

        def builder(
            luminance_lq: torch.Tensor,
            luminance_gt: torch.Tensor,
            dataset_name: str,
            low_pass_kernel_size: int = kernel_size,
            low_pass_sigma: float = sigma,
            fixed_tau_a: float = tau_a,
        ) -> torch.Tensor:
            """使用当前数据集冻结的 tau_A 构造新版目标。"""

            del dataset_name
            return build_revised_demand_target(
                luminance_lq=luminance_lq,
                luminance_gt=luminance_gt,
                kernel_size=low_pass_kernel_size,
                sigma=low_pass_sigma,
                tau_a=fixed_tau_a,
            )

        builders[spec.name] = builder
        parameters[spec.name] = {
            "definition": "clip(relu(G(Y_gt)-G(Y_lq))/tau_A,0,1)",
            "low_pass_kernel_size": kernel_size,
            "low_pass_sigma": sigma,
            "tau_a": tau_a,
            "tau_source": tau_source,
            "tau_quantile": float(args.tau_quantile),
            "tau_histogram_bins": int(args.tau_histogram_bins),
            "tau_positive_pixel_count": positive_count,
        }

    run_selected_datasets(
        args=args,
        formula_name="revised_low_frequency_absolute_gap",
        demand_builders=builders,
        formula_parameters=parameters,
    )


if __name__ == "__main__":
    main()
