"""实现 BEAR-BioIR 的范围预算—证据非对称路由恢复网络。

本文件独立于原始 ``BioIR_arch.py``，以便 BioIR baseline 与方案 3 能在不同
配置中直接对照。网络在入口将图像右侧、下侧反射补齐到 64 的倍数；所有
BEPR 路由网格与 SARI token 都在该补边坐标系中构造，输出再裁回原始尺寸。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.models.archs.BioIR_arch import (Downsample, FeedForward, Fuse,
                                              LayerNorm, OverlapPatchEmbed,
                                              Upsample)


def _reflect_pad_right_bottom(
        image: torch.Tensor, pad_height: int, pad_width: int) -> torch.Tensor:
    """仅向右侧和下侧做可分段的反射补边。

    Args:
        image: 输入特征，形状为 ``B×C×H×W``。
        pad_height: 需要在下侧增加的像素数。
        pad_width: 需要在右侧增加的像素数。

    Returns:
        反射补边后的特征。对极小输入采用 replicate 兜底，避免 PyTorch
        reflection padding 的 ``padding < size`` 限制。
    """
    if pad_height == 0 and pad_width == 0:
        return image
    if image.shape[-2] <= 1 or image.shape[-1] <= 1:
        return F.pad(image, (0, pad_width, 0, pad_height), mode='replicate')

    padded = image
    remaining_height, remaining_width = pad_height, pad_width
    # 一次 reflection padding 不能大于当前尺寸减一，因此对任意尺寸分段处理。
    while remaining_height > 0 or remaining_width > 0:
        step_height = min(remaining_height, padded.shape[-2] - 1)
        step_width = min(remaining_width, padded.shape[-1] - 1)
        padded = F.pad(
            padded, (0, step_width, 0, step_height), mode='reflect')
        remaining_height -= step_height
        remaining_width -= step_width
    return padded


def pad_to_multiple(image: torch.Tensor,
                    multiple: int) -> Tuple[torch.Tensor, int, int]:
    """将 ``B×C×H×W`` 图像右下补齐到指定整数倍。

    Args:
        image: 待补边的四维 Tensor。
        multiple: 目标高度和宽度必须整除的正整数。

    Returns:
        ``(padded, original_height, original_width)``。后两个值用于在输出端
        精确裁回原始有效区域。
    """
    if image.ndim != 4:
        raise ValueError(f'Expected BxCxHxW Tensor, got {tuple(image.shape)}')
    if multiple <= 0:
        raise ValueError(f'multiple must be positive, got {multiple}')
    height, width = image.shape[-2:]
    pad_height = (multiple - height % multiple) % multiple
    pad_width = (multiple - width % multiple) % multiple
    return _reflect_pad_right_bottom(image, pad_height, pad_width), height, width


def rgb_to_luminance(image: torch.Tensor) -> torch.Tensor:
    """按 Rec.709 系数将 RGB 图像变换为单通道亮度。

    Args:
        image: 值域为 ``[0, 1]`` 的 ``B×3×H×W`` RGB Tensor。

    Returns:
        形状为 ``B×1×H×W`` 的 Rec.709 亮度 Tensor。
    """
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(
            f'BEAR-BioIR expects Bx3xHxW RGB input, got {tuple(image.shape)}')
    return (0.2126 * image[:, 0:1] + 0.7152 * image[:, 1:2]
            + 0.0722 * image[:, 2:3])


def patch_average(feature: torch.Tensor, patch_size: int) -> torch.Tensor:
    """对非重叠方形区域平均池化并保留二维网格。

    Args:
        feature: 输入特征，形状为 ``B×C×H×W``。
        patch_size: 当前尺度特征坐标中的方形区域边长。

    Returns:
        形状为 ``B×C×(H/p)×(W/p)`` 的区域平均特征。
    """
    height, width = feature.shape[-2:]
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            f'Feature {height}x{width} is not divisible by patch {patch_size}')
    return F.avg_pool2d(feature, kernel_size=patch_size, stride=patch_size)


def topk_patch_mean(feature: torch.Tensor, patch_size: int,
                    topk: int) -> torch.Tensor:
    """在每个非重叠区域选择 Top-k 响应并返回区域均值。

    Args:
        feature: 形状为 ``B×C×H×W`` 的候选结构响应。
        patch_size: 输入坐标中路由区域的边长。
        topk: 每个区域参与平均的最大响应数量。

    Returns:
        形状为 ``B×C×(H/p)×(W/p)`` 的稀疏区域证据。
    """
    if topk <= 0 or topk > patch_size * patch_size:
        raise ValueError(
            f'topk must be in [1, {patch_size * patch_size}], got {topk}')
    height, width = feature.shape[-2:]
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            f'Feature {height}x{width} is not divisible by patch {patch_size}')
    batch, channels = feature.shape[:2]
    patches = F.unfold(feature, kernel_size=patch_size, stride=patch_size)
    patches = patches.view(batch, channels, patch_size * patch_size, -1)
    selected = patches.topk(topk, dim=2).values.mean(dim=2)
    return selected.view(batch, channels, height // patch_size,
                         width // patch_size)


class FixedStructureEvidence(nn.Module):
    """从低光 RGB 图生成无可学习参数的结构强度与方向一致性证据。

    高斯平滑、对数亮度、Sobel 梯度和结构张量完全依照方案 3 的固定观测
    路径实现。该模块不把 coherence 当作噪声真值；其输出只作为 BEPR 的
    输入证据和 token 内结构加权的先验。
    """

    def __init__(self, log_epsilon: float = 0.02,
                 structure_scale: float = 0.10) -> None:
        """注册固定高斯与 Sobel 核。

        Args:
            log_epsilon: 对数亮度的数值稳定常数 ``ε_Y``。
            structure_scale: 结构强度软压缩尺度 ``τ_s``。
        """
        super().__init__()
        gaussian_1d = torch.tensor([1.0, 2.0, 1.0]) / 4.0
        gaussian = torch.outer(gaussian_1d, gaussian_1d).view(1, 1, 3, 3)
        sobel_x = torch.tensor([
            [-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]
        ]).view(1, 1, 3, 3) / 8.0
        sobel_y = sobel_x.transpose(-1, -2).contiguous()
        self.register_buffer('gaussian_kernel', gaussian, persistent=False)
        self.register_buffer('sobel_x_kernel', sobel_x, persistent=False)
        self.register_buffer('sobel_y_kernel', sobel_y, persistent=False)
        self.log_epsilon = float(log_epsilon)
        self.structure_scale = float(structure_scale)

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算亮度、二维梯度、结构强度与结构张量一致性。

        Args:
            image: ``[0, 1]`` 范围的 ``B×3×H×W`` RGB 图像。

        Returns:
            包含 ``luminance``、``gradient``、``magnitude``、``strength`` 与
            ``coherence`` 的字典；除 ``gradient`` 外均为单通道特征。
        """
        luminance = rgb_to_luminance(image)
        smoothed = F.conv2d(
            F.pad(luminance, (1, 1, 1, 1), mode='reflect'),
            self.gaussian_kernel)
        log_luminance = torch.log(smoothed + self.log_epsilon)
        grad_x = F.conv2d(
            F.pad(log_luminance, (1, 1, 1, 1), mode='reflect'),
            self.sobel_x_kernel)
        grad_y = F.conv2d(
            F.pad(log_luminance, (1, 1, 1, 1), mode='reflect'),
            self.sobel_y_kernel)
        magnitude = torch.sqrt(grad_x.square() + grad_y.square() + 1e-12)
        strength = 1.0 - torch.exp(-magnitude / self.structure_scale)

        # 结构张量在 3×3 邻域内统计梯度方向是否稳定。
        j_xx = F.avg_pool2d(grad_x.square(), kernel_size=3, stride=1, padding=1)
        j_yy = F.avg_pool2d(grad_y.square(), kernel_size=3, stride=1, padding=1)
        j_xy = F.avg_pool2d(grad_x * grad_y, kernel_size=3, stride=1, padding=1)
        coherence = torch.sqrt((j_xx - j_yy).square() + 4.0 * j_xy.square())
        coherence = coherence / (j_xx + j_yy + 1e-6)
        return {
            'luminance': luminance,
            'gradient': torch.cat([grad_x, grad_y], dim=1),
            'magnitude': magnitude,
            'strength': strength,
            'coherence': coherence.clamp(0.0, 1.0),
        }


class DepthwisePointwiseHead(nn.Module):
    """用于路由预测的轻量 depthwise–pointwise 卷积头。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """构造 ``3×3 DWConv + GELU + 1×1 Conv`` 预测器。

        Args:
            in_channels: 输入路由证据通道数。
            out_channels: 输出 logit 或权重的通道数。
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, padding=1,
                groups=in_channels), nn.GELU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1))

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        """将路由证据映射为目标通道的空间 logit。

        Args:
            feature: 形状为 ``B×C×H_r×W_r`` 的路由特征。

        Returns:
            形状为 ``B×C_out×H_r×W_r`` 的预测结果。
        """
        return self.layers(feature)


class BEPR(nn.Module):
    """预测 BEAR 的整图预算、分尺度范围和细节可靠性图。

    BEPR 将三种亮度欠曝观测融合为 ``E_f``，先预测整图范围先验 ``b``，
    再由每个 encoder 尺度的压缩状态对范围 ``A_s`` 和可靠性 ``R_s`` 做
    有界修正。输入图证据只在一次前向中计算一次，decoder 与 refinement
    复用其对应 encoder 尺度的路由结果。
    """

    def __init__(self, scale_channels: Tuple[int, int, int],
                 route_channels: int = 8, route_patch: int = 16,
                 topk: int = 16) -> None:
        """初始化三尺度路由头与固定结构证据模块。

        Args:
            scale_channels: 三个 BioIR encoder 尺度的通道数。
            route_channels: 每个尺度压缩到路由网格后的通道数 ``C_r``。
            route_patch: 输入坐标中一个路由区域的边长 ``q_r``。
            topk: 构造稀疏结构/可靠性标签时每个区域保留的像素数。
        """
        super().__init__()
        self.route_patch = int(route_patch)
        self.topk = int(topk)
        self.structure = FixedStructureEvidence()
        self.evidence_weight = DepthwisePointwiseHead(3, 3)
        self.budget_head = nn.Sequential(
            nn.Linear(4, 16), nn.GELU(), nn.Linear(16, 1))
        # [E_bar(3), E_f(1), alpha(3)] 与 [P_str, P_coh, E_f]。
        self.scope_base = DepthwisePointwiseHead(7, 1)
        self.reliability_base = DepthwisePointwiseHead(3, 1)
        self.scale_projectors = nn.ModuleList([
            nn.Conv2d(channels, route_channels, kernel_size=1)
            for channels in scale_channels
        ])
        self.scope_refiners = nn.ModuleList([
            DepthwisePointwiseHead(route_channels + 1, 1)
            for _ in scale_channels
        ])
        self.reliability_refiners = nn.ModuleList([
            DepthwisePointwiseHead(route_channels + 1, 1)
            for _ in scale_channels
        ])

    def prepare(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """由补边低光输入构造所有可共享的 BEPR 证据。

        Args:
            image: 已被补到 64 倍数的 ``B×3×H_p×W_p`` 低光图。

        Returns:
            包含整图范围预算、区域亮度/结构证据及 token 内发送先验的字典。
        """
        height, width = image.shape[-2:]
        if height % self.route_patch != 0 or width % self.route_patch != 0:
            raise ValueError('BEPR input must be divisible by route_patch')
        luminance = rgb_to_luminance(image)
        maximum = image.max(dim=1, keepdim=True).values
        hsl_lightness = (maximum + image.min(dim=1, keepdim=True).values) / 2.0
        underexposure = torch.cat(
            [1.0 - luminance, 1.0 - maximum, 1.0 - hsl_lightness], dim=1)
        exposure_regions = patch_average(underexposure, self.route_patch)
        alpha = self.evidence_weight(exposure_regions).softmax(dim=1)
        fused_exposure = (alpha * exposure_regions).sum(dim=1, keepdim=True)
        budget_input = torch.cat([exposure_regions, fused_exposure], dim=1)
        budget = torch.sigmoid(self.budget_head(budget_input.mean(dim=(-2, -1))))

        structure = self.structure(image)
        sparse_structure = topk_patch_mean(
            structure['strength'] * structure['coherence'], self.route_patch,
            self.topk)
        mean_coherence = patch_average(structure['coherence'], self.route_patch)
        scope_base = self.scope_base(
            torch.cat([exposure_regions, fused_exposure, alpha], dim=1))
        reliability_base = self.reliability_base(
            torch.cat([sparse_structure, mean_coherence, fused_exposure], dim=1))
        return {
            'budget': budget,
            'scope_base': scope_base,
            'reliability_base': reliability_base,
            'sender_prior': structure['strength'] * structure['coherence'],
        }

    def route_scale(self, scale_index: int, feature: torch.Tensor,
                    context: Dict[str, torch.Tensor],
                    downsample_factor: int) -> Dict[str, torch.Tensor]:
        """用当前尺度状态将整图先验分配成 ``A_s``、``R_s``。

        Args:
            scale_index: 0、1、2 分别代表 BioIR 的三个 encoder 尺度。
            feature: 首个 SARI 前的 ``B×C_s×H_s×W_s`` 特征。
            context: :meth:`prepare` 返回的共享输入证据。
            downsample_factor: 当前尺度相对补边输入的下采样倍率 ``r_s``。

        Returns:
            包含路由范围 ``scope``、发送可靠性 ``reliability`` 和压缩状态
            ``state`` 的字典，三者均对齐到 ``H_r×W_r`` 路由网格。
        """
        if scale_index not in (0, 1, 2):
            raise ValueError(f'Unsupported scale_index: {scale_index}')
        patch_size = self.route_patch // downsample_factor
        if patch_size <= 0:
            raise ValueError('downsample_factor exceeds route_patch')
        state = self.scale_projectors[scale_index](
            patch_average(feature, patch_size))
        scope_delta = self.scope_refiners[scale_index](
            torch.cat([context['scope_base'], state], dim=1))
        # 减去每张图的空间均值，使尺度头主要做区域重新分配而非重估总预算。
        scope_delta = scope_delta - scope_delta.mean(dim=(-2, -1), keepdim=True)
        budget = context['budget'].clamp(1e-4, 1.0 - 1e-4)
        budget_logit = torch.logit(budget).view(-1, 1, 1, 1)
        scope = torch.sigmoid(budget_logit + torch.tanh(scope_delta))

        reliability_delta = self.reliability_refiners[scale_index](
            torch.cat([context['reliability_base'], state], dim=1))
        reliability = torch.sigmoid(
            context['reliability_base'] + torch.tanh(reliability_delta))
        return {
            'scope': scope,
            'reliability': reliability,
            'state': state,
        }

    def build_targets(self, low: torch.Tensor,
                      gt: torch.Tensor) -> Dict[str, torch.Tensor]:
        """仅训练时由低光/GT 构造预算与可靠性代理目标。

        Args:
            low: 与网络路由坐标对齐的补边低光 RGB 图。
            gt: 使用同一补边规则得到的配对正常曝光 RGB 图。

        Returns:
            ``budget`` 与 ``reliability`` 两个不参与梯度计算的监督目标。
        """
        if low.shape != gt.shape:
            raise ValueError(
                f'Low/GT shapes must match, got {tuple(low.shape)} and '
                f'{tuple(gt.shape)}')
        low_luminance = rgb_to_luminance(low)
        gt_luminance = rgb_to_luminance(gt)
        relative_gap = ((gt_luminance - low_luminance).clamp_min(0.0)
                        / (gt_luminance + 0.05)).clamp(0.0, 1.0)
        budget_target = relative_gap.mean(dim=(-2, -1))

        low_structure = self.structure(low)
        gt_structure = self.structure(gt)
        low_gradient = low_structure['gradient']
        gt_gradient = gt_structure['gradient']
        low_magnitude = low_structure['magnitude']
        gt_magnitude = gt_structure['magnitude']
        gt_structure_presence = 1.0 - torch.exp(-gt_magnitude / 0.10)
        magnitude_match = torch.exp(
            -(low_magnitude - gt_magnitude).abs() / (gt_magnitude + 0.05))
        direction_match = (low_gradient * gt_gradient).sum(dim=1, keepdim=True)
        direction_match = direction_match / (
            low_magnitude * gt_magnitude + 1e-6)
        pixel_reliability = (
            gt_structure_presence * magnitude_match
            * direction_match.clamp(0.0, 1.0))
        reliability_target = topk_patch_mean(
            pixel_reliability, self.route_patch, self.topk)
        return {
            'budget': budget_target.detach(),
            'reliability': reliability_target.detach(),
        }


class SARIInteraction(nn.Module):
    """执行范围感知的上下文检索与可靠细节几何聚合。

    ``G→D`` 仅用一次细节到全局 token 注意力完成不确定来源的检索；
    ``D→G`` 利用已知的 4×4 空间归属做可靠性门控均值，不重复计算反向
    cross-attention。这样两条消息的方向、问题和算子都不对称。
    """

    def __init__(self, dim: int, heads: int, route_patch: int,
                 global_patch: int, sender_temperature: float = 0.10,
                 bias: bool = False) -> None:
        """构造一个尺度无关、由 patch 参数对齐的 SARI 交互模块。

        Args:
            dim: 当前尺度的通道数 ``C_s``。
            heads: 全局检索注意力头数。
            route_patch: 当前特征图中细节 token 的边长 ``p_d^s``。
            global_patch: 当前特征图中全局 token 的边长 ``p_g^s``。
            sender_temperature: token 内结构加权池化的 Softmax 温度。
            bias: 是否在卷积/线性投影中启用 bias。
        """
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f'dim={dim} must be divisible by heads={heads}')
        if global_patch % route_patch != 0:
            raise ValueError('global_patch must be an integer multiple of route_patch')
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = dim // heads
        self.route_patch = int(route_patch)
        self.global_patch = int(global_patch)
        self.group_size = global_patch // route_patch
        self.sender_temperature = float(sender_temperature)

        self.context_projection = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.detail_projection = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.context_dw = nn.Conv2d(
            dim, dim, kernel_size=5, padding=2, groups=dim, bias=bias)
        self.detail_dw = nn.Conv2d(
            dim, dim, kernel_size=3, padding=1, groups=dim, bias=bias)
        self.query = nn.Linear(dim, dim, bias=bias)
        self.key = nn.Linear(dim, dim, bias=bias)
        self.value = nn.Linear(dim, dim, bias=bias)
        self.local_value = nn.Linear(dim, dim, bias=bias)
        self.sender_value = nn.Linear(dim, dim, bias=bias)
        self.detail_output = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.context_output = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.fuse_output = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.detail_scale = nn.Parameter(torch.full((1, dim, 1, 1), 1e-2))
        self.context_scale = nn.Parameter(torch.full((1, dim, 1, 1), 1e-2))
        self.output_scale = nn.Parameter(torch.full((1, dim, 1, 1), 1e-2))

    @staticmethod
    def _tokens(feature: torch.Tensor, patch_size: int) -> torch.Tensor:
        """把区域平均特征从二维网格变为行优先 token 序列。"""
        pooled = patch_average(feature, patch_size)
        return pooled.flatten(2).transpose(1, 2).contiguous()

    @staticmethod
    def _grid(tokens: torch.Tensor, grid_height: int,
              grid_width: int) -> torch.Tensor:
        """把行优先 token 序列恢复为 ``B×C×H×W`` 网格。"""
        batch, count, channels = tokens.shape
        if count != grid_height * grid_width:
            raise ValueError(
                f'Token count {count} does not match {grid_height}x{grid_width}')
        return tokens.transpose(1, 2).reshape(batch, channels, grid_height,
                                               grid_width)

    def _structure_weighted_tokens(self, detail: torch.Tensor,
                                   sender_prior: torch.Tensor) -> torch.Tensor:
        """在每个细节区域内用共享结构证据选择实际发送内容。

        ``R_s`` 仍只决定区域是否有发送资格；该无参数池化只解决“Top-k
        可靠性标签很高但普通 PatchAvg 抹掉那条边缘”的 token 内容失配。
        """
        height, width = detail.shape[-2:]
        route_height, route_width = (height // self.route_patch,
                                     width // self.route_patch)
        prior = F.interpolate(
            sender_prior, size=(height, width), mode='area').clamp_min(0.0)
        batch, channels = detail.shape[:2]
        pixels_per_token = self.route_patch * self.route_patch
        detail_patches = detail.reshape(
            batch, channels, route_height, self.route_patch, route_width,
            self.route_patch).permute(0, 2, 4, 1, 3, 5).reshape(
                batch, route_height, route_width, channels, pixels_per_token)
        prior_patches = prior.reshape(
            batch, 1, route_height, self.route_patch, route_width,
            self.route_patch).permute(0, 2, 4, 1, 3, 5).reshape(
                batch, route_height, route_width, 1, pixels_per_token)
        weights = (prior_patches / self.sender_temperature).softmax(dim=-1)
        tokens = (detail_patches * weights).sum(dim=-1)
        return tokens.reshape(batch, route_height * route_width, channels)

    def _aggregate_detail_to_global(self, sender_tokens: torch.Tensor,
                                    reliability: torch.Tensor,
                                    route_height: int,
                                    route_width: int) -> torch.Tensor:
        """把 4×4 细节 token 做可靠性门控均值并归属于固定全局区域。"""
        batch = sender_tokens.shape[0]
        global_height = route_height // self.group_size
        global_width = route_width // self.group_size
        values = self.sender_value(sender_tokens).reshape(
            batch, route_height, route_width, self.dim)
        weights = reliability.permute(0, 2, 3, 1).reshape(
            batch, route_height, route_width, 1)
        values = values.reshape(
            batch, global_height, self.group_size, global_width,
            self.group_size, self.dim)
        weights = weights.reshape(
            batch, global_height, self.group_size, global_width,
            self.group_size, 1)
        # 固定除以 token 数而非可靠性和，低可靠区域的消息幅值会真正减弱。
        messages = (values * weights).mean(dim=(2, 4))
        return messages.reshape(batch, global_height * global_width, self.dim)

    def forward(self, feature: torch.Tensor, scope: torch.Tensor,
                reliability: torch.Tensor,
                sender_prior: torch.Tensor) -> torch.Tensor:
        """输出一个 SARI 残差消息，随后由 :class:`SARIBlock` 接入 GDFN。

        Args:
            feature: 当前尺度 ``B×C_s×H_s×W_s`` 的已归一化特征。
            scope: ``B×1×H_r×W_r`` 范围坐标图 ``A_s``。
            reliability: ``B×1×H_r×W_r`` 发送可靠性图 ``R_s``。
            sender_prior: 由 BEPR 一次计算的像素级结构候选图。

        Returns:
            与 ``feature`` 同形状的、LayerScale 抑制过的交互残差。
        """
        height, width = feature.shape[-2:]
        if height % self.global_patch != 0 or width % self.global_patch != 0:
            raise ValueError(
                f'SARI feature {height}x{width} must divide {self.global_patch}')
        route_height, route_width = (height // self.route_patch,
                                     width // self.route_patch)
        if scope.shape[-2:] != (route_height, route_width):
            raise ValueError('Scope grid is not aligned with current SARI scale')
        if reliability.shape[-2:] != (route_height, route_width):
            raise ValueError('Reliability grid is not aligned with current SARI scale')

        context = self.context_dw(self.context_projection(feature))
        detail = self.detail_dw(self.detail_projection(feature))
        detail_tokens = self._tokens(detail, self.route_patch)
        local_tokens = self._tokens(context, self.route_patch)
        global_tokens = self._tokens(context, self.global_patch)

        batch, route_count, _ = detail_tokens.shape
        global_count = global_tokens.shape[1]
        query = self.query(detail_tokens).reshape(
            batch, route_count, self.heads, self.head_dim).transpose(1, 2)
        key = self.key(global_tokens).reshape(
            batch, global_count, self.heads, self.head_dim).transpose(1, 2)
        value = self.value(global_tokens).reshape(
            batch, global_count, self.heads, self.head_dim).transpose(1, 2)
        attention = (query @ key.transpose(-2, -1)) * (self.head_dim ** -0.5)
        global_message = (attention.softmax(dim=-1) @ value).transpose(1, 2)
        global_message = global_message.reshape(batch, route_count, self.dim)
        local_message = self.local_value(local_tokens)

        scope_tokens = scope.flatten(2).transpose(1, 2)
        local_weight = 2.0 * scope_tokens * (1.0 - scope_tokens)
        global_weight = scope_tokens.square()
        mixed_message = local_weight * local_message + global_weight * global_message
        detail_message = self._grid(mixed_message, route_height, route_width)
        detail_message = F.interpolate(
            detail_message, size=(height, width), mode='nearest')
        updated_detail = detail + self.detail_scale * self.detail_output(detail_message)

        sender_tokens = self._structure_weighted_tokens(detail, sender_prior)
        global_detail_message = self._aggregate_detail_to_global(
            sender_tokens, reliability, route_height, route_width)
        global_height = height // self.global_patch
        global_width = width // self.global_patch
        context_message = self._grid(
            global_detail_message, global_height, global_width)
        context_message = F.interpolate(
            context_message, size=(height, width), mode='nearest')
        updated_context = context + self.context_scale * self.context_output(
            context_message)
        return self.output_scale * self.fuse_output(updated_context * updated_detail)


class SARIBlock(nn.Module):
    """以 SARI 替换原 BioIR ``AttBlock``，并保留原普通 GDFN。"""

    def __init__(self, dim: int, route_patch: int, global_patch: int,
                 heads: int = 4, ffn_expansion_factor: float = 3.0,
                 bias: bool = False,
                 layer_norm_type: str = 'WithBias') -> None:
        """初始化通道归一化、SARI 交互和原 BioIR FeedForward。

        Args:
            dim: 当前特征通道数。
            route_patch: 当前尺度的细节 token 边长。
            global_patch: 当前尺度的全局 token 边长。
            heads: 全局检索注意力头数。
            ffn_expansion_factor: 原 BioIR GDFN 的通道扩张倍数。
            bias: 是否为卷积和线性层启用 bias。
            layer_norm_type: 与原 BioIR 一致的 LayerNorm 类型。
        """
        super().__init__()
        self.norm1 = LayerNorm(dim, layer_norm_type)
        self.interaction = SARIInteraction(
            dim, heads, route_patch, global_patch, bias=bias)
        self.norm2 = LayerNorm(dim, layer_norm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, feature: torch.Tensor, scope: torch.Tensor,
                reliability: torch.Tensor,
                sender_prior: torch.Tensor) -> torch.Tensor:
        """先写入 SARI 非对称消息，再执行未改动职责的原 GDFN。

        Args:
            feature: 当前尺度输入特征。
            scope: 该尺度共享的范围路由图 ``A_s``。
            reliability: 该尺度共享的可靠性图 ``R_s``。
            sender_prior: 共享的像素级结构候选图。

        Returns:
            SARI 和 GDFN 两次残差更新后的同形状恢复特征。
        """
        feature = feature + self.interaction(
            self.norm1(feature), scope, reliability, sender_prior)
        return feature + self.ffn(self.norm2(feature))


class BEARBioIR(nn.Module):
    """方案 3 的 BioIR 主干：BEPR 路由器加 12 个 SARIBlock。

    保留 BioIR 的三尺度 encoder–decoder、skip fusion、上/下采样、普通
    GDFN 和 RGB 残差输出。原始 12 个 ``AttBlock`` 被替换为 12 个独立参数
    的 ``SARIBlock``；同一尺度的 encoder/decoder 复用一次预测的 ``A_s``、
    ``R_s``，一级路由同时供 4 个 refinement block 使用。
    """

    def __init__(self, inp_channels: int = 3, out_channels: int = 3,
                 dim: int = 32, num_blocks: List[int] = None,
                 num_refinement_blocks: int = 4,
                 ffn_expansion_factor: float = 3.0, bias: bool = False,
                 route_channels: int = 8, route_patch: int = 16,
                 global_patch: int = 64, attention_heads: int = 4,
                 topk: int = 16) -> None:
        """构造方案固定的 BioIR 三尺度恢复网络。

        Args:
            inp_channels: 输入 RGB 通道数，方案固定为 3。
            out_channels: 输出 RGB 通道数，方案固定为 3。
            dim: 第一尺度基础通道数。
            num_blocks: 三个 encoder 尺度的 block 数，默认 ``[1, 1, 2]``。
            num_refinement_blocks: 第一尺度 refinement SARI 数，默认 4。
            ffn_expansion_factor: 保留的 BioIR GDFN 通道扩张倍数。
            bias: 是否启用卷积/线性 bias。
            route_channels: BEPR 路由压缩状态通道数。
            route_patch: 输入坐标细节路由区域边长，方案默认 16。
            global_patch: 输入坐标全局区域边长，方案默认 64。
            attention_heads: SARI ``G→D`` 检索注意力头数。
            topk: 每个路由区域的结构/可靠性 Top-k 聚合数。
        """
        super().__init__()
        num_blocks = [1, 1, 2] if num_blocks is None else list(num_blocks)
        if len(num_blocks) != 3:
            raise ValueError('num_blocks must contain exactly three scales')
        if inp_channels != 3 or out_channels != 3:
            raise ValueError('BEAR-BioIR is defined for 3-channel RGB input/output')
        if global_patch != route_patch * 4:
            raise ValueError('Scheme 3 requires global_patch = 4 * route_patch')
        self.pad_multiple = int(global_patch)
        self.route_patch = int(route_patch)
        self.global_patch = int(global_patch)
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        scale_dims = (dim, dim * 2, dim * 4)
        self.router = BEPR(
            scale_dims, route_channels=route_channels, route_patch=route_patch,
            topk=topk)

        def make_blocks(scale: int, count: int) -> nn.ModuleList:
            """创建同一尺度但不共享参数的一组 SARIBlock。"""
            downsample_factor = 2 ** scale
            return nn.ModuleList([
                SARIBlock(
                    dim=scale_dims[scale],
                    route_patch=route_patch // downsample_factor,
                    global_patch=global_patch // downsample_factor,
                    heads=attention_heads,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias) for _ in range(count)
            ])

        self.encoder_level1 = make_blocks(0, num_blocks[0])
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = make_blocks(1, num_blocks[1])
        self.down2_3 = Downsample(dim * 2)
        self.encoder_level3 = make_blocks(2, num_blocks[2])
        self.decoder_level3 = make_blocks(2, num_blocks[2])
        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1,
                                            bias=bias)
        self.decoder_level2 = make_blocks(1, num_blocks[1])
        self.up2_1 = Upsample(dim * 2)
        self.decoder_level1 = make_blocks(0, num_blocks[0])
        self.refinement = make_blocks(0, num_refinement_blocks)
        self.fuse2 = Fuse(dim * 2)
        self.fuse1 = Fuse(dim)
        self.output = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1,
                                bias=bias)

    @staticmethod
    def _run_blocks(blocks: nn.ModuleList, feature: torch.Tensor,
                    route: Dict[str, torch.Tensor],
                    sender_prior: torch.Tensor) -> torch.Tensor:
        """让同一尺度所有 block 复用同一张 ``A_s``、``R_s`` 路由图。"""
        for block in blocks:
            feature = block(feature, route['scope'], route['reliability'],
                            sender_prior)
        return feature

    def build_routing_targets(self, low: torch.Tensor,
                              gt: torch.Tensor) -> Dict[str, torch.Tensor]:
        """将低光/GT 用与网络一致的入口补边后构造辅助监督目标。

        Args:
            low: 训练输入低光 RGB patch。
            gt: 与 ``low`` 同空间尺寸的配对 GT patch。

        Returns:
            适配 BEPR 路由网格的预算和可靠性目标字典。
        """
        padded_low, original_height, original_width = pad_to_multiple(
            low, self.pad_multiple)
        if gt.shape[-2:] != (original_height, original_width):
            raise ValueError('GT spatial size must match low-light input')
        padded_gt = _reflect_pad_right_bottom(
            gt, padded_low.shape[-2] - original_height,
            padded_low.shape[-1] - original_width)
        return self.router.build_targets(padded_low, padded_gt)

    def forward(self, inp_img: torch.Tensor,
                return_aux: bool = False):
        """执行一次 BEPR+SARI 前向，并在需要时返回训练辅助预测。

        Args:
            inp_img: ``B×3×H×W`` 低光 RGB 图像；测试可直接使用完整图。
            return_aux: 为 ``True`` 时同时返回 ``b``、三尺度 ``A/R``，仅供
                :class:`BEARBioIRModel` 计算训练期路由损失。

        Returns:
            默认仅返回裁回原尺寸的增强图；训练模式返回 ``(image, aux)``。
        """
        padded_input, original_height, original_width = pad_to_multiple(
            inp_img, self.pad_multiple)
        context = self.router.prepare(padded_input)
        sender_prior = context['sender_prior']

        enc1_input = self.patch_embed(padded_input)
        route1 = self.router.route_scale(0, enc1_input, context, 1)
        enc1 = self._run_blocks(
            self.encoder_level1, enc1_input, route1, sender_prior)

        enc2_input = self.down1_2(enc1)
        route2 = self.router.route_scale(1, enc2_input, context, 2)
        enc2 = self._run_blocks(
            self.encoder_level2, enc2_input, route2, sender_prior)

        enc3_input = self.down2_3(enc2)
        route3 = self.router.route_scale(2, enc3_input, context, 4)
        enc3 = self._run_blocks(
            self.encoder_level3, enc3_input, route3, sender_prior)
        dec3 = self._run_blocks(self.decoder_level3, enc3, route3, sender_prior)

        dec2_input = self.fuse2(self.up3_2(dec3), enc2)
        dec2 = self._run_blocks(
            self.decoder_level2, dec2_input, route2, sender_prior)
        dec1_input = self.fuse1(self.up2_1(dec2), enc1)
        dec1 = self._run_blocks(
            self.decoder_level1, dec1_input, route1, sender_prior)
        refined = self._run_blocks(self.refinement, dec1, route1, sender_prior)
        restored = self.output(refined) + padded_input
        restored = restored[:, :, :original_height, :original_width]

        if not return_aux:
            return restored
        return restored, {
            'budget': context['budget'],
            'scopes': [route1['scope'], route2['scope'], route3['scope']],
            'reliabilities': [
                route1['reliability'], route2['reliability'],
                route3['reliability']
            ],
        }
