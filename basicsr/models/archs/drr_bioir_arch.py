"""DRR-BioIR v0 网络结构。

该文件在不改动原始 BioIR 实现的前提下，提供带需求—可靠性路由的低照度
增强网络。默认前向仅返回恢复图，以保持现有测试入口兼容；训练时可显式请求
辅助先验输出。
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.models.archs.BioIR_arch import (Downsample, Fuse, LayerNorm,
                                              OverlapPatchEmbed, Upsample)


def _safe_reflect_pad(tensor: torch.Tensor, padding: Tuple[int, int, int,
                                                            int]) -> torch.Tensor:
    """优先使用反射补边；极小张量退化为复制补边以避免 PyTorch 报错。"""
    left, right, top, bottom = padding
    height, width = tensor.shape[-2:]
    if (left < width and right < width and top < height and bottom < height):
        return F.pad(tensor, padding, mode='reflect')
    return F.pad(tensor, padding, mode='replicate')


def calculate_luminance(image: torch.Tensor) -> torch.Tensor:
    """按固定 RGB 权重计算单通道亮度图。

    Args:
        image: 形状为 ``B×3×H×W``、值域通常为 ``[0, 1]`` 的 RGB 图像。

    Returns:
        形状为 ``B×1×H×W`` 的亮度图。
    """
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError('DRR-BioIR 仅支持形状为 B×3×H×W 的 RGB 输入。')
    weights = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return (image * weights).sum(dim=1, keepdim=True)


def gaussian_smooth_luminance(luminance: torch.Tensor,
                              kernel_size: int = 15,
                              sigma: float = 3.0) -> torch.Tensor:
    """使用固定可分离 Gaussian 核平滑单通道亮度图。

    该算子仅用于构造训练阶段的 A* 监督目标，不含可学习参数。先沿宽度、再沿
    高度卷积与二维 Gaussian 等价，能减少成对图像纹理差异、暗噪声和微小配准误差。

    Args:
        luminance: 形状为 ``B×1×H×W`` 的亮度图。
        kernel_size: 正奇数 Gaussian 核边长。
        sigma: Gaussian 标准差，单位为像素。

    Returns:
        与输入同形状的 Gaussian 平滑亮度图。
    """
    if luminance.ndim != 4 or luminance.shape[1] != 1:
        raise ValueError('Gaussian 低通输入必须为 B×1×H×W。')
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError('Gaussian kernel_size 必须为正奇数。')
    if sigma <= 0.0:
        raise ValueError('Gaussian sigma 必须大于 0。')

    radius = kernel_size // 2
    positions = torch.arange(-radius,
                             radius + 1,
                             device=luminance.device,
                             dtype=luminance.dtype)
    kernel_1d = torch.exp(-positions.square() / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    horizontal_kernel = kernel_1d.view(1, 1, 1, kernel_size)
    vertical_kernel = kernel_1d.view(1, 1, kernel_size, 1)

    smoothed = F.conv2d(
        _safe_reflect_pad(luminance, (radius, radius, 0, 0)),
        horizontal_kernel)
    return F.conv2d(_safe_reflect_pad(smoothed, (0, 0, radius, radius)),
                    vertical_kernel)


def build_relative_demand_target(luminance_lq: torch.Tensor,
                                 luminance_gt: torch.Tensor,
                                 demand_epsilon: float = 0.05) -> torch.Tensor:
    """按原相对正向亮度缺口公式构造 A*。

    Args:
        luminance_lq: 低照度亮度图，形状为 ``B×1×H×W``。
        luminance_gt: 配对 GT 亮度图，形状同 ``luminance_lq``。
        demand_epsilon: GT 亮度分母的数值稳定常数。

    Returns:
        值域为 ``[0,1]`` 的原版相对需求目标。
    """
    if demand_epsilon <= 0.0:
        raise ValueError('relative_gap 的 demand_epsilon 必须大于 0。')
    positive_gap = (luminance_gt - luminance_lq).clamp_min(0.0)
    return (positive_gap / (luminance_gt + demand_epsilon)).clamp(0.0, 1.0)


def build_gaussian_smoothed_absolute_demand_target(
        luminance_lq: torch.Tensor,
        luminance_gt: torch.Tensor,
        kernel_size: int = 15,
        sigma: float = 3.0,
        demand_tau: float = 1.0) -> torch.Tensor:
    """按 Gaussian 平滑后的绝对正亮度缺口公式构造新版 A*。

    先分别低通 LQ/GT 亮度，再保留 GT 比 LQ 更亮的区域差值，最后用训练集固定
    ``demand_tau`` 归一化。该定义保留原有的“只接收正向提亮上下文”语义。

    Args:
        luminance_lq: 低照度亮度图，形状为 ``B×1×H×W``。
        luminance_gt: 配对 GT 亮度图，形状同 ``luminance_lq``。
        kernel_size: 固定 Gaussian 低通核边长。
        sigma: 固定 Gaussian 标准差。
        demand_tau: 训练集正 Gaussian 平滑亮度缺口的冻结归一化尺度 tau_A。

    Returns:
        值域为 ``[0,1]`` 的新版 Gaussian 平滑绝对需求目标。
    """
    if demand_tau <= 0.0:
        raise ValueError('gaussian_smoothed_absolute_gap 的 demand_tau 必须大于 0。')
    smoothed_lq = gaussian_smooth_luminance(luminance_lq, kernel_size, sigma)
    smoothed_gt = gaussian_smooth_luminance(luminance_gt, kernel_size, sigma)
    positive_gap = (smoothed_gt - smoothed_lq).clamp_min(0.0)
    return (positive_gap / demand_tau).clamp(0.0, 1.0)


def log_sobel_features(luminance: torch.Tensor,
                       log_epsilon: float = 0.02) -> Tuple[torch.Tensor,
                                                            torch.Tensor,
                                                            torch.Tensor]:
    """从亮度图生成经高斯平滑与对数压缩的 Sobel 特征。

    Args:
        luminance: 形状为 ``B×1×H×W`` 的亮度图。
        log_epsilon: 对数变换的数值稳定常数。

    Returns:
        水平梯度、垂直梯度及其二范数强度，三者形状均为 ``B×1×H×W``。
    """
    if luminance.ndim != 4 or luminance.shape[1] != 1:
        raise ValueError('亮度图必须为 B×1×H×W。')
    gaussian = luminance.new_tensor(
        [[0.07511361, 0.1238414, 0.07511361],
         [0.1238414, 0.20417996, 0.1238414],
         [0.07511361, 0.1238414, 0.07511361]]).view(1, 1, 3, 3)
    sobel_x = luminance.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0],
         [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3) / 8.0
    sobel_y = luminance.new_tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0],
         [1.0, 2.0, 1.0]]).view(1, 1, 3, 3) / 8.0

    # 反射补边避免固定梯度核在边界处引入零填充伪影。
    smoothed = F.conv2d(_safe_reflect_pad(luminance, (1, 1, 1, 1)), gaussian)
    log_luminance = torch.log(smoothed.clamp_min(0.0) + log_epsilon)
    padded_log = _safe_reflect_pad(log_luminance, (1, 1, 1, 1))
    gradient_x = F.conv2d(padded_log, sobel_x)
    gradient_y = F.conv2d(padded_log, sobel_y)
    magnitude = torch.sqrt(gradient_x.square() + gradient_y.square())
    return gradient_x, gradient_y, magnitude


def calculate_structure_observation(luminance: torch.Tensor,
                                    structure_tau: float = 0.10,
                                    log_epsilon: float = 0.02) -> torch.Tensor:
    """计算只用于候选细节分支的输入结构观测图。

    该观测仅表达低光输入中的梯度强弱，是否为可信结构由训练阶段的可靠性目标
    与 ADRI 路由共同决定。
    """
    _, _, magnitude = log_sobel_features(luminance, log_epsilon)
    return 1.0 - torch.exp(-magnitude / structure_tau)


def build_drr_targets(low_quality: torch.Tensor,
                       ground_truth: torch.Tensor,
                       demand_epsilon: float = 0.05,
                       structure_tau: float = 0.10,
                       magnitude_tau: float = 0.05,
                       log_epsilon: float = 0.02,
                       reliability_epsilon: float = 1e-6,
                       demand_target_type: str = 'relative_gap',
                       demand_gaussian_kernel_size: int = 15,
                       demand_gaussian_sigma: float = 3.0,
                       demand_tau: float = 1.0) -> Dict[str, torch.Tensor]:
    """由配对 LQ/GT 构造需求、可靠性及结构存在度训练目标。

    Args:
        low_quality: 低照度 RGB 图像，形状为 ``B×3×H×W``。
        ground_truth: 配对正常曝光 RGB 图像，形状与 ``low_quality`` 相同。
        demand_epsilon: 原相对需求公式的分母稳定常数。
        structure_tau: GT 梯度成为明显结构的软阈值。
        magnitude_tau: 两图梯度幅值的一致性容忍尺度。
        log_epsilon: 对数 Sobel 特征的稳定常数。
        reliability_epsilon: 方向余弦分母的稳定常数。
        demand_target_type: ``relative_gap`` 或 ``gaussian_smoothed_absolute_gap``。
        demand_gaussian_kernel_size: 新版 A* 的固定 Gaussian 核边长。
        demand_gaussian_sigma: 新版 A* 的固定 Gaussian 标准差。
        demand_tau: 新版 A* 使用的冻结归一化尺度 tau_A。

    Returns:
        包含 ``demand``、``reliability`` 与 ``edge_presence`` 的字典。
    """
    if low_quality.shape != ground_truth.shape:
        raise ValueError('构造 DRR 目标时，LQ 与 GT 的形状必须完全一致。')
    luminance_lq = calculate_luminance(low_quality)
    luminance_gt = calculate_luminance(ground_truth)
    grad_lq_x, grad_lq_y, magnitude_lq = log_sobel_features(
        luminance_lq, log_epsilon)
    grad_gt_x, grad_gt_y, magnitude_gt = log_sobel_features(
        luminance_gt, log_epsilon)

    if demand_target_type == 'relative_gap':
        demand = build_relative_demand_target(luminance_lq, luminance_gt,
                                               demand_epsilon)
    elif demand_target_type == 'gaussian_smoothed_absolute_gap':
        demand = build_gaussian_smoothed_absolute_demand_target(
            luminance_lq,
            luminance_gt,
            kernel_size=demand_gaussian_kernel_size,
            sigma=demand_gaussian_sigma,
            demand_tau=demand_tau)
    else:
        raise ValueError(
            'demand_target_type 只能是 relative_gap 或 '
            'gaussian_smoothed_absolute_gap。')

    edge_presence = 1.0 - torch.exp(-magnitude_gt / structure_tau)
    magnitude_consistency = torch.exp(
        -(magnitude_lq - magnitude_gt).abs() / (magnitude_gt + magnitude_tau))
    direction_dot = grad_lq_x * grad_gt_x + grad_lq_y * grad_gt_y
    direction_consistency = (direction_dot /
                             (magnitude_lq * magnitude_gt + reliability_epsilon))
    direction_consistency = direction_consistency.clamp(0.0, 1.0)
    reliability = edge_presence * magnitude_consistency * direction_consistency
    return {
        'demand': demand,
        'reliability': reliability.clamp(0.0, 1.0),
        'edge_presence': edge_presence.clamp(0.0, 1.0)
    }


def pad_to_factor(image: torch.Tensor,
                  factor: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """将图像右侧和下侧反射补至指定因子的整数倍。

    Args:
        image: 待补边的 ``B×C×H×W`` 张量。
        factor: 需要整除的空间因子。

    Returns:
        补边后的张量，以及 ``(original_height, original_width)``。
    """
    if factor <= 0:
        raise ValueError('补边因子必须为正整数。')
    height, width = image.shape[-2:]
    pad_height = (factor - height % factor) % factor
    pad_width = (factor - width % factor) % factor
    if pad_height or pad_width:
        image = _safe_reflect_pad(image, (0, pad_width, 0, pad_height))
    return image, (height, width)


def aggregate_detail_gates(demand: torch.Tensor,
                           reliability: torch.Tensor,
                           detail_patch_size: int = 16,
                           reliability_topk: int = 16) -> Tuple[torch.Tensor,
                                                                torch.Tensor]:
    """将原分辨率 A/R 聚合为所有尺度共享的细节 token gate。

    ``A`` 使用窗口平均表达区域总体需求；``R`` 使用 Top-k 均值，避免稀疏可靠
    边缘被普通平均池化稀释。
    """
    if demand.shape != reliability.shape or demand.shape[1] != 1:
        raise ValueError('A/R 必须是形状一致的 B×1×H×W 单通道图。')
    height, width = demand.shape[-2:]
    if height % detail_patch_size or width % detail_patch_size:
        raise ValueError('聚合 A/R 前，空间尺寸必须能被 detail patch 整除。')
    window_area = detail_patch_size * detail_patch_size
    if not 1 <= reliability_topk <= window_area:
        raise ValueError('R 的 Top-k 必须位于每个 detail patch 的像素数范围内。')

    demand_grid = F.avg_pool2d(demand, detail_patch_size, detail_patch_size)
    batch, channels, _, _ = reliability.shape
    grid_height, grid_width = height // detail_patch_size, width // detail_patch_size
    # ``unfold`` 保持行优先窗口顺序，后续与 detail token 的展平顺序一致。
    reliability_windows = reliability.unfold(2, detail_patch_size,
                                             detail_patch_size).unfold(
                                                 3, detail_patch_size,
                                                 detail_patch_size)
    reliability_windows = reliability_windows.reshape(batch, channels,
                                                      grid_height, grid_width,
                                                      window_area)
    reliability_grid = reliability_windows.topk(
        reliability_topk, dim=-1).values.mean(dim=-1)
    demand_tokens = demand_grid.flatten(2).transpose(1, 2).contiguous()
    reliability_tokens = reliability_grid.flatten(2).transpose(1,
                                                                 2).contiguous()
    return demand_tokens, reliability_tokens


class DemandReliabilityHead(nn.Module):
    """从浅层特征、亮度和候选结构图预测独立的 A/R 控制图。"""

    def __init__(self, feature_channels: int, hidden_channels: int = 16):
        """初始化局部—粗尺度上下文预测头。

        Args:
            feature_channels: 输入浅层特征的通道数。
            hidden_channels: 预测头的中间通道数。
        """
        super().__init__()
        self.local = nn.Conv2d(feature_channels + 2,
                               hidden_channels,
                               kernel_size=3,
                               padding=1)
        self.coarse = nn.Conv2d(hidden_channels,
                                hidden_channels,
                                kernel_size=3,
                                padding=1)
        self.output = nn.Conv2d(hidden_channels, 2, kernel_size=1)

    def forward(self, features: torch.Tensor, luminance: torch.Tensor,
                structure: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """预测需求图 A 与结构可靠性图 R。

        Args:
            features: 浅层特征，形状为 ``B×C×H×W``。
            luminance: 同尺寸亮度图，形状为 ``B×1×H×W``。
            structure: 同尺寸候选结构图，形状为 ``B×1×H×W``。

        Returns:
            经 Sigmoid 限制到 ``[0, 1]`` 的 ``(A, R)``。
        """
        if features.shape[-2:] != luminance.shape[-2:] or features.shape[-2:] != structure.shape[-2:]:
            raise ValueError('预测头的特征、亮度和结构观测必须具有相同空间尺寸。')
        local_feature = F.gelu(
            self.local(torch.cat([features, luminance, structure], dim=1)))
        coarse_feature = F.avg_pool2d(local_feature, kernel_size=4, stride=4)
        coarse_feature = F.gelu(self.coarse(coarse_feature))
        context_feature = F.interpolate(coarse_feature,
                                        size=local_feature.shape[-2:],
                                        mode='bilinear',
                                        align_corners=False)
        demand, reliability = torch.sigmoid(
            self.output(local_feature + context_feature)).chunk(2, dim=1)
        return demand, reliability


class AsymmetricDemandReliabilityInteraction(nn.Module):
    """以 A/R 分别约束接收与发送方向的双向跨粒度交互。"""

    def __init__(self,
                 dim: int,
                 context_patch_size: int,
                 detail_patch_size: int,
                 num_heads: int = 1,
                 layer_scale_init: float = 1e-2):
        """初始化独立 QKV、patch token 化与消息回写模块。

        Args:
            dim: 当前尺度特征通道数。
            context_patch_size: 上下文分支的大 patch 边长。
            detail_patch_size: 细节分支的小 patch 边长。
            num_heads: 交叉注意力头数，必须整除 ``dim``。
            layer_scale_init: 双向消息注入的初始残差系数。
        """
        super().__init__()
        if dim % num_heads:
            raise ValueError('ADRI 的 num_heads 必须整除特征通道数。')
        if context_patch_size <= 0 or detail_patch_size <= 0:
            raise ValueError('ADRI patch 尺寸必须为正整数。')
        self.dim = dim
        self.context_patch_size = context_patch_size
        self.detail_patch_size = detail_patch_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.context_norm = nn.LayerNorm(dim)
        self.detail_norm = nn.LayerNorm(dim)
        self.detail_embed_dw = nn.Conv2d(dim,
                                         dim,
                                         kernel_size=detail_patch_size,
                                         stride=detail_patch_size,
                                         groups=dim,
                                         bias=True)
        self.detail_embed_pw = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.qkv_context = nn.Linear(dim, dim * 3)
        self.qkv_detail = nn.Linear(dim, dim * 3)
        self.detail_message_out = nn.Conv2d(dim, dim, kernel_size=1)
        self.context_message_out = nn.Conv2d(dim, dim, kernel_size=1)
        self.detail_message_scale = nn.Parameter(
            torch.full((), layer_scale_init))
        self.context_message_scale = nn.Parameter(
            torch.full((), layer_scale_init))

    @staticmethod
    def _pad_for_patch(feature: torch.Tensor,
                       patch_size: int) -> Tuple[torch.Tensor, int, int]:
        """仅向右和向下复制补边，并返回补边后的空间尺寸。"""
        height, width = feature.shape[-2:]
        pad_height = (patch_size - height % patch_size) % patch_size
        pad_width = (patch_size - width % patch_size) % patch_size
        if pad_height or pad_width:
            feature = F.pad(feature, (0, pad_width, 0, pad_height), mode='replicate')
        return feature, feature.shape[-2], feature.shape[-1]

    @staticmethod
    def _tokens_to_feature(tokens: torch.Tensor, grid_height: int,
                           grid_width: int, padded_height: int,
                           padded_width: int, height: int,
                           width: int) -> torch.Tensor:
        """将 patch 级消息恢复为当前尺度的稠密空间消息。"""
        batch, token_count, channels = tokens.shape
        if token_count != grid_height * grid_width:
            raise ValueError('消息 token 数量与恢复网格尺寸不匹配。')
        message_grid = tokens.transpose(1, 2).reshape(batch, channels,
                                                       grid_height, grid_width)
        message = F.interpolate(message_grid,
                                size=(padded_height, padded_width),
                                mode='bilinear',
                                align_corners=False)
        return message[:, :, :height, :width]

    def _context_tokens(self, feature: torch.Tensor) -> Tuple[torch.Tensor,
                                                               Tuple[int, int,
                                                                     int, int]]:
        """通过平均池化把上下文特征转为大粒度 token。"""
        padded, padded_height, padded_width = self._pad_for_patch(
            feature, self.context_patch_size)
        context_grid = F.avg_pool2d(padded,
                                    kernel_size=self.context_patch_size,
                                    stride=self.context_patch_size)
        grid_height, grid_width = context_grid.shape[-2:]
        tokens = context_grid.flatten(2).transpose(1, 2).contiguous()
        return self.context_norm(tokens), (grid_height, grid_width,
                                           padded_height, padded_width)

    def _detail_tokens(self, feature: torch.Tensor) -> Tuple[torch.Tensor,
                                                              Tuple[int, int,
                                                                    int, int]]:
        """用可学习深度可分离卷积把细节 patch 转为小粒度 token。"""
        padded, padded_height, padded_width = self._pad_for_patch(
            feature, self.detail_patch_size)
        detail_grid = self.detail_embed_pw(self.detail_embed_dw(padded))
        grid_height, grid_width = detail_grid.shape[-2:]
        tokens = detail_grid.flatten(2).transpose(1, 2).contiguous()
        return self.detail_norm(tokens), (grid_height, grid_width,
                                          padded_height, padded_width)

    def _split_qkv(self, tokens: torch.Tensor,
                   projection: nn.Linear) -> Tuple[torch.Tensor, torch.Tensor,
                                                   torch.Tensor]:
        """投影并将 token 拆分为多头 Q、K、V。"""
        batch, token_count, _ = tokens.shape
        qkv = projection(tokens).reshape(batch, token_count, 3,
                                         self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        return qkv[0], qkv[1], qkv[2]

    def forward(self, context: torch.Tensor, detail: torch.Tensor,
                demand_tokens: torch.Tensor, reliability_tokens: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                           torch.Tensor]:
        """同步计算 ``G→D`` 与 ``D→G`` 路由并回写两个原分支。

        Args:
            context: DCBC 构造的上下文分支 ``G``。
            detail: DCBC 构造的细节分支 ``D``。
            demand_tokens: 形状为 ``B×N_d×1`` 的接收需求 gate。
            reliability_tokens: 形状为 ``B×N_d×1`` 的发送可靠性 gate。

        Returns:
            更新后的 ``(G, D)``，以及恢复到当前尺度的两条路由消息。
        """
        if context.shape != detail.shape:
            raise ValueError('ADRI 的上下文与细节分支必须具有相同形状。')
        batch, _, height, width = context.shape
        context_tokens, context_meta = self._context_tokens(context)
        detail_tokens, detail_meta = self._detail_tokens(detail)
        if demand_tokens.shape != reliability_tokens.shape:
            raise ValueError('ADRI 的 A_D 与 R_D token 形状必须一致。')
        if demand_tokens.shape[:2] != detail_tokens.shape[:2] or demand_tokens.shape[-1] != 1:
            raise ValueError('ADRI gate 必须与当前细节 token 网格逐一对应。')

        query_context, key_context, value_context = self._split_qkv(
            context_tokens, self.qkv_context)
        query_detail, key_detail, value_detail = self._split_qkv(
            detail_tokens, self.qkv_detail)

        # 两个方向都基于更新前的 G/D 构造，避免同一 block 内的信息回声。
        attention_g_to_d = torch.matmul(
            query_detail, key_context.transpose(-2, -1)) * self.scale
        attention_g_to_d = attention_g_to_d.softmax(dim=-1)
        message_g_to_d = torch.matmul(attention_g_to_d, value_context)
        message_g_to_d = message_g_to_d.transpose(1, 2).reshape(
            batch, detail_tokens.shape[1], self.dim)
        message_g_to_d = message_g_to_d * demand_tokens

        attention_d_to_g = torch.matmul(
            query_context, key_detail.transpose(-2, -1)) * self.scale
        attention_d_to_g = attention_d_to_g.softmax(dim=-1)
        reliable_values = value_detail * reliability_tokens[:, None, :, :]
        message_d_to_g = torch.matmul(attention_d_to_g, reliable_values)
        message_d_to_g = message_d_to_g.transpose(1, 2).reshape(
            batch, context_tokens.shape[1], self.dim)

        detail_message = self._tokens_to_feature(message_g_to_d, *detail_meta,
                                                  height, width)
        context_message = self._tokens_to_feature(message_d_to_g,
                                                   *context_meta, height, width)
        updated_detail = detail + self.detail_message_scale * self.detail_message_out(
            detail_message)
        updated_context = context + self.context_message_scale * self.context_message_out(
            context_message)
        return updated_context, updated_detail, detail_message, context_message


class RoutingAwareFeedForward(nn.Module):
    """由当前 ADRI 双向消息调制门值的可选 GDFN。"""

    def __init__(self, dim: int, ffn_expansion_factor: float, bias: bool):
        """初始化主 GDFN 与路由调制分支。

        路由末层置零初始化，故训练起点处 ``Gamma=0``，模块严格退化为同权重的
        普通 GDFN。
        """
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim,
                                    hidden_features * 2,
                                    kernel_size=1,
                                    bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2,
                                hidden_features * 2,
                                kernel_size=3,
                                padding=1,
                                groups=hidden_features * 2,
                                bias=bias)
        self.project_out = nn.Conv2d(hidden_features,
                                     dim,
                                     kernel_size=1,
                                     bias=bias)
        self.route_in = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.route_dwconv = nn.Conv2d(dim,
                                      dim,
                                      kernel_size=3,
                                      padding=1,
                                      groups=dim,
                                      bias=bias)
        self.route_out = nn.Conv2d(dim,
                                   hidden_features,
                                   kernel_size=1,
                                   bias=True)
        nn.init.zeros_(self.route_out.weight)
        nn.init.zeros_(self.route_out.bias)

    def forward(self, feature: torch.Tensor, message_g_to_d: torch.Tensor,
                message_d_to_g: torch.Tensor) -> torch.Tensor:
        """根据两条方向可区分的消息计算路由感知 GDFN 输出。"""
        if feature.shape != message_g_to_d.shape or feature.shape != message_d_to_g.shape:
            raise ValueError('RA-GDFN 的输入特征与两条路由消息必须形状一致。')
        route = torch.cat([message_g_to_d, message_d_to_g], dim=1)
        gamma = torch.tanh(
            self.route_out(self.route_dwconv(F.gelu(self.route_in(route)))))
        branch_u, branch_v = self.dwconv(self.project_in(feature)).chunk(2,
                                                                          dim=1)
        gated = F.gelu(branch_u) * branch_v * (1.0 + gamma)
        return self.project_out(gated)


class DRRBlock(nn.Module):
    """由 DCBC、可选 ADRI 与 GDFN 组成的统一 DRR 恢复块。"""

    def __init__(self,
                 dim: int,
                 context_patch_size: int,
                 detail_patch_size: int,
                 ffn_expansion_factor: float = 3.0,
                 bias: bool = False,
                 num_heads: int = 1,
                 use_adri: bool = True,
                 use_ra_gdfn: bool = False,
                 layer_scale_init: float = 1e-2):
        """初始化当前尺度的退化条件分支与路由交互模块。"""
        super().__init__()
        self.use_adri = use_adri
        self.use_ra_gdfn = use_ra_gdfn
        self.norm1 = LayerNorm(dim, 'WithBias')
        self.context_in = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.detail_in = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.context_dwconv = nn.Conv2d(dim,
                                        dim,
                                        kernel_size=5,
                                        padding=2,
                                        groups=dim,
                                        bias=bias)
        # 3×3、dilation=1 时 padding=1 才能保持 H×W，与先验和残差融合对齐。
        self.detail_dwconv = nn.Conv2d(dim,
                                       dim,
                                       kernel_size=3,
                                       padding=1,
                                       groups=dim,
                                       bias=bias)
        self.luminance_embed = nn.Sequential(nn.Conv2d(1, dim, 3, padding=1),
                                              nn.GELU())
        self.structure_embed = nn.Sequential(nn.Conv2d(1, dim, 3, padding=1),
                                             nn.GELU())
        self.context_prior_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.detail_prior_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.context_prior_scale = nn.Parameter(torch.full((), layer_scale_init))
        self.detail_prior_scale = nn.Parameter(torch.full((), layer_scale_init))
        self.fuse = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.fuse_scale = nn.Parameter(torch.full((), layer_scale_init))
        self.norm2 = LayerNorm(dim, 'WithBias')

        if use_adri:
            self.adri = AsymmetricDemandReliabilityInteraction(
                dim, context_patch_size, detail_patch_size, num_heads,
                layer_scale_init)
        if use_ra_gdfn:
            self.ffn = RoutingAwareFeedForward(dim, ffn_expansion_factor, bias)
        else:
            hidden_features = int(dim * ffn_expansion_factor)
            self.ffn = nn.Sequential(
                nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias),
                nn.Conv2d(hidden_features * 2,
                          hidden_features * 2,
                          kernel_size=3,
                          padding=1,
                          groups=hidden_features * 2,
                          bias=bias))
            self.ffn_out = nn.Conv2d(hidden_features,
                                     dim,
                                     kernel_size=1,
                                     bias=bias)

    def _run_plain_gdfn(self, feature: torch.Tensor) -> torch.Tensor:
        """执行与原 BioIR 公式一致的普通 GDFN。"""
        branch_u, branch_v = self.ffn(feature).chunk(2, dim=1)
        return self.ffn_out(F.gelu(branch_u) * branch_v)

    def forward(self, feature: torch.Tensor, luminance: torch.Tensor,
                structure: torch.Tensor, demand_tokens: torch.Tensor,
                reliability_tokens: torch.Tensor) -> torch.Tensor:
        """以当前尺度先验执行一次完整 DRR 恢复块。

        Args:
            feature: 主干输入特征 ``B×C×H×W``。
            luminance: 当前尺度亮度图。
            structure: 当前尺度候选结构图。
            demand_tokens: 共享的细节接收需求 gate。
            reliability_tokens: 共享的细节发送可靠性 gate。

        Returns:
            与输入 ``feature`` 形状一致的恢复特征。
        """
        normalized = self.norm1(feature)
        context_base = self.context_dwconv(self.context_in(normalized))
        detail_base = self.detail_dwconv(self.detail_in(normalized))
        luminance_feature = self.luminance_embed(luminance)
        structure_feature = self.structure_embed(structure)
        context = context_base + self.context_prior_scale * self.context_prior_out(
            torch.cat([context_base, luminance_feature], dim=1))
        detail = detail_base + self.detail_prior_scale * self.detail_prior_out(
            torch.cat([detail_base, structure_feature], dim=1))

        if self.use_adri:
            context, detail, message_g_to_d, message_d_to_g = self.adri(
                context, detail, demand_tokens, reliability_tokens)
        else:
            message_g_to_d = torch.zeros_like(feature)
            message_d_to_g = torch.zeros_like(feature)

        routed_feature = feature + self.fuse_scale * self.fuse(
            torch.cat([context, detail], dim=1))
        normalized_routed = self.norm2(routed_feature)
        if self.use_ra_gdfn:
            ffn_feature = self.ffn(normalized_routed, message_g_to_d,
                                   message_d_to_g)
        else:
            ffn_feature = self._run_plain_gdfn(normalized_routed)
        return routed_feature + ffn_feature


class DRRBioIR(nn.Module):
    """保留 BioIR 三尺度 U-Net 骨架的 DRR-BioIR v0 生成网络。"""

    def __init__(self,
                 inp_channels: int = 3,
                 out_channels: int = 3,
                 dim: int = 32,
                 num_blocks: Iterable[int] = (1, 1, 2),
                 num_refinement_blocks: int = 4,
                 ffn_expansion_factor: float = 3.0,
                 bias: bool = False,
                 prior_hidden_channels: int = 16,
                 input_pad_factor: int = 32,
                 adri_context_patch_sizes: Iterable[int] = (64, 32, 16),
                 adri_detail_patch_sizes: Iterable[int] = (16, 8, 4),
                 adri_num_heads: int = 1,
                 adri_scales: Iterable[int] = (1, 2, 3),
                 use_ra_gdfn: bool = False,
                 structure_tau: float = 0.10,
                 log_epsilon: float = 0.02,
                 gate_topk: int = 16):
        """初始化 DRR-BioIR v0。

        Args:
            inp_channels: 输入 RGB 通道数，v0 固定为 3。
            out_channels: 输出 RGB 通道数，v0 固定为 3。
            dim: 第一尺度基础通道数。
            num_blocks: 三个编码/解码尺度的 block 数量。
            num_refinement_blocks: 高分辨率 refinement block 数量。
            ffn_expansion_factor: GDFN 隐藏通道扩张倍数。
            bias: 卷积是否启用 bias。
            prior_hidden_channels: A/R 预测头中间通道数。
            input_pad_factor: 输入和 gate 网格对齐所需补边因子。
            adri_context_patch_sizes: 三尺度上下文 patch 尺寸。
            adri_detail_patch_sizes: 三尺度细节 patch 尺寸。
            adri_num_heads: ADRI 注意力头数。
            adri_scales: 启用 ADRI 的 1-based 尺度编号。
            use_ra_gdfn: 是否启用可选的路由感知 GDFN。
            structure_tau: 结构观测的软阈值。
            log_epsilon: log-Sobel 的稳定常数。
            gate_topk: 每个 16×16 可靠性窗口的 Top-k 均值参数。
        """
        super().__init__()
        num_blocks = tuple(num_blocks)
        context_patch_sizes = tuple(adri_context_patch_sizes)
        detail_patch_sizes = tuple(adri_detail_patch_sizes)
        if len(num_blocks) != 3 or len(context_patch_sizes) != 3 or len(
                detail_patch_sizes) != 3:
            raise ValueError('DRR-BioIR 的三尺度配置必须均为长度为 3 的序列。')
        if inp_channels != 3 or out_channels != 3:
            raise ValueError('DRR-BioIR v0 当前仅支持 RGB 三通道输入输出。')
        if input_pad_factor % detail_patch_sizes[0]:
            raise ValueError('input_pad_factor 必须整除第一尺度 detail patch。')
        self.input_pad_factor = input_pad_factor
        self.structure_tau = structure_tau
        self.log_epsilon = log_epsilon
        self.gate_topk = gate_topk
        self.adri_scales = {int(level) for level in adri_scales}
        if not self.adri_scales.issubset({1, 2, 3}):
            raise ValueError('adri_scales 仅允许 1、2、3。')

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.prior_head = DemandReliabilityHead(dim, prior_hidden_channels)
        channels = (dim, dim * 2, dim * 4)

        def make_blocks(level: int, count: int) -> nn.ModuleList:
            """按固定尺度构建一组共享先验输入的 DRR Block。"""
            return nn.ModuleList([
                DRRBlock(channels[level - 1],
                         context_patch_sizes[level - 1],
                         detail_patch_sizes[level - 1],
                         ffn_expansion_factor,
                         bias,
                         adri_num_heads,
                         level in self.adri_scales,
                         use_ra_gdfn) for _ in range(count)
            ])

        self.encoder_level1 = make_blocks(1, num_blocks[0])
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = make_blocks(2, num_blocks[1])
        self.down2_3 = Downsample(dim * 2)
        self.encoder_level3 = make_blocks(3, num_blocks[2])
        self.decoder_level3 = make_blocks(3, num_blocks[2])
        self.up3_2 = Upsample(dim * 4)
        self.fuse2 = Fuse(dim * 2)
        self.decoder_level2 = make_blocks(2, num_blocks[1])
        self.up2_1 = Upsample(dim * 2)
        self.fuse1 = Fuse(dim)
        self.decoder_level1 = make_blocks(1, num_blocks[0])
        self.refinement = make_blocks(1, num_refinement_blocks)
        self.output = nn.Conv2d(dim,
                                out_channels,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=bias)

    @staticmethod
    def _run_blocks(blocks: nn.ModuleList, feature: torch.Tensor,
                    luminance: torch.Tensor, structure: torch.Tensor,
                    demand_tokens: torch.Tensor,
                    reliability_tokens: torch.Tensor) -> torch.Tensor:
        """使同一尺度的多个 DRR Block 共享对应的先验和 token gate。"""
        for block in blocks:
            feature = block(feature, luminance, structure, demand_tokens,
                            reliability_tokens)
        return feature

    def _scale_priors(self, luminance: torch.Tensor,
                      feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """将亮度降到特征尺寸并在该尺度重新计算候选结构观测。"""
        if luminance.shape[-2:] != feature.shape[-2:]:
            luminance = F.interpolate(luminance,
                                      size=feature.shape[-2:],
                                      mode='area')
        structure = calculate_structure_observation(luminance,
                                                    self.structure_tau,
                                                    self.log_epsilon)
        return luminance, structure

    def forward(self,
                inp_img: torch.Tensor,
                return_aux: bool = False):
        """恢复低照度图像，并在训练时按需返回 A/R 预测。

        Args:
            inp_img: 形状为 ``B×3×H×W`` 的低照度 RGB 输入。
            return_aux: 为真时额外返回包含 ``demand``、``reliability`` 的字典。

        Returns:
            默认返回裁回原尺寸的 RGB 恢复图；训练模式返回 ``(restored, aux)``。
        """
        padded_input, original_size = pad_to_factor(inp_img,
                                                    self.input_pad_factor)
        luminance_level1 = calculate_luminance(padded_input)
        structure_level1 = calculate_structure_observation(
            luminance_level1, self.structure_tau, self.log_epsilon)
        input_feature = self.patch_embed(padded_input)
        demand, reliability = self.prior_head(input_feature, luminance_level1,
                                               structure_level1)
        demand_tokens, reliability_tokens = aggregate_detail_gates(
            demand, reliability, detail_patch_size=16, reliability_topk=self.gate_topk)

        encoder_level1 = self._run_blocks(self.encoder_level1, input_feature,
                                           luminance_level1,
                                           structure_level1, demand_tokens,
                                           reliability_tokens)
        input_level2 = self.down1_2(encoder_level1)
        luminance_level2, structure_level2 = self._scale_priors(
            luminance_level1, input_level2)
        encoder_level2 = self._run_blocks(self.encoder_level2, input_level2,
                                           luminance_level2,
                                           structure_level2, demand_tokens,
                                           reliability_tokens)
        input_level3 = self.down2_3(encoder_level2)
        luminance_level3, structure_level3 = self._scale_priors(
            luminance_level1, input_level3)
        encoder_level3 = self._run_blocks(self.encoder_level3, input_level3,
                                           luminance_level3,
                                           structure_level3, demand_tokens,
                                           reliability_tokens)
        decoder_level3 = self._run_blocks(self.decoder_level3, encoder_level3,
                                           luminance_level3,
                                           structure_level3, demand_tokens,
                                           reliability_tokens)
        decoder_level2_input = self.fuse2(self.up3_2(decoder_level3),
                                           encoder_level2)
        decoder_level2 = self._run_blocks(self.decoder_level2,
                                           decoder_level2_input,
                                           luminance_level2,
                                           structure_level2, demand_tokens,
                                           reliability_tokens)
        decoder_level1_input = self.fuse1(self.up2_1(decoder_level2),
                                           encoder_level1)
        decoder_level1 = self._run_blocks(self.decoder_level1,
                                           decoder_level1_input,
                                           luminance_level1,
                                           structure_level1, demand_tokens,
                                           reliability_tokens)
        refined = self._run_blocks(self.refinement, decoder_level1,
                                   luminance_level1, structure_level1,
                                   demand_tokens, reliability_tokens)
        restored = self.output(refined) + padded_input
        original_height, original_width = original_size
        restored = restored[:, :, :original_height, :original_width]
        if not return_aux:
            return restored
        return restored, {
            'demand': demand[:, :, :original_height, :original_width],
            'reliability': reliability[:, :, :original_height, :original_width]
        }
