"""DRR-BioIR 的训练模型封装。"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict

import torch
import torch.nn.functional as F

from basicsr.models.archs.drr_bioir_arch import (aggregate_detail_gates,
                                                  build_drr_targets,
                                                  pad_to_factor)
from basicsr.models.image_restoration_model import ImageRestorationModel


class DRRImageRestorationModel(ImageRestorationModel):
    """在原有恢复训练流程中加入 A/R 有监督路由损失与诊断。"""

    def init_training_settings(self):
        """复用基础训练配置，并读取 DRR 专用损失和可视化参数。"""
        super().init_training_settings()
        drr_opt = self.opt['train'].get('drr_loss_opt', {})
        self.demand_loss_weight = float(drr_opt.get('demand_loss_weight', 0.05))
        self.reliability_loss_weight = float(
            drr_opt.get('reliability_loss_weight', 0.05))
        self.smooth_l1_beta = float(drr_opt.get('smooth_l1_beta', 0.10))
        self.demand_target_type = str(
            drr_opt.get('demand_target_type', 'relative_gap'))
        self.demand_supervision_type = str(
            drr_opt.get('demand_supervision_type', 'pixel'))
        self.demand_epsilon = float(drr_opt.get('demand_epsilon', 0.05))
        self.demand_gaussian_kernel_size = int(
            drr_opt.get('demand_gaussian_kernel_size', 15))
        self.demand_gaussian_sigma = float(
            drr_opt.get('demand_gaussian_sigma', 3.0))
        self.demand_tau = float(drr_opt.get('demand_tau', 1.0))
        if self.demand_target_type not in {
                'relative_gap', 'gaussian_smoothed_absolute_gap'}:
            raise ValueError(
                'demand_target_type 只能是 relative_gap 或 '
                'gaussian_smoothed_absolute_gap。')
        if self.demand_supervision_type not in {'pixel', 'gate'}:
            raise ValueError('demand_supervision_type 只能是 pixel 或 gate。')
        self.structure_tau = float(drr_opt.get('structure_tau', 0.10))
        self.magnitude_tau = float(drr_opt.get('magnitude_tau', 0.05))
        self.log_epsilon = float(drr_opt.get('log_epsilon', 0.02))
        self.reliability_epsilon = float(
            drr_opt.get('reliability_epsilon', 1e-6))
        self.edge_weight = float(drr_opt.get('edge_weight', 4.0))
        self.prior_visualization_freq = int(
            drr_opt.get('prior_visualization_freq', 1000))
        self._probe_lq = None
        self._probe_targets = None

    @staticmethod
    def _spearman_correlation(prediction: torch.Tensor,
                              target: torch.Tensor) -> torch.Tensor:
        """按图计算并平均 Spearman 相关性，用于检查 A/R 是否退化为常数图。"""
        pred_flat = prediction.detach().flatten(1)
        target_flat = target.detach().flatten(1)
        pred_rank = torch.argsort(torch.argsort(pred_flat, dim=1), dim=1).float()
        target_rank = torch.argsort(torch.argsort(target_flat, dim=1), dim=1).float()
        pred_rank = pred_rank - pred_rank.mean(dim=1, keepdim=True)
        target_rank = target_rank - target_rank.mean(dim=1, keepdim=True)
        denominator = torch.sqrt(pred_rank.square().sum(dim=1) *
                                 target_rank.square().sum(dim=1)).clamp_min(1e-12)
        return (pred_rank * target_rank).sum(dim=1).div(denominator).mean()

    @staticmethod
    def _prior_statistics(prefix: str, prediction: torch.Tensor,
                          target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """统计 A/R 的分布、饱和比例和监督一致性。"""
        flat = prediction.detach().flatten()
        target_detached = target.detach()
        return {
            f'{prefix}_mean': flat.mean(),
            f'{prefix}_std': flat.std(unbiased=False),
            f'{prefix}_q05': torch.quantile(flat, 0.05),
            f'{prefix}_q25': torch.quantile(flat, 0.25),
            f'{prefix}_q50': torch.quantile(flat, 0.50),
            f'{prefix}_q75': torch.quantile(flat, 0.75),
            f'{prefix}_q95': torch.quantile(flat, 0.95),
            f'{prefix}_ratio_low': (flat < 0.05).float().mean(),
            f'{prefix}_ratio_mid': ((flat >= 0.05) & (flat <= 0.95)).float().mean(),
            f'{prefix}_ratio_high': (flat > 0.95).float().mean(),
            f'{prefix}_mae': F.l1_loss(prediction.detach(), target_detached),
            f'{prefix}_spearman': DRRImageRestorationModel._spearman_correlation(
                prediction, target_detached)
        }

    def _cache_probe(self, targets: Dict[str, torch.Tensor]):
        """缓存首个训练 batch，供后续 TensorBoard 对比同一张 A/R 热图。"""
        if self._probe_lq is None:
            self._probe_lq = self.lq[:1].detach().clone()
            self._probe_targets = {
                name: value[:1].detach().clone()
                for name, value in targets.items() if name != 'edge_presence'
            }

    def _log_probe_images(self, current_iter: int, tb_logger):
        """定期记录固定训练样本的输入、预测 A/R 和目标热图。"""
        if (tb_logger is None or self._probe_lq is None or
                self.prior_visualization_freq <= 0 or
                current_iter % self.prior_visualization_freq != 0):
            return
        with torch.no_grad():
            _, auxiliary = self.net_g(self._probe_lq, return_aux=True)
        tb_logger.add_image('drr_priors/input_lq', self._probe_lq[0].clamp(0, 1),
                            current_iter)
        for name in ('demand', 'reliability'):
            tb_logger.add_image(f'drr_priors/{name}_prediction',
                                auxiliary[name][0].clamp(0, 1), current_iter)
            tb_logger.add_image(f'drr_priors/{name}_target',
                                self._probe_targets[name][0].clamp(0, 1),
                                current_iter)

    def _build_demand_gate_target(self) -> torch.Tensor:
        """按网络补边与池化规则构造真实 A_D 的配对监督目标。"""
        bare_net = self.get_bare_model(self.net_g)
        padded_lq, _ = pad_to_factor(self.lq, bare_net.input_pad_factor)
        padded_gt, _ = pad_to_factor(self.gt, bare_net.input_pad_factor)
        padded_targets = build_drr_targets(
            padded_lq,
            padded_gt,
            demand_epsilon=self.demand_epsilon,
            demand_target_type=self.demand_target_type,
            demand_gaussian_kernel_size=self.demand_gaussian_kernel_size,
            demand_gaussian_sigma=self.demand_gaussian_sigma,
            demand_tau=self.demand_tau,
            structure_tau=self.structure_tau,
            magnitude_tau=self.magnitude_tau,
            log_epsilon=self.log_epsilon,
            reliability_epsilon=self.reliability_epsilon)
        demand_gate_target, _ = aggregate_detail_gates(
            padded_targets['demand'],
            padded_targets['reliability'],
            detail_patch_size=bare_net.gate_detail_patch_size,
            reliability_topk=bare_net.gate_topk)
        return demand_gate_target

    def optimize_parameters(self, current_iter, tb_logger):
        """执行恢复损失、A/R 辅助监督及参数更新。"""
        self.optimizer_g.zero_grad()
        restored, auxiliary = self.net_g(self.lq, return_aux=True)
        self.output = restored
        targets = build_drr_targets(
            self.lq,
            self.gt,
            demand_epsilon=self.demand_epsilon,
            demand_target_type=self.demand_target_type,
            demand_gaussian_kernel_size=self.demand_gaussian_kernel_size,
            demand_gaussian_sigma=self.demand_gaussian_sigma,
            demand_tau=self.demand_tau,
            structure_tau=self.structure_tau,
            magnitude_tau=self.magnitude_tau,
            log_epsilon=self.log_epsilon,
            reliability_epsilon=self.reliability_epsilon)

        total_loss = restored.new_zeros(())
        loss_dict = OrderedDict()
        if self.cri_pix:
            pixel_loss = self.cri_pix(restored, self.gt)
            total_loss = total_loss + pixel_loss
            loss_dict['l_pix'] = pixel_loss
        if self.cri_fft:
            fft_loss = self.cri_fft(restored, self.gt)
            total_loss = total_loss + fft_loss
            loss_dict['l_fft'] = fft_loss

        if self.demand_supervision_type == 'gate':
            demand_loss = F.smooth_l1_loss(
                auxiliary['demand_gate'],
                self._build_demand_gate_target(),
                beta=self.smooth_l1_beta)
        else:
            demand_loss = F.smooth_l1_loss(auxiliary['demand'],
                                           targets['demand'],
                                           beta=self.smooth_l1_beta)
        reliability_error = F.smooth_l1_loss(auxiliary['reliability'],
                                             targets['reliability'],
                                             beta=self.smooth_l1_beta,
                                             reduction='none')
        reliability_weight = 1.0 + self.edge_weight * targets['edge_presence']
        reliability_loss = (reliability_error * reliability_weight).sum() / (
            reliability_weight.sum() + self.reliability_epsilon)
        total_loss = total_loss + self.demand_loss_weight * demand_loss
        total_loss = total_loss + self.reliability_loss_weight * reliability_loss
        loss_dict['l_demand'] = demand_loss
        loss_dict['l_reliability'] = reliability_loss
        loss_dict['l_total'] = total_loss

        # 保留零梯度连接，兼容 BasicSR 对所有参数参与图构建的既有约定。
        total_loss = total_loss + 0.0 * sum(
            parameter.sum() for parameter in self.net_g.parameters())
        total_loss.backward()
        if self.opt['train'].get('use_grad_clip', True):
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()

        self._cache_probe(targets)
        loss_dict.update(
            self._prior_statistics('demand', auxiliary['demand'],
                                   targets['demand']))
        loss_dict.update(
            self._prior_statistics('reliability', auxiliary['reliability'],
                                   targets['reliability']))
        self.log_dict = self.reduce_loss_dict(loss_dict)
        self._log_probe_images(current_iter, tb_logger)
