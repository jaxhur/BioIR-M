"""为 BEAR-BioIR 增加路由与稠密结构监督的 BasicSR 训练适配层。"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict

import torch
import torch.nn.functional as F

from basicsr.models.image_restoration_model import ImageRestorationModel


class BEARBioIRModel(ImageRestorationModel):
    """复用原 ImageRestorationModel，加入 BEPR 与结构头辅助损失。

    主恢复路径仍使用配置中的 RGB L1 和原 FFTLoss；本类不改写 BasicSR 的
    数据加载、优化器、scheduler、日志、验证、断点与 checkpoint 规则。
    预测阶段的 ``net_g`` 仍返回普通 RGB Tensor，因此 ``test_lol.py`` 无需
    依赖训练期辅助输出。
    """

    def __init__(self, opt: Dict) -> None:
        """读取路由及结构损失权重后初始化原 BasicSR 图像恢复模型。

        Args:
            opt: BasicSR 解析后的实验配置，其中 ``routing_loss_opt`` 可选。
        """
        super().__init__(opt)
        routing_opt = opt.get('train', {}).get('routing_loss_opt', {})
        self.budget_weight = float(routing_opt.get('budget_weight', 0.05))
        self.scope_weight = float(routing_opt.get('scope_weight', 0.01))
        self.reliability_weight = float(
            routing_opt.get('reliability_weight', 0.05))
        self.structure_weight = float(
            routing_opt.get('structure_weight', 0.05))
        self.structure_scale_weights = tuple(float(value) for value in
                                             routing_opt.get(
                                                 'structure_scale_weights',
                                                 [1.0, 0.5, 0.25]))
        if len(self.structure_scale_weights) != 3:
            raise ValueError(
                'structure_scale_weights must contain weights for 1x/2x/4x')
        if any(value < 0 for value in self.structure_scale_weights):
            raise ValueError('structure_scale_weights cannot be negative')
        if sum(self.structure_scale_weights) <= 0:
            raise ValueError('At least one structure scale weight must be positive')

    def _structure_loss(self, prediction: torch.Tensor,
                        target: torch.Tensor) -> torch.Tensor:
        """按 ``1×/2×/4×`` area 尺度计算归一化 SmoothL1。

        Args:
            prediction: 结构头预测的 ``B×1×H_p×W_p`` 稠密结构图。
            target: 由 GT 线性亮度生成、与预测对齐的软 Sobel 目标。

        Returns:
            尚未乘 ``lambda_S`` 的多尺度结构监督标量。
        """
        if prediction.shape != target.shape:
            raise ValueError(
                f'Structure shapes differ: {prediction.shape} vs {target.shape}')
        weighted_losses = []
        for factor, weight in zip((1, 2, 4), self.structure_scale_weights):
            if factor == 1:
                scaled_prediction, scaled_target = prediction, target
            else:
                output_size = (prediction.shape[-2] // factor,
                               prediction.shape[-1] // factor)
                scaled_prediction = F.interpolate(
                    prediction, size=output_size, mode='area')
                scaled_target = F.interpolate(
                    target, size=output_size, mode='area')
            weighted_losses.append(
                weight * F.smooth_l1_loss(scaled_prediction, scaled_target))
        return sum(weighted_losses) / sum(self.structure_scale_weights)

    def _routing_losses(self, aux: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """计算 ``L_b``、``L_scope``、``L_R`` 与 ``L_S`` 辅助损失。

        Args:
            aux: 网络 ``return_aux=True`` 时输出的结构、预算、范围和可靠性预测。

        Returns:
            四项已经乘配置权重的标量 Tensor，名称可直接写入训练日志。
        """
        bare_network = self.get_bare_model(self.net_g)
        # 目标路径只服务监督，不应保留梯度图或反向影响固定观测计算。
        with torch.no_grad():
            targets = bare_network.build_routing_targets(self.lq, self.gt)

        budget_loss = F.smooth_l1_loss(aux['budget'], targets['budget'])
        scope_losses = [
            (scope.mean(dim=(-2, -1)) - aux['budget']).abs().mean()
            for scope in aux['scopes']
        ]
        scope_loss = torch.stack(scope_losses).mean()
        reliability_losses = [
            F.smooth_l1_loss(reliability, targets['reliability'])
            for reliability in aux['reliabilities']
        ]
        reliability_loss = torch.stack(reliability_losses).mean()
        structure_loss = self._structure_loss(
            aux['structure'], targets['structure'])
        return {
            'l_budget': self.budget_weight * budget_loss,
            'l_scope': self.scope_weight * scope_loss,
            'l_reliability': self.reliability_weight * reliability_loss,
            'l_structure': self.structure_weight * structure_loss,
        }

    def optimize_parameters(self, current_iter: int, tb_logger) -> None:
        """以原恢复损失、路由监督和稠密结构监督联合优化生成网络。

        Args:
            current_iter: 当前 BasicSR global iteration，用于保持接口一致。
            tb_logger: 原训练入口创建的 TensorBoard logger，可为 ``None``。
        """
        del current_iter, tb_logger
        self.optimizer_g.zero_grad()
        restored, aux = self.net_g(self.lq, return_aux=True)
        self.output = restored
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

        routing_losses = self._routing_losses(aux)
        for name, loss in routing_losses.items():
            total_loss = total_loss + loss
            loss_dict[name] = loss
        loss_dict['l_total'] = total_loss

        # 保留原项目的梯度裁剪默认值与优化器更新方式。
        total_loss.backward()
        if self.opt['train'].get('use_grad_clip', True):
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self) -> None:
        """验证时直接交给网络执行一次 64 倍数入口补边与精确裁回。

        原模型的测试方法会先按 32 倍数补边；BEAR-BioIR 自身必须唯一地按
        64 倍数定义路由网格，因此此处避免外部二次补边，确保训练、验证和
        ``test_lol.py`` 的几何坐标完全一致。
        """
        was_training = self.net_g.training
        self.net_g.eval()
        with torch.no_grad():
            output = self.net_g(self.lq)
            self.output = output[-1] if isinstance(output, list) else output
        if was_training:
            self.net_g.train()
