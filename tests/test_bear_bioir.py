"""验证 BEAR-BioIR 的结构开关、路由与反向传播闭环。"""

import unittest
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from basicsr.models.archs import define_network
from basicsr.models.archs.BEAR_BioIR_arch import (
    BEARBioIR, DenseStructurePredictor, FixedCIConvW, SARIInteraction,
    StructureChannelRefinement, pad_to_multiple)
from basicsr.models.bear_bioir_model import BEARBioIRModel


class FakeTensorBoardWriter:
    """记录 ``add_image`` 调用，避免单元测试依赖真实事件文件。"""

    def __init__(self):
        self.images = []

    def add_image(self, tag, image, global_step, dataformats):
        """保存一次图片记录调用的标签、形状与迭代信息。"""
        self.images.append((tag, image.clone(), global_step, dataformats))


class TestBEARBioIR(unittest.TestCase):
    """在 CPU 合成 Tensor 上检查方案 3 的关键几何与梯度连接。"""

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

    def test_structure_configuration_matrix(self):
        """验证两项 YAML 开关的四种组合均保持独立且可前向。"""
        low = torch.rand(1, 3, 64, 64)
        gt = torch.rand(1, 3, 64, 64)
        for structure_source in ('fixed', 'predicted'):
            for refine_with_structure in (False, True):
                with self.subTest(
                        structure_source=structure_source,
                        refine_with_structure=refine_with_structure):
                    model = BEARBioIR(
                        dim=8, num_blocks=[0, 0, 0],
                        num_refinement_blocks=1, attention_heads=4,
                        structure_source=structure_source,
                        refine_with_structure=refine_with_structure).eval()
                    with torch.no_grad():
                        restored, aux = model(low, return_aux=True)
                        targets = model.build_routing_targets(low, gt)

                    self.assertEqual(restored.shape, low.shape)
                    self.assertEqual(aux['structure'].shape, (1, 1, 64, 64))
                    self.assertEqual(
                        model.structure_predictor is not None,
                        structure_source == 'predicted')
                    self.assertEqual(
                        'structure' in targets,
                        structure_source == 'predicted')
                    self.assertTrue(all(
                        (block.interaction.structure_refinement is not None)
                        == refine_with_structure
                        for block in model.refinement))

                    if structure_source == 'fixed':
                        fixed_prior = model.router.prepare(low)['sender_prior']
                        torch.testing.assert_close(
                            aux['structure'], fixed_prior)

                    wrapper = object.__new__(BEARBioIRModel)
                    wrapper.net_g = model
                    wrapper.lq = low
                    wrapper.gt = gt
                    wrapper.budget_weight = 0.05
                    wrapper.scope_weight = 0.01
                    wrapper.reliability_weight = 0.05
                    wrapper.structure_weight = 0.05
                    wrapper.structure_scale_weights = (1.0, 0.5, 0.25)
                    routing_losses = wrapper._routing_losses(aux)
                    self.assertEqual(
                        'l_structure' in routing_losses,
                        structure_source == 'predicted')

    def test_constant_route_and_average_sender_for_m1(self):
        """验证 M1 不创建 BEPR/结构头，并严格使用常数 A/R 与 PatchAvg。"""
        model = BEARBioIR(
            dim=8, num_blocks=[1, 0, 0], num_refinement_blocks=0,
            attention_heads=4, routing_mode='constant',
            constant_scope=0.5, constant_reliability=1.0,
            sender_pooling='average', structure_source='fixed',
            refine_with_structure=False).eval()
        low = torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            restored, aux = model(low, return_aux=True)

        self.assertIsNone(model.router)
        self.assertIsNone(model.structure_predictor)
        self.assertIsNone(aux['structure'])
        self.assertEqual(restored.shape, low.shape)
        for scope, reliability in zip(
                aux['scopes'], aux['reliabilities']):
            torch.testing.assert_close(scope, torch.full_like(scope, 0.5))
            torch.testing.assert_close(
                reliability, torch.ones_like(reliability))
        with self.assertRaises(RuntimeError):
            model.build_routing_targets(low, low)

    def test_sender_pooling_and_direction_switches(self):
        """验证发送池化与两个消息方向能独立切换且保持输出形状。"""
        detail = torch.arange(8 * 8, dtype=torch.float32).reshape(
            1, 1, 8, 8).repeat(1, 4, 1, 1)
        structure = torch.zeros(1, 1, 8, 8)
        structure[:, :, 1::2, 1::2] = 1.0
        average_interaction = SARIInteraction(
            dim=4, heads=1, route_patch=2, global_patch=8,
            sender_pooling='average')
        structure_interaction = SARIInteraction(
            dim=4, heads=1, route_patch=2, global_patch=8,
            sender_pooling='structure')
        average_tokens = average_interaction._sender_tokens(detail, None)
        structure_tokens = structure_interaction._sender_tokens(
            detail, structure)
        torch.testing.assert_close(
            average_tokens, average_interaction._tokens(detail, 2))
        self.assertFalse(torch.allclose(average_tokens, structure_tokens))

        low = torch.rand(1, 3, 64, 64)
        for enable_g_to_d, enable_d_to_g in ((False, True), (True, False)):
            with self.subTest(
                    enable_g_to_d=enable_g_to_d,
                    enable_d_to_g=enable_d_to_g):
                model = BEARBioIR(
                    dim=8, num_blocks=[1, 0, 0],
                    num_refinement_blocks=0, attention_heads=4,
                    enable_g_to_d=enable_g_to_d,
                    enable_d_to_g=enable_d_to_g).eval()
                with torch.no_grad():
                    restored = model(low)
                self.assertEqual(restored.shape, low.shape)
                interaction = model.encoder_level1[0].interaction
                self.assertIs(interaction.enable_g_to_d, enable_g_to_d)
                self.assertIs(interaction.enable_d_to_g, enable_d_to_g)

    def test_invalid_structure_configuration_is_rejected(self):
        """验证错误的结构来源或非布尔 refinement 开关会立即报错。"""
        with self.assertRaises(ValueError):
            BEARBioIR(structure_source='unknown')
        with self.assertRaises(TypeError):
            BEARBioIR(refine_with_structure='true')
        with self.assertRaises(ValueError):
            BEARBioIR(routing_mode='unknown')
        with self.assertRaises(ValueError):
            BEARBioIR(sender_pooling='unknown')
        with self.assertRaises(TypeError):
            BEARBioIR(enable_g_to_d='false')
        with self.assertRaises(ValueError):
            BEARBioIR(route_patch=10, global_patch=40)

    def test_lol_yaml_defaults_keep_current_v2_behavior(self):
        """验证三套 LOL YAML 均能构建当前 C 版默认组合。"""
        option_directory = Path(__file__).resolve().parents[1] / 'options'
        option_names = (
            'BEAR-LOLv1.yml',
            'BEAR-LOLv2-real.yml',
            'BEAR-LOLv2-syn.yml',
        )
        for option_name in option_names:
            with self.subTest(option_name=option_name):
                with (option_directory / option_name).open(
                        'r', encoding='utf-8') as option_file:
                    options = yaml.safe_load(option_file)
                network_options = dict(options['network_g'])
                self.assertEqual(
                    network_options['structure_source'], 'predicted')
                self.assertIs(
                    network_options['refine_with_structure'], False)
                network = define_network(network_options)
                self.assertIsNotNone(network.structure_predictor)
                self.assertTrue(all(
                    block.interaction.structure_refinement is None
                    for block in network.refinement))

    def test_ablation_yaml_matrix(self):
        """逐项核对八份 YAML 的实验隔离、开关和共同训练口径。"""
        option_directory = Path(__file__).resolve().parents[1] / 'options'
        expected = {
            'ablation_m0_lolv1.yml': {
                'model_type': 'ImageRestorationModel',
                'network_type': 'BioIR',
            },
            'ablation_m1_lolv1.yml': {
                'model_type': 'ImageRestorationModel',
                'routing_mode': 'constant',
                'sender_pooling': 'average',
            },
            'ablation_m2_lolv1.yml': {
                'model_type': 'BEARBioIRModel',
                'routing_mode': 'bepr',
                'sender_pooling': 'average',
            },
            'ablation_ours_lolv1.yml': {
                'model_type': 'BEARBioIRModel',
                'routing_mode': 'bepr',
                'sender_pooling': 'structure',
            },
            'ablation_wo_g2d_lolv1.yml': {
                'model_type': 'BEARBioIRModel',
                'enable_g_to_d': False,
                'enable_d_to_g': True,
            },
            'ablation_wo_d2g_lolv1.yml': {
                'model_type': 'BEARBioIRModel',
                'enable_g_to_d': True,
                'enable_d_to_g': False,
            },
            'sensitivity_fine_lolv1.yml': {
                'route_patch': 8,
                'global_patch': 32,
                'topk': 4,
            },
            'sensitivity_coarse_lolv1.yml': {
                'route_patch': 32,
                'global_patch': 128,
                'topk': 64,
            },
        }
        experiment_names = set()
        for option_name, assertions in expected.items():
            with self.subTest(option_name=option_name):
                with (option_directory / option_name).open(
                        'r', encoding='utf-8') as option_file:
                    options = yaml.safe_load(option_file)
                network_options = options['network_g']
                experiment_names.add(options['name'])
                self.assertEqual(options['datasets']['train']['gt_size'], 256)
                self.assertEqual(
                    options['datasets']['train']['batch_size_per_gpu'], 4)
                self.assertEqual(options['train']['total_iter'], 150000)
                self.assertEqual(options['logger']['print_freq'], 20)
                self.assertEqual(options['val']['val_freq'], 1000.0)
                self.assertIs(
                    options['val']['metrics']['ssim']['ssim3d'], False)
                for key, value in assertions.items():
                    if key == 'model_type':
                        actual = options['model_type']
                    elif key == 'network_type':
                        actual = network_options['type']
                    else:
                        actual = network_options[key]
                    self.assertEqual(actual, value)

                if option_name.startswith('ablation_wo_'):
                    self.assertIs(options['find_unused_parameters'], True)
                if network_options['type'] == 'BEARBioIR':
                    network = define_network(dict(network_options))
                    self.assertEqual(
                        network.global_patch, 4 * network.route_patch)
        self.assertEqual(len(experiment_names), len(expected))

    def test_sensitivity_padding_matches_plan(self):
        """验证三档 q_g 对 LOL-v1 400×600 完整图得到计划中的补边尺寸。"""
        image = torch.zeros(1, 3, 400, 600)
        expected_sizes = {
            32: (416, 608),
            64: (448, 640),
            128: (512, 640),
        }
        for global_patch, expected_size in expected_sizes.items():
            with self.subTest(global_patch=global_patch):
                padded, height, width = pad_to_multiple(image, global_patch)
                self.assertEqual((height, width), (400, 600))
                self.assertEqual(padded.shape[-2:], expected_size)

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

    def test_tensorboard_validation_visuals(self):
        """验证 TensorBoard 记录五类图像并裁回原始空间尺寸。"""
        wrapper = object.__new__(BEARBioIRModel)
        wrapper.opt = {
            'val': {
                'val_freq': 1000,
                'tensorboard_images': {
                    'enabled': True,
                    'interval': 1000,
                    'max_samples': 1,
                    'include_structure_target': True,
                },
            },
        }
        wrapper.net_g = BEARBioIR(
            dim=8, num_blocks=[0, 0, 0], num_refinement_blocks=0,
            attention_heads=4)
        wrapper.lq = torch.rand(1, 3, 65, 67)
        wrapper.gt = torch.rand(1, 3, 65, 67)
        wrapper._collect_tensorboard_visuals = True
        wrapper.test()

        self.assertEqual(wrapper.output.shape, wrapper.lq.shape)
        self.assertEqual(
            wrapper.structure_prediction.shape, (1, 1, 65, 67))
        visuals = OrderedDict([
            ('lq', wrapper.lq.detach().cpu()),
            ('result', wrapper.output.detach().cpu()),
            ('gt', wrapper.gt.detach().cpu()),
        ])
        writer = FakeTensorBoardWriter()
        wrapper._log_validation_images_to_tensorboard(
            writer, 1000, 0, 'LOLv1-test', '1', visuals)

        expected_tags = [
            'validation/LOLv1-test/1/01_input_lq',
            'validation/LOLv1-test/1/02_prediction',
            'validation/LOLv1-test/1/03_ground_truth',
            'validation/LOLv1-test/1/04_structure_prediction',
            'validation/LOLv1-test/1/05_structure_target',
        ]
        self.assertEqual(
            [record[0] for record in writer.images], expected_tags)
        self.assertTrue(all(record[2] == 1000 for record in writer.images))
        self.assertTrue(all(record[3] == 'CHW' for record in writer.images))
        self.assertEqual(writer.images[3][1].shape, (1, 65, 67))
        self.assertEqual(writer.images[4][1].shape, (1, 65, 67))

        wrapper._log_validation_images_to_tensorboard(
            writer, 1500, 0, 'LOLv1-test', '1', visuals)
        wrapper._log_validation_images_to_tensorboard(
            writer, 2000, 1, 'LOLv1-test', '2', visuals)
        self.assertEqual(len(writer.images), 5)


if __name__ == '__main__':
    unittest.main()
