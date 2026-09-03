# BEAR-BioIR 方案 3 实现说明

## 实现范围

本实现位于独立分支 `codex/bear-bioir`，保留原 `BioIR_arch.py`、原 YAML 与
原恢复流程作为 baseline 对照。新增的 `BEAR_BioIR_arch.py` 使用原 BioIR 的
三尺度 encoder–decoder、skip fusion、上/下采样、普通 GDFN 与 RGB 残差输出，
仅将原有 12 个 `AttBlock` 替换为 12 个独立参数的 `SARIBlock`。

`BEPR` 在一次前向中从补边低光输入计算三种欠曝观测、融合曝光证据、整图范围
先验 `b`、结构强度与方向一致性。encoder 三个尺度第一次进入 SARI 前分别生成
`(A_1,R_1)`、`(A_2,R_2)`、`(A_3,R_3)`；对应 decoder 重用同尺度的图，四个
refinement block 重用一级的图。`SARI` 的 `G→D` 只执行一次细节 token 查询全局
token 的注意力，并用 Bernstein 权重混合 null/local/global 消息；`D→G` 不执行
第二次 attention，而是在已知的 4×4 区域归属中做可靠性门控均值。

## 为使方案可执行而明确的处理

1. 方案正文的个别段落仅写 `R_2,R_3`，但总览、使用位置表和 `L_R` 公式要求
   `R_1,R_2,R_3`。代码按后者的完整定义实现三张可靠性图，并让三者共同接受
   `R_D*` 监督。
2. 原方案已指出：`R_D*` 用 Top-16 强调稀疏可靠边缘，而发送 token 若直接做
   PatchAvg 会把该边缘稀释。代码不新增第四个学习分支，而是复用已计算的
   `S_l·C_l`，在每个细节区域内部做无参数 Softmax 加权池化，再由 `R_s` 控制
   该 token 向所属全局区域发送的资格。因此 `R_s` 的区域语义不变，发送内容
   与稀疏结构标签更一致。
3. 方案将 `L_R` 称为“加权”损失，但给出的公式没有 GT 权重项。代码严格实现
   公式中的三尺度 SmoothL1 平均，未杜撰额外加权规则。若要验证“GT 结构更重”
   的说法，应先补充明确权重公式，再单独做消融。
4. `L_scope` 公式确实对三尺度求平均；正文中“两个尺度”的表述视为笔误，代码
   对三张 `A_s` 都约束其空间均值接近 `b`。

## 入口、配置与产物

- 训练入口不变：`train.sh`。三套新增配置为
  `options/BEAR-LOLv1.yml`、`options/BEAR-LOLv2-syn.yml`、
  `options/BEAR-LOLv2-real.yml`。
- 测试入口不变：`test_lol.py`。其默认推理补边已改为方案规定的 64 倍数；模型
  内部也会保证直接调用时采用同一规则，输出和指标前裁回原图尺寸。
- 三套配置均为单卡 `BatchSize=4`、训练 `PatchSize=256×256`、`total_iter=150000`、
  每 20 iter 训练日志、每约 1000 iter 完整测试集验证。`train.py` 已有的
  latest/best/定时权重、state、`train.log`、`val.log` 和 TensorBoard 机制不变。
- 原项目的 `test_lol.py` 已按 RGB PSNR/SSIM、LPIPS-Alex-v0.1、THOP 的
  Params(M)/GMACs(G)/GFLOPs(G) 保存增强图和 `metric.csv`，本次没有重写它。

## 4090/5090 环境基线

`train.sh` 不再固定 `CUDA_VISIBLE_DEVICES=0`，由服务器调用命令决定可见单卡。
环境应使用 PyTorch 官方 CUDA 12.8 或更高 build；仓库 README 使用官方 cu128
索引安装命令。每次切换租用服务器后，记录 `torch.__version__`、
`torch.version.cuda`、NVIDIA driver、Python 和实际 GPU，并先完成一个真实 batch
的前向、反向与 AMP 冒烟，才开始完整训练。本网络没有自定义 CUDA 扩展；4090 和
5090 的兼容性尚未在目标服务器实测，不能据本地 CPU 测试宣称已验证。

## 后续实验风险

- `b*` 只由 Rec.709 亮度差构造，而 BEPR 输入还融合最大通道与 HSL lightness；
  应做单亮度/三亮度证据消融，确认多亮度输入没有被 `L_b` 压成 Rec.709 单一路径。
- 三尺度 `R_s` 共享 `R_D*`，独立尺度头只能降低、不能完全排除三张图趋同。训练后
  应记录三对路由图的相关性，并比较共享头、无尺度修正和当前三尺度头。
- Top-k、路由网格、LayerScale 和路由损失权重都是首版超参数，当前没有实验结果，
  不应写成已验证最优设置。
