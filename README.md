# BioIR原始论文

<img src="img/README_img/image-20260630192144846.png" alt="image-20260630192144846" style="zoom:80%;" />

<img src="img/README_img/image-20260716224418604.png" alt="image-20260716224418604" style="zoom:67%;" />



# 创建环境

## 一键创建环境与数据集：

```bash
git clone https://github.com/jaxhur/BioIR-M.git
cd BioIR-M
git switch codex/drr-v0

bash scripts/bootstrap_environment_and_datasets.sh
```

```bash
# 明确删除并重建已有 bioir 环境。
bash scripts/bootstrap_environment_and_datasets.sh --recreate-env

# 使用 AutoDL 已缓存的两个 ZIP。
bash scripts/bootstrap_environment_and_datasets.sh --data-source autodl

# 已手动放好数据时，只校验目录与配对。
bash scripts/bootstrap_environment_and_datasets.sh --skip-env --skip-data
```

## 手动创建

创建环境：

```
git clone https://github.com/jaxhur/BioIR-M.git
cd BioIR-M
git switch codex/drr-v0

conda remove -n bioir --all -y
conda create -n bioir python=3.9 -y 
conda activate bioir

# 安装依赖
# conda install pytorch=2.4.0 torchvision pytorch-cuda=12.4 -c pytorch -c nvidia -y
pip install --no-cache-dir torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install opencv-python lmdb tqdm einops scipy scikit-image tensorboard natsort pyiqa joblib lpips ptflops scikit-learn pandas thop

python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"


# 安装basicsr
cd BioIR-M
python -m pip install -e .
# 旧版命令：python setup.py develop --no_cuda_ext
# 验证
python -c "import basicsr; print(basicsr.__file__)"
```



兼容5090、4090

```
git clone https://github.com/jaxhur/BioIR-M.git
cd BioIR-M
git switch codex/drr-v0

conda create -n bioir-cu128 python=3.9 -y
# 强制走官方源
conda create -n bioir-cu128 python=3.9 -y  --override-channels -c https://repo.anaconda.com/pkgs/main
conda activate bioir-cu128

# 安装依赖
# conda install pytorch=2.4.0 torchvision pytorch-cuda=12.4 -c pytorch -c nvidia -y
python -m pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install opencv-python lmdb tqdm einops scipy scikit-image tensorboard natsort pyiqa joblib lpips ptflops scikit-learn pandas thop


python -c "import torch; print('torch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('available:', torch.cuda.is_available()); print('arch list:', torch.cuda.get_arch_list()); [print(f'gpu{i}: {torch.cuda.get_device_name(i)}, sm_{torch.cuda.get_device_capability(i)[0]}{torch.cuda.get_device_capability(i)[1]}') for i in range(torch.cuda.device_count())]"

# 安装basicsr
cd BioIR-M
python -m pip install -e .
# 旧版命令：python setup.py develop --no_cuda_ext
# 验证
python -c "import basicsr; print(basicsr.__file__)"
```



# 数据集

数据集：LOLv1、LOLv2-real、LOLv2-syn

```
pip install -U gdown
apt install -y unzip

cd ./datasets
# LOL-v1
gdown "https://drive.google.com/uc?id=1mAN3ll5wWwt1Xz0C7uio31-NJu-50S8Z"
# LOL-v2
gdown "https://drive.google.com/uc?id=1L0UnJg6gZ4Eb7It2EuNxP0L3lQNmKMaP"

# 如果是AUtoDL平台
cp /root/autodl-fs/LOL-v1.zip /root/BioIR/Single_Composite/datasets
cp /root/autodl-fs/LOL-v2-renamed.zip /root/BioIR/Single_Composite/datasets

# 解压
unzip LOL-v1.zip -d LOL-v1
unzip LOL-v2-renamed.zip -d LOL-v2

rm LOL-v1.zip LOL-v2-renamed.zip
cd ../
```

目录结构

```
datasets/LOL-v1/our485/{low,high}
datasets/LOL-v1/eval15/{low,high}

datasets/LOL-v2/Synthetic/Train/{Low,Normal}
datasets/LOL-v2/Synthetic/Test/{Low,Normal}

datasets/LOL-v2/Real_captured/Train/{Low,Normal}
datasets/LOL-v2/Real_captured/Test/{Low,Normal}
```





# 训练

## 训练配置

| Dataset     | BatchSize | PatchSize | Actual iterations |
| ----------- | --------: | --------: | ----------------: |
| LOL-v1      |         4 |       256 |           150,000 |
| LOL-v2-syn  |         4 |       256 |           150,000 |
| LOL-v2-real |         4 |       256 |           150,000 |

不同方案应使用不同的 YAML `name`。训练产物和测试产物会按该实验名隔离，便于在同一服务器顺序切换分支训练。

## 训练产物

**周期性输出评价指标、保存模型权重、断点状态**：

```
experiments/<实验名/
  models/
    latest_G.pth
    best_G.pth
    1000_G.pth
  training_state/
    1000.state
  logs/
    train.log
    val.log
  tb_looger/
  visualization/
```



```
tensorboard --logdir ./BioIR-M/experiments/DRR-BioIR-v0-LOLv1/tb_looger --port 6009

```



# 测试

## 预训练权重

预训练权重，放到`pretrained_models/`

## 测试产物

`test_lol.py` 是唯一测试入口：同时完成推理、保存增强图、按同名 GT 计算 PSNR/SSIM/LPIPS，统计模型 Params(M) 和输入 `1x3x256x256` 的单次前向复杂度，并写入逐图与汇总指标。

- PSNR/SSIM 固定使用 BasicSR 的 RGB 全图口径（`crop_border=0`）；
- LPIPS 固定使用 AlexNet v0.1，RGB 输入归一化到 `[-1, 1]`。
- Params 按全部生成网络参数除以 `1e6` 统计；
- 复杂度使用 THOP，`GMACs=MACs/1e9`、`GFLOPs=2×MACs/1e9`。

```
# 测试
python test_lol.py --opt options/LOL-v2-syn.yml --weights pretrained_models/LOL-v2-syn.pth
# 额外保存低光图/增强图/GT 的横向拼接对比图
python test_lol.py --opt options/LOL-v2-syn.yml --weights pretrained_models/LOL-v2-syn.pth --save_comparison
```



## 测试产物

```text
test_result/<实验名>/<数据集名>/
  enhanced/              # 增强后图片
  per_image_metrics.csv  # 每张图的 PSNR/SSIM/LPIPS
  metric.csv             # 全测试集平均指标和 Params/GMACs/GFLOPs
```





# LOLv1

训练

- 耗时：
  - 4090：最多15h、至少13h
    - 一般都是0.25$，有时能弄到0.22\$，总成本=14\*0.25\*6.8=23.8
  - 5090
    - 一般是0.35\$，有时能弄到0.32\$，总成本=23.8/0.35/6.8=10，只要5090上的耗时小于10小时，就使用5090

```
sh train.sh options/DRR-BioIR-v0-LOL-v1.yml

tensorboard --logdir ./BioIR-M/experiments/DRR-BioIR-v0-gaussianA-LOLv1/tb_looger --port 6009
```

先验

- `input_lq`：固定的一张低照度训练样本。后面四张都对应它。
- `demand_target`：训练阶段由 LQ 与 GT 亮度差计算的 \(A^*\)。白色表示“这里确实需要更多提亮/上下文校正”，黑色表示不需要大幅改。
- `demand_prediction`：预测头仅根据低照度输入预测的 \(A\)。测试时真正使用的是它，不会看 GT。
- `reliability_target`：由 LQ 与 GT 的结构一致性生成的 \(R^*\)。白色边缘表示“GT 中确实有结构，且低光图里方向、强度也还可信”；黑色既可能是平坦区域，也可能是噪声或丢失的边缘。
- `reliability_prediction`：模型从低照度图预测的 \(R\)。



测试

- PSNR：
- SSIM：
- LPIPS：
- 参数量(M)：
- FLOPS(G)：

```
# LOL-v1
python test_lol.py --opt options/DRR-BioIR-v0-LOL-v1.yml --weights experiments/DRR-BioIR-v0-LOLv1/models/best_G.pth --dataset LOL-v1
```



# LOLv2-real

训练

- 耗时
  - 4090：15h

```
sh train.sh options/DRR-BioIR-v0-LOL-v2-real.yml
tensorboard --logdir ./BioIR-M/experiments/DRR-BioIR-v0-gaussianA-LOLv2-real/tb_looger --port 6009
```

测试

- PSNR：
- SSIM：
- LPIPS：
- 参数量(M)：
- FLOPS(G)：

```
python test_lol.py --opt options/DRR-BioIR-v0-LOL-v2-real.yml --weights ./experiments/DRR-BioIR-v0-LOLv2-real/models/best_G.pth --dataset LOL-v2-real

```



# LOLv2-syn

训练

- 耗时
  - 5090：11h

```
sh train.sh options/DRR-BioIR-v0-LOL-v2-syn.yml
tensorboard --logdir ./BioIR-M/experiments/DRR-BioIR-v0-gaussianA-LOLv2-syn/tb_looger --port 6010
```

测试

- PSNR：
- SSIM：
- LPIPS：
- 参数量(M)：
- FLOPS(G)：

```
python test_lol.py --opt options/DRR-BioIR-v0-LOL-v2-syn.yml --weights experiments/DRR-BioIR-v0-LOLv2-syn/models/best_G.pth --dataset LOL-v2-syn

ln -s "./BioIR-M/experiments/DRR-BioIR-v0-gaussianA-LOLv2-syn/tb_looger" "./tf_dir"
```

