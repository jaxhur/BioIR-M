# BioIR原始论文

<img src="img/README_img/image-20260630192144846.png" alt="image-20260630192144846" style="zoom:80%;" />

<img src="img/README_img/image-20260716224418604.png" alt="image-20260716224418604" style="zoom:67%;" />



# 配置环境

配置conda环境：

```
git clone https://github.com/jaxhur/BioIR.git
git clone https://gitee.com/wallcaptain/BioIR.git

conda remove -n bioir --all -y
conda create -n bioir python=3.9 -y 
conda activate bioir

# 安装依赖
# conda install pytorch=2.4.0 torchvision pytorch-cuda=12.4 -c pytorch -c nvidia -y
pip install --no-cache-dir torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install opencv-python lmdb tqdm einops scipy scikit-image tensorboard natsort pyiqa joblib lpips ptflops scikit-learn pandas

python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"


# 安装basicsr
cd BioIR/Single_Composite
# 把 setuptools 降到 64 以下，并确保有 wheel。原因是你当前的新版 setuptools/pip 会用“隔离构建环境”，那个临时环境里看不到你已经安装好的 torch，所以报 No module named 'torch'。
python -m pip install "setuptools<64" wheel
python setup.py develop --no_cuda_ext
```





# 数据集

数据集：LOLv1、LOLv2

```
pip install -U gdown
apt install -y unzip

cd ./datasets
# LOL-v1
gdown "https://drive.google.com/uc?id=1mAN3ll5wWwt1Xz0C7uio31-NJu-50S8Z"
# LOL-v2原始
# gdown "https://drive.google.com/uc?id=1dzLJFz0svHXYHvAe-Tl52miChhF4BXXE"
# LOL-v2重命名
gdown "https://drive.google.com/uc?id=1L0UnJg6gZ4Eb7It2EuNxP0L3lQNmKMaP"

# AUtoDL
cp /root/autodl-fs/LOL-v1.zip /root/BioIR/Single_Composite/datasets
cp /root/autodl-fs/LOL-v2-renamed.zip /root/BioIR/Single_Composite/datasets


# 
cd /root/BioIR/Single_Composite/datasets
unzip LOL-v1.zip -d LOL-v1
unzip LOL-v2-renamed.zip -d LOL-v2

rm LOL-v1.zip LOL-v2-renamed.zip
cd ../
```



目录结构

```
Single_Composite/
  datasets/
    LOL-v1/
      our485/
        low/
        high/
      eval15/
        low/
        high/
    LOL-v2/
      Synthetic/
        Train/
          Low/
          Normal/
        Test/
          Low/
          Normal/
      Real_captured/
        Train/
          Low/
          Normal/
        Test/
          Low/
          Normal/
```





# 训练

原始 README用 `torchrun`。它是 PyTorch 分布式启动器，即使只有 1 张 GPU，也按“单进程分布式”方式跑。普通单卡实验不需要优先用它；原生 Windows 上还可能因为 `nccl` 分布式后端不可用而报错。

```
# win
torchrun --nproc_per_node=1 --master_port=4322 basicsr/train.py -opt options/LOL-v1.yml --launcher pytorch
torchrun --nproc_per_node=1 --master_port=4322 basicsr/train.py -opt options/LOL-v2-syn.yml --launcher pytorch
torchrun --nproc_per_node=1 --master_port=4322 basicsr/train.py -opt options/LOL-v2-real.yml --launcher pytorch

# linux
sh train.sh options/LOL-v1.yml
sh train.sh options/LOL-v2-syn.yml
sh train.sh options/LOL-v2-real.yml
```

**周期性输出评价指标、保存模型权重和断点状态**：训练中断后，原训练脚本会自动从 `experiments/<实验名>/training_states/` 里最新的 `.state` 恢复

```
val:
  val_freq: 1e3

logger:
  save_checkpoint_freq: 1e3
```

**训练产物**：

```
experiments\<实验名>\
  models\
  training_states\
  
# 示例
Single_Composite\experiments\BioIR-LOLv1\models\net_g_1000.pth
Single_Composite\experiments\BioIR-LOLv1\models\latest_G.pth
Single_Composite\experiments\BioIR-LOLv1\training_states\1000.state
```





# 测试

`test_lol.py` 是唯一测试入口：同时完成推理、保存增强图、按同名 GT 计算 PSNR/SSIM/LPIPS，统计模型 Params(M) 和输入 `1x3x256x256` 的单次前向复杂度，并写入逐图与汇总指标。

- 下载预训练权重，放到`pretrained_models/`
- 计算口径
  - PSNR/SSIM 固定使用 BasicSR 的 RGB 全图口径（`crop_border=0`）；
  - LPIPS 固定使用 AlexNet v0.1，RGB 输入归一化到 `[-1, 1]`。
  - Params 按全部生成网络参数除以 `1e6` 统计；
  - 复杂度使用 THOP，`GMACs=MACs/1e9`、`GFLOPs=2×MACs/1e9`。

```
# 测试
python test_lol.py --opt options/LOL-v2-syn.yml --weights pretrained_models/LOL-v2-syn.pth
# 额外保存低光图/增强图/GT 的横向拼接对比图
python test_lol.py --opt options/LOL-v2-syn.yml --weights pretrained_models/LOL-v2-syn.pth --save_comparison

# 测试自己训练出的权重
# LOL-v1
python test_lol.py --opt options/LOL-v1.yml --weights experiments/BioIR-LOLv1/models/latest_G.pth
# LOL-v2-syn
python test_lol.py --opt options/LOL-v2-syn.yml --weights experiments/BioIR-LOLv2-syn/models/latest_G.pth
# LOL-v2-real
python test_lol.py --opt options/LOL-v2-real.yml --weights experiments/BioIR-LOLv2-real/models/latest_G.pth
```

测试产物：

```text
test_result/<数据集名>/
  enhanced/              # 增强后图片
  per_image_metrics.csv  # 每张图的 PSNR/SSIM/LPIPS
  metric.csv             # 全测试集平均指标和 Params/GMACs/GFLOPs
```









