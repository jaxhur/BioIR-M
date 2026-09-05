# 创建环境

创建环境：

```
git clone https://github.com/jaxhur/BioIR-M.git

cd BioIR-M
git switch codex/bear-bioir

conda create -n bear-bioir python=3.10 -y --override-channels -c https://repo.anaconda.com/pkgs/main
conda activate bear-bioir

# 安装依赖
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install opencv-python lmdb tqdm einops scipy scikit-image tensorboard natsort pyiqa joblib lpips ptflops scikit-learn pandas thop

python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"


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

#hyperai
cp /openbayes/input/input0/LOL-v1.zip /openbayes/input/input0/LOL-v2-renamed.zip /openbayes/home/BioIR-M/datasets

# AUtoDL
cp /root/autodl-fs/LOL-v1.zip /root/BioIR/Single_Composite/datasets
cp /root/autodl-fs/LOL-v2-renamed.zip /root/BioIR/Single_Composite/datasets

# 解压
cd /openbayes/home/BioIR-M
cd ./datasets
unzip LOL-v1.zip -d LOL-v1
unzip LOL-v2-renamed.zip -d LOL-v2

rm LOL-v1.zip LOL-v2-renamed.zip
cd ../
```

目录结构

```
datasets/LOL-v1-Fr/our485/{low,high}
datasets/LOL-v1-Fr/eval15/{low,high}

datasets/LOL-v2-Fr/Synthetic/Train/{Low,Normal}
datasets/LOL-v2-Fr/Synthetic/Test/{Low,Normal}

datasets/LOL-v2-Fr/Real_captured/Train/{Low,Normal}
datasets/LOL-v2-Fr/Real_captured/Test/{Low,Normal}
```





# 训练

## 训练配置

| Dataset     | BatchSize | PatchSize | Actual iterations |
| ----------- | --------: | --------: | ----------------: |
| LOL-v1      |         4 |       256 |           150,000 |
| LOL-v2-syn  |         4 |       256 |           150,000 |
| LOL-v2-real |         4 |       256 |           150,000 |









## Tensorboard

```
tensorboard --logdir ./BioIR-M/experiments/BioIR-LOLv1/tb_looger --port 6008
```



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

```
CUDA_VISIBLE_DEVICES=0 sh train.sh options/BEAR-LOLv1.yml ; \
CUDA_VISIBLE_DEVICES=0 sh train.sh options/BEAR-LOLv2-real.yml ; \
CUDA_VISIBLE_DEVICES=0 sh train.sh options/BEAR-LOLv2-syn.yml
```

train_all.sh

```
#!/bin/bash

set -e

export CUDA_VISIBLE_DEVICES=0

echo "===== Training LOLv1 ====="
sh train.sh options/BEAR-LOLv1.yml

echo "===== Training LOLv2-real ====="
sh train.sh options/BEAR-LOLv2-real.yml

echo "===== Training LOLv2-syn ====="
sh train.sh options/BEAR-LOLv2-syn.yml

echo "===== All training finished ====="
```

训练

```
CUDA_VISIBLE_DEVICES=0 sh train.sh options/BEAR-LOLv1.yml
```

```
# 查看当前错误链接
ls -ld ./tf_dir
readlink ./tf_dir

# 只删除软连接
unlink ./tf_dir

# 重新建立正确链接
ln -s "./BioIR-M/experiments/BEAR-BioIR-LOLv1/tb_looger" "./tf_dir"
ln -s "./BioIR-M/experiments/BEAR-BioIR-v2-LOLv1/tb_looger" "./tf_dir"

# 验证最终指向
readlink -f ./tf_dir
ls ./tf_dir
```

测试

```
python test_lol.py --opt options/BEAR-LOLv1.yml --weights ./experiments/BEAR-BioIR-LOLv1/models/best_G.pth
```

<img src="img/README_img/image-20260903212234348.png" alt="image-20260903212234348" style="zoom:67%;" />

<img src="img/README_img/image-20260903212255751.png" alt="image-20260903212255751" style="zoom:80%;" />

# LOLv2-real

训练

```
CUDA_VISIBLE_DEVICES=0 sh train.sh options/BEAR-LOLv2-real.yml
```

```
# 查看当前错误链接
ls -ld ./tf_dir
readlink ./tf_dir

# 只删除软连接
unlink ./tf_dir

# 重新建立正确链接
ln -s "./BioIR-M/experiments/BEAR-BioIR-LOLv2-real/tb_looger" "./tf_dir"


# 验证最终指向
readlink -f ./tf_dir
ls ./tf_dir
```

测试

```
python test_lol.py --opt options/BEAR-LOLv2-real.yml --weights experiments/BEAR-BioIR-LOLv2-real/models/best_G.pth
```



# LOLv2-syn

训练

```
CUDA_VISIBLE_DEVICES=0 sh train.sh options/BEAR-LOLv2-syn.yml
```

测试

```
python test_lol.py --opt options/BEAR-LOLv2-syn.yml --weights experiments/BEAR-BioIR-LOLv2-syn/models/best_G.pth
```

