# zly滴原创毕业设计项目完成模型部分实验喽！嘻嘻😁

## 🚀 快速开始

### 环境配置

**Mac系统（推荐MPS加速）**：
```bash
# 启用MPS回退（如果不支持MPS自动使用CPU）
echo 'export PYTORCH_ENABLE_MPS_FALLBACK=1' >> ~/.zshrc
source ~/.zshrc
```

**安装依赖**：
```bash
pip install torch torchvision
pip install lpips
pip install bchlib
pip install cryptography
pip install diffusers transformers  
```

### 数据准备

```bash
# 下载DIV2K数据集（推荐）
python utils/download_div2k.py

# 或下载COCO数据集
python utils/download_coco.py
```

### 训练

```bash
python experiments/train.py
```

### 测试

```bash
python experiments/test.py
```

## 📚 参考项目

本项目参考了以下优秀工作（详见 `REFERENCES.md`）：

- [StegaStamp](https://github.com/tancik/StegaStamp.git) - CVPR 2020
- [HiDDeN](https://github.com/ando-khachatryan/HiDDeN.git) - WACV 2018
- [RoSteALS](https://github.com/guanzhichen/RoSteALS.git) - ICCV 2023
- [Tree-Ring Watermarks](https://github.com/YuxinWenRick/tree-ring-watermark.git) - NeurIPS 2024
- [Stable Signature](https://github.com/facebookresearch/stable_signature.git) - Meta 2023
- [LPIPS](https://github.com/richzhang/PerceptualSimilarity.git) - 感知损失

## 📖 项目结构

```
watermark/
├── experiments/          # 训练和测试脚本
│   ├── train.py         # 训练主程序
│   └── test.py          # 测试/提取程序
├── models/              # 模型定义
│   ├── encoder.py       # 编码器网络
│   ├── decoder.py       # 解码器网络
│   ├── sync_net.py      # 同步网络
│   └── attacks.py       # 攻击模块
├── utils/               # 工具函数
│   ├── watermark_utils.py  # 水印预处理（BCH+ECDSA）
│   ├── sync_pattern.py      # 星环同步模板
│   ├── losses.py            # 损失函数
│   └── dataset.py           # 数据集加载
└── results/             # 训练结果
    ├── checkpoints/     # 模型检查点
    ├── visualizations/  # 可视化图像
    └── logs/            # TensorBoard日志
```

## 🔬 技术特点

### 1. 多域分层嵌入
- 载荷嵌入到低频语义域（U-Net瓶颈层）
- 同步模板嵌入到中高频纹理域（解码路径）

### 2. 异构攻击模块
- 传统攻击：JPEG、仿射变换、高斯噪声
- AI攻击：GAN风格、DDIM去噪（模拟/真实）
- 可微攻击：支持端到端训练

### 3. 可信溯源
- BCH纠错编码：抵抗比特错误
- ECDSA数字签名：确保来源真实性
- 结构化证据包：完整的法证链条