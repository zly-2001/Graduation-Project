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
# 必需库
pip install torch torchvision
pip install lpips                    # 感知损失
pip install bchlib                   # BCH纠错编码
pip install cryptography             # ECDSA数字签名
pip install pillow numpy tqdm tensorboard
pip install diffusers transformers   # 可选
```

**实际使用的库**：详见 [实际使用的库和参考.md](实际使用的库和参考.md)

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

### 经典方法
- [StegaStamp](https://github.com/tancik/StegaStamp.git) - CVPR 2020
- [HiDDeN](https://github.com/ando-khachatryan/HiDDeN.git) - WACV 2018
- [RoSteALS](https://github.com/guanzhichen/RoSteALS.git) - ICCV 2023
- [Stable Signature](https://github.com/facebookresearch/stable_signature.git) - Meta 2023

### 最新方法（2024-2025）
- [Tree-Ring Watermarks](https://github.com/YuxinWenRick/tree-ring-watermark.git) - NeurIPS 2024
- [TrustMark](https://github.com/adobe/trustmark) (2025, ICCV) - Adobe开源，空谱损失函数
- [InvisMark](https://github.com/microsoft/InvisMark) (2025, WACV) - Microsoft开源，AIGC图像溯源
- [Hidden in the Noise](https://github.com/Kasraarabi/Hidden-in-the-Noise) (2025) - 两阶段水印框架
- [SFWMark](https://github.com/thomas11809/SFWMark) (2025) - 语义水印框架
- [VINE](https://github.com/Shilin-LU/VINE) (2025) - 生成先验水印
- **WaterFlow** (2025) - 潜在空间傅里叶域水印（代码待发布）
- **GaussMarker** (2025) - 双域水印策略（代码待发布）
- **SEAL** (2025, ICCV) - 语义感知水印（代码待发布）

### 使用的开源库
- [LPIPS](https://github.com/richzhang/PerceptualSimilarity.git) - 感知损失
- [python-bchlib](https://github.com/jkent/python-bchlib.git) - BCH纠错编码
- [diffusers](https://github.com/huggingface/diffusers) - 扩散模型

**完整参考列表请查看**: [REFERENCES.md](REFERENCES.md)

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