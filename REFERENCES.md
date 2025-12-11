# 参考项目与开源库

本项目是基于专利实现的原创水印系统，但在开发过程中参考了以下优秀的工作和使用了以下开源库。

## 📚 参考论文与实现

### 经典方法（2018-2023）

### 1. StegaStamp (2020, CVPR)
- **GitHub**: https://github.com/tancik/StegaStamp.git
- **论文**: "Learning to Invert: Signal Recovery via Deep Convolutional Networks"
- **参考内容**: U-Net架构设计思路、对抗训练方法

### 2. HiDDeN (2018, WACV)
- **GitHub**: https://github.com/ando-khachatryan/HiDDeN.git
- **论文**: "Hiding Images in Plain Sight: Deep Steganography"
- **参考内容**: 可微攻击层设计、噪声层实现

### 3. RoSteALS (2023, ICCV)
- **GitHub**: https://github.com/guanzhichen/RoSteALS.git
- **论文**: "Robust Steganography Using Steganographic Adversarial Networks"
- **参考内容**: 鲁棒性攻击模拟、VAE架构

### 4. Stable Signature (2023, Meta)
- **GitHub**: https://github.com/facebookresearch/stable_signature.git
- **论文**: "Stable Signature: Rooting Watermarks in Latent Diffusion Models"
- **参考内容**: 水印预处理、签名验证流程

### 最新方法（2024-2025）

### 5. Tree-Ring Watermarks (2024, NeurIPS)
- **GitHub**: https://github.com/YuxinWenRick/tree-ring-watermark.git
- **论文**: "Tree-Ring Watermarks: Fingerprints for Diffusion Images that are Invisible and Robust"
- **参考内容**: 扩散模型攻击、频域同步模板

### 6. WaterFlow (2025)
- **arXiv**: https://arxiv.org/abs/2504.12354
- **论文**: "WaterFlow: Robust Watermarking in Latent Diffusion Models"
- **参考内容**: 傅里叶域水印嵌入、潜在空间水印
- **代码**: ⚠️ 暂未找到公开代码

### 7. GaussMarker (2025)
- **arXiv**: https://arxiv.org/abs/2506.11444
- **论文**: "GaussMarker: Dual-Domain Watermarking for Diffusion Models"
- **参考内容**: 空间域+频域双域水印、高斯噪声恢复器
- **代码**: ⚠️ 暂未找到公开代码

### 8. WMAdapter (2025, ICML)
- **论文**: "WMAdapter: Plug-and-Play Watermarking for Diffusion Models"
- **参考内容**: 即插即用水印适配器、上下文自适应嵌入
- **代码**: ⚠️ 暂未找到公开代码

### 9. MaXsive (2025)
- **论文**: "MaXsive: Training-Free Generative Watermarking"
- **参考内容**: 无训练水印、X形模板抗几何攻击
- **代码**: ⚠️ 暂未找到公开代码

### 10. SimuFreeMark (2025)
- **arXiv**: https://arxiv.org/abs/2511.11295
- **论文**: "SimuFreeMark: A Noise-Simulation-Free Robust Watermarking Against Image Editing"
- **参考内容**: 无噪声模拟训练、VAE低频嵌入
- **代码**: ⚠️ 暂未找到公开代码

### 11. TrustMark (2025, ICCV) ✅ **已开源**
- **论文**: "TrustMark: Robust Watermarking and Watermark Removal for Arbitrary Resolution Images"
- **GitHub**: https://github.com/adobe/trustmark
- **参考内容**: 空谱损失函数、任意分辨率支持
- **开发者**: Adobe

### 12. InvisMark (2025, WACV) ✅ **已开源**
- **论文**: "InvisMark: Invisible and Robust Watermarking for AI-Generated Image Provenance"
- **GitHub**: https://github.com/microsoft/InvisMark
- **参考内容**: 高分辨率AIGC图像水印、神经网络架构
- **开发者**: Microsoft

### 13. SEAL (2025, ICCV)
- **论文**: "SEAL: Semantic Aware Image Watermarking"
- **参考内容**: 语义感知水印、内容语义信息利用
- **代码**: ⚠️ 暂未找到公开代码

### 14. Text-Guided Invariant Feature Learning (2025)
- **arXiv**: https://arxiv.org/abs/2503.13805
- **论文**: "Text-Guided Invariant Feature Learning for Robust Watermarking"
- **参考内容**: CLIP多模态、文本嵌入作为语义锚点
- **代码**: ⚠️ 暂未找到公开代码

### 15. Hidden in the Noise (2025) ✅ **已开源**
- **GitHub**: https://github.com/Kasraarabi/Hidden-in-the-Noise
- **论文**: "Hidden in the Noise: Two-Stage Robust Watermarking for Images"
- **参考内容**: 两阶段水印框架、扩散模型初始噪声利用

### 16. SFWMark (2025) ✅ **已开源**
- **GitHub**: https://github.com/thomas11809/SFWMark
- **论文**: "SFWMark: Semantic Watermarking Framework"
- **参考内容**: 语义水印框架、傅里叶域分析

### 17. VINE (2025) ✅ **已开源**
- **GitHub**: https://github.com/Shilin-LU/VINE
- **论文**: "VINE: Robust Watermarking Using Generative Priors"
- **参考内容**: 生成先验、图像编辑鲁棒性

## 📅 时间线

- **2018**: HiDDeN - 深度隐写术的开创性工作
- **2020**: StegaStamp - U-Net架构的隐写术
- **2023**: RoSteALS, Stable Signature - 鲁棒性和扩散模型水印
- **2024**: Tree-Ring Watermarks - 频域同步模板
- **2025**: WaterFlow, GaussMarker, WMAdapter, TrustMark, InvisMark, SEAL - 最新AIGC水印方法

## 🔧 使用的开源库

### 核心深度学习库
- **PyTorch**: https://pytorch.org/
  - 深度学习框架
- **torchvision**: https://pytorch.org/vision/
  - 图像处理和数据集

### 感知损失
- **LPIPS (PerceptualSimilarity)**: https://github.com/richzhang/PerceptualSimilarity.git
  - 学习感知图像块相似度指标
  - 用于计算感知损失

### 扩散模型
- **diffusers** (Hugging Face): https://github.com/huggingface/diffusers
  - Stable Diffusion模型加载
  - DDIM调度器和UNet
- **transformers** (Hugging Face): https://github.com/huggingface/transformers
  - CLIP文本编码器

### 密码学与纠错
- **cryptography**: https://github.com/pyca/cryptography
  - ECDSA数字签名
  - 密钥管理
- **python-bchlib**: https://github.com/jkent/python-bchlib.git
  - BCH纠错编码

### 工具库
- **PIL/Pillow**: 图像处理
- **numpy**: 数值计算
- **tqdm**: 进度条
- **tensorboard**: 训练可视化

## 🎯 项目原创性说明

本项目是**基于专利实现的原创系统**，具有以下特点：

### 原创实现
1. **专利驱动的架构设计**
   - 完全按照专利权利要求和实施方式实现
   - 多域分层嵌入（功能分离）
   - 星环同步模板（多尺度、多方向）
   - 异构可微攻击模块

2. **独特的创新点**
   - 结合BCH纠错和ECDSA签名的可信溯源
   - 针对AIGC重绘攻击的鲁棒性设计
   - 同步网络几何校正（Frobenius范数损失）

3. **独立实现的模块**
   - 编码器/解码器网络（U-Net变体）
   - 同步网络（SyncNet）
   - 攻击模块（包括模拟DDIM）
   - 完整的训练和测试流程

### 参考与借鉴
- **架构思路**：参考了StegaStamp、HiDDeN等经典方法
- **攻击模拟**：参考了RoSteALS、Tree-Ring等鲁棒性设计
- **工具库**：使用了LPIPS、bchlib等开源库

## 📝 引用建议

如果使用本项目，建议引用：

```bibtex
@misc{watermark-patent-system,
  title={一种面向生成式模型重绘攻击的图像鲁棒水印嵌入与可信溯源系统},
  author={Your Name},
  year={2024},
  note={基于专利实现的原创系统}
}
```

## 🙏 致谢

感谢以下开源项目和论文作者：
- StegaStamp团队
- HiDDeN作者
- RoSteALS作者
- Tree-Ring Watermarks作者
- Stable Signature (Meta)团队
- LPIPS作者
- 所有开源库的维护者

## 📄 许可证

本项目代码为原创实现，遵循相应的开源许可证。
使用的第三方库遵循各自的许可证。
