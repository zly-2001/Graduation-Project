"""
专利步骤S3：异构可微攻击模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torchvision import transforms

class HeterogeneousAttack(nn.Module):
    def __init__(self, use_light_aigc: bool = True, use_inpaint: bool = False, use_ip2p: bool = False, use_ddim: bool = True, no_attack: bool = False):
        """
        异构攻击模块：模拟多种攻击
        
        Args:
            use_light_aigc: 是否使用轻量AIGC攻击（模拟DDIM，使用相同公式但轻量实现）
            use_inpaint: 是否使用inpainting攻击
            use_ip2p: 是否使用InstructPix2Pix攻击
            use_ddim: 是否使用真正的DDIM攻击（实施方式要求）
            no_attack: 是否禁用所有攻击（第一阶段训练：让模型先学会基础嵌入）
        """
        super().__init__()
        
        # 攻击池权重（可学习）
        self.attack_weights = nn.Parameter(torch.ones(3))

        # 轮询指针，确保每个攻击在若干 batch 内都被用到
        self._attack_index = 0
        self.use_light_aigc = use_light_aigc
        self.use_inpaint = use_inpaint
        self.use_ip2p = use_ip2p
        self.use_ddim = use_ddim
        self.no_attack = no_attack  # 无攻击模式
        
        if no_attack:
            print("⚠️  无攻击模式：所有攻击已禁用，直接返回原图（用于第一阶段训练）")

        # 延迟加载重模型，避免无依赖时报错
        self._inpaint_pipe = None
        self._ip2p_pipe = None
        self._ddim_unet = None
        self._ddim_scheduler = None
        self._ddim_vae = None  # VAE编码器/解码器
    
    def jpeg_compression(self, image, quality=None):
        """
        JPEG压缩近似（可微）
        """
        if quality is None:
            quality = random.uniform(30, 95)
        
        # 简化版：使用卷积模拟压缩
        kernel_size = int(100 / quality)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = max(3, min(kernel_size, 7))
        
        blurred = F.avg_pool2d(
            image, 
            kernel_size=kernel_size, 
            stride=1, 
            padding=kernel_size//2
        )
        return blurred
    
    def affine_transform(self, image):
        """
        仿射变换（旋转、缩放）
        """
        B, C, H, W = image.shape
        
        # 随机生成变换参数
        angle = random.uniform(-15, 15) * 3.14159 / 180
        scale = random.uniform(0.8, 1.2)
        
        # 构建变换矩阵
        cos_a = scale * torch.cos(torch.tensor(angle))
        sin_a = scale * torch.sin(torch.tensor(angle))
        
        theta = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0]
        ]).unsqueeze(0).repeat(B, 1, 1).to(image.device)
        
        # 生成采样网格
        grid = F.affine_grid(theta, image.size(), align_corners=False)
        
        # 应用变换
        transformed = F.grid_sample(image, grid, align_corners=False)
        
        return transformed
    
    def gaussian_noise(self, image):
        """
        高斯噪声
        """
        noise_level = random.uniform(0.01, 0.05)
        noise = torch.randn_like(image) * noise_level
        return torch.clamp(image + noise, -1, 1)
    
    def gan_style_attack(self, image):
        """
        GAN风格攻击
        模拟生成模型的图像重绘
        """
        # 简化实现：颜色扰动 + 纹理模糊
        
        # 1. 颜色扰动
        color_shift = torch.randn(1, 3, 1, 1).to(image.device) * 0.1
        image_shifted = image + color_shift
        
        # 2. 纹理模糊
        image_blurred = F.avg_pool2d(
            image_shifted, 
            kernel_size=3, 
            stride=1, 
            padding=1
        )
        
        # 3. 混合
        alpha = random.uniform(0.3, 0.7)
        return alpha * image_shifted + (1 - alpha) * image_blurred
    
    def diffusion_attack(self, image):
        """
        扩散模型攻击
        模拟加噪-去噪过程
        """
        # 1. 加噪（模拟前向过程）
        noise_level = random.uniform(0.1, 0.3)
        noise = torch.randn_like(image) * noise_level
        noisy_image = image + noise
        
        # 2. 去噪（简化的反向过程）
        denoised = F.conv2d(
            noisy_image,
            # depthwise 3x3 blur per通道，避免groups维度不匹配
            torch.ones(3, 1, 5, 5, device=image.device) / 25,
            padding=2,
            groups=3
        )
        
        return torch.clamp(denoised, -1, 1)

    def _load_ddim_model(self, device):
        """
        加载预训练的DDIM模型（延迟加载）
        
        实施方式要求：使用预训练且参数冻结的噪声预测网络ε_θ
        """
        if self._ddim_unet is None:
            try:
                from diffusers import UNet2DConditionModel, DDIMScheduler, AutoencoderKL
                from transformers import CLIPTextModel, CLIPTokenizer
                import os
                
                # 加载预训练的Stable Diffusion模型组件
                model_id = "runwayml/stable-diffusion-v1-5"
                
                # 尝试使用本地缓存（避免网络请求）
                cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
                local_path = None
                
                # 检查缓存中是否有模型
                if os.path.exists(cache_dir):
                    # 查找模型快照目录
                    model_cache = os.path.join(cache_dir, "models--runwayml--stable-diffusion-v1-5", "snapshots")
                    if os.path.exists(model_cache):
                        snapshots = [d for d in os.listdir(model_cache) if os.path.isdir(os.path.join(model_cache, d))]
                        if snapshots:
                            local_path = os.path.join(model_cache, snapshots[0])
                            print(f"📂 使用本地缓存模型: {local_path}")
                
                # Mac M4优化：大型模型放到CPU以避免MPS内存不足
                # MPS设备检测
                use_cpu_for_large_models = (device.type == "mps")
                ddim_device = "cpu" if use_cpu_for_large_models else device
                if use_cpu_for_large_models:
                    print("💡 Mac M4检测：将DDIM模型放到CPU以避免内存不足（速度较慢但稳定）")
                
                # 加载VAE（用于RGB <-> Latent转换）
                # 注意：VAE可能没有下载，如果下载失败则使用简化版本
                vae_loaded = False
                if local_path and os.path.exists(os.path.join(local_path, "vae")):
                    try:
                        self._ddim_vae = AutoencoderKL.from_pretrained(
                            local_path,
                            subfolder="vae",
                            local_files_only=True
                        ).to(ddim_device)  # 使用CPU或MPS
                        vae_loaded = True
                    except Exception as e:
                        print(f"⚠️ VAE本地加载失败: {e}")
                
                if not vae_loaded:
                    print("⚠️ VAE未在本地缓存中找到，尝试从网络下载...")
                    try:
                        self._ddim_vae = AutoencoderKL.from_pretrained(
                            model_id,
                            subfolder="vae"
                        ).to(ddim_device)  # 使用CPU或MPS
                        vae_loaded = True
                    except Exception as e:
                        print(f"⚠️ VAE下载失败: {e}")
                        print("⚠️ 将使用简化版DDIM攻击（不使用VAE，直接在RGB空间操作）")
                        self._ddim_vae = None
                
                if vae_loaded:
                    for param in self._ddim_vae.parameters():
                        param.requires_grad = False
                    self._ddim_vae.eval()
                
                # 加载UNet（噪声预测网络ε_θ）
                if local_path and os.path.exists(os.path.join(local_path, "unet")):
                    # 使用本地路径，设置local_files_only=True避免网络请求
                    self._ddim_unet = UNet2DConditionModel.from_pretrained(
                        local_path,
                        subfolder="unet",
                        local_files_only=True
                    ).to(ddim_device)  # 使用CPU或MPS
                else:
                    # 如果本地没有，尝试从网络下载（可能失败）
                    self._ddim_unet = UNet2DConditionModel.from_pretrained(
                        model_id, 
                        subfolder="unet"
                    ).to(ddim_device)  # 使用CPU或MPS
                # 冻结参数（实施方式要求）
                for param in self._ddim_unet.parameters():
                    param.requires_grad = False
                self._ddim_unet.eval()
                
                # 加载DDIM调度器（用于计算ᾱ_t）
                if local_path and os.path.exists(os.path.join(local_path, "scheduler")):
                    self._ddim_scheduler = DDIMScheduler.from_pretrained(
                        local_path,
                        subfolder="scheduler",
                        local_files_only=True
                    )
                else:
                    self._ddim_scheduler = DDIMScheduler.from_pretrained(
                        model_id,
                        subfolder="scheduler"
                    )
                
                # 加载文本编码器（用于条件生成，但单步去噪可以不用）
                if local_path and os.path.exists(os.path.join(local_path, "tokenizer")):
                    self._ddim_tokenizer = CLIPTokenizer.from_pretrained(
                        local_path,
                        subfolder="tokenizer",
                        local_files_only=True
                    )
                    self._ddim_text_encoder = CLIPTextModel.from_pretrained(
                        local_path,
                        subfolder="text_encoder",
                        local_files_only=True
                    ).to(ddim_device)  # 使用CPU或MPS
                else:
                    self._ddim_tokenizer = CLIPTokenizer.from_pretrained(
                        model_id,
                        subfolder="tokenizer"
                    )
                    self._ddim_text_encoder = CLIPTextModel.from_pretrained(
                        model_id,
                        subfolder="text_encoder"
                    ).to(ddim_device)  # 使用CPU或MPS
                for param in self._ddim_text_encoder.parameters():
                    param.requires_grad = False
                self._ddim_text_encoder.eval()
                
                print("✅ DDIM模型加载成功（包含VAE）")
            except ImportError:
                print("⚠️ diffusers库未安装，将使用简化版DDIM攻击")
                self.use_ddim = False
            except Exception as e:
                print(f"⚠️ DDIM模型加载失败: {e}，将使用简化版DDIM攻击")
                self.use_ddim = False
    
    def ddim_attack(self, image):
        """
        实施方式（3）：基于扩散模型的单步去噪攻击
        
        使用预训练且参数冻结的噪声预测网络ε_θ，根据DDIM公式计算：
        x_t = I_w
        x_{0_pred} = (x_t - √(1-ᾱ_t) • ε_θ(x_t, t)) / √(ᾱ_t)
        
        Args:
            image: [B, 3, H, W] 带水印图像I_w，范围[-1, 1]
        
        Returns:
            x_0_pred: [B, 3, H, W] 攻击后的图像，范围[-1, 1]
        """
        device = image.device
        B, C, H, W = image.shape
        
        # 如果未启用DDIM或加载失败，使用简化版本
        if not self.use_ddim:
            return self.pseudo_ddim_denoise(image)
        
        # 延迟加载模型
        self._load_ddim_model(device)
        
        if self._ddim_unet is None:
            return self.pseudo_ddim_denoise(image)
        
        # 如果VAE未加载，使用简化版本（直接在RGB空间操作，不符合标准但可用）
        if self._ddim_vae is None:
            print("⚠️ VAE未加载，使用简化版DDIM（RGB空间近似）")
            return self.pseudo_ddim_denoise(image)
        
        # 将图像从[-1, 1]转换到[0, 1]（Stable Diffusion的输入范围）
        image_01 = (image + 1.0) / 2.0
        
        # 将图像resize到512x512（Stable Diffusion的标准输入尺寸）
        if H != 512 or W != 512:
            image_512 = F.interpolate(image_01, size=(512, 512), mode='bilinear', align_corners=False)
        else:
            image_512 = image_01
        
        # 使用VAE编码器将RGB图像转换为latent space（4通道）
        # 注意：如果VAE在CPU上，需要将图像移到CPU
        vae_device = next(self._ddim_vae.parameters()).device
        image_512_vae = image_512.to(vae_device) if image_512.device != vae_device else image_512
        
        with torch.no_grad():
            # VAE编码：RGB [B,3,512,512] -> Latent [B,4,64,64]
            latent = self._ddim_vae.encode(image_512_vae).latent_dist.sample()
            # 缩放因子（Stable Diffusion标准）
            latent = latent * self._ddim_vae.config.scaling_factor
        
        # 将latent移回原设备（如果VAE在CPU上）
        latent = latent.to(device)
        
        # 随机选择时间步t（实施方式要求）
        # 使用较大的时间步以模拟更强的攻击
        t = torch.randint(
            low=int(0.3 * self._ddim_scheduler.config.num_train_timesteps),
            high=int(0.7 * self._ddim_scheduler.config.num_train_timesteps),
            size=(B,),
            device=device
        )
        
        # 计算ᾱ_t（累积噪声调度系数）
        alphas_cumprod = self._ddim_scheduler.alphas_cumprod.to(device)
        alpha_bar_t = alphas_cumprod[t].view(B, 1, 1, 1)
        
        # 准备输入：将latent视为x_t（加噪后的latent）
        # 为了模拟攻击，我们假设latent已经是某个时间步的加噪latent
        x_t_latent = latent
        
        # 准备文本条件（使用空提示词，因为单步去噪不需要强条件）
        # 注意：text_encoder可能在CPU上，需要确保输入在正确设备
        text_encoder_device = next(self._ddim_text_encoder.parameters()).device
        
        prompt = [""] * B
        text_inputs = self._ddim_tokenizer(
            prompt,
            padding="max_length",
            max_length=self._ddim_tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        # 将输入移到text_encoder所在的设备
        text_inputs = {k: v.to(text_encoder_device) for k, v in text_inputs.items()}
        
        with torch.no_grad():
            # 获取文本编码
            text_embeddings = self._ddim_text_encoder(text_inputs['input_ids'])[0]
        
        # 将text_embeddings移回原设备（用于UNet）
        text_embeddings = text_embeddings.to(device)
        
        # 使用噪声预测网络ε_θ预测噪声（实施方式要求）
        # 注意：如果UNet在CPU上，需要将输入移到CPU
        unet_device = next(self._ddim_unet.parameters()).device
        x_t_latent_unet = x_t_latent.to(unet_device) if x_t_latent.device != unet_device else x_t_latent
        text_embeddings_unet = text_embeddings.to(unet_device) if text_embeddings.device != unet_device else text_embeddings
        
        # 注意：这里需要启用梯度以支持端到端训练
        noise_pred = self._ddim_unet(
            x_t_latent_unet,
            t.to(unet_device),
            encoder_hidden_states=text_embeddings_unet
        ).sample
        
        # 将结果移回原设备
        noise_pred = noise_pred.to(device)
        
        # 根据DDIM公式计算x_0_pred（实施方式公式）
        # x_{0_pred} = (x_t - √(1-ᾱ_t) • ε_θ(x_t, t)) / √(ᾱ_t)
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)
        
        x_0_pred_latent = (x_t_latent - sqrt_one_minus_alpha_bar_t * noise_pred) / sqrt_alpha_bar_t
        
        # 使用VAE解码器将latent转换回RGB图像
        # 注意：如果VAE在CPU上，需要将latent移到CPU
        x_0_pred_latent_vae = x_0_pred_latent.to(vae_device) if x_0_pred_latent.device != vae_device else x_0_pred_latent
        
        with torch.no_grad():
            # 反缩放
            x_0_pred_latent_vae = x_0_pred_latent_vae / self._ddim_vae.config.scaling_factor
            # VAE解码：Latent [B,4,64,64] -> RGB [B,3,512,512]
            x_0_pred = self._ddim_vae.decode(x_0_pred_latent_vae).sample
        
        # 将结果移回原设备
        x_0_pred = x_0_pred.to(device)
        
        # 将结果resize回原始尺寸
        if H != 512 or W != 512:
            x_0_pred = F.interpolate(x_0_pred, size=(H, W), mode='bilinear', align_corners=False)
        
        # 将图像从[0, 1]转换回[-1, 1]
        x_0_pred = x_0_pred * 2.0 - 1.0
        
        return torch.clamp(x_0_pred, -1, 1)

    def pseudo_ddim_denoise(self, image):
        """
        轻量“DDIM去噪”占位：先加噪，再进行可微平滑，近似重绘/去噪效果
        """
        # 模拟DDIM单步去噪公式：x_{0_pred} = (x_t - √(1-ᾱ_t) • ε_θ(x_t, t)) / √(ᾱ_t)
        B, C, H, W = image.shape
        device = image.device
        dtype = image.dtype
        
        # 模拟时间步t和ᾱ_t（DDIM调度）
        # 随机选择"时间步"（模拟30%-70%范围，对应真正的DDIM）
        t_ratio = random.uniform(0.3, 0.7)
        alpha_bar_t = 1.0 - t_ratio  # 简化的ᾱ_t
        sqrt_alpha_bar_t = torch.sqrt(torch.tensor(alpha_bar_t, device=device, dtype=dtype))
        sqrt_one_minus_alpha_bar_t = torch.sqrt(torch.tensor(1.0 - alpha_bar_t, device=device, dtype=dtype))
        
        # 将I_w视为x_t（加噪后的图像）
        x_t = image
        
        # 模拟噪声预测网络ε_θ(x_t, t) - 使用轻量卷积网络
        # 初始化固定的随机权重（轻量，不需要训练）
        if not hasattr(self, '_pseudo_ddim_conv1'):
            torch.manual_seed(42)  # 固定随机种子确保可重复
            self._pseudo_ddim_conv1 = torch.randn(32, 3, 3, 3, device=device, dtype=dtype, requires_grad=False) * 0.1
            self._pseudo_ddim_conv2 = torch.randn(64, 32, 3, 3, device=device, dtype=dtype, requires_grad=False) * 0.1
            self._pseudo_ddim_conv3 = torch.randn(3, 32+64, 3, 3, device=device, dtype=dtype, requires_grad=False) * 0.1
        
        # 特征提取（模拟UNet编码器）
        feat1 = F.conv2d(x_t, self._pseudo_ddim_conv1, padding=1)
        feat1 = F.relu(feat1)
        
        # 下采样（模拟UNet瓶颈层）
        feat2 = F.avg_pool2d(feat1, 2)
        feat2 = F.conv2d(feat2, self._pseudo_ddim_conv2, padding=1)
        feat2 = F.relu(feat2)
        
        # 上采样并预测噪声（模拟UNet解码器）
        feat2_up = F.interpolate(feat2, size=(H, W), mode='bilinear', align_corners=False)
        noise_pred = F.conv2d(
            torch.cat([feat1, feat2_up], dim=1),
            self._pseudo_ddim_conv3,
            padding=1
        )
        
        # 归一化噪声预测
        noise_pred = noise_pred * 0.15  # 缩放因子
        
        # 根据DDIM公式计算x_0_pred（实施方式公式）
        x_0_pred = (x_t - sqrt_one_minus_alpha_bar_t * noise_pred) / sqrt_alpha_bar_t
        
        return torch.clamp(x_0_pred, -1, 1)

    def inpaint_attack(self, image):
        """
        轻量占位：若无 diffusers 依赖或未启用，则直接返回
        """
        if not self.use_inpaint:
            return image
        try:
            if self._inpaint_pipe is None:
                from diffusers import StableDiffusionInpaintPipeline
                self._inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
                    "runwayml/stable-diffusion-inpainting"
                ).to(image.device)
                self._inpaint_pipe.set_progress_bar_config(disable=True)
            B, C, H, W = image.shape
            # 构造随机遮罩（20%-40%区域）
            mask = torch.zeros_like(image[:, :1])
            ratio = random.uniform(0.2, 0.4)
            h = int(H * ratio)
            w = int(W * ratio)
            top = random.randint(0, H - h)
            left = random.randint(0, W - w)
            mask[:, :, top:top+h, left:left+w] = 1.0
            # 转为 PIL 批处理较重，这里仅在开启时才运行
            imgs = (image.clamp(-1, 1) * 127.5 + 127.5).byte().cpu()
            masks = (mask * 255).byte().cpu()
            pil_imgs = [transforms.ToPILImage()(imgs[i]) for i in range(B)]
            pil_masks = [transforms.ToPILImage()(masks[i]) for i in range(B)]
            results = []
            for img_pil, m_pil in zip(pil_imgs, pil_masks):
                out = self._inpaint_pipe(prompt="restore the image", image=img_pil, mask_image=m_pil).images[0]
                tensor = transforms.ToTensor()(out).to(image.device) * 2 - 1
                results.append(tensor)
            return torch.stack(results, dim=0)
        except Exception:
            return image

    def ip2p_attack(self, image):
        """
        轻量占位：InstructPix2Pix 风格编辑（需 diffusers），默认关闭
        """
        if not self.use_ip2p:
            return image
        try:
            if self._ip2p_pipe is None:
                from diffusers import StableDiffusionInstructPix2PixPipeline
                self._ip2p_pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                    "timbrooks/instruct-pix2pix"
                ).to(image.device)
                self._ip2p_pipe.set_progress_bar_config(disable=True)
            prompt = random.choice([
                "make it oil painting style",
                "convert to watercolor",
                "add cinematic lighting",
                "make it sketch style",
            ])
            imgs = (image.clamp(-1, 1) * 127.5 + 127.5).byte().cpu()
            pil_imgs = [transforms.ToPILImage()(imgs[i]) for i in range(imgs.size(0))]
            results = []
            for img_pil in pil_imgs:
                out = self._ip2p_pipe(prompt=prompt, image=img_pil, num_inference_steps=4, guidance_scale=1.5).images[0]
                tensor = transforms.ToTensor()(out).to(image.device) * 2 - 1
                results.append(tensor)
            return torch.stack(results, dim=0)
        except Exception:
            return image
    
    def forward(self, image):
        """
        随机选择攻击
        
        Args:
            image: [B, 3, H, W]
        
        Returns:
            attacked_image: [B, 3, H, W]
        """
        # 无攻击模式：直接返回原图（用于第一阶段训练）
        if self.no_attack:
            return image
        
        # 攻击池
        attacks = [
            self.jpeg_compression,
            self.affine_transform,
            self.gaussian_noise,
            self.gan_style_attack,
            self.diffusion_attack
        ]
        # 实施方式要求：使用真正的DDIM攻击
        if self.use_ddim:
            attacks.append(self.ddim_attack)
        elif self.use_light_aigc:
            # 如果DDIM不可用，使用简化版本
            attacks.append(self.pseudo_ddim_denoise)
        if self.use_inpaint:
            attacks.append(self.inpaint_attack)
        if self.use_ip2p:
            attacks.append(self.ip2p_attack)

        # 轮询保证覆盖，再随机补充第二个
        first_attack = attacks[self._attack_index % len(attacks)]
        self._attack_index += 1

        num_attacks = random.choice([1, 2])
        if num_attacks == 2:
            remaining = [a for a in attacks if a is not first_attack]
            second_attack = random.choice(remaining)
            selected_attacks = [first_attack, second_attack]
        else:
            selected_attacks = [first_attack]
        
        # 依次应用攻击
        attacked_image = image
        for attack in selected_attacks:
            attacked_image = attack(attacked_image)
        
        return attacked_image


# 测试代码
if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    attack_module = HeterogeneousAttack().to(device)
    
    # 测试输入
    image = torch.randn(4, 3, 256, 256).to(device)
    
    # 测试攻击
    attacked = attack_module(image)
    
    print(f"✅ 攻击模块测试通过")
    print(f"   输入: {image.shape}")
    print(f"   输出: {attacked.shape}")
    print(f"   设备: {device}")