"""
专利步骤S3：异构可微攻击模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torchvision import transforms

class HeterogeneousAttack(nn.Module):
    def __init__(self, use_light_aigc: bool = True, use_inpaint: bool = False, use_ip2p: bool = False):
        """
        异构攻击模块：模拟多种攻击
        """
        super().__init__()
        
        # 攻击池权重（可学习）
        self.attack_weights = nn.Parameter(torch.ones(3))

        # 轮询指针，确保每个攻击在若干 batch 内都被用到
        self._attack_index = 0
        self.use_light_aigc = use_light_aigc
        self.use_inpaint = use_inpaint
        self.use_ip2p = use_ip2p

        # 延迟加载重模型，避免无依赖时报错
        self._inpaint_pipe = None
        self._ip2p_pipe = None
    
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

    def pseudo_ddim_denoise(self, image):
        """
        轻量“DDIM去噪”占位：先加噪，再进行可微平滑，近似重绘/去噪效果
        """
        noise_level = random.uniform(0.05, 0.12)
        noisy = image + torch.randn_like(image) * noise_level
        # 小核高斯样平滑
        kernel = torch.tensor(
            [[[1, 2, 1],
              [2, 4, 2],
              [1, 2, 1]]] , device=image.device, dtype=image.dtype
        )
        kernel = kernel / kernel.sum()
        kernel = kernel.unsqueeze(0).expand(3,1,3,3)
        denoised = F.conv2d(noisy, kernel, padding=1, groups=3)
        # 轻微锐化混合
        alpha = 0.2
        out = alpha * noisy + (1 - alpha) * denoised
        return torch.clamp(out, -1, 1)

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
        # 攻击池
        attacks = [
            self.jpeg_compression,
            self.affine_transform,
            self.gaussian_noise,
            self.gan_style_attack,
            self.diffusion_attack
        ]
        if self.use_light_aigc:
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