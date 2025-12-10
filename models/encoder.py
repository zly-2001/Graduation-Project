"""
专利步骤S2：编码器网络（5层U-Net，多域分层嵌入，输出扰动残差 + 星环模板叠加）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class Encoder(nn.Module):
    def __init__(self, watermark_length=128, alpha=1.0, beta=1.0):
        """
        编码器：生成扰动残差并叠加星环同步模板
        
        Args:
            watermark_length: 水印比特长度
            alpha: 残差权重
            beta: 同步模板权重
        """
        super().__init__()
        self.watermark_length = watermark_length
        self.alpha = alpha
        self.beta = beta
        
        # U-Net 通道配置（5层下采样/上采样）
        chs = [64, 128, 256, 512, 512]
        
        # 下采样
        self.down1 = conv_block(3, chs[0])
        self.down2 = conv_block(chs[0], chs[1])
        self.down3 = conv_block(chs[1], chs[2])
        self.down4 = conv_block(chs[2], chs[3])
        self.down5 = conv_block(chs[3], chs[4])
        self.pool = nn.MaxPool2d(2)
        
        # 载荷映射到瓶颈通道
        bottleneck_ch = chs[4]
        self.watermark_fc = nn.Sequential(
            nn.Linear(watermark_length, bottleneck_ch),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_ch, bottleneck_ch),
            nn.ReLU(inplace=True)
        )
        
        # 上采样
        self.up5 = nn.ConvTranspose2d(chs[4]*2, chs[3], 2, stride=2)
        self.dec5 = conv_block(chs[3]*2, chs[3])
        
        self.up4 = nn.ConvTranspose2d(chs[3], chs[2], 2, stride=2)
        self.dec4 = conv_block(chs[2]*2, chs[2])
        
        self.up3 = nn.ConvTranspose2d(chs[2], chs[1], 2, stride=2)
        self.dec3 = conv_block(chs[1]*2, chs[1])
        
        self.up2 = nn.ConvTranspose2d(chs[1], chs[0], 2, stride=2)
        self.dec2 = conv_block(chs[0]*2, chs[0])
        
        # 输出扰动残差
        self.out_conv = nn.Conv2d(chs[0], 3, 1)
    
    def forward(self, image, watermark_bits, sync_pattern):
        """
        Args:
            image: [B,3,H,W]
            watermark_bits: [B, L]
            sync_pattern: [B,1,H,W]
        Returns:
            watermarked_image: I + alpha*R + beta*S
        """
        B = image.size(0)
        # 编码
        x1 = self.down1(image)
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))
        x4 = self.down4(self.pool(x3))
        x5 = self.down5(self.pool(x4))
        
        # 载荷注入瓶颈
        wm = self.watermark_fc(watermark_bits)  # [B, bottleneck_ch]
        wm = wm.view(B, -1, 1, 1).expand_as(x5)
        x5 = torch.cat([x5, wm], dim=1)  # 通道翻倍
        
        # 解码
        d5 = self.up5(x5)
        d5 = self.dec5(torch.cat([d5, x4], dim=1))
        
        d4 = self.up4(d5)
        d4 = self.dec4(torch.cat([d4, x3], dim=1))
        
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, x2], dim=1))
        
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, x1], dim=1))
        
        residual = torch.tanh(self.out_conv(d2))  # [-1,1]，尺寸与输入一致（256x256）
        
        # 同步模板叠加（插值到图像尺寸）
        if sync_pattern.shape[-1] != image.shape[-1]:
            sync_pattern = F.interpolate(sync_pattern, size=image.shape[2:], mode='bilinear', align_corners=False)
        # 水印图像
        watermarked_image = image + self.alpha * residual + self.beta * sync_pattern
        watermarked_image = torch.clamp(watermarked_image, -1, 1)
        return watermarked_image


# 测试代码
if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    encoder = Encoder(watermark_length=128).to(device)
    
    # 测试输入
    batch_size = 4
    image = torch.randn(batch_size, 3, 256, 256).to(device)
    watermark = torch.randint(0, 2, (batch_size, 128)).float().to(device)
    sync_pattern = torch.randn(batch_size, 1, 256, 256).to(device)
    
    # 前向传播
    output = encoder(image, watermark, sync_pattern)
    
    print(f"✅ 编码器测试通过")
    print(f"   输入: {image.shape}")
    print(f"   输出: {output.shape}")
    print(f"   设备: {device}")