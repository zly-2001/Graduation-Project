"""
专利步骤S4：解码器网络（水印提取，5层对称U-Net）
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


class Decoder(nn.Module):
    def __init__(self, watermark_length=128):
        """
        解码器：对带水印图做编码-解码，输出水印概率
        
        Args:
            watermark_length: 水印比特长度
        """
        super().__init__()
        self.watermark_length = watermark_length
        
        chs = [64, 128, 256, 512, 512]
        # 下采样
        self.down1 = conv_block(3, chs[0])
        self.down2 = conv_block(chs[0], chs[1])
        self.down3 = conv_block(chs[1], chs[2])
        self.down4 = conv_block(chs[2], chs[3])
        self.down5 = conv_block(chs[3], chs[4])
        self.pool = nn.MaxPool2d(2)

        # 上采样
        self.up5 = nn.ConvTranspose2d(chs[4], chs[3], 2, stride=2)
        self.dec5 = conv_block(chs[3]*2, chs[3])

        self.up4 = nn.ConvTranspose2d(chs[3], chs[2], 2, stride=2)
        self.dec4 = conv_block(chs[2]*2, chs[2])

        self.up3 = nn.ConvTranspose2d(chs[2], chs[1], 2, stride=2)
        self.dec3 = conv_block(chs[1]*2, chs[1])

        self.up2 = nn.ConvTranspose2d(chs[1], chs[0], 2, stride=2)
        self.dec2 = conv_block(chs[0]*2, chs[0])

        # 输出特征到水印概率
        self.out_conv = nn.Conv2d(chs[0], 32, 1)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, watermark_length),
            nn.Sigmoid()
        )
    
    def forward(self, watermarked_image):
        """
        前向传播
        
        Args:
            watermarked_image: [B, 3, H, W]
        
        Returns:
            watermark_prob: [B, watermark_length] 水印比特概率
        """
        # 编码
        x1 = self.down1(watermarked_image)
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))
        x4 = self.down4(self.pool(x3))
        x5 = self.down5(self.pool(x4))

        # 解码
        d5 = self.up5(x5)
        d5 = self.dec5(torch.cat([d5, x4], dim=1))

        d4 = self.up4(d5)
        d4 = self.dec4(torch.cat([d4, x3], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, x2], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, x1], dim=1))

        feat = self.out_conv(d2)
        watermark_prob = self.head(feat)
        return watermark_prob


# 测试代码
if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    decoder = Decoder(watermark_length=128).to(device)
    
    # 测试输入
    image = torch.randn(4, 3, 256, 256).to(device)
    
    # 前向传播
    output = decoder(image)
    
    print(f"✅ 解码器测试通过")
    print(f"   输入: {image.shape}")
    print(f"   输出: {output.shape} (概率)")
    print(f"   设备: {device}")