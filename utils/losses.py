"""
专利权利要求5：复合损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips

class CompositeLoss(nn.Module):
    def __init__(self, lambda_p=1.0, lambda_w=10.0):
        """
        复合损失函数
        
        Args:
            lambda_p: 感知损失权重
            lambda_w: 水印重建损失权重
        """
        super().__init__()
        self.lambda_p = lambda_p
        self.lambda_w = lambda_w
        
        # LPIPS感知损失
        self.lpips_loss = lpips.LPIPS(net='alex').eval()
        for param in self.lpips_loss.parameters():
            param.requires_grad = False
    
    def perceptual_loss(self, watermarked, original):
        """
        感知损失（LPIPS）
        """
        # LPIPS要求输入范围[-1, 1]
        loss = self.lpips_loss(watermarked, original)
        return loss.mean()
    
    def watermark_loss(self, pred_watermark, true_watermark):
        """
        水印重建损失（二进制交叉熵）
        
        Args:
            pred_watermark: [B, L] 预测概率
            true_watermark: [B, L] 真实比特(0/1)
        """
        bce_loss = F.binary_cross_entropy(
            pred_watermark, 
            true_watermark, 
            reduction='mean'
        )
        return bce_loss
    
    def forward(self, watermarked, original, pred_watermark, true_watermark):
        """
        计算总损失
        """
        # L_p: 感知损失
        loss_p = self.perceptual_loss(watermarked, original)
        
        # L_w: 水印损失
        loss_w = self.watermark_loss(pred_watermark, true_watermark)
        
        # 总损失
        total_loss = self.lambda_p * loss_p + self.lambda_w * loss_w
        
        return {
            'total': total_loss,
            'perceptual': loss_p,
            'watermark': loss_w
        }


# 测试代码
if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 注意：LPIPS可能不支持MPS，使用CPU
    criterion = CompositeLoss().to("cpu")
    
    # 测试输入
    watermarked = torch.randn(2, 3, 256, 256)
    original = torch.randn(2, 3, 256, 256)
    pred_wm = torch.rand(2, 128)
    true_wm = torch.randint(0, 2, (2, 128)).float()
    
    # 计算损失
    losses = criterion(watermarked, original, pred_wm, true_wm)
    
    print(f"✅ 损失函数测试通过")
    for k, v in losses.items():
        print(f"   {k}: {v.item():.4f}")