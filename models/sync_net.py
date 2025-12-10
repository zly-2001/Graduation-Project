"""
专利步骤S61：几何同步卷积神经网络
"""

import torch
import torch.nn as nn

class SyncNet(nn.Module):
    def __init__(self):
        """
        几何同步网络：估计仿射变换矩阵
        """
        super().__init__()
        
        # 特征提取
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # 回归仿射矩阵
        self.regressor = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 6)  # 2x3仿射矩阵的6个参数
        )
        
        # 初始化为单位矩阵
        self.regressor[-1].weight.data.zero_()
        self.regressor[-1].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )
    
    def forward(self, sync_pattern):
        """
        预测仿射变换矩阵
        
        Args:
            sync_pattern: [B, 1, H, W] 同步模板或灰度图
        
        Returns:
            theta: [B, 2, 3] 仿射变换矩阵
        """
        # 特征提取
        features = self.features(sync_pattern)
        features = features.view(features.size(0), -1)
        
        # 回归矩阵参数
        theta_params = self.regressor(features)
        theta = theta_params.view(-1, 2, 3)
        
        return theta


# 测试代码
if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    sync_net = SyncNet().to(device)
    
    # 测试输入
    pattern = torch.randn(4, 1, 256, 256).to(device)
    
    # 前向传播
    theta = sync_net(pattern)
    
    print(f"✅ 同步网络测试通过")
    print(f"   输入: {pattern.shape}")
    print(f"   输出: {theta.shape} (仿射矩阵)")
    print(f"   设备: {device}")