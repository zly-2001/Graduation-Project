"""
专利权利要求3：多尺度、多方向冗余同步模板
"""

import torch
import numpy as np

class SyncPatternGenerator:
    def __init__(self, image_size=256):
        """
        生成“星环”同步模板：多频同心圆 + 多方向平面波
        """
        self.size = image_size
    
    def generate(
        self,
        circle_freqs=(0.05, 0.10, 0.20),
        line_thetas=(0, 45, 90, 135),
        line_freq=0.10,
        amplitude=0.08,
    ):
        """
        生成多尺度、多方向同步模板
        
        Args:
            circle_freqs: 同心圆频率集合
            line_thetas: 平面波方向集合（度）
            line_freq: 平面波频率
            amplitude: 总幅度因子
        Returns:
            sync_pattern: [H, W] 同步模板
        """
        H, W = self.size, self.size
        
        # 创建坐标网格
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H),
            torch.linspace(-1, 1, W),
            indexing='ij'
        )
        
        # 同心圆正弦波叠加
        r = torch.sqrt(x**2 + y**2)
        circles = torch.zeros_like(r)
        for f in circle_freqs:
            phase = 0.0
            circles += torch.sin(2 * np.pi * f * r + phase)
        
        # 多方向平面正弦波叠加
        lines = torch.zeros_like(r)
        for deg in line_thetas:
            theta = np.deg2rad(deg)
            phase = 0.0
            x_rot = x * np.cos(theta) - y * np.sin(theta)
            lines += torch.sin(2 * np.pi * line_freq * x_rot + phase)
        
        sync_pattern = circles + lines
        sync_pattern = sync_pattern / torch.abs(sync_pattern).max() * amplitude
        
        return sync_pattern


# 测试代码
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    generator = SyncPatternGenerator(256)
    pattern = generator.generate()
    
    plt.figure(figsize=(6, 6))
    plt.imshow(pattern, cmap='gray')
    plt.title('同步模板')
    plt.colorbar()
    plt.savefig('sync_pattern.png')
    print("✅ 同步模板已生成: sync_pattern.png")