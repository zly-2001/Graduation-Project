"""
数据集加载器
"""

import os
import sys
import hashlib
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from utils.watermark_utils import WatermarkPreprocessor

class WatermarkDataset(Dataset):
    def __init__(
        self,
        image_dir,
        image_size=256,
        preprocessor: WatermarkPreprocessor | None = None,
        watermark_length: int = 640,
    ):
        """
        Args:
            image_dir: 图像目录
            image_size: 图像尺寸
            watermark_length: 模型水印长度，需与预处理保持一致
        """
        self.image_dir = image_dir
        self.image_size = image_size
        self.watermark_length = watermark_length
        
        # 获取所有图像路径
        self.image_paths = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.endswith(('.jpg', '.png', '.jpeg'))
        ]
        
        # 预处理器：身份+时间戳+纠错+签名 -> 比特载荷
        self.preprocessor = preprocessor or WatermarkPreprocessor(target_bit_len=watermark_length)
        
        # 图像变换
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)  # 归一化到[-1, 1]
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 加载图像
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        # 基于文件名生成可追溯水印载荷
        source_info = os.path.basename(image_path)
        # 计算图像hash（SHA-256），取前32bit用于载荷
        with open(image_path, 'rb') as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        payload_bits, _ = self.preprocessor.preprocess(source_info, image_hash_hex=sha)
        payload_bits = payload_bits[:self.watermark_length]
        watermark = torch.tensor([int(b) for b in payload_bits], dtype=torch.float32)
        
        return {
            'image': image,
            'watermark': watermark
        }


def get_dataloader(
    image_dir,
    batch_size=8,
    num_workers=4,
    preprocessor: WatermarkPreprocessor | None = None,
    pin_memory=True,
    watermark_length: int = 640,
):
    """
    创建数据加载器
    """
    dataset = WatermarkDataset(image_dir, preprocessor=preprocessor, watermark_length=watermark_length)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return dataloader


# 测试代码
if __name__ == "__main__":
    # 需要准备一些测试图像
    # 可以下载COCO或DIV2K数据集
    pass