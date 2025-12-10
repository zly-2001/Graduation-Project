"""
专利提取验证流程
"""

import sys
import hashlib
import json
from pathlib import Path

# 确保可通过绝对路径找到项目内模块
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import numpy as np

from models.encoder import Encoder
from models.decoder import Decoder
from models.sync_net import SyncNet
from utils.watermark_utils import WatermarkPreprocessor
from utils.sync_pattern import SyncPatternGenerator

class WatermarkExtractor:
    def __init__(self, checkpoint_path):
        """
        初始化提取器
        """
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        
        # 加载模型
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        wm_len = checkpoint.get('config', {}).get('watermark_length', 640)
        self.watermark_length = wm_len

        self.decoder = Decoder(wm_len).to(self.device)
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.decoder.eval()
        
        # 同步网络
        self.sync_net = SyncNet().to(self.device)
        if 'sync_net' in checkpoint:
            self.sync_net.load_state_dict(checkpoint['sync_net'])
        else:
            print("⚠️ checkpoint 未包含 sync_net，几何校正将使用未训练模型")
        self.sync_net.eval()
        
        # 预处理器
        # 仅加载公钥用于验签；key路径默认在 results/keys
        key_dir = project_root / 'results/keys'
        public_key_path = key_dir / 'public.pem'
        self.preprocessor = WatermarkPreprocessor(
            public_key_path=str(public_key_path),
            private_key_path=None,
            target_bit_len=wm_len,
        )
        
        # 图像变换
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
    
    def extract(self, image_path):
        """
        提取水印
        
        Args:
            image_path: 图像路径
        
        Returns:
            watermark_bits: 提取的水印比特
            confidence: 置信度
        """
        # S61: 几何同步校正
        image = Image.open(image_path).convert('RGB')
        # 计算图像哈希（用于证据）
        with open(image_path, 'rb') as f:
            sha_hex = hashlib.sha256(f.read()).hexdigest()
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 这里简化处理，实际需要同步网络
        # corrected_image = self.geometric_correction(image_tensor)
        corrected_image = image_tensor  # 简化
        
        # 同步校正：使用SyncNet估计仿射并逆变换
        with torch.no_grad():
            # 灰度输入给同步网络
            gray = corrected_image.mean(dim=1, keepdim=True)
            if hasattr(self, 'sync_net'):
                theta = self.sync_net(gray)
                inv_theta = self._invert_theta(theta)
                grid = F.affine_grid(inv_theta, corrected_image.size(), align_corners=False)
                corrected_image = F.grid_sample(corrected_image, grid, align_corners=False)

            # S63: 盲提取
            watermark_prob = self.decoder(corrected_image)
        
        # 转为比特
        watermark_bits = (watermark_prob > 0.5).float().cpu().numpy()[0]
        confidence = torch.mean(
            torch.max(watermark_prob, 1 - watermark_prob)
        ).item()
        
        # 验签+纠错解码
        bits_str = ''.join(str(int(b)) for b in watermark_bits[:self.watermark_length])
        verify_result = self.preprocessor.decode_and_verify(bits_str)
        evidence = {
            "image_name": Path(image_path).name,
            "image_hash_sha256": sha_hex,
            "bitstring_prefix": bits_str[:64],
            "verified": verify_result.get("verified", False),
            "identity_bits": verify_result.get("identity_bits", ""),
            "timestamp": verify_result.get("timestamp", ""),
            "hash_prefix": verify_result.get("hash_prefix", ""),
            "message": verify_result.get("message", ""),
            "model_watermark_length": self.watermark_length,
        }
        
        return watermark_bits, confidence, evidence

    def _invert_theta(self, theta):
        """
        计算2x3仿射矩阵的逆，用于校正
        """
        B = theta.size(0)
        device = theta.device
        A = torch.eye(3, device=device).unsqueeze(0).repeat(B,1,1)
        A[:,:2,:3] = theta
        A_inv = torch.inverse(A)
        return A_inv[:,:2,:3]
    
    def verify(self, extracted_bits):
        """
        S64: 验证水印
        """
        # 简化实现
        # 实际需要BCH解码和ECDSA验签
        return True


# 测试代码
if __name__ == "__main__":
    checkpoint_path = project_root / 'results/checkpoints/best.pth'
    test_dir = project_root / 'data/test_images'
    # 选取测试图像（优先jpg/png/jpeg）
    candidates = sorted(
        list(test_dir.glob("*.jpg")) + 
        list(test_dir.glob("*.png")) + 
        list(test_dir.glob("*.jpeg"))
    )
    if not candidates:
        raise FileNotFoundError(f"测试目录为空: {test_dir}")
    test_image = candidates[0]

    extractor = WatermarkExtractor(checkpoint_path)
    
    # 提取水印
    bits, conf, evidence = extractor.extract(test_image)
    
    print(f"✅ 提取完成: {test_image.name}")
    print(f"   置信度: {conf:.2%}")
    print(f"   前10比特: {bits[:10]}")
    print(f"   法证证据: {json.dumps(evidence, ensure_ascii=False, indent=2)}")