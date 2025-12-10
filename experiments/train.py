"""
专利完整训练流程
"""
import sys
import os
from pathlib import Path
current_file = os.path.abspath(__file__)           # train.py 的绝对路径
experiments_dir = os.path.dirname(current_file)    # experiments/
project_root = os.path.dirname(experiments_dir)    # watermark/

# 添加到搜索路径
sys.path.insert(0, project_root)

print(f"📁 项目根目录: {project_root}")
# import sys
# sys.path.append('..')

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.encoder import Encoder
from models.decoder import Decoder
from models.attacks import HeterogeneousAttack
from models.sync_net import SyncNet
from utils.sync_pattern import SyncPatternGenerator
from utils.losses import CompositeLoss
from utils.dataset import get_dataloader
from utils.watermark_utils import WatermarkPreprocessor

class WatermarkTrainer:
    def __init__(self, config):
        """
        初始化训练器
        """
        self.config = config
        
        # 设备
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        print(f"🚀 使用设备: {self.device}")
        
        # 模型
        self.encoder = Encoder(config['watermark_length']).to(self.device)
        self.decoder = Decoder(config['watermark_length']).to(self.device)
        self.attack = HeterogeneousAttack().to(self.device)
        
        # 同步模板生成器
        self.sync_generator = SyncPatternGenerator(config['image_size'])
        # 同步网络
        self.sync_net = SyncNet().to(self.device)
        
        # 损失函数（LPIPS可能需要CPU）
        self.criterion = CompositeLoss(
            lambda_p=config['lambda_p'],
            lambda_w=config['lambda_w']
        )

        # 预处理器（身份+时间戳+纠错+签名 -> 比特载荷），使用持久化密钥
        key_dir = Path(config['save_dir']).parent / "keys"
        key_dir.mkdir(parents=True, exist_ok=True)
        private_key_path = key_dir / "private.pem"
        public_key_path = key_dir / "public.pem"
        self.preprocessor = WatermarkPreprocessor(
            private_key_path=str(private_key_path),
            public_key_path=str(public_key_path),
            target_bit_len=config['watermark_length'],
        )
        # 首次生成时保存密钥，便于提取端验签
        if not private_key_path.exists() or not public_key_path.exists():
            self.preprocessor.save_keys(private_key_path, public_key_path)
        
        # 优化器
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + 
            list(self.decoder.parameters()),
            lr=config['lr']
        )
        # 同步网络优化器
        self.sync_optimizer = optim.Adam(
            self.sync_net.parameters(),
            lr=config.get('sync_lr', config['lr'])
        )
        
        # 学习率调度
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config['lr_step'],
            gamma=0.5
        )
        self.sync_scheduler = optim.lr_scheduler.StepLR(
            self.sync_optimizer,
            step_size=config['lr_step'],
            gamma=0.5
        )
        
        # TensorBoard
        self.writer = SummaryWriter(config['log_dir'])
        
        # 数据加载器
        self.train_loader = get_dataloader(
            config['train_dir'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            preprocessor=self.preprocessor,
            pin_memory=config.get('pin_memory', True),
            watermark_length=config['watermark_length']
        )
    
    def train_epoch(self, epoch):
        """
        训练一个epoch
        """
        self.encoder.train()
        self.decoder.train()
        self.sync_net.train()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        epoch_losses = {'total': 0, 'perceptual': 0, 'watermark': 0}
        
        for batch_idx, batch in enumerate(pbar):
            # 数据移到设备
            images = batch['image'].to(self.device)
            watermarks = batch['watermark'].to(self.device)
            
            # 生成同步模板
            sync_patterns = torch.stack([
                self.sync_generator.generate()
                for _ in range(images.size(0))
            ]).unsqueeze(1).to(self.device)

            # ===== 同步网络训练（随机仿射 -> 回归逆变换）=====
            # 随机仿射参数
            angle = (torch.rand(images.size(0)) * 30 - 15).to(self.device)  # [-15,15]度
            scale = (torch.rand(images.size(0)) * 0.4 + 0.8).to(self.device)  # [0.8,1.2]
            tx = (torch.rand(images.size(0)) * 0.1 - 0.05).to(self.device)    # [-0.05,0.05] 归一化平移
            ty = (torch.rand(images.size(0)) * 0.1 - 0.05).to(self.device)
            rad = angle * torch.pi / 180
            cos_a = torch.cos(rad) * scale
            sin_a = torch.sin(rad) * scale
            theta_true = torch.zeros(images.size(0), 2, 3, device=self.device)
            theta_true[:,0,0] = cos_a; theta_true[:,0,1] = -sin_a; theta_true[:,0,2] = tx
            theta_true[:,1,0] = sin_a; theta_true[:,1,1] =  cos_a; theta_true[:,1,2] = ty

            grid = torch.nn.functional.affine_grid(theta_true, sync_patterns.size(), align_corners=False)
            warped = torch.nn.functional.grid_sample(sync_patterns, grid, align_corners=False)

            pred_theta = self.sync_net(warped)
            sync_loss = torch.nn.functional.mse_loss(pred_theta, theta_true)
            self.sync_optimizer.zero_grad()
            sync_loss.backward()
            self.sync_optimizer.step()
            
            # 前向传播
            # S2: 嵌入水印
            watermarked = self.encoder(images, watermarks, sync_patterns)
            
            # S3: 模拟攻击
            attacked = self.attack(watermarked)
            
            # S4: 提取水印
            pred_watermarks = self.decoder(attacked)
            
            # 计算损失
            # 注意：LPIPS需要CPU
            losses = self.criterion(
                watermarked.cpu(),
                images.cpu(),
                pred_watermarks,
                watermarks
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            losses['total'].backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + 
                list(self.decoder.parameters()),
                max_norm=1.0
            )
            
            self.optimizer.step()
            
            # 累计损失
            for k in epoch_losses:
                epoch_losses[k] += losses[k].item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'p_loss': f"{losses['perceptual'].item():.4f}",
                'w_loss': f"{losses['watermark'].item():.4f}"
            })
            
            # TensorBoard记录
            global_step = epoch * len(self.train_loader) + batch_idx
            for k, v in losses.items():
                self.writer.add_scalar(f'train/{k}', v.item(), global_step)
        
        # 平均损失
        for k in epoch_losses:
            epoch_losses[k] /= len(self.train_loader)
        
        # 学习率调度
        self.sync_scheduler.step()
        return epoch_losses
    
    def train(self):
        """
        完整训练流程
        """
        print(f"\n{'='*50}")
        print(f"🎯 开始训练水印系统")
        print(f"{'='*50}\n")
        
        best_loss = float('inf')
        
        for epoch in range(1, self.config['epochs'] + 1):
            # 训练
            losses = self.train_epoch(epoch)
            
            # 学习率调整
            self.scheduler.step()
            
            # 打印
            print(f"\nEpoch {epoch}/{self.config['epochs']}:")
            print(f"  Total Loss: {losses['total']:.4f}")
            print(f"  Perceptual: {losses['perceptual']:.4f}")
            print(f"  Watermark:  {losses['watermark']:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 保存模型
            if losses['total'] < best_loss:
                best_loss = losses['total']
                self.save_checkpoint(epoch, 'best')
                print(f"  ✅ 保存最佳模型 (loss={best_loss:.4f})")
            
            # 定期保存
            if epoch % self.config['save_interval'] == 0:
                self.save_checkpoint(epoch, f'epoch_{epoch}')
        
        print(f"\n🎉 训练完成！")
        self.writer.close()
    
    def save_checkpoint(self, epoch, name):
        """
        保存模型
        """
        os.makedirs(self.config['save_dir'], exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'sync_net': self.sync_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'sync_optimizer': self.sync_optimizer.state_dict(),
            'config': self.config
        }
        
        path = os.path.join(self.config['save_dir'], f'{name}.pth')
        torch.save(checkpoint, path)


if __name__ == "__main__":
    # 训练配置
    data_dir = os.path.join(project_root, 'data/train_images')
    save_dir = os.path.join(project_root, 'results/checkpoints')
    log_dir = os.path.join(project_root, 'results/logs')
    config = {
        # 数据
        'train_dir': data_dir,  # 需要准备图像
        'image_size': 256,
        'batch_size': 8,  # M4可以开到16-32
        'num_workers': 0,  # MPS 下避免多进程 pickle bch 对象；需并发可改>0并重构预处理实例化方式
        
        # 模型
        'watermark_length': 640,  # 64bit原文 + BCH(127,64,10) + 512bit签名
        
        # 训练
        'epochs': 10,
        'lr': 0.0001,
        'lr_step': 20,
        'lambda_p': 1.0,
        'lambda_w': 10.0,
        
        # 保存
        'save_dir': save_dir,
        'log_dir': log_dir,
        'save_interval': 10
    }
    
    # 创建训练器
    trainer = WatermarkTrainer(config)
    
    # 开始训练
    trainer.train()