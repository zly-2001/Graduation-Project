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
import math
import torchvision.utils as vutils
from PIL import Image

from models.encoder import Encoder
from models.decoder import Decoder
from models.attacks import HeterogeneousAttack
from models.sync_net import SyncNet
from utils.sync_pattern import SyncPatternGenerator
from utils.losses import CompositeLoss
from utils.dataset import get_dataloader
from utils.watermark_utils import WatermarkPreprocessor
def compute_psnr(a, b, max_val=2.0):
    # 输入范围[-1,1]，max_val=2.0
    mse = torch.mean((a - b) ** 2)
    if mse == 0:
        return 99.0
    return 10 * math.log10((max_val ** 2) / mse.item())


def _gaussian_window(window_size=11, sigma=1.5, channels=3, device="cpu"):
    coords = torch.arange(window_size, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    window_1d = g
    window_2d = window_1d.T @ window_1d
    window = window_2d.expand(channels, 1, window_size, window_size)
    return window


def compute_ssim(img1, img2, window_size=11, sigma=1.5):
    # 简化版 SSIM，假设输入范围[-1,1]
    device = img1.device
    channel = img1.size(1)
    window = _gaussian_window(window_size, sigma, channel, device=device)
    padding = window_size // 2
    mu1 = torch.nn.functional.conv2d(img1, window, padding=padding, groups=channel)
    mu2 = torch.nn.functional.conv2d(img2, window, padding=padding, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = torch.nn.functional.conv2d(img1 * img1, window, padding=padding, groups=channel) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(img2 * img2, window, padding=padding, groups=channel) - mu2_sq
    sigma12 = torch.nn.functional.conv2d(img1 * img2, window, padding=padding, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()

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
        # 简化版本：使用模拟DDIM以提升训练速度（Mac M4优化）
        # 注意：模拟DDIM使用相同的数学公式，但使用轻量卷积网络代替预训练UNet
        # 完全符合实施方式需要启用真正的DDIM（use_ddim=True），但可以先验证流程
        self.attack = HeterogeneousAttack(
            use_ddim=False,       # ⚠️ 暂时禁用真正的DDIM（使用模拟版本，训练更快）
            use_light_aigc=True,  # ✅ 启用模拟DDIM（使用相同公式，轻量实现）
            use_inpaint=False,    # ❌ Mac上太慢，默认关闭
            use_ip2p=False        # ❌ Mac上太慢，默认关闭
        ).to(self.device)
        
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
        
        # 优化器（AdamW）
        self.optimizer = optim.AdamW(
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
        
        # 测试数据加载器（如果存在，用于验证，不会参与训练）
        test_dir = os.path.join(Path(config['train_dir']).parent, 'test_images')
        self.test_loader = None
        if os.path.exists(test_dir) and len(os.listdir(test_dir)) > 0:
            print(f"📊 检测到测试集: {test_dir} (用于验证，不会参与训练)")
            self.test_loader = get_dataloader(
                test_dir,
                batch_size=min(4, config['batch_size']),  # 测试时batch可以小一点
                num_workers=0,  # 测试时不需要多进程
                preprocessor=self.preprocessor,
                watermark_length=config['watermark_length']
            )
        else:
            print(f"⚠️  未找到测试集 ({test_dir})，将只在训练集上验证")
        
        # 可视化保存目录
        self.vis_dir = os.path.join(Path(config['save_dir']).parent, 'visualizations')
        os.makedirs(self.vis_dir, exist_ok=True)
    
    def train_epoch(self, epoch):
        """
        训练一个epoch
        """
        self.encoder.train()
        self.decoder.train()
        self.sync_net.train()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        epoch_losses = {'total': 0, 'perceptual': 0, 'watermark': 0}
        epoch_metrics = {'ber': 0, 'psnr': 0, 'ssim': 0}
        
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
            # 权利要求7：L_sync = ||M - M_gt||F (Frobenius范数)
            sync_loss = torch.norm(pred_theta - theta_true, p='fro')
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
            # 指标：BER/PSNR/SSIM
            with torch.no_grad():
                pred_bits = (pred_watermarks > 0.5).float()
                ber = (pred_bits != watermarks).float().mean().item()
                psnr = compute_psnr(watermarked.detach(), images.detach())
                ssim = compute_ssim(watermarked.detach(), images.detach())
                epoch_metrics['ber'] += ber
                epoch_metrics['psnr'] += psnr
                epoch_metrics['ssim'] += ssim
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'p_loss': f"{losses['perceptual'].item():.4f}",
                'w_loss': f"{losses['watermark'].item():.4f}",
                'ber': f"{ber:.3f}"
            })
            
            # TensorBoard记录
            global_step = epoch * len(self.train_loader) + batch_idx
            for k, v in losses.items():
                self.writer.add_scalar(f'train/{k}', v.item(), global_step)
            self.writer.add_scalar('train/ber', ber, global_step)
            self.writer.add_scalar('train/psnr', psnr, global_step)
            self.writer.add_scalar('train/ssim', ssim, global_step)
            
            # 每个epoch的第一个batch保存可视化（避免保存太多）
            if batch_idx == 0:
                self._save_visualization(
                    images, watermarked, attacked, pred_watermarks, watermarks,
                    epoch, global_step
                )
        
        # 平均损失
        for k in epoch_losses:
            epoch_losses[k] /= len(self.train_loader)
        for k in epoch_metrics:
            epoch_metrics[k] /= len(self.train_loader)
        
        # 学习率调度
        self.sync_scheduler.step()
        
        # 合并指标
        all_metrics = {**epoch_losses, **epoch_metrics}
        
        # 在测试集上验证（如果存在）
        if self.test_loader is not None:
            test_metrics = self._validate_on_test_set(epoch)
            all_metrics.update({f'test_{k}': v for k, v in test_metrics.items()})
        
        return all_metrics
    
    def train(self, resume_from=None, train_batch_epochs=None):
        """
        完整训练流程
        
        Args:
            resume_from: 从哪个checkpoint恢复（None=自动检测, 'best'=最佳模型, 'latest'=最新epoch, 或具体路径）
            train_batch_epochs: 每次训练的轮数（None=训练到config['epochs']，否则只训练指定轮数后保存并退出）
        """
        print(f"\n{'='*50}")
        print(f"🎯 开始训练水印系统")
        print(f"{'='*50}\n")
        
        # 恢复训练
        start_epoch = 1
        best_loss = float('inf')
        
        # 自动检测checkpoint（如果用户没有指定resume_from）
        if resume_from is None:
            # 自动检测：优先使用latest，如果没有则使用best
            if os.path.exists(self.config['save_dir']):
                checkpoints = [f for f in os.listdir(self.config['save_dir']) if f.startswith('epoch_') and f.endswith('.pth')]
                if checkpoints:
                    # 找到最新的epoch checkpoint
                    latest = max(checkpoints, key=lambda x: int(x.split('_')[1].split('.')[0]))
                    checkpoint_path = os.path.join(self.config['save_dir'], latest)
                    resume_from = checkpoint_path
                    print(f"🔍 自动检测到checkpoint: {latest}")
                elif os.path.exists(os.path.join(self.config['save_dir'], 'best.pth')):
                    checkpoint_path = os.path.join(self.config['save_dir'], 'best.pth')
                    resume_from = checkpoint_path
                    print(f"🔍 自动检测到checkpoint: best.pth")
        
        if resume_from:
            if resume_from == 'best':
                checkpoint_path = os.path.join(self.config['save_dir'], 'best.pth')
            elif resume_from == 'latest':
                # 找到最新的epoch checkpoint
                if os.path.exists(self.config['save_dir']):
                    checkpoints = [f for f in os.listdir(self.config['save_dir']) if f.startswith('epoch_') and f.endswith('.pth')]
                    if checkpoints:
                        latest = max(checkpoints, key=lambda x: int(x.split('_')[1].split('.')[0]))
                        checkpoint_path = os.path.join(self.config['save_dir'], latest)
                    else:
                        checkpoint_path = os.path.join(self.config['save_dir'], 'best.pth')
                else:
                    checkpoint_path = None
            else:
                checkpoint_path = resume_from
            
            if checkpoint_path and os.path.exists(checkpoint_path):
                start_epoch, best_loss = self.load_checkpoint(checkpoint_path, resume_training=True)
        
        # 确定训练结束的epoch
        if train_batch_epochs is not None:
            # 分批次训练：只训练指定轮数
            end_epoch = start_epoch + train_batch_epochs - 1
            max_epoch = self.config['epochs']
            if end_epoch > max_epoch:
                end_epoch = max_epoch
            print(f"📌 分批次训练模式：从epoch {start_epoch} 训练到 epoch {end_epoch} (共{train_batch_epochs}轮)")
        else:
            # 正常训练：训练到配置的epochs
            end_epoch = self.config['epochs']
            print(f"📌 完整训练模式：从epoch {start_epoch} 训练到 epoch {end_epoch}")
        
        for epoch in range(start_epoch, end_epoch + 1):
            # 训练
            losses = self.train_epoch(epoch)
            
            # 学习率调整
            self.scheduler.step()
            
            # 打印
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{self.config['epochs']}:")
            print(f"  📊 训练集指标:")
            print(f"    Total Loss: {losses['total']:.4f}")
            print(f"    Perceptual: {losses['perceptual']:.4f}")
            print(f"    Watermark:  {losses['watermark']:.4f}")
            print(f"    BER:        {losses['ber']:.4f}")
            print(f"    PSNR:       {losses['psnr']:.2f} dB")
            print(f"    SSIM:       {losses['ssim']:.4f}")
            print(f"    LR:         {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 如果有测试集，打印测试指标
            if self.test_loader is not None and f'test_ber' in losses:
                print(f"  🧪 测试集指标:")
                print(f"    BER:        {losses.get('test_ber', 0):.4f}")
                print(f"    PSNR:       {losses.get('test_psnr', 0):.2f} dB")
                print(f"    SSIM:       {losses.get('test_ssim', 0):.4f}")
            
            print(f"  💾 可视化已保存: {self.vis_dir}/epoch_{epoch:03d}_*.png")
            print(f"{'='*60}")
            
            # 保存模型
            if losses['total'] < best_loss:
                best_loss = losses['total']
                self.save_checkpoint(epoch, 'best', best_loss=best_loss)
                print(f"  ✅ 保存最佳模型 (loss={best_loss:.4f})")
            
            # 定期保存
            if epoch % self.config['save_interval'] == 0:
                self.save_checkpoint(epoch, f'epoch_{epoch}')
        
        # 分批次训练：训练完指定轮数后保存并退出
        if train_batch_epochs is not None:
            # 保存当前进度
            self.save_checkpoint(epoch, f'epoch_{epoch}')
            print(f"\n✅ 本次训练完成！已训练 {train_batch_epochs} 轮 (epoch {start_epoch} → {epoch})")
            print(f"💾 已保存checkpoint: epoch_{epoch}.pth")
            if epoch < self.config['epochs']:
                remaining = self.config['epochs'] - epoch
                print(f"📌 剩余 {remaining} 轮，下次运行会自动继续训练")
                print(f"   下次运行: python experiments/train.py (会自动从epoch {epoch+1}继续)")
            else:
                print(f"🎉 所有训练已完成！(共{self.config['epochs']}轮)")
        else:
            print(f"\n🎉 训练完成！")
        
        self.writer.close()
    
    def save_checkpoint(self, epoch, name, best_loss=None):
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
            'scheduler': self.scheduler.state_dict(),
            'sync_scheduler': self.sync_scheduler.state_dict(),
            'config': self.config
        }
        if best_loss is not None:
            checkpoint['best_loss'] = best_loss
        
        path = os.path.join(self.config['save_dir'], f'{name}.pth')
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, checkpoint_path, resume_training=True):
        """
        加载检查点，支持恢复训练
        
        Args:
            checkpoint_path: checkpoint文件路径
            resume_training: 是否恢复训练（True=继续训练，False=只加载模型）
        
        Returns:
            start_epoch: 开始的epoch（如果恢复训练）
            best_loss: 最佳loss（如果checkpoint中有）
        """
        if not os.path.exists(checkpoint_path):
            print(f"⚠️ Checkpoint不存在: {checkpoint_path}")
            return 0, float('inf')
        
        print(f"📂 加载checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 加载模型权重
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        if 'sync_net' in checkpoint:
            self.sync_net.load_state_dict(checkpoint['sync_net'])
        
        best_loss = checkpoint.get('best_loss', float('inf'))
        
        if resume_training:
            # 恢复训练状态
            if 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            if 'sync_optimizer' in checkpoint:
                self.sync_optimizer.load_state_dict(checkpoint['sync_optimizer'])
            if 'scheduler' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            if 'sync_scheduler' in checkpoint:
                self.sync_scheduler.load_state_dict(checkpoint['sync_scheduler'])
            
            start_epoch = checkpoint.get('epoch', 0) + 1
            print(f"✅ 从epoch {start_epoch}恢复训练 (best_loss={best_loss:.4f})")
            return start_epoch, best_loss
        else:
            print(f"✅ 加载模型权重（epoch {checkpoint.get('epoch', 0)}）")
            return 0, best_loss
    
    def _save_visualization(self, images, watermarked, attacked, pred_watermarks, true_watermarks, epoch, step):
        """
        保存训练效果可视化
        
        保存：
        1. 原始图像 vs 带水印图像（对比不可见性）
        2. 带水印图像 vs 攻击后图像（展示攻击效果）
        3. 水印提取对比（真实 vs 预测）
        """
        with torch.no_grad():
            # 只保存第一个样本
            img_orig = images[0:1].cpu()
            img_wm = watermarked[0:1].cpu()
            img_att = attacked[0:1].cpu()
            
            # 转换为[0,1]范围用于保存
            img_orig_vis = (img_orig + 1.0) / 2.0
            img_wm_vis = (img_wm + 1.0) / 2.0
            img_att_vis = (img_att + 1.0) / 2.0
            
            # 保存对比图：原始 vs 带水印
            comparison1 = torch.cat([img_orig_vis, img_wm_vis], dim=3)  # 水平拼接
            vutils.save_image(
                comparison1,
                os.path.join(self.vis_dir, f'epoch_{epoch:03d}_original_vs_watermarked.png'),
                nrow=1,
                normalize=False
            )
            
            # 保存对比图：带水印 vs 攻击后
            comparison2 = torch.cat([img_wm_vis, img_att_vis], dim=3)
            vutils.save_image(
                comparison2,
                os.path.join(self.vis_dir, f'epoch_{epoch:03d}_watermarked_vs_attacked.png'),
                nrow=1,
                normalize=False
            )
            
            # 保存完整流程：原始 -> 带水印 -> 攻击后
            comparison3 = torch.cat([img_orig_vis, img_wm_vis, img_att_vis], dim=3)
            vutils.save_image(
                comparison3,
                os.path.join(self.vis_dir, f'epoch_{epoch:03d}_full_pipeline.png'),
                nrow=1,
                normalize=False
            )
            
            # 记录到TensorBoard
            self.writer.add_image('visualization/original_vs_watermarked', comparison1[0], step)
            self.writer.add_image('visualization/watermarked_vs_attacked', comparison2[0], step)
            self.writer.add_image('visualization/full_pipeline', comparison3[0], step)
            
            # 计算并保存水印提取准确率（前64位，原始信息部分）
            pred_bits = (pred_watermarks[0] > 0.5).float().cpu().numpy()
            true_bits = true_watermarks[0].cpu().numpy()
            accuracy = 1.0 - (pred_bits != true_bits).mean()
            
            # 保存水印对比文本
            with open(os.path.join(self.vis_dir, f'epoch_{epoch:03d}_watermark_info.txt'), 'w') as f:
                f.write(f"Epoch {epoch}, Step {step}\n")
                f.write(f"水印提取准确率: {accuracy*100:.2f}%\n")
                f.write(f"前32位真实: {''.join([str(int(b)) for b in true_bits[:32]])}\n")
                f.write(f"前32位预测: {''.join([str(int(b)) for b in pred_bits[:32]])}\n")
    
    def _validate_on_test_set(self, epoch):
        """
        在测试集上验证（不会影响训练，只是评估）
        
        注意：测试数据不会被污染，因为：
        1. 测试集只用于前向传播（torch.no_grad()）
        2. 不会进行反向传播和参数更新
        3. 测试集和训练集完全分离
        """
        self.encoder.eval()
        self.decoder.eval()
        self.attack.eval()
        
        test_metrics = {
            'ber': 0.0,
            'psnr': 0.0,
            'ssim': 0.0
        }
        
        with torch.no_grad():
            for batch in self.test_loader:
                images = batch['image'].to(self.device)
                watermarks = batch['watermark'].to(self.device)
                
                # 生成同步模板
                sync_patterns = []
                for _ in range(images.size(0)):
                    pattern = self.sync_generator.generate()
                    sync_patterns.append(pattern)
                sync_patterns = torch.stack(sync_patterns, dim=0).unsqueeze(1).to(self.device)
                
                # 前向传播（不训练）
                watermarked = self.encoder(images, watermarks, sync_patterns)
                attacked = self.attack(watermarked)
                pred_watermarks = self.decoder(attacked)
                
                # 计算指标
                pred_bits = (pred_watermarks > 0.5).float()
                ber = (pred_bits != watermarks).float().mean().item()
                psnr = compute_psnr(watermarked.cpu(), images.cpu())
                ssim = compute_ssim(watermarked.cpu(), images.cpu())
                
                test_metrics['ber'] += ber
                test_metrics['psnr'] += psnr
                test_metrics['ssim'] += ssim
        
        # 平均
        num_batches = len(self.test_loader)
        for k in test_metrics:
            test_metrics[k] /= num_batches
        
        # 记录到TensorBoard
        for k, v in test_metrics.items():
            self.writer.add_scalar(f'test/{k}', v, epoch)
        
        # 恢复训练模式
        self.encoder.train()
        self.decoder.train()
        self.attack.train()
        
        return test_metrics


if __name__ == "__main__":
    # 训练配置
    data_dir = os.path.join(project_root, 'data/train_images')
    save_dir = os.path.join(project_root, 'results/checkpoints')
    log_dir = os.path.join(project_root, 'results/logs')
    
    # 数据隔离说明
    print("\n" + "="*60)
    print("📁 数据目录说明:")
    print(f"  训练集: {data_dir} (用于训练，会被模型学习)")
    print(f"  测试集: {os.path.join(project_root, 'data/test_images')} (用于验证，不会参与训练)")
    print("  ✅ 训练数据和测试数据完全隔离，不会相互污染")
    print("  ✅ 测试集只用于评估，不会进行反向传播")
    print("="*60 + "\n")
    config = {
        # 数据
        'train_dir': data_dir,  # 需要准备图像
        'image_size': 256,
        'batch_size': 2,  # Mac M4建议2-4（DDIM模型较大，需要更多内存）
        'num_workers': 0,  # MPS 下避免多进程 pickle bch 对象；需并发可改>0并重构预处理实例化方式
        
        # 模型
        'watermark_length': 640,  # 64bit原文 + BCH(127,64,10) + 512bit签名
        
        # 训练
        'epochs': 100,  # 总训练轮数（可以设置大一点，比如100轮）
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
    
    # ========== 训练模式选择 ==========
    
    # 模式1：分批次训练（推荐）- 每次训练10轮，自动保存并退出
    # 下次运行时会自动从上次的checkpoint继续
    trainer.train(train_batch_epochs=10)
    
    # 模式2：完整训练 - 一次性训练完所有轮数
    # trainer.train()
    
    # 模式3：手动指定恢复点
    # trainer.train(resume_from='best')   # 从最佳模型恢复
    # trainer.train(resume_from='latest') # 从最新epoch恢复
    # trainer.train(resume_from=None)     # 强制从头开始
    # trainer.train(resume_from='path/to/checkpoint.pth')  # 从指定路径恢复