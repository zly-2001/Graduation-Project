"""
下载COCO 2017数据集
"""

import os
import requests
from tqdm import tqdm
import zipfile

def download_file(url, save_path):
    """下载文件并显示进度"""
    if os.path.exists(save_path):
        print(f"   ✅ 文件已存在: {os.path.basename(save_path)}")
        return
    
    print(f"   📥 下载: {os.path.basename(save_path)}")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(save_path, 'wb') as f, tqdm(
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

def download_coco():
    """下载COCO 2017数据集"""
    
    print("=" * 60)
    print("�� 下载COCO 2017数据集")
    print("=" * 60)
    
    # COCO官方镜像
    base_url = "http://images.cocodataset.org/zips"
    
    # 训练集和验证集
    files = {
        'train': 'train2017.zip',    # 18GB
        'val': 'val2017.zip'          # 1GB
    }
    
    # 创建目录
    os.makedirs('data/coco', exist_ok=True)
    os.makedirs('data/train_images', exist_ok=True)
    os.makedirs('data/test_images', exist_ok=True)
    
    # 下载
    for split, filename in files.items():
        print(f"\n{'='*60}")
        print(f"📦 {split.upper()} 数据集")
        print(f"{'='*60}")
        
        zip_path = f'data/coco/{filename}'
        url = f"{base_url}/{filename}"
        
        # 下载
        download_file(url, zip_path)
        
        # 解压
        print(f"   📂 解压中...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('data/coco/')
        
        print(f"   ✅ {split.upper()} 完成")
    
    # 移动文件
    print("\n" + "=" * 60)
    print("📁 整理文件...")
    print("=" * 60)
    
    import shutil
    
    # 移动训练集
    src_train = 'data/coco/train2017'
    if os.path.exists(src_train):
        print("   移动训练集...")
        for img in tqdm(os.listdir(src_train)):
            shutil.move(
                os.path.join(src_train, img),
                'data/train_images/'
            )
    
    # 移动验证集
    src_val = 'data/coco/val2017'
    if os.path.exists(src_val):
        print("   移动验证集...")
        for img in tqdm(os.listdir(src_val)):
            shutil.move(
                os.path.join(src_val, img),
                'data/test_images/'
            )
    
    print("\n" + "=" * 60)
    print("✅ COCO 2017数据集准备完成！")
    print("=" * 60)
    print(f"   训练集: data/train_images/ (118,287张)")
    print(f"   测试集: data/test_images/ (5,000张)")
    print(f"   总大小: ~25GB")
    print("=" * 60)


if __name__ == "__main__":
    download_coco()
