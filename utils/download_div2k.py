"""
下载DIV2K数据集（CVPR 2017）
专业图像超分辨率数据集，高质量
"""

import os
import requests
from tqdm import tqdm
import zipfile

def download_file(url, save_path):
    """下载文件并显示进度"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(save_path, 'wb') as f, tqdm(
        total=total_size,
        unit='B',
        unit_scale=True,
        desc=os.path.basename(save_path)
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

def download_div2k():
    """下载并解压DIV2K数据集"""
    
    print("=" * 60)
    print("📥 下载DIV2K数据集")
    print("=" * 60)
    
    # 数据集URL
    urls = {
        'train_hr': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip',
        'valid_hr': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip'
    }
    
    # 创建目录
    os.makedirs('data/div2k', exist_ok=True)
    os.makedirs('data/train_images', exist_ok=True)
    
    # 下载训练集
    print("\n1️⃣  下载训练集 (800张, 约3.5GB)...")
    train_zip = 'data/div2k/DIV2K_train_HR.zip'
    if not os.path.exists(train_zip):
        download_file(urls['train_hr'], train_zip)
    else:
        print("   ✅ 训练集已存在，跳过下载")
    
    # 下载验证集
    print("\n2️⃣  下载验证集 (100张, 约500MB)...")
    valid_zip = 'data/div2k/DIV2K_valid_HR.zip'
    if not os.path.exists(valid_zip):
        download_file(urls['valid_hr'], valid_zip)
    else:
        print("   ✅ 验证集已存在，跳过下载")
    
    # 解压
    print("\n3️⃣  解压数据...")
    with zipfile.ZipFile(train_zip, 'r') as zip_ref:
        zip_ref.extractall('data/div2k/')
    
    with zipfile.ZipFile(valid_zip, 'r') as zip_ref:
        zip_ref.extractall('data/div2k/')
    
    # 移动到训练目录
    print("\n4️⃣  整理文件...")
    import shutil
    
    # 移动训练集
    src_train = 'data/div2k/DIV2K_train_HR'
    if os.path.exists(src_train):
        for img in os.listdir(src_train):
            shutil.move(
                os.path.join(src_train, img),
                'data/train_images/'
            )
    
    # 移动验证集
    src_valid = 'data/div2k/DIV2K_valid_HR'
    if os.path.exists(src_valid):
        os.makedirs('data/test_images', exist_ok=True)
        for img in os.listdir(src_valid):
            shutil.move(
                os.path.join(src_valid, img),
                'data/test_images/'
            )
    
    print("\n" + "=" * 60)
    print("✅ DIV2K数据集准备完成！")
    print("=" * 60)
    print(f"   训练集: data/train_images/ (800张)")
    print(f"   测试集: data/test_images/ (100张)")
    print(f"   总大小: ~5GB")
    print("=" * 60)


if __name__ == "__main__":
    download_div2k()
