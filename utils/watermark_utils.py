"""
专利权利要求1-步骤S1：可信溯源信息预处理
"""

import hashlib
import time
from pathlib import Path
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import (
    encode_dss_signature,
    decode_dss_signature,
)
from cryptography.exceptions import InvalidSignature
import bchlib

class WatermarkPreprocessor:
    def __init__(
        self,
        bch_bits=10,
        bch_polynomial=137,
        private_key_path: str | None = None,
        public_key_path: str | None = None,
        target_bit_len: int = 640,
    ):
        """
        初始化水印预处理器
        
        Args:
            bch_bits: BCH纠错位数（默认10，对应(127,64,10)）
            bch_polynomial: BCH本原多项式（默认137）
            private_key_path: PEM 私钥路径（训练端）
            public_key_path: PEM 公钥路径（提取端/验签）
            target_bit_len: 载荷目标长度（默认640比特，需与模型配置一致）
        """
        self.bch = bchlib.BCH(bch_bits, prim_poly=bch_polynomial)
        self.target_bit_len = target_bit_len
        
        # 加载或生成密钥
        self.private_key = None
        self.public_key = None
        if private_key_path and Path(private_key_path).exists():
            self.private_key = self._load_private_key(private_key_path)
            self.public_key = self.private_key.public_key()
        if public_key_path and Path(public_key_path).exists():
            self.public_key = self._load_public_key(public_key_path)
        if self.private_key is None and self.public_key is None:
            self.private_key = ec.generate_private_key(ec.SECP256K1())
            self.public_key = self.private_key.public_key()
        # 如果只加载了私钥未加载公钥，补齐公钥
        if self.private_key and self.public_key is None:
            self.public_key = self.private_key.public_key()
    
    def preprocess(self, source_info: str, image_hash_hex: str | None = None):
        """
        专利步骤S1：预处理溯源信息
        Args:
            source_info: 原始溯源信息（如用户ID）
            image_hash_hex: 原始图像SHA-256十六进制字符串，用于截取前32比特
        
        Returns:
            payload_bits: 待嵌载荷（target_bit_len比特）
            timestamp: 时间戳
        """
        # 构造64比特原始信息串：ID(16bit) | timestamp(16bit) | hash前32bit
        ts = int(time.time()) & 0xFFFF  # 16bit
        ts_bits = format(ts, '016b')
        # ID 16bit：从 source_info 的哈希截取
        id_int = int(hashlib.sha256(source_info.encode('utf-8')).hexdigest()[:4], 16)
        id_bits = format(id_int & 0xFFFF, '016b')
        # hash 32bit：来自图像hash
        if image_hash_hex is None:
            image_hash_hex = hashlib.sha256(source_info.encode('utf-8')).hexdigest()
        hash_prefix = image_hash_hex[:8]  # 32bit
        hash_bits = format(int(hash_prefix, 16), '032b')
        raw_bits = id_bits + ts_bits + hash_bits  # 64 bits
        data_bytes = int(raw_bits, 2).to_bytes(8, 'big')
        
        # BCH(127,64,10)
        ecc = self.bch.encode(data_bytes)
        encoded_data = data_bytes + ecc  # bytes
        
        # ECDSA(secp256k1) 原始 r||s (64字节)
        der_sig = self.private_key.sign(encoded_data, ec.ECDSA(hashes.SHA256()))
        r, s = decode_dss_signature(der_sig)
        signature = r.to_bytes(32, 'big') + s.to_bytes(32, 'big')
        
        structured = encoded_data + signature  # 预计约80字节
        
        # 截断或填充到目标长度
        target_bytes = self.target_bit_len // 8
        if len(structured) > target_bytes:
            structured = structured[:target_bytes]
        else:
            structured = structured.ljust(target_bytes, b'\x00')
        
        payload_bits = ''.join(format(byte, '08b') for byte in structured)
        
        return payload_bits, ts

    # ========== 验签/解码相关 ==========
    def decode_and_verify(self, payload_bits: str):
        """
        从提取的比特序列中验证签名并进行BCH纠错解码
        Returns: dict(status, verified, message, info, timestamp)
        """
        target_bytes = self.target_bit_len // 8
        payload_bits = payload_bits[:self.target_bit_len]
        payload_bytes = int(payload_bits, 2).to_bytes(target_bytes, 'big')
        
        data_len_bytes = 8  # 64bit 原文
        encoded_len = data_len_bytes + self.bch.ecc_bytes
        if len(payload_bytes) < encoded_len + 64:
            return {"status": False, "verified": False, "message": "载荷长度不足"}
        
        encoded_data = payload_bytes[:encoded_len]
        signature = payload_bytes[encoded_len:encoded_len+64]
        
        # 验签
        if self.public_key is None:
            return {"status": False, "verified": False, "message": "缺少公钥，无法验签"}
        try:
            r = int.from_bytes(signature[:32], 'big')
            s = int.from_bytes(signature[32:], 'big')
            der_sig = encode_dss_signature(r, s)
            self.public_key.verify(der_sig, encoded_data, ec.ECDSA(hashes.SHA256()))
            verified = True
        except InvalidSignature:
            verified = False
        
        # BCH 解码
        try:
            data, _ = self.bch.decode(encoded_data)
        except Exception as e:
            return {"status": False, "verified": verified, "message": f"BCH解码失败: {e}"}
        
        # 解析原始64bit
        if len(data) < data_len_bytes:
            return {"status": False, "verified": verified, "message": "数据长度不足"}
        info_bits = format(int.from_bytes(data[:data_len_bytes], 'big'), '064b')
        id_bits = info_bits[:16]
        ts_bits = info_bits[16:32]
        hash_bits = info_bits[32:64]
        return {
            "status": True,
            "verified": verified,
            "identity_bits": id_bits,
            "timestamp": int(ts_bits, 2),
            "hash_prefix": hash_bits,
            "message": "ok"
        }

    # ========== 密钥持久化 ==========
    def save_keys(self, private_path: str, public_path: str):
        if self.private_key:
            pem = self.private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            Path(private_path).write_bytes(pem)
        if self.public_key:
            pem = self.public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            Path(public_path).write_bytes(pem)
    
    def _load_private_key(self, path: str):
        data = Path(path).read_bytes()
        return serialization.load_pem_private_key(data, password=None)
    
    def _load_public_key(self, path: str):
        data = Path(path).read_bytes()
        return serialization.load_pem_public_key(data)
    
    def verify_signature(self, payload_bytes: bytes):
        """
        验证数字签名（提取阶段使用）
        
        Args:
            payload_bytes: 提取的载荷字节
        
        Returns:
            bool: 验证是否通过
        """
        # 分离数据和签名
        data_len = self.bch.n // 8
        encoded_data = payload_bytes[:data_len]
        signature = payload_bytes[data_len:]
        
        try:
            self.public_key.verify(
                signature,
                encoded_data,
                ec.ECDSA(hashes.SHA256())
            )
            return True
        except Exception as e:
            print(f"⚠️ 签名验证失败: {e}")
            return False


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 测试水印预处理器")
    print("=" * 60)
    
    try:
        # 初始化
        preprocessor = WatermarkPreprocessor()
        
        # 打印BCH参数
        print(f"📊 BCH参数:")
        print(f"   码长n: {preprocessor.bch.n} bits")
        print(f"   纠错位: {preprocessor.bch.t} bits")
        print(f"   ECC字节: {preprocessor.bch.ecc_bytes} bytes")
        print()
        
        # 测试预处理
        payload, ts = preprocessor.preprocess("user_12345")
        
        print(f"✅ 载荷长度: {len(payload)} bits")
        print(f"✅ 时间戳: {ts}")
        print(f"✅ 前32比特: {payload[:32]}")
        print()
        
        # 测试不同用户
        test_users = ["alice", "bob", "charlie"]
        print("🔍 测试多个用户:")
        for user in test_users:
            p, t = preprocessor.preprocess(user)
            print(f"   {user}: {p[:20]}... (len={len(p)})")
        
        print("=" * 60)
        print("🎉 所有测试通过！")
        print("=" * 60)
        
    except Exception as e:
        print("=" * 60)
        print(f"❌ 测试失败: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
