"""
差分隐私和数据脱敏模块
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import hashlib
import hmac
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PrivacyConfig:
    """隐私配置"""
    epsilon: float = 1.0  # 隐私预算
    delta: float = 1e-5   # 隐私参数
    clip_norm: float = 1.0  # 梯度裁剪范数
    noise_multiplier: float = 1.0  # 噪声乘数
    secure_aggregation: bool = True  # 是否使用安全聚合


class DifferentialPrivacy:
    """差分隐私实现"""
    
    def __init__(self, config: PrivacyConfig):
        self.config = config
        self.privacy_budget_used = 0.0
        
    def add_noise_to_gradient(self, gradients: np.ndarray, 
                             batch_size: int = 1) -> Tuple[np.ndarray, float]:
        """
        向梯度添加差分隐私噪声
        
        Args:
            gradients: 原始梯度
            batch_size: 批次大小
            
        Returns:
            (noisy_gradients, privacy_cost)
        """
        # 梯度裁剪
        clipped_grads = self._clip_gradients(gradients)
        
        # 计算敏感度
        sensitivity = 2 * self.config.clip_norm / batch_size
        
        # 计算噪声标准差
        noise_std = sensitivity * self.config.noise_multiplier
        
        # 添加高斯噪声
        noise = np.random.normal(0, noise_std, gradients.shape)
        noisy_grads = clipped_grads + noise
        
        # 计算隐私成本
        privacy_cost = self._compute_privacy_cost(
            self.config.noise_multiplier, 
            batch_size
        )
        
        self.privacy_budget_used += privacy_cost
        
        return noisy_grads, privacy_cost
    
    def _clip_gradients(self, gradients: np.ndarray) -> np.ndarray:
        """梯度裁剪"""
        grad_norm = np.linalg.norm(gradients)
        
        if grad_norm > self.config.clip_norm:
            scale = self.config.clip_norm / grad_norm
            return gradients * scale
        
        return gradients
    
    def _compute_privacy_cost(self, noise_multiplier: float, 
                             batch_size: int) -> float:
        """计算隐私成本（简化版）"""
        # 使用RDP (Rényi Differential Privacy) 分析
        # 这是一个简化的计算
        q = batch_size / 10000  # 采样率（假设总数据量为10000）
        
        # 简化的隐私成本计算
        privacy_cost = q * np.sqrt(2 * np.log(1.25 / self.config.delta)) / noise_multiplier
        
        return min(privacy_cost, self.config.epsilon)
    
    def add_noise_to_data(self, data: Dict, 
                         sensitivity_map: Optional[Dict] = None) -> Dict:
        """向数据添加拉普拉斯噪声"""
        noisy_data = data.copy()
        
        # 默认敏感度映射
        if sensitivity_map is None:
            sensitivity_map = {
                'temperature': 0.5,
                'humidity': 1.0,
                'occupancy': 1,
                'power_consumption': 0.2
            }
        
        for key, value in data.items():
            if key in sensitivity_map:
                sensitivity = sensitivity_map[key]
                # 拉普拉斯噪声
                scale = sensitivity / self.config.epsilon
                noise = np.random.laplace(0, scale)
                
                if isinstance(value, (int, float)):
                    noisy_data[key] = value + noise
                elif isinstance(value, np.ndarray):
                    noise_array = np.random.laplace(0, scale, value.shape)
                    noisy_data[key] = value + noise_array
        
        return noisy_data
    
    def check_privacy_budget(self) -> bool:
        """检查隐私预算是否超限"""
        return self.privacy_budget_used < self.config.epsilon
    
    def reset_privacy_budget(self):
        """重置隐私预算"""
        self.privacy_budget_used = 0.0
        logger.info("Privacy budget reset")


class DataAnonymizer:
    """数据匿名化器"""
    
    def __init__(self, secret_key: str = "hvac_secret_2024"):
        self.secret_key = secret_key.encode()
        
    def anonymize_device_id(self, device_id: str) -> str:
        """匿名化设备ID"""
        h = hmac.new(self.secret_key, device_id.encode(), hashlib.sha256)
        return h.hexdigest()[:16]  # 返回前16个字符
    
    def generalize_location(self, zone_id: str) -> str:
        """泛化位置信息"""
        # 将具体区域泛化为楼层或建筑
        if 'zone' in zone_id:
            floor_num = int(zone_id.split('_')[-1]) // 10
            return f"floor_{floor_num}"
        return "building"
    
    def suppress_sensitive_data(self, data: Dict, 
                               sensitive_fields: List[str]) -> Dict:
        """抑制敏感数据"""
        cleaned_data = data.copy()
        
        for field in sensitive_fields:
            if field in cleaned_data:
                del cleaned_data[field]
                
        return cleaned_data
    
    def k_anonymize_occupancy(self, occupancy: int, k: int = 5) -> int:
        """K-匿名化占用率"""
        # 将占用率舍入到最近的k的倍数
        return (occupancy // k) * k
    
    def add_dummy_data(self, real_data: List[Dict], 
                      dummy_ratio: float = 0.2) -> List[Dict]:
        """添加虚拟数据以混淆真实数据"""
        num_dummy = int(len(real_data) * dummy_ratio)
        all_data = real_data.copy()
        
        for _ in range(num_dummy):
            # 生成虚拟数据
            dummy = {
                'device_id': f"dummy_{np.random.randint(1000, 9999)}",
                'temperature': np.random.uniform(18, 26),
                'humidity': np.random.uniform(30, 70),
                'is_dummy': True  # 标记为虚拟数据
            }
            all_data.append(dummy)
            
        # 随机打乱顺序
        np.random.shuffle(all_data)
        
        return all_data


class SecureAggregation:
    """安全聚合协议（简化版）"""
    
    def __init__(self, num_clients: int, threshold: int):
        """
        Args:
            num_clients: 客户端数量
            threshold: 恢复秘密所需的最小客户端数
        """
        self.num_clients = num_clients
        self.threshold = threshold
        self.client_masks = {}
        
    def generate_mask(self, client_id: str, seed: int) -> np.ndarray:
        """生成随机掩码"""
        np.random.seed(seed)
        mask = np.random.randn(1000)  # 假设模型有1000个参数
        self.client_masks[client_id] = mask
        return mask
    
    def mask_gradients(self, gradients: np.ndarray, 
                       client_id: str) -> np.ndarray:
        """使用掩码加密梯度"""
        if client_id not in self.client_masks:
            mask = self.generate_mask(client_id, hash(client_id) % 10000)
        else:
            mask = self.client_masks[client_id]
            
        # 确保掩码和梯度形状匹配
        if mask.shape != gradients.shape:
            mask = np.resize(mask, gradients.shape)
            
        return gradients + mask
    
    def aggregate_masked_gradients(self, masked_grads_list: List[np.ndarray],
                                  client_ids: List[str]) -> np.ndarray:
        """聚合掩码梯度"""
        # 在实际实现中，这里会使用同态加密或秘密分享
        # 简化版：假设所有客户端都参与，掩码会相互抵消
        
        aggregated = np.zeros_like(masked_grads_list[0])
        
        for masked_grads in masked_grads_list:
            aggregated += masked_grads
            
        # 移除掩码（在真实场景中，掩码会通过安全协议移除）
        for client_id in client_ids:
            if client_id in self.client_masks:
                mask = self.client_masks[client_id]
                if mask.shape != aggregated.shape:
                    mask = np.resize(mask, aggregated.shape)
                aggregated -= mask
                
        return aggregated / len(masked_grads_list)


class PrivacyAccountant:
    """隐私预算记账器"""
    
    def __init__(self, total_budget: float):
        self.total_budget = total_budget
        self.used_budget = 0.0
        self.history = []
        
    def spend_budget(self, amount: float, operation: str) -> bool:
        """
        消费隐私预算
        
        Returns:
            是否成功（预算充足）
        """
        if self.used_budget + amount > self.total_budget:
            logger.warning(f"Privacy budget exceeded for {operation}")
            return False
            
        self.used_budget += amount
        self.history.append({
            'operation': operation,
            'amount': amount,
            'total_used': self.used_budget,
            'remaining': self.total_budget - self.used_budget
        })
        
        return True
    
    def get_remaining_budget(self) -> float:
        """获取剩余预算"""
        return self.total_budget - self.used_budget
    
    def reset(self):
        """重置预算"""
        self.used_budget = 0.0
        self.history = []
        
    def get_report(self) -> Dict:
        """获取隐私预算报告"""
        return {
            'total_budget': self.total_budget,
            'used_budget': self.used_budget,
            'remaining_budget': self.get_remaining_budget(),
            'usage_percentage': (self.used_budget / self.total_budget) * 100,
            'num_operations': len(self.history),
            'history': self.history[-10:]  # 最近10条记录
        }


if __name__ == "__main__":
    # 测试差分隐私
    print("Testing Differential Privacy...")
    config = PrivacyConfig(epsilon=1.0, noise_multiplier=1.0)
    dp = DifferentialPrivacy(config)
    
    # 测试梯度噪声
    gradients = np.random.randn(100)
    noisy_grads, privacy_cost = dp.add_noise_to_gradient(gradients, batch_size=32)
    print(f"Original gradient norm: {np.linalg.norm(gradients):.4f}")
    print(f"Noisy gradient norm: {np.linalg.norm(noisy_grads):.4f}")
    print(f"Privacy cost: {privacy_cost:.4f}")
    
    # 测试数据匿名化
    print("\nTesting Data Anonymization...")
    anonymizer = DataAnonymizer()
    
    device_id = "device_001"
    anon_id = anonymizer.anonymize_device_id(device_id)
    print(f"Original ID: {device_id}")
    print(f"Anonymized ID: {anon_id}")
    
    # 测试安全聚合
    print("\nTesting Secure Aggregation...")
    secure_agg = SecureAggregation(num_clients=3, threshold=2)
    
    # 模拟三个客户端的梯度
    grads1 = np.array([1.0, 2.0, 3.0])
    grads2 = np.array([2.0, 3.0, 4.0])
    grads3 = np.array([3.0, 4.0, 5.0])
    
    # 加掩码
    masked1 = secure_agg.mask_gradients(grads1, "client1")
    masked2 = secure_agg.mask_gradients(grads2, "client2")
    masked3 = secure_agg.mask_gradients(grads3, "client3")
    
    # 聚合
    aggregated = secure_agg.aggregate_masked_gradients(
        [masked1, masked2, masked3],
        ["client1", "client2", "client3"]
    )
    
    print(f"Expected average: {(grads1 + grads2 + grads3) / 3}")
    print(f"Aggregated result: {aggregated}")
    
    # 测试隐私预算记账
    print("\nTesting Privacy Accountant...")
    accountant = PrivacyAccountant(total_budget=10.0)
    
    accountant.spend_budget(2.0, "gradient_update")
    accountant.spend_budget(1.5, "data_release")
    
    report = accountant.get_report()
    print(f"Privacy budget report:")
    print(f"  Total: {report['total_budget']}")
    print(f"  Used: {report['used_budget']}")
    print(f"  Remaining: {report['remaining_budget']}")
    print(f"  Usage: {report['usage_percentage']:.1f}%")