"""
本地训练逻辑模块
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Callable
import logging
from collections import deque
import time

from ..data_collector.privacy_guard import DifferentialPrivacy, PrivacyConfig

logger = logging.getLogger(__name__)


class HVACDataset(Dataset):
    """HVAC数据集"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class LocalTrainer:
    """本地训练器"""
    
    def __init__(self, device_id: str, model: nn.Module,
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 enable_privacy: bool = True):
        """
        Args:
            device_id: 设备ID
            model: 模型实例
            learning_rate: 学习率
            batch_size: 批次大小
            enable_privacy: 是否启用差分隐私
        """
        self.device_id = device_id
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.enable_privacy = enable_privacy
        
        # 优化器
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
        # 差分隐私
        if enable_privacy:
            privacy_config = PrivacyConfig(epsilon=1.0, clip_norm=1.0)
            self.privacy_engine = DifferentialPrivacy(privacy_config)
        else:
            self.privacy_engine = None
            
        # 训练历史
        self.training_history = {
            'loss': [],
            'accuracy': [],
            'privacy_cost': []
        }
        
        # 早停机制
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.early_stop_patience = 10
        
    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_privacy_cost = 0
        num_batches = 0
        
        for batch_features, batch_labels in dataloader:
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(batch_features)
            loss = self.criterion(outputs, batch_labels)
            
            # 反向传播
            loss.backward()
            
            # 差分隐私处理
            if self.privacy_engine:
                for param in self.model.parameters():
                    if param.grad is not None:
                        noisy_grad, privacy_cost = self.privacy_engine.add_noise_to_gradient(
                            param.grad.numpy(),
                            batch_size=len(batch_features)
                        )
                        param.grad = torch.FloatTensor(noisy_grad)
                        total_privacy_cost += privacy_cost
                        
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 更新参数
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        avg_loss = total_loss / num_batches
        avg_privacy_cost = total_privacy_cost / num_batches if self.privacy_engine else 0
        
        return avg_loss, avg_privacy_cost
    
    def validate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """验证模型"""
        self.model.eval()
        total_loss = 0
        total_accuracy = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_features, batch_labels in dataloader:
                outputs = self.model(batch_features)
                loss = self.criterion(outputs, batch_labels)
                
                # 计算准确率（对于回归问题，使用相对误差）
                relative_error = torch.abs(outputs - batch_labels) / (torch.abs(batch_labels) + 1e-6)
                accuracy = (relative_error < 0.1).float().mean()  # 10%误差内认为准确
                
                total_loss += loss.item()
                total_accuracy += accuracy.item()
                num_batches += 1
                
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        return avg_loss, avg_accuracy
    
    def train(self, train_data: Tuple[np.ndarray, np.ndarray],
             val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
             epochs: int = 10,
             verbose: bool = True) -> Dict:
        """
        训练模型
        
        Args:
            train_data: (features, labels) 训练数据
            val_data: (features, labels) 验证数据
            epochs: 训练轮数
            verbose: 是否打印训练信息
            
        Returns:
            训练结果字典
        """
        # 创建数据加载器
        train_dataset = HVACDataset(train_data[0], train_data[1])
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, 
                                shuffle=True)
        
        val_loader = None
        if val_data:
            val_dataset = HVACDataset(val_data[0], val_data[1])
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, 
                                  shuffle=False)
        
        # 训练循环
        start_time = time.time()
        
        for epoch in range(epochs):
            # 训练
            train_loss, privacy_cost = self.train_epoch(train_loader)
            
            # 验证
            if val_loader:
                val_loss, val_accuracy = self.validate(val_loader)
            else:
                val_loss, val_accuracy = train_loss, 0.0
                
            # 记录历史
            self.training_history['loss'].append(train_loss)
            self.training_history['accuracy'].append(val_accuracy)
            self.training_history['privacy_cost'].append(privacy_cost)
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            # 早停检查
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.early_stop_patience:
                if verbose:
                    logger.info(f"Early stopping at epoch {epoch}")
                break
                
            # 打印进度
            if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
                logger.info(
                    f"Epoch {epoch}/{epochs} - "
                    f"Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Accuracy: {val_accuracy:.2%}, "
                    f"Privacy Cost: {privacy_cost:.4f}"
                )
        
        training_time = time.time() - start_time
        
        # 返回训练结果
        return {
            'final_loss': self.training_history['loss'][-1],
            'final_accuracy': self.training_history['accuracy'][-1] if val_loader else 0,
            'best_loss': self.best_loss,
            'epochs_trained': len(self.training_history['loss']),
            'training_time': training_time,
            'total_privacy_cost': sum(self.training_history['privacy_cost']),
            'model_state': self.model.state_dict()
        }
    
    def incremental_train(self, new_data: Tuple[np.ndarray, np.ndarray],
                         epochs: int = 5) -> Dict:
        """增量训练（在线学习）"""
        # 使用较小的学习率进行增量训练
        original_lr = self.optimizer.param_groups[0]['lr']
        self.optimizer.param_groups[0]['lr'] = original_lr * 0.1
        
        # 训练
        result = self.train(new_data, epochs=epochs, verbose=False)
        
        # 恢复原始学习率
        self.optimizer.param_groups[0]['lr'] = original_lr
        
        return result
    
    def get_gradients(self) -> Dict[str, np.ndarray]:
        """获取模型梯度（用于联邦学习）"""
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.numpy().copy()
        return gradients
    
    def apply_gradients(self, gradients: Dict[str, np.ndarray]):
        """应用梯度更新"""
        for name, param in self.model.named_parameters():
            if name in gradients:
                param.data -= self.learning_rate * torch.FloatTensor(gradients[name])


class AdaptiveTrainer(LocalTrainer):
    """自适应训练器"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 自适应参数
        self.performance_buffer = deque(maxlen=10)
        self.lr_adjustment_factor = 1.0
        
    def adaptive_adjust(self, performance_metric: float):
        """根据性能自适应调整训练参数"""
        self.performance_buffer.append(performance_metric)
        
        if len(self.performance_buffer) >= 5:
            recent_performance = list(self.performance_buffer)[-5:]
            
            # 检查性能趋势
            if all(recent_performance[i] <= recent_performance[i-1] 
                  for i in range(1, len(recent_performance))):
                # 性能持续改善，增加学习率
                self.lr_adjustment_factor = min(1.5, self.lr_adjustment_factor * 1.1)
            elif all(recent_performance[i] >= recent_performance[i-1] 
                    for i in range(1, len(recent_performance))):
                # 性能持续恶化，降低学习率
                self.lr_adjustment_factor = max(0.5, self.lr_adjustment_factor * 0.9)
                
        # 应用调整
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate * self.lr_adjustment_factor
            
    def train_with_adaptation(self, *args, **kwargs) -> Dict:
        """带自适应的训练"""
        result = self.train(*args, **kwargs)
        
        # 根据最终损失调整参数
        self.adaptive_adjust(result['final_loss'])
        
        return result


class FederatedLocalTrainer(LocalTrainer):
    """联邦学习本地训练器"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_model_state = None
        self.local_steps = 0
        self.max_local_steps = 10
        
    def set_global_model(self, global_state: Dict[str, torch.Tensor]):
        """设置全局模型状态"""
        self.global_model_state = global_state
        
        # 只更新共享层
        for name, param in self.model.named_parameters():
            if 'personalization' not in name and name in global_state:
                param.data = global_state[name].clone()
                
    def get_model_update(self) -> Dict[str, np.ndarray]:
        """获取模型更新（相对于全局模型的差异）"""
        if self.global_model_state is None:
            return self.get_gradients()
            
        updates = {}
        for name, param in self.model.named_parameters():
            if 'personalization' not in name and name in self.global_model_state:
                update = param.data - self.global_model_state[name]
                updates[name] = update.numpy()
                
        return updates
    
    def fedprox_regularization(self, mu: float = 0.01) -> torch.Tensor:
        """FedProx正则化项"""
        if self.global_model_state is None:
            return torch.tensor(0.0)
            
        reg_loss = 0
        for name, param in self.model.named_parameters():
            if 'personalization' not in name and name in self.global_model_state:
                reg_loss += mu * torch.norm(param - self.global_model_state[name]) ** 2
                
        return reg_loss
    
    def train_federated_round(self, train_data: Tuple[np.ndarray, np.ndarray],
                            local_epochs: int = 5) -> Dict:
        """执行一轮联邦训练"""
        result = self.train(train_data, epochs=local_epochs, verbose=False)
        
        # 获取模型更新
        model_update = self.get_model_update()
        
        result['model_update'] = model_update
        result['num_samples'] = len(train_data[0])
        
        return result


if __name__ == "__main__":
    # 测试训练器
    from model_manager import HVACModel
    
    # 创建模型
    model = HVACModel(input_dim=50, hidden_dims=[64, 32], output_dim=4)
    
    # 创建训练器
    trainer = LocalTrainer("device_001", model, enable_privacy=True)
    
    # 生成模拟数据
    num_samples = 1000
    train_features = np.random.randn(num_samples, 50)
    train_labels = np.random.randn(num_samples, 4)
    
    val_features = np.random.randn(200, 50)
    val_labels = np.random.randn(200, 4)
    
    # 训练
    print("Starting training...")
    result = trainer.train(
        train_data=(train_features, train_labels),
        val_data=(val_features, val_labels),
        epochs=20
    )
    
    print(f"\nTraining completed:")
    print(f"  Final loss: {result['final_loss']:.4f}")
    print(f"  Final accuracy: {result['final_accuracy']:.2%}")
    print(f"  Training time: {result['training_time']:.2f}s")
    print(f"  Total privacy cost: {result['total_privacy_cost']:.4f}")