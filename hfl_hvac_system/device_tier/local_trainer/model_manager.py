"""
模型版本管理和加载模块
"""
import os
import json
import pickle
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """模型元数据"""
    model_id: str
    version: str
    created_at: str
    architecture: str
    parameters_count: int
    accuracy: float
    loss: float
    training_samples: int
    device_id: str
    checksum: str


class HVACModel(nn.Module):
    """HVAC控制模型"""
    
    def __init__(self, input_dim: int = 50, hidden_dims: List[int] = [128, 64, 32],
                 output_dim: int = 4, use_batch_norm: bool = False):
        """
        Args:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表
            output_dim: 输出维度 (温度设定, 湿度设定, 通风率, 功率)
            use_batch_norm: 是否使用BatchNorm（对小批次可能有问题）
        """
        super(HVACModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 构建隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            # 只在批次大小足够时使用BatchNorm
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
            
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # 个性化层（用于联邦学习）
        self.personalization_layer = nn.Linear(output_dim, output_dim)
        
    def forward(self, x: torch.Tensor, use_personalization: bool = True) -> torch.Tensor:
        """前向传播"""
        out = self.network(x)
        
        if use_personalization:
            out = self.personalization_layer(out)
            
        return out
    
    def get_shared_parameters(self) -> Dict[str, torch.Tensor]:
        """获取共享参数（用于联邦学习）"""
        shared_params = {}
        for name, param in self.network.named_parameters():
            shared_params[name] = param.data.clone()
        return shared_params
    
    def set_shared_parameters(self, params: Dict[str, torch.Tensor]):
        """设置共享参数"""
        for name, param in self.network.named_parameters():
            if name in params:
                param.data = params[name].clone()
                
    def get_personalized_parameters(self) -> Dict[str, torch.Tensor]:
        """获取个性化参数"""
        personal_params = {}
        for name, param in self.personalization_layer.named_parameters():
            personal_params[name] = param.data.clone()
        return personal_params


class ModelManager:
    """模型管理器"""
    
    def __init__(self, device_id: str, model_dir: str = "models"):
        self.device_id = device_id
        self.model_dir = os.path.join(model_dir, device_id)
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.current_model: Optional[HVACModel] = None
        self.model_history: List[ModelMetadata] = []
        self.best_model_path: Optional[str] = None
        
        # 加载模型历史
        self._load_model_history()
        
    def _load_model_history(self):
        """加载模型历史"""
        history_file = os.path.join(self.model_dir, "model_history.json")
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    history_data = json.load(f)
                    self.model_history = [
                        ModelMetadata(**item) for item in history_data
                    ]
            except Exception as e:
                logger.error(f"Failed to load model history: {e}")
                
    def _save_model_history(self):
        """保存模型历史"""
        history_file = os.path.join(self.model_dir, "model_history.json")
        try:
            history_data = [asdict(meta) for meta in self.model_history]
            with open(history_file, 'w') as f:
                json.dump(history_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save model history: {e}")
            
    def create_model(self, config: Optional[Dict] = None) -> HVACModel:
        """创建新模型"""
        if config is None:
            config = {
                'input_dim': 42,  # 改为实际的特征维度
                'hidden_dims': [128, 64, 32],
                'output_dim': 4
            }
            
        model = HVACModel(**config)
        self.current_model = model
        
        # 初始化权重
        self._initialize_weights(model)
        
        logger.info(f"Created new model with config: {config}")
        return model
    
    def _initialize_weights(self, model: nn.Module):
        """初始化模型权重"""
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def save_model(self, model: HVACModel, metrics: Dict, 
                  version: Optional[str] = None) -> str:
        """保存模型"""
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        model_filename = f"model_{version}.pt"
        model_path = os.path.join(self.model_dir, model_filename)
        
        # 保存模型状态
        torch.save({
            'model_state_dict': model.state_dict(),
            'architecture': model.__class__.__name__,
            'metrics': metrics,
            'version': version,
            'device_id': self.device_id
        }, model_path)
        
        # 计算校验和
        checksum = self._calculate_checksum(model_path)
        
        # 创建元数据
        metadata = ModelMetadata(
            model_id=f"{self.device_id}_{version}",
            version=version,
            created_at=datetime.now().isoformat(),
            architecture=model.__class__.__name__,
            parameters_count=sum(p.numel() for p in model.parameters()),
            accuracy=metrics.get('accuracy', 0.0),
            loss=metrics.get('loss', float('inf')),
            training_samples=metrics.get('training_samples', 0),
            device_id=self.device_id,
            checksum=checksum
        )
        
        self.model_history.append(metadata)
        self._save_model_history()
        
        # 更新最佳模型
        if self._is_better_model(metadata):
            self.best_model_path = model_path
            logger.info(f"New best model saved: {model_filename}")
            
        logger.info(f"Model saved: {model_filename}")
        return model_path
    
    def _calculate_checksum(self, file_path: str) -> str:
        """计算文件校验和"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _is_better_model(self, new_metadata: ModelMetadata) -> bool:
        """判断是否是更好的模型"""
        if not self.model_history or len(self.model_history) == 1:
            return True
            
        # 比较损失值
        best_loss = min(m.loss for m in self.model_history[:-1])
        return new_metadata.loss < best_loss
    
    def load_model(self, model_path: Optional[str] = None) -> HVACModel:
        """加载模型"""
        if model_path is None:
            model_path = self.best_model_path
            
        if model_path is None or not os.path.exists(model_path):
            logger.warning("No model found, creating new model")
            return self.create_model()
            
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # 创建模型实例
            model = self.create_model()
            
            # 加载权重
            model.load_state_dict(checkpoint['model_state_dict'])
            
            self.current_model = model
            logger.info(f"Model loaded from: {model_path}")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return self.create_model()
            
    def get_model_diff(self, model1_path: str, model2_path: str) -> Dict:
        """计算两个模型的差异"""
        checkpoint1 = torch.load(model1_path, map_location='cpu')
        checkpoint2 = torch.load(model2_path, map_location='cpu')
        
        state1 = checkpoint1['model_state_dict']
        state2 = checkpoint2['model_state_dict']
        
        diff = {}
        for key in state1.keys():
            if key in state2:
                param_diff = state2[key] - state1[key]
                diff[key] = {
                    'mean': float(param_diff.mean()),
                    'std': float(param_diff.std()),
                    'max': float(param_diff.max()),
                    'min': float(param_diff.min())
                }
                
        return diff
    
    def merge_models(self, model_paths: List[str], 
                    weights: Optional[List[float]] = None) -> HVACModel:
        """合并多个模型"""
        if weights is None:
            weights = [1.0 / len(model_paths)] * len(model_paths)
            
        # 加载所有模型
        states = []
        for path in model_paths:
            checkpoint = torch.load(path, map_location='cpu')
            states.append(checkpoint['model_state_dict'])
            
        # 加权平均
        merged_state = {}
        for key in states[0].keys():
            merged_param = torch.zeros_like(states[0][key])
            for state, weight in zip(states, weights):
                if key in state:
                    merged_param += weight * state[key]
            merged_state[key] = merged_param
            
        # 创建新模型并加载合并的权重
        model = self.create_model()
        model.load_state_dict(merged_state)
        
        return model
    
    def export_model(self, model: HVACModel, export_format: str = "onnx") -> str:
        """导出模型"""
        export_path = os.path.join(self.model_dir, f"exported_model.{export_format}")
        
        if export_format == "onnx":
            # 导出为ONNX格式
            dummy_input = torch.randn(1, 50)  # 假设输入维度为50
            torch.onnx.export(
                model,
                dummy_input,
                export_path,
                export_params=True,
                opset_version=11,
                input_names=['input'],
                output_names=['output']
            )
        elif export_format == "torchscript":
            # 导出为TorchScript
            scripted_model = torch.jit.script(model)
            scripted_model.save(export_path)
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
            
        logger.info(f"Model exported to: {export_path}")
        return export_path
    
    def get_model_summary(self) -> Dict:
        """获取模型摘要"""
        if self.current_model is None:
            return {}
            
        return {
            'architecture': self.current_model.__class__.__name__,
            'parameters_count': sum(p.numel() for p in self.current_model.parameters()),
            'trainable_parameters': sum(
                p.numel() for p in self.current_model.parameters() if p.requires_grad
            ),
            'model_size_mb': sum(
                p.numel() * p.element_size() for p in self.current_model.parameters()
            ) / (1024 * 1024),
            'history_length': len(self.model_history),
            'best_loss': min(m.loss for m in self.model_history) if self.model_history else None
        }
        
    def cleanup_old_models(self, keep_n: int = 5):
        """清理旧模型，只保留最近的n个"""
        if len(self.model_history) <= keep_n:
            return
            
        # 按时间排序
        self.model_history.sort(key=lambda x: x.created_at, reverse=True)
        
        # 删除旧模型文件
        for metadata in self.model_history[keep_n:]:
            model_path = os.path.join(self.model_dir, f"model_{metadata.version}.pt")
            if os.path.exists(model_path):
                os.remove(model_path)
                logger.info(f"Removed old model: {model_path}")
                
        # 更新历史记录
        self.model_history = self.model_history[:keep_n]
        self._save_model_history()


if __name__ == "__main__":
    # 测试模型管理器
    manager = ModelManager("device_001")
    
    # 创建模型
    model = manager.create_model()
    print(f"Model created: {manager.get_model_summary()}")
    
    # 保存模型
    metrics = {
        'accuracy': 0.95,
        'loss': 0.05,
        'training_samples': 1000
    }
    model_path = manager.save_model(model, metrics)
    print(f"Model saved to: {model_path}")
    
    # 加载模型
    loaded_model = manager.load_model(model_path)
    print("Model loaded successfully")
    
    # 测试模型前向传播
    dummy_input = torch.randn(32, 50)
    output = loaded_model(dummy_input)
    print(f"Output shape: {output.shape}")