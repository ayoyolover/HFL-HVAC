"""
联邦学习聚合算法实现
"""
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from collections import defaultdict
import copy

logger = logging.getLogger(__name__)


@dataclass
class ClientUpdate:
    """客户端更新"""
    device_id: str
    model_weights: Dict[str, np.ndarray]
    num_samples: int
    metrics: Dict[str, float]
    timestamp: float


class FederatedAggregator:
    """联邦聚合器基类"""
    
    def __init__(self, aggregation_method: str = "fedavg"):
        self.aggregation_method = aggregation_method
        self.global_model = None
        self.round_number = 0
        self.client_updates_history = defaultdict(list)
        
    def aggregate(self, client_updates: List[ClientUpdate]) -> Dict[str, np.ndarray]:
        """执行聚合"""
        if self.aggregation_method == "fedavg":
            return self.federated_averaging(client_updates)
        elif self.aggregation_method == "fedprox":
            return self.fedprox_aggregation(client_updates)
        elif self.aggregation_method == "scaffold":
            return self.scaffold_aggregation(client_updates)
        elif self.aggregation_method == "fednova":
            return self.fednova_aggregation(client_updates)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
            
    def federated_averaging(self, client_updates: List[ClientUpdate]) -> Dict[str, np.ndarray]:
        """
        FedAvg: 加权平均聚合
        McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data"
        """
        if not client_updates:
            return self.global_model if self.global_model else {}
            
        # 计算总样本数
        total_samples = sum(update.num_samples for update in client_updates)
        
        # 初始化聚合模型
        aggregated_weights = {}
        
        # 加权平均
        for update in client_updates:
            weight = update.num_samples / total_samples
            
            for param_name, param_value in update.model_weights.items():
                if param_name not in aggregated_weights:
                    aggregated_weights[param_name] = np.zeros_like(param_value)
                aggregated_weights[param_name] += weight * param_value
                
        self.global_model = aggregated_weights
        self.round_number += 1
        
        logger.info(f"FedAvg aggregation completed for round {self.round_number}")
        return aggregated_weights
    
    def fedprox_aggregation(self, client_updates: List[ClientUpdate],
                          mu: float = 0.01) -> Dict[str, np.ndarray]:
        """
        FedProx: 带近端项的联邦优化
        Li et al., "Federated Optimization in Heterogeneous Networks"
        """
        # 首先执行标准FedAvg
        aggregated_weights = self.federated_averaging(client_updates)
        
        # 如果有全局模型，添加近端正则化
        if self.global_model is not None:
            for param_name in aggregated_weights:
                if param_name in self.global_model:
                    # 添加近端项：拉向前一轮的全局模型
                    aggregated_weights[param_name] = (
                        (1 - mu) * aggregated_weights[param_name] +
                        mu * self.global_model[param_name]
                    )
                    
        return aggregated_weights
    
    def scaffold_aggregation(self, client_updates: List[ClientUpdate]) -> Dict[str, np.ndarray]:
        """
        SCAFFOLD: 带控制变量的随机客户端漂移修正
        Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning"
        """
        if not hasattr(self, 'control_variates'):
            self.control_variates = defaultdict(lambda: None)
            self.server_control = None
            
        aggregated_weights = {}
        total_samples = sum(update.num_samples for update in client_updates)
        
        for update in client_updates:
            weight = update.num_samples / total_samples
            device_id = update.device_id
            
            # 获取或初始化控制变量
            if self.control_variates[device_id] is None:
                self.control_variates[device_id] = {
                    k: np.zeros_like(v) for k, v in update.model_weights.items()
                }
                
            for param_name, param_value in update.model_weights.items():
                # 应用控制变量修正
                if self.server_control and param_name in self.server_control:
                    correction = self.control_variates[device_id][param_name] - self.server_control[param_name]
                    corrected_update = param_value - correction
                else:
                    corrected_update = param_value
                    
                if param_name not in aggregated_weights:
                    aggregated_weights[param_name] = np.zeros_like(param_value)
                aggregated_weights[param_name] += weight * corrected_update
                
        # 更新服务器控制变量
        if self.global_model:
            self.server_control = {
                k: aggregated_weights[k] - self.global_model[k]
                for k in aggregated_weights
            }
            
        self.global_model = aggregated_weights
        return aggregated_weights
    
    def fednova_aggregation(self, client_updates: List[ClientUpdate]) -> Dict[str, np.ndarray]:
        """
        FedNova: 归一化平均的联邦学习
        Wang et al., "Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization"
        """
        if not client_updates:
            return self.global_model if self.global_model else {}
            
        aggregated_weights = {}
        total_normalized_grads = {}
        
        for update in client_updates:
            # 假设客户端提供了本地步数信息
            local_steps = update.metrics.get('local_steps', 1)
            
            for param_name, param_value in update.model_weights.items():
                if self.global_model and param_name in self.global_model:
                    # 计算归一化梯度
                    grad = (self.global_model[param_name] - param_value) / local_steps
                else:
                    grad = param_value
                    
                if param_name not in total_normalized_grads:
                    total_normalized_grads[param_name] = np.zeros_like(grad)
                total_normalized_grads[param_name] += grad
                
        # 计算平均归一化梯度并更新模型
        num_clients = len(client_updates)
        for param_name in total_normalized_grads:
            avg_grad = total_normalized_grads[param_name] / num_clients
            if self.global_model and param_name in self.global_model:
                aggregated_weights[param_name] = self.global_model[param_name] - avg_grad
            else:
                aggregated_weights[param_name] = -avg_grad
                
        self.global_model = aggregated_weights
        return aggregated_weights
    
    def weighted_aggregation(self, client_updates: List[ClientUpdate],
                           weights: Optional[List[float]] = None) -> Dict[str, np.ndarray]:
        """自定义权重聚合"""
        if not client_updates:
            return {}
            
        if weights is None:
            # 默认使用样本数作为权重
            total_samples = sum(update.num_samples for update in client_updates)
            weights = [update.num_samples / total_samples for update in client_updates]
            
        aggregated_weights = {}
        
        for update, weight in zip(client_updates, weights):
            for param_name, param_value in update.model_weights.items():
                if param_name not in aggregated_weights:
                    aggregated_weights[param_name] = np.zeros_like(param_value)
                aggregated_weights[param_name] += weight * param_value
                
        return aggregated_weights


class PersonalizedFederatedAggregator(FederatedAggregator):
    """个性化联邦学习聚合器"""
    
    def __init__(self, personalization_method: str = "fedper"):
        super().__init__()
        self.personalization_method = personalization_method
        self.personalized_layers = ['personalization_layer']
        
    def aggregate_with_personalization(self, client_updates: List[ClientUpdate]) -> Dict:
        """带个性化的聚合"""
        # 分离共享层和个性化层
        shared_updates = []
        personalized_weights = {}
        
        for update in client_updates:
            shared_weights = {}
            personal_weights = {}
            
            for param_name, param_value in update.model_weights.items():
                if any(layer in param_name for layer in self.personalized_layers):
                    personal_weights[param_name] = param_value
                else:
                    shared_weights[param_name] = param_value
                    
            # 创建只包含共享层的更新
            shared_update = ClientUpdate(
                device_id=update.device_id,
                model_weights=shared_weights,
                num_samples=update.num_samples,
                metrics=update.metrics,
                timestamp=update.timestamp
            )
            shared_updates.append(shared_update)
            
            # 保存个性化层权重
            personalized_weights[update.device_id] = personal_weights
            
        # 聚合共享层
        aggregated_shared = self.aggregate(shared_updates)
        
        return {
            'shared_weights': aggregated_shared,
            'personalized_weights': personalized_weights
        }
        
    def ditto_aggregation(self, client_updates: List[ClientUpdate],
                         lambda_reg: float = 0.1) -> Dict:
        """
        Ditto: 公平异构联邦学习
        Li et al., "Ditto: Fair and Robust Federated Learning Through Personalization"
        """
        # 全局模型聚合
        global_weights = self.federated_averaging(client_updates)
        
        # 每个客户端维护个性化模型
        personalized_models = {}
        
        for update in client_updates:
            # 个性化模型 = (1-λ) * 本地模型 + λ * 全局模型
            personalized_weights = {}
            
            for param_name, param_value in update.model_weights.items():
                if param_name in global_weights:
                    personalized_weights[param_name] = (
                        (1 - lambda_reg) * param_value +
                        lambda_reg * global_weights[param_name]
                    )
                else:
                    personalized_weights[param_name] = param_value
                    
            personalized_models[update.device_id] = personalized_weights
            
        return {
            'global_model': global_weights,
            'personalized_models': personalized_models
        }


class ClusteredFederatedAggregator(FederatedAggregator):
    """基于聚类的联邦聚合器"""
    
    def __init__(self, num_clusters: int = 3):
        super().__init__()
        self.num_clusters = num_clusters
        self.cluster_models = {}
        self.device_clusters = {}
        
    def cluster_clients(self, client_updates: List[ClientUpdate]) -> Dict[int, List[ClientUpdate]]:
        """基于模型相似度聚类客户端"""
        if len(client_updates) <= self.num_clusters:
            # 客户端数量少于聚类数，每个客户端一个簇
            return {i: [update] for i, update in enumerate(client_updates)}
            
        # 计算模型相似度矩阵
        similarity_matrix = self._compute_similarity_matrix(client_updates)
        
        # 简化的聚类（实际应用中可以使用K-means等）
        clusters = defaultdict(list)
        for i, update in enumerate(client_updates):
            cluster_id = i % self.num_clusters
            clusters[cluster_id].append(update)
            self.device_clusters[update.device_id] = cluster_id
            
        return clusters
    
    def _compute_similarity_matrix(self, client_updates: List[ClientUpdate]) -> np.ndarray:
        """计算客户端模型之间的相似度"""
        n = len(client_updates)
        similarity = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                # 计算余弦相似度
                sim = self._cosine_similarity(
                    client_updates[i].model_weights,
                    client_updates[j].model_weights
                )
                similarity[i, j] = similarity[j, i] = sim
                
        return similarity
    
    def _cosine_similarity(self, weights1: Dict, weights2: Dict) -> float:
        """计算两个模型权重的余弦相似度"""
        dot_product = 0
        norm1 = 0
        norm2 = 0
        
        for param_name in weights1:
            if param_name in weights2:
                w1 = weights1[param_name].flatten()
                w2 = weights2[param_name].flatten()
                dot_product += np.dot(w1, w2)
                norm1 += np.linalg.norm(w1) ** 2
                norm2 += np.linalg.norm(w2) ** 2
                
        if norm1 == 0 or norm2 == 0:
            return 0
            
        return dot_product / (np.sqrt(norm1) * np.sqrt(norm2))
    
    def hierarchical_aggregation(self, client_updates: List[ClientUpdate]) -> Dict:
        """分层聚合：先聚类内聚合，再跨聚类聚合"""
        # 聚类客户端
        clusters = self.cluster_clients(client_updates)
        
        # 每个聚类内部聚合
        cluster_models = {}
        for cluster_id, cluster_updates in clusters.items():
            cluster_model = self.federated_averaging(cluster_updates)
            cluster_models[cluster_id] = cluster_model
            self.cluster_models[cluster_id] = cluster_model
            
        # 跨聚类聚合得到全局模型
        global_updates = []
        for cluster_id, model in cluster_models.items():
            # 计算聚类大小作为权重
            cluster_size = len(clusters[cluster_id])
            total_samples = sum(u.num_samples for u in clusters[cluster_id])
            
            global_updates.append(ClientUpdate(
                device_id=f"cluster_{cluster_id}",
                model_weights=model,
                num_samples=total_samples,
                metrics={},
                timestamp=0
            ))
            
        global_model = self.federated_averaging(global_updates)
        
        return {
            'global_model': global_model,
            'cluster_models': cluster_models,
            'device_clusters': self.device_clusters
        }


if __name__ == "__main__":
    # 测试聚合器
    print("Testing Federated Aggregators...")
    
    # 创建模拟客户端更新
    def create_mock_update(device_id: str, base_value: float) -> ClientUpdate:
        weights = {
            'layer1.weight': np.random.randn(10, 10) * 0.1 + base_value,
            'layer2.weight': np.random.randn(5, 10) * 0.1 + base_value,
            'output.weight': np.random.randn(1, 5) * 0.1 + base_value
        }
        return ClientUpdate(
            device_id=device_id,
            model_weights=weights,
            num_samples=np.random.randint(100, 1000),
            metrics={'loss': np.random.random()},
            timestamp=time.time()
        )
    
    import time
    
    # 创建多个客户端更新
    updates = [
        create_mock_update(f"device_{i}", i * 0.1) 
        for i in range(5)
    ]
    
    # 测试FedAvg
    print("\n1. Testing FedAvg:")
    aggregator = FederatedAggregator("fedavg")
    result = aggregator.aggregate(updates)
    print(f"   Aggregated {len(result)} parameters")
    
    # 测试FedProx
    print("\n2. Testing FedProx:")
    aggregator = FederatedAggregator("fedprox")
    result = aggregator.aggregate(updates)
    print(f"   Aggregated {len(result)} parameters")
    
    # 测试个性化联邦学习
    print("\n3. Testing Personalized FL:")
    pers_aggregator = PersonalizedFederatedAggregator()
    result = pers_aggregator.aggregate_with_personalization(updates)
    print(f"   Shared parameters: {len(result['shared_weights'])}")
    print(f"   Personalized models: {len(result['personalized_weights'])}")
    
    # 测试聚类联邦学习
    print("\n4. Testing Clustered FL:")
    cluster_aggregator = ClusteredFederatedAggregator(num_clusters=2)
    result = cluster_aggregator.hierarchical_aggregation(updates)
    print(f"   Global model parameters: {len(result['global_model'])}")
    print(f"   Number of clusters: {len(result['cluster_models'])}")
    print(f"   Device clustering: {result['device_clusters']}")