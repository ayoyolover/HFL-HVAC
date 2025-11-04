"""
数据预处理和特征工程模块
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collections import deque
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self, window_size: int = 12, feature_config: Optional[Dict] = None):
        """
        Args:
            window_size: 时间窗口大小（用于时序特征）
            feature_config: 特征配置
        """
        self.window_size = window_size
        self.feature_config = feature_config or self._default_feature_config()
        
        # 数据缓冲区
        self.data_buffer = deque(maxlen=window_size * 2)
        
        # 归一化器
        self.scalers = {
            'temperature': StandardScaler(),
            'humidity': StandardScaler(),
            'co2': StandardScaler(),
            'power': MinMaxScaler()
        }
        
        # 特征统计
        self.feature_stats = {}
        
    def _default_feature_config(self) -> Dict:
        """默认特征配置"""
        return {
            'raw_features': ['temperature', 'humidity', 'co2', 'occupancy', 'power_consumption'],
            'time_features': ['hour', 'day_of_week', 'is_weekend'],
            'statistical_features': ['mean', 'std', 'min', 'max', 'change_rate'],
            'lag_features': [1, 3, 6, 12],  # 滞后步数
            'interaction_features': True
        }
        
    def preprocess_batch(self, data_batch: List[Dict]) -> np.ndarray:
        """批量预处理数据"""
        # 转换为DataFrame
        df = pd.DataFrame(data_batch)
        
        # 提取原始特征
        raw_features = self._extract_raw_features(df)
        
        # 提取时间特征
        time_features = self._extract_time_features(df)
        
        # 提取统计特征
        stat_features = self._extract_statistical_features(df)
        
        # 提取滞后特征
        lag_features = self._extract_lag_features(df)
        
        # 组合所有特征
        all_features = np.hstack([
            raw_features,
            time_features,
            stat_features,
            lag_features
        ])
        
        # 添加交互特征
        if self.feature_config.get('interaction_features'):
            interaction_features = self._extract_interaction_features(raw_features)
            all_features = np.hstack([all_features, interaction_features])
            
        return all_features
    
    def preprocess_single(self, data: Dict) -> np.ndarray:
        """预处理单条数据"""
        # 添加到缓冲区
        self.data_buffer.append(data)
        
        # 如果缓冲区不足，返回零向量
        if len(self.data_buffer) < self.window_size:
            return np.zeros(self._get_feature_dim())
            
        # 使用窗口数据进行预处理
        window_data = list(self.data_buffer)[-self.window_size:]
        return self.preprocess_batch(window_data)[-1]  # 返回最后一个样本的特征
    
    def _extract_raw_features(self, df: pd.DataFrame) -> np.ndarray:
        """提取原始特征"""
        features = []
        
        for col in self.feature_config['raw_features']:
            if col in df.columns:
                values = df[col].values
                # 处理缺失值
                values = np.nan_to_num(values, nan=0)
                # 归一化
                if col in self.scalers:
                    try:
                        values = self.scalers[col].fit_transform(values.reshape(-1, 1)).flatten()
                    except:
                        pass
                features.append(values)
            else:
                features.append(np.zeros(len(df)))
                
        return np.column_stack(features)
    
    def _extract_time_features(self, df: pd.DataFrame) -> np.ndarray:
        """提取时间特征"""
        features = []
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 小时（循环编码）
            hour = df['timestamp'].dt.hour
            features.append(np.sin(2 * np.pi * hour / 24))
            features.append(np.cos(2 * np.pi * hour / 24))
            
            # 星期几（循环编码）
            day_of_week = df['timestamp'].dt.dayofweek
            features.append(np.sin(2 * np.pi * day_of_week / 7))
            features.append(np.cos(2 * np.pi * day_of_week / 7))
            
            # 是否周末
            is_weekend = (day_of_week >= 5).astype(float)
            features.append(is_weekend)
            
            # 是否工作时间
            is_working = ((hour >= 8) & (hour <= 18)).astype(float)
            features.append(is_working)
        else:
            # 如果没有时间戳，返回零特征
            features = [np.zeros(len(df)) for _ in range(6)]
            
        return np.column_stack(features)
    
    def _extract_statistical_features(self, df: pd.DataFrame) -> np.ndarray:
        """提取统计特征"""
        features = []
        
        for col in ['temperature', 'humidity', 'co2', 'power_consumption']:
            if col in df.columns:
                values = df[col].values
                values = np.nan_to_num(values, nan=0)
                
                # 滚动窗口统计
                if len(values) >= self.window_size:
                    window_features = []
                    for i in range(len(values) - self.window_size + 1):
                        window = values[i:i+self.window_size]
                        window_features.append([
                            np.mean(window),
                            np.std(window),
                            np.min(window),
                            np.max(window),
                            window[-1] - window[0]  # 变化量
                        ])
                    # 填充前面的数据
                    padding = [[0] * 5] * (self.window_size - 1)
                    window_features = padding + window_features
                    features.append(np.array(window_features))
                else:
                    features.append(np.zeros((len(df), 5)))
            else:
                features.append(np.zeros((len(df), 5)))
                
        return np.hstack(features)
    
    def _extract_lag_features(self, df: pd.DataFrame) -> np.ndarray:
        """提取滞后特征"""
        features = []
        
        for col in ['temperature', 'humidity']:
            if col in df.columns:
                values = df[col].values
                values = np.nan_to_num(values, nan=0)
                
                for lag in self.feature_config['lag_features']:
                    lagged = np.zeros_like(values)
                    if len(values) > lag:
                        lagged[lag:] = values[:-lag]
                    features.append(lagged)
            else:
                for lag in self.feature_config['lag_features']:
                    features.append(np.zeros(len(df)))
                    
        return np.column_stack(features)
    
    def _extract_interaction_features(self, raw_features: np.ndarray) -> np.ndarray:
        """提取交互特征"""
        features = []
        
        # 温度-湿度交互
        if raw_features.shape[1] >= 2:
            features.append(raw_features[:, 0] * raw_features[:, 1])
            
        # 温度-占用率交互
        if raw_features.shape[1] >= 4:
            features.append(raw_features[:, 0] * raw_features[:, 3])
            
        # 功率-占用率交互
        if raw_features.shape[1] >= 5:
            features.append(raw_features[:, 4] * raw_features[:, 3])
            
        if features:
            return np.column_stack(features)
        else:
            return np.zeros((raw_features.shape[0], 3))
    
    def _get_feature_dim(self) -> int:
        """获取特征维度"""
        # 这是一个估算，实际维度取决于配置
        dim = 0
        dim += len(self.feature_config['raw_features'])  # 原始特征
        dim += 6  # 时间特征
        dim += 4 * 5  # 统计特征（4个变量，每个5个统计量）
        dim += 2 * len(self.feature_config['lag_features'])  # 滞后特征
        if self.feature_config.get('interaction_features'):
            dim += 3  # 交互特征
        return dim
    
    def update_statistics(self, data: Dict):
        """更新特征统计信息"""
        for key in ['temperature', 'humidity', 'co2']:
            if key in data:
                if key not in self.feature_stats:
                    self.feature_stats[key] = {
                        'count': 0,
                        'sum': 0,
                        'sum_sq': 0,
                        'min': float('inf'),
                        'max': float('-inf')
                    }
                    
                stats = self.feature_stats[key]
                value = data[key]
                
                stats['count'] += 1
                stats['sum'] += value
                stats['sum_sq'] += value ** 2
                stats['min'] = min(stats['min'], value)
                stats['max'] = max(stats['max'], value)
                
                # 计算均值和标准差
                stats['mean'] = stats['sum'] / stats['count']
                stats['std'] = np.sqrt(
                    stats['sum_sq'] / stats['count'] - stats['mean'] ** 2
                )
    
    def get_feature_importance(self) -> Dict:
        """获取特征重要性（简化版）"""
        # 实际应用中，这里会使用模型的特征重要性
        importance = {
            'temperature': 0.25,
            'humidity': 0.15,
            'co2': 0.10,
            'occupancy': 0.20,
            'power_consumption': 0.15,
            'time_features': 0.10,
            'lag_features': 0.05
        }
        return importance


class FeatureEngineering:
    """高级特征工程"""
    
    def __init__(self):
        self.derived_features = {}
        
    def create_comfort_index(self, temp: float, humidity: float) -> float:
        """创建舒适度指数"""
        # PMV (Predicted Mean Vote) 简化计算
        temp_comfort = 1 - abs(temp - 22) / 10
        humidity_comfort = 1 - abs(humidity - 50) / 40
        return (temp_comfort + humidity_comfort) / 2
    
    def create_energy_efficiency_ratio(self, power: float, 
                                      temp_diff: float) -> float:
        """创建能效比"""
        if temp_diff == 0:
            return 1.0
        return min(1.0, abs(temp_diff) / (power + 1e-6))
    
    def create_occupancy_pattern(self, occupancy_history: List[int]) -> str:
        """创建占用模式"""
        if not occupancy_history:
            return "empty"
            
        avg_occupancy = np.mean(occupancy_history)
        
        if avg_occupancy < 2:
            return "low"
        elif avg_occupancy < 10:
            return "medium"
        else:
            return "high"
    
    def create_anomaly_score(self, current_value: float, 
                            historical_values: List[float]) -> float:
        """创建异常分数"""
        if not historical_values:
            return 0.0
            
        mean = np.mean(historical_values)
        std = np.std(historical_values)
        
        if std == 0:
            return 0.0
            
        z_score = abs(current_value - mean) / std
        return min(1.0, z_score / 3)  # 归一化到[0, 1]
    
    def create_trend_features(self, values: List[float]) -> Dict:
        """创建趋势特征"""
        if len(values) < 2:
            return {'trend': 0, 'acceleration': 0}
            
        # 一阶差分（趋势）
        trend = np.mean(np.diff(values))
        
        # 二阶差分（加速度）
        if len(values) >= 3:
            acceleration = np.mean(np.diff(np.diff(values)))
        else:
            acceleration = 0
            
        return {
            'trend': trend,
            'acceleration': acceleration
        }


if __name__ == "__main__":
    # 测试预处理器
    preprocessor = DataPreprocessor(window_size=6)
    
    # 生成测试数据
    test_data = []
    for i in range(20):
        data = {
            'timestamp': pd.Timestamp.now() + pd.Timedelta(minutes=i*5),
            'temperature': 22 + np.random.normal(0, 1),
            'humidity': 50 + np.random.normal(0, 5),
            'co2': 450 + np.random.normal(0, 50),
            'occupancy': np.random.randint(0, 10),
            'power_consumption': 3 + np.random.normal(0, 0.5)
        }
        test_data.append(data)
    
    # 批量预处理
    features = preprocessor.preprocess_batch(test_data)
    print(f"Feature shape: {features.shape}")
    print(f"Sample features: {features[0][:10]}")
    
    # 测试特征工程
    fe = FeatureEngineering()
    comfort = fe.create_comfort_index(23, 55)
    print(f"Comfort index: {comfort:.2f}")
    
    trend = fe.create_trend_features([20, 21, 22, 23, 22, 21])
    print(f"Trend features: {trend}")