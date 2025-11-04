"""
传感器数据采集接口 - 支持真实设备和仿真数据
"""
import asyncio
import json
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Callable, Tuple
from datetime import datetime
import numpy as np
from collections import deque
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SensorInterface(ABC):
    """传感器接口基类"""
    
    @abstractmethod
    async def read_data(self) -> Dict:
        """读取传感器数据"""
        pass
    
    @abstractmethod
    async def calibrate(self) -> bool:
        """校准传感器"""
        pass
        

class SimulatedSensor(SensorInterface):
    """仿真传感器接口"""
    
    def __init__(self, device_id: str, simulator=None):
        self.device_id = device_id
        self.simulator = simulator
        self.last_read_time = time.time()
        
    async def read_data(self) -> Dict:
        """从仿真器读取数据"""
        if self.simulator:
            data = self.simulator.get_sensor_data()
        else:
            # 生成随机数据用于测试
            data = {
                'device_id': self.device_id,
                'timestamp': datetime.now().isoformat(),
                'temperature': np.random.uniform(18, 28),
                'humidity': np.random.uniform(30, 70),
                'co2': np.random.uniform(400, 1200),
                'occupancy': np.random.randint(0, 20),
                'power_consumption': np.random.uniform(1, 10)
            }
        
        self.last_read_time = time.time()
        return data
    
    async def calibrate(self) -> bool:
        """仿真传感器无需校准"""
        await asyncio.sleep(0.1)  # 模拟校准过程
        return True


class RealSensor(SensorInterface):
    """真实传感器接口（预留）"""
    
    def __init__(self, device_id: str, sensor_config: Dict):
        self.device_id = device_id
        self.sensor_config = sensor_config
        self.connection = None
        
    async def connect(self):
        """建立与传感器的连接"""
        # 实际实现中这里会连接到真实硬件
        pass
        
    async def read_data(self) -> Dict:
        """从真实传感器读取数据"""
        # 实际实现中这里会读取真实传感器数据
        # 现在返回模拟数据
        return {
            'device_id': self.device_id,
            'timestamp': datetime.now().isoformat(),
            'temperature': 22.5,
            'humidity': 50.0,
            'co2': 450,
            'occupancy': 5,
            'power_consumption': 3.2
        }
    
    async def calibrate(self) -> bool:
        """校准真实传感器"""
        # 实际实现中这里会执行校准程序
        return True


class DataCollector:
    """数据采集器 - 管理多个传感器"""
    
    def __init__(self, device_id: str, collection_interval: float = 300):
        """
        Args:
            device_id: 设备ID
            collection_interval: 采集间隔（秒）
        """
        self.device_id = device_id
        self.collection_interval = collection_interval
        self.sensors: List[SensorInterface] = []
        self.data_buffer = deque(maxlen=1000)  # 缓存最近1000条数据
        self.is_running = False
        self.callbacks: List[Callable] = []
        
        # 数据质量监控
        self.error_count = 0
        self.success_count = 0
        
    def add_sensor(self, sensor: SensorInterface):
        """添加传感器"""
        self.sensors.append(sensor)
        logger.info(f"Added sensor for device {self.device_id}")
        
    def register_callback(self, callback: Callable):
        """注册数据回调函数"""
        self.callbacks.append(callback)
        
    async def collect_once(self) -> Dict:
        """执行一次数据采集"""
        all_data = {
            'device_id': self.device_id,
            'timestamp': datetime.now().isoformat(),
            'sensors': []
        }
        
        for sensor in self.sensors:
            try:
                data = await sensor.read_data()
                all_data['sensors'].append(data)
                self.success_count += 1
            except Exception as e:
                logger.error(f"Error reading sensor: {e}")
                self.error_count += 1
                
        # 聚合多个传感器数据（如果有多个）
        if all_data['sensors']:
            aggregated = self._aggregate_sensor_data(all_data['sensors'])
            all_data.update(aggregated)
            
        return all_data
    
    def _aggregate_sensor_data(self, sensor_data: List[Dict]) -> Dict:
        """聚合多个传感器的数据"""
        if len(sensor_data) == 1:
            return sensor_data[0]
            
        # 对数值型数据取平均
        aggregated = {}
        numeric_fields = ['temperature', 'humidity', 'co2', 'power_consumption']
        
        for field in numeric_fields:
            values = [d.get(field) for d in sensor_data if field in d]
            if values:
                aggregated[field] = np.mean(values)
                
        # 对占用率取最大值
        occupancy_values = [d.get('occupancy', 0) for d in sensor_data]
        aggregated['occupancy'] = max(occupancy_values)
        
        return aggregated
        
    async def start_collection(self):
        """开始连续数据采集"""
        self.is_running = True
        logger.info(f"Starting data collection for device {self.device_id}")
        
        while self.is_running:
            try:
                # 采集数据
                data = await self.collect_once()
                
                # 存入缓冲区
                self.data_buffer.append(data)
                
                # 触发回调
                for callback in self.callbacks:
                    asyncio.create_task(callback(data))
                    
                # 等待下一次采集
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Collection error: {e}")
                await asyncio.sleep(10)  # 错误后短暂等待
                
    def stop_collection(self):
        """停止数据采集"""
        self.is_running = False
        logger.info(f"Stopped data collection for device {self.device_id}")
        
    def get_recent_data(self, n: int = 10) -> List[Dict]:
        """获取最近n条数据"""
        return list(self.data_buffer)[-n:]
        
    def get_statistics(self) -> Dict:
        """获取采集统计信息"""
        if not self.data_buffer:
            return {}
            
        recent_data = list(self.data_buffer)
        temps = [d.get('temperature', 0) for d in recent_data]
        
        return {
            'device_id': self.device_id,
            'total_samples': len(self.data_buffer),
            'success_rate': self.success_count / (self.success_count + self.error_count + 1e-6),
            'avg_temperature': np.mean(temps),
            'std_temperature': np.std(temps),
            'min_temperature': np.min(temps),
            'max_temperature': np.max(temps)
        }
        
    async def calibrate_all_sensors(self):
        """校准所有传感器"""
        logger.info(f"Calibrating sensors for device {self.device_id}")
        results = []
        
        for sensor in self.sensors:
            try:
                result = await sensor.calibrate()
                results.append(result)
            except Exception as e:
                logger.error(f"Calibration error: {e}")
                results.append(False)
                
        return all(results)


class DataQualityChecker:
    """数据质量检查器"""
    
    def __init__(self):
        self.thresholds = {
            'temperature': (10, 40),  # °C
            'humidity': (10, 90),      # %
            'co2': (300, 5000),        # ppm
            'power_consumption': (0, 100)  # kW
        }
        
    def check_data(self, data: Dict) -> Tuple[bool, List[str]]:
        """
        检查数据质量
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        # 检查必要字段
        required_fields = ['temperature', 'humidity', 'timestamp']
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
                
        # 检查数值范围
        for field, (min_val, max_val) in self.thresholds.items():
            if field in data:
                value = data[field]
                if not min_val <= value <= max_val:
                    errors.append(f"{field} out of range: {value}")
                    
        # 检查时间戳
        if 'timestamp' in data:
            try:
                timestamp = datetime.fromisoformat(data['timestamp'])
                # 检查时间戳是否太旧（超过1小时）
                age = (datetime.now() - timestamp).total_seconds()
                if abs(age) > 3600:
                    errors.append(f"Timestamp too old: {age} seconds")
            except:
                errors.append("Invalid timestamp format")
                
        return len(errors) == 0, errors
        
    def clean_data(self, data: Dict) -> Dict:
        """清理和修正数据"""
        cleaned = data.copy()
        
        # 限制数值范围
        for field, (min_val, max_val) in self.thresholds.items():
            if field in cleaned:
                cleaned[field] = np.clip(cleaned[field], min_val, max_val)
                
        # 填充缺失值
        defaults = {
            'humidity': 50.0,
            'co2': 450.0,
            'occupancy': 0
        }
        
        for field, default_value in defaults.items():
            if field not in cleaned:
                cleaned[field] = default_value
                
        return cleaned


if __name__ == "__main__":
    async def test_collector():
        """测试数据采集器"""
        # 创建采集器
        collector = DataCollector("device_001", collection_interval=5)
        
        # 添加仿真传感器
        sim_sensor = SimulatedSensor("sensor_001")
        collector.add_sensor(sim_sensor)
        
        # 定义数据回调
        async def data_callback(data):
            print(f"New data: T={data.get('temperature', 0):.1f}°C, "
                  f"H={data.get('humidity', 0):.1f}%")
        
        collector.register_callback(data_callback)
        
        # 运行采集
        collection_task = asyncio.create_task(collector.start_collection())
        
        # 运行30秒
        await asyncio.sleep(30)
        
        # 停止采集
        collector.stop_collection()
        await collection_task
        
        # 显示统计
        stats = collector.get_statistics()
        print(f"\nCollection statistics: {stats}")
    
    # 运行测试
    asyncio.run(test_collector())