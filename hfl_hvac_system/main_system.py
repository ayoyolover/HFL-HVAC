"""
分层联邦学习HVAC控制系统主程序
"""
import asyncio
import logging
import json
import time
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import torch
from dataclasses import dataclass, asdict
from pathlib import Path

# 导入各层组件
from simulator.device_simulator import BuildingSimulator
from device_tier.data_collector.sensor_interface import DataCollector, SimulatedSensor
from device_tier.data_collector.data_preprocessor import DataPreprocessor
from device_tier.data_collector.privacy_guard import DifferentialPrivacy, PrivacyConfig
from device_tier.local_trainer.model_manager import ModelManager, HVACModel
from device_tier.local_trainer.trainer import FederatedLocalTrainer
from edge_tier.aggregation.federated_aggregator import FederatedAggregator, ClientUpdate
from cloud_tier.drl_agent.environment import HVACEnvironment
from cloud_tier.drl_agent.sac_agent import HVACController

# 尝试导入可视化模块（可选）
try:
    from visualization.dashboard import HVACDashboard, SystemArchitecturePlot, PerformanceVisualizer
    VISUALIZATION_ENABLED = True
except ImportError:
    VISUALIZATION_ENABLED = False
    logger = logging.getLogger(__name__)
    logger.warning("Visualization module not available. Running without dashboard.")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SystemConfig:
    """系统配置"""
    num_devices: int = 10
    num_edge_servers: int = 3
    simulation_hours: int = 24
    fl_rounds: int = 100
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.001
    privacy_epsilon: float = 1.0
    aggregation_method: str = "fedavg"
    drl_update_interval: int = 10  # DRL更新间隔（FL轮次）


class DeviceTier:
    """设备层管理器"""
    
    def __init__(self, device_id: str, zone_id: str, config: SystemConfig):
        self.device_id = device_id
        self.zone_id = zone_id
        self.config = config
        
        # 数据收集器
        self.data_collector = DataCollector(device_id)
        
        # 数据预处理器
        self.preprocessor = DataPreprocessor()
        
        # 隐私保护
        privacy_config = PrivacyConfig(epsilon=config.privacy_epsilon)
        self.privacy_guard = DifferentialPrivacy(privacy_config)
        
        # 模型管理器
        self.model_manager = ModelManager(device_id)
        self.model = self.model_manager.create_model()
        
        # 本地训练器
        self.trainer = FederatedLocalTrainer(
            device_id, 
            self.model,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            enable_privacy=True
        )
        
        # 数据缓冲
        self.data_buffer = []
        
    async def collect_data(self, simulator):
        """收集数据"""
        sensor = SimulatedSensor(self.device_id, simulator)
        self.data_collector.add_sensor(sensor)
        
        # 收集一次数据
        data = await self.data_collector.collect_once()
        
        # 预处理
        if data and 'sensors' in data:
            processed = self.preprocessor.preprocess_single(data)
            self.data_buffer.append((processed, data))
            
        return data
    
    def train_local_model(self) -> ClientUpdate:
        """训练本地模型"""
        if len(self.data_buffer) < 10:
            logger.warning(f"Device {self.device_id}: Insufficient data for training")
            return None
            
        # 准备训练数据
        features = np.array([d[0] for d in self.data_buffer[-100:]])
        
        # 生成标签（这里使用模拟的控制目标）
        labels = np.array([
            [
                d[1].get('temperature', 22) - 22,  # 温度偏差
                d[1].get('humidity', 50) - 50,      # 湿度偏差
                d[1].get('ventilation_rate', 0.5),  # 通风率
                d[1].get('power_consumption', 5)    # 功率
            ]
            for d in self.data_buffer[-100:]
        ])
        
        # 本地训练
        result = self.trainer.train_federated_round(
            train_data=(features, labels),
            local_epochs=self.config.local_epochs
        )
        
        # 创建客户端更新
        return ClientUpdate(
            device_id=self.device_id,
            model_weights=result['model_update'],
            num_samples=result['num_samples'],
            metrics={'loss': result['final_loss']},
            timestamp=time.time()
        )
        
    def apply_global_model(self, global_weights: Dict):
        """应用全局模型"""
        self.trainer.set_global_model(global_weights)
        logger.info(f"Device {self.device_id}: Applied global model")


class EdgeTier:
    """边缘层管理器"""
    
    def __init__(self, server_id: str, config: SystemConfig):
        self.server_id = server_id
        self.config = config
        
        # 联邦聚合器
        self.aggregator = FederatedAggregator(config.aggregation_method)
        
        # 设备管理
        self.connected_devices: List[str] = []
        self.device_updates: List[ClientUpdate] = []
        
    def register_device(self, device_id: str):
        """注册设备"""
        if device_id not in self.connected_devices:
            self.connected_devices.append(device_id)
            logger.info(f"Edge {self.server_id}: Registered device {device_id}")
            
    def receive_update(self, update: ClientUpdate):
        """接收设备更新"""
        if update:
            self.device_updates.append(update)
            
    def aggregate_updates(self) -> Dict:
        """聚合设备更新"""
        if not self.device_updates:
            return None
            
        # 执行聚合
        aggregated = self.aggregator.aggregate(self.device_updates)
        
        # 清空更新列表
        self.device_updates = []
        
        logger.info(f"Edge {self.server_id}: Aggregated {len(self.connected_devices)} device updates")
        
        return aggregated


class CloudTier:
    """云层管理器"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        
        # 全局聚合器
        self.global_aggregator = FederatedAggregator("fedavg")
        
        # DRL控制器
        self.drl_controller = HVACController(num_zones=config.num_devices)
        
        # 全局模型
        self.global_model = None
        
        # 性能监控
        self.performance_history = []
        
    def aggregate_edge_models(self, edge_models: List[Dict]) -> Dict:
        """聚合边缘模型"""
        if not edge_models:
            return self.global_model
            
        # 创建边缘更新
        edge_updates = []
        for i, model in enumerate(edge_models):
            if model:
                update = ClientUpdate(
                    device_id=f"edge_{i}",
                    model_weights=model,
                    num_samples=1000,  # 假设每个边缘有1000个样本
                    metrics={},
                    timestamp=time.time()
                )
                edge_updates.append(update)
                
        # 全局聚合
        self.global_model = self.global_aggregator.aggregate(edge_updates)
        
        logger.info(f"Cloud: Aggregated {len(edge_updates)} edge models")
        
        return self.global_model
    
    def get_drl_control(self, building_state: Dict) -> Dict:
        """获取DRL控制指令"""
        return self.drl_controller.get_control_action(building_state)
    
    def update_drl_agent(self, experiences: List):
        """更新DRL智能体"""
        for exp in experiences:
            self.drl_controller.update_from_feedback(*exp)
            
    def evaluate_performance(self, metrics: Dict):
        """评估系统性能"""
        self.performance_history.append({
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        })
        
        # 计算平均性能
        if len(self.performance_history) > 10:
            recent = self.performance_history[-10:]
            
            # 正确提取能耗数据 (kW -> kWh转换，5分钟间隔)
            avg_power_kw = np.mean([m['metrics'].get('total_power_kw', 0) for m in recent])
            avg_energy_kwh = avg_power_kw * 5 / 60  # 5分钟的能耗
            
            # 计算舒适度得分 (基于违规率)
            total_zones = 10  # 假设10个区域
            avg_violations = np.mean([m['metrics'].get('comfort_violations', 0) for m in recent])
            comfort_score = max(0, 1 - avg_violations / (total_zones * 3))  # 3个舒适度指标
            
            logger.info(f"Cloud: Avg Power={avg_power_kw:.2f}kW, Avg Energy={avg_energy_kwh:.2f}kWh per 5min, Comfort Score={comfort_score:.2%}")


class HFLHVACSystem:
    """分层联邦学习HVAC控制系统"""
    
    def __init__(self, config: SystemConfig, enable_visualization: bool = True):
        self.config = config
        self.enable_visualization = enable_visualization and VISUALIZATION_ENABLED
        
        # 初始化仿真器
        self.simulator = BuildingSimulator(num_zones=config.num_devices)
        
        # 初始化可视化仪表板
        self.dashboard = None
        if self.enable_visualization:
            try:
                self.dashboard = HVACDashboard(num_zones=config.num_devices)
                logger.info("Visualization dashboard initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize dashboard: {e}")
                self.dashboard = None
        
        # 初始化设备层
        self.devices = []
        for i in range(config.num_devices):
            device = DeviceTier(
                device_id=f"device_{i:03d}",
                zone_id=f"zone_{i:03d}",
                config=config
            )
            self.devices.append(device)
            
        # 初始化边缘层
        self.edge_servers = []
        for i in range(config.num_edge_servers):
            edge = EdgeTier(f"edge_{i:03d}", config)
            self.edge_servers.append(edge)
            
        # 分配设备到边缘服务器
        for i, device in enumerate(self.devices):
            edge_idx = i % config.num_edge_servers
            self.edge_servers[edge_idx].register_device(device.device_id)
            
        # 初始化云层
        self.cloud = CloudTier(config)
        
        # 系统状态
        self.is_running = False
        self.fl_round = 0
        
    async def run_simulation_step(self):
        """运行一个仿真步骤"""
        # 获取建筑状态
        sensor_data = self.simulator.step()
        
        # 设备层数据收集
        for i, device in enumerate(self.devices):
            if i < len(self.simulator.devices):
                await device.collect_data(self.simulator.devices[i])
                
        return sensor_data
    
    def run_fl_round(self):
        """运行一轮联邦学习"""
        logger.info(f"\n{'='*50}")
        logger.info(f"Starting FL Round {self.fl_round + 1}")
        
        # 1. 设备本地训练
        device_updates = []
        for device in self.devices:
            update = device.train_local_model()
            if update:
                device_updates.append((device.device_id, update))
                
        # 2. 边缘聚合
        edge_models = []
        for edge in self.edge_servers:
            # 收集属于该边缘的设备更新
            for device_id, update in device_updates:
                if device_id in edge.connected_devices:
                    edge.receive_update(update)
                    
            # 边缘聚合
            edge_model = edge.aggregate_updates()
            if edge_model:
                edge_models.append(edge_model)
                
        # 3. 云端全局聚合
        global_model = self.cloud.aggregate_edge_models(edge_models)
        
        # 4. 模型下发
        if global_model:
            # 将全局模型转换为torch tensors
            global_state = {}
            for name, weights in global_model.items():
                global_state[name] = torch.FloatTensor(weights)
                
            # 下发到设备
            for device in self.devices:
                device.apply_global_model(global_state)
                
        self.fl_round += 1
        
        logger.info(f"FL Round {self.fl_round} completed")
        
    def run_drl_control(self):
        """运行DRL控制"""
        # 获取当前建筑状态
        weather_obj = self.simulator.weather_sim.get_weather(
            self.simulator.current_time.hour,
            self.simulator.current_time.timetuple().tm_yday
        )
        
        # 将WeatherCondition对象转换为字典
        building_state = {
            'zones': [],
            'weather': {
                'outdoor_temp': weather_obj.outdoor_temp,
                'outdoor_humidity': weather_obj.humidity,
                'solar_radiation': weather_obj.solar_radiation
            },
            'energy_price': 0.1,
            'hour': self.simulator.current_time.hour
        }
        
        for device in self.simulator.devices:
            zone_state = {
                'temperature': device.current_temp,
                'humidity': device.current_humidity,
                'co2': device.current_co2,
                'occupancy': device.current_occupancy,
                'power': device.power_consumption
            }
            building_state['zones'].append(zone_state)
            
        # 获取DRL控制指令
        control_commands = self.cloud.get_drl_control(building_state)
        
        # 应用控制指令
        for i, zone_cmd in enumerate(control_commands.get('zones', [])):
            if i < len(self.simulator.devices):
                device = self.simulator.devices[i]
                device.temp_setpoint += zone_cmd['temperature_adjustment']
                device.ventilation_rate += zone_cmd['ventilation_adjustment']
                
        logger.info(f"DRL control applied to {len(control_commands.get('zones', []))} zones")
        
    async def run(self):
        """运行系统"""
        self.is_running = True
        logger.info("Starting HFL-HVAC System...")
        
        simulation_steps = self.config.simulation_hours * 12  # 5分钟步长
        
        for step in range(simulation_steps):
            # 运行仿真步骤
            await self.run_simulation_step()
            
            # 每小时运行一次FL
            if step % 12 == 0 and step > 0:
                self.run_fl_round()
                
            # 每10轮FL更新一次DRL
            if self.fl_round % self.config.drl_update_interval == 0 and self.fl_round > 0:
                self.run_drl_control()
                
            # 评估性能
            if step % 12 == 0:
                building_status = self.simulator.get_building_status()
                self.cloud.evaluate_performance(building_status)
                
                logger.info(f"Step {step}/{simulation_steps}: "
                          f"Power={building_status['total_power_kw']:.1f}kW, "
                          f"Temp={building_status['average_temperature']:.1f}°C")
                
                # 更新可视化仪表板
                if self.dashboard:
                    try:
                        dashboard_data = {
                            'zones': [{'temperature': d.current_temp,
                                      'humidity': d.current_humidity,
                                      'co2': d.current_co2,
                                      'occupancy': d.current_occupancy,
                                      'power': d.power_consumption}
                                     for d in self.simulator.devices],
                            'total_power': building_status['total_power_kw'],
                            'comfort_violations': building_status['comfort_violations'],
                            'fl_round': self.fl_round,
                            'active_devices': len(self.devices),
                            'privacy_budget': 0.1 * self.fl_round  # 简化计算
                        }
                        
                        # 添加FL损失（如果有）
                        if hasattr(self, 'last_fl_loss'):
                            dashboard_data['fl_loss'] = self.last_fl_loss
                            
                        # 添加DRL奖励（如果有）
                        if hasattr(self, 'last_drl_reward'):
                            dashboard_data['drl_reward'] = self.last_drl_reward
                            
                        self.dashboard.update(dashboard_data)
                    except Exception as e:
                        logger.warning(f"Dashboard update failed: {e}")
                
        self.is_running = False
        logger.info("System stopped")
        
        # 保存结果
        self.save_results()
        
        # 保存可视化快照
        if self.dashboard:
            try:
                self.dashboard.save_snapshot("dashboard_final.png")
                logger.info("Dashboard snapshot saved")
            except Exception as e:
                logger.warning(f"Failed to save dashboard snapshot: {e}")
        
    def save_results(self):
        """保存运行结果"""
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存性能历史
        with open(results_dir / f"performance_{timestamp}.json", 'w') as f:
            json.dump(self.cloud.performance_history, f, indent=2)
            
        # 保存最终模型
        if self.cloud.global_model:
            torch.save(
                self.cloud.global_model,
                results_dir / f"global_model_{timestamp}.pt"
            )
            
        logger.info(f"Results saved to {results_dir}")


async def main():
    """主函数"""
    # 加载配置
    config = SystemConfig(
        num_devices=5,
        num_edge_servers=2,
        simulation_hours=24,
        fl_rounds=10,
        local_epochs=5,
        privacy_epsilon=1.0
    )
    
    # 创建系统
    system = HFLHVACSystem(config)
    
    # 运行系统
    await system.run()


if __name__ == "__main__":
    asyncio.run(main())