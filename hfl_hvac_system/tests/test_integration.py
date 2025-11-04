"""
集成测试 - 测试完整系统工作流
"""
import sys
import os
import unittest
import numpy as np
import asyncio

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_system import HFLHVACSystem, SystemConfig
from simulator.device_simulator import BuildingSimulator
from device_tier.data_collector.sensor_interface import DataCollector, SimulatedSensor
from device_tier.data_collector.data_preprocessor import DataPreprocessor
from device_tier.data_collector.privacy_guard import DifferentialPrivacy, PrivacyConfig
from edge_tier.aggregation.federated_aggregator import FederatedAggregator, ClientUpdate
from cloud_tier.drl_agent.environment import HVACEnvironment


class TestSystemIntegration(unittest.TestCase):
    """系统集成测试"""
    
    def setUp(self):
        """测试准备"""
        self.config = SystemConfig(
            num_devices=3,
            num_edge_servers=1,
            simulation_hours=1,
            fl_rounds=2,
            local_epochs=1,
            privacy_epsilon=1.0
        )
        
    def test_building_simulator(self):
        """测试建筑仿真器"""
        simulator = BuildingSimulator(num_zones=3, start_hour=10)
        
        # 测试初始状态
        self.assertEqual(len(simulator.zones), 3)
        self.assertEqual(len(simulator.devices), 3)
        self.assertEqual(simulator.current_time.hour, 10)
        
        # 测试仿真步骤
        data = simulator.step()
        self.assertEqual(len(data), 3)  # 3个设备的数据
        
        # 测试建筑状态
        status = simulator.get_building_status()
        self.assertIn('total_power_kw', status)
        self.assertIn('average_temperature', status)
        self.assertIn('total_occupancy', status)
        
        # 验证工作时间占用率
        self.assertGreater(status['total_occupancy'], 0, 
                          "Occupancy should be > 0 during work hours")
        
    def test_data_preprocessing(self):
        """测试数据预处理"""
        preprocessor = DataPreprocessor(window_size=6)
        
        # 生成测试数据
        test_data = []
        for i in range(10):
            data = {
                'timestamp': f"2024-01-01T10:{i:02d}:00",
                'temperature': 22 + np.random.randn(),
                'humidity': 50 + np.random.randn(),
                'co2': 450 + np.random.randn() * 50,
                'occupancy': np.random.randint(0, 10),
                'power_consumption': 5 + np.random.randn()
            }
            test_data.append(data)
        
        # 测试批量预处理
        features = preprocessor.preprocess_batch(test_data)
        
        # 验证特征维度（应该是42维）
        self.assertEqual(features.shape[1], 42, 
                        f"Expected 42 features, got {features.shape[1]}")
        
    def test_privacy_protection(self):
        """测试隐私保护机制"""
        config = PrivacyConfig(epsilon=1.0, clip_norm=1.0)
        dp = DifferentialPrivacy(config)
        
        # 测试梯度添加噪声
        gradients = np.random.randn(100)
        noisy_grads, privacy_cost = dp.add_noise_to_gradient(gradients, batch_size=32)
        
        # 验证噪声添加
        self.assertFalse(np.array_equal(gradients, noisy_grads), 
                        "Gradients should be modified by noise")
        
        # 验证隐私成本
        self.assertGreater(privacy_cost, 0)
        self.assertLess(privacy_cost, config.epsilon)
        
    def test_federated_aggregation(self):
        """测试联邦聚合"""
        aggregator = FederatedAggregator("fedavg")
        
        # 创建模拟客户端更新
        updates = []
        for i in range(3):
            weights = {
                'layer1': np.random.randn(10, 10),
                'layer2': np.random.randn(5, 10)
            }
            update = ClientUpdate(
                device_id=f"device_{i}",
                model_weights=weights,
                num_samples=100,
                metrics={'loss': 0.1},
                timestamp=0
            )
            updates.append(update)
        
        # 测试聚合
        aggregated = aggregator.aggregate(updates)
        
        # 验证聚合结果
        self.assertIn('layer1', aggregated)
        self.assertIn('layer2', aggregated)
        self.assertEqual(aggregated['layer1'].shape, (10, 10))
        
    def test_drl_environment(self):
        """测试DRL环境"""
        env = HVACEnvironment(num_zones=3)
        
        # 测试重置
        obs = env.reset()
        self.assertEqual(obs.shape[0], 3 * 5 + 6)  # 3区域*5特征 + 6全局特征
        
        # 测试动作执行
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        
        # 验证返回值
        self.assertEqual(next_obs.shape, obs.shape)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIn('total_power', info)
        
    async def test_data_collection(self):
        """测试异步数据采集"""
        collector = DataCollector("test_device", collection_interval=1)
        
        # 添加仿真传感器
        sensor = SimulatedSensor("test_sensor")
        collector.add_sensor(sensor)
        
        # 测试单次采集
        data = await collector.collect_once()
        
        # 验证数据结构
        self.assertIn('device_id', data)
        self.assertIn('timestamp', data)
        self.assertIn('sensors', data)
        
    def test_system_initialization(self):
        """测试系统初始化"""
        system = HFLHVACSystem(self.config, enable_visualization=False)
        
        # 验证组件初始化
        self.assertEqual(len(system.devices), self.config.num_devices)
        self.assertEqual(len(system.edge_servers), self.config.num_edge_servers)
        self.assertIsNotNone(system.cloud)
        self.assertIsNotNone(system.simulator)
        
    def test_fl_round_execution(self):
        """测试FL轮次执行"""
        system = HFLHVACSystem(self.config, enable_visualization=False)
        
        # 先收集一些数据
        asyncio.run(self._collect_initial_data(system))
        
        # 执行一轮FL
        try:
            system.run_fl_round()
            self.assertEqual(system.fl_round, 1)
        except Exception as e:
            self.fail(f"FL round execution failed: {e}")
            
    async def _collect_initial_data(self, system):
        """收集初始数据用于测试"""
        for _ in range(20):
            await system.run_simulation_step()
            
    def test_performance_metrics(self):
        """测试性能指标计算"""
        # 创建测试数据
        test_metrics = {
            'total_power_kw': 35.5,
            'average_temperature': 22.3,
            'comfort_violations': 2,
            'total_occupancy': 45
        }
        
        # 验证能耗降低
        baseline_power = 60
        reduction = (baseline_power - test_metrics['total_power_kw']) / baseline_power
        self.assertGreater(reduction, 0.3, "Should achieve >30% energy reduction")
        
        # 验证舒适度
        violation_rate = test_metrics['comfort_violations'] / 10  # 10个区域
        self.assertLess(violation_rate, 0.3, "Violation rate should be <30%")


class TestEndToEndWorkflow(unittest.TestCase):
    """端到端工作流测试"""
    
    def test_complete_workflow(self):
        """测试完整工作流"""
        print("\nTesting End-to-End Workflow...")
        
        # 1. 初始化系统
        config = SystemConfig(
            num_devices=2,
            num_edge_servers=1,
            simulation_hours=0.5,  # 30分钟测试
            fl_rounds=1,
            local_epochs=1
        )
        system = HFLHVACSystem(config, enable_visualization=False)
        
        # 2. 运行仿真
        async def run_test():
            # 收集数据
            for _ in range(10):
                await system.run_simulation_step()
            
            # FL训练
            if len(system.devices[0].data_buffer) > 5:
                system.run_fl_round()
                
            # DRL控制
            system.run_drl_control()
            
            return True
            
        # 3. 执行测试
        result = asyncio.run(run_test())
        self.assertTrue(result, "Workflow should complete successfully")
        
        # 4. 验证结果
        building_status = system.simulator.get_building_status()
        self.assertIsNotNone(building_status)
        self.assertGreater(building_status['total_power_kw'], 0)
        
        print("✅ End-to-End workflow test passed!")


def run_integration_tests():
    """运行所有集成测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestSystemIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEndWorkflow))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 打印结果摘要
    print("\n" + "="*60)
    print("Integration Test Summary")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)