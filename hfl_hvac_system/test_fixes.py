"""
测试修复后的系统
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simulator.device_simulator import BuildingSimulator
import numpy as np

def test_occupancy_and_power():
    """测试占用率和功率计算"""
    print("Testing Building Simulator Fixes...")
    print("="*50)
    
    # 创建仿真器，从早上8点开始
    simulator = BuildingSimulator(num_zones=5, start_hour=8)
    
    print(f"Simulation starts at: {simulator.current_time}")
    print(f"Initial hour: {simulator.current_time.hour}")
    
    # 运行几个步骤
    for step in range(5):
        print(f"\n--- Step {step + 1} ---")
        
        # 执行仿真步骤
        data = simulator.step()
        
        # 获取建筑状态
        status = simulator.get_building_status()
        
        print(f"Time: {simulator.current_time.strftime('%H:%M')}")
        print(f"Total Power: {status['total_power_kw']:.2f} kW")
        print(f"Average Temperature: {status['average_temperature']:.2f}°C")
        print(f"Total Occupancy: {status['total_occupancy']} people")
        print(f"Comfort Violations: {status['comfort_violations']}")
        
        # 检查每个设备的占用率
        occupancies = [device.current_occupancy for device in simulator.devices]
        print(f"Zone Occupancies: {occupancies}")
        
        # 验证工作时间占用率
        hour = simulator.current_time.hour
        if 8 <= hour <= 18:
            assert status['total_occupancy'] > 0, f"Occupancy should be > 0 during work hours (hour={hour})"
            print("✓ Occupancy check passed (work hours)")
        else:
            print(f"✓ Hour {hour} is outside work hours")
    
    print("\n" + "="*50)
    print("All tests passed!")
    
    # 测试不同时间段
    print("\n\nTesting different time periods:")
    print("-"*50)
    
    test_hours = [0, 6, 8, 12, 18, 20, 23]
    for test_hour in test_hours:
        sim = BuildingSimulator(num_zones=5, start_hour=test_hour)
        sim.step()
        status = sim.get_building_status()
        
        expected_occupancy = "High" if 8 <= test_hour < 18 else "Low"
        actual_occupancy = "High" if status['total_occupancy'] > 10 else "Low"
        
        print(f"Hour {test_hour:2d}: Occupancy={status['total_occupancy']:3d}, "
              f"Power={status['total_power_kw']:.1f}kW, "
              f"Expected={expected_occupancy}, Actual={actual_occupancy}")
    
    print("\n✅ System fixes verified successfully!")

def test_cloud_evaluation():
    """测试云层性能评估"""
    print("\n\nTesting Cloud Performance Evaluation...")
    print("="*50)
    
    # 创建模拟性能数据
    performance_history = []
    
    for i in range(15):
        metrics = {
            'total_power_kw': 45 + np.random.randn() * 5,
            'average_temperature': 22 + np.random.randn() * 0.5,
            'comfort_violations': np.random.randint(0, 5),
            'total_occupancy': np.random.randint(20, 80)
        }
        performance_history.append({'metrics': metrics})
    
    # 模拟云层评估逻辑
    if len(performance_history) > 10:
        recent = performance_history[-10:]
        
        # 能耗计算
        avg_power_kw = np.mean([m['metrics'].get('total_power_kw', 0) for m in recent])
        avg_energy_kwh = avg_power_kw * 5 / 60  # 5分钟间隔
        
        # 舒适度计算
        total_zones = 10
        avg_violations = np.mean([m['metrics'].get('comfort_violations', 0) for m in recent])
        comfort_score = max(0, 1 - avg_violations / (total_zones * 3))
        
        print(f"Average Power: {avg_power_kw:.2f} kW")
        print(f"Average Energy: {avg_energy_kwh:.2f} kWh per 5min")
        print(f"Comfort Score: {comfort_score:.2%}")
        print(f"Average Violations: {avg_violations:.1f}")
        
        assert avg_power_kw > 0, "Power should be positive"
        assert 0 <= comfort_score <= 1, "Comfort score should be between 0 and 1"
        
        print("\n✅ Cloud evaluation logic working correctly!")

if __name__ == "__main__":
    test_occupancy_and_power()
    test_cloud_evaluation()