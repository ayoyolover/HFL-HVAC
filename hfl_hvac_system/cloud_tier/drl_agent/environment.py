"""
HVAC控制的强化学习环境
"""
import gym
from gym import spaces
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class BuildingState:
    """建筑状态"""
    zones: List[Dict]  # 各区域状态
    weather: Dict  # 天气条件
    occupancy: List[int]  # 占用情况
    energy_price: float  # 电价
    timestamp: float


class HVACEnvironment(gym.Env):
    """HVAC控制环境"""
    
    def __init__(self, num_zones: int = 10, 
                 comfort_weight: float = 0.4,
                 energy_weight: float = 0.4,
                 peak_weight: float = 0.2):
        """
        Args:
            num_zones: 区域数量
            comfort_weight: 舒适度权重
            energy_weight: 能耗权重
            peak_weight: 峰值削减权重
        """
        super(HVACEnvironment, self).__init__()
        
        self.num_zones = num_zones
        self.comfort_weight = comfort_weight
        self.energy_weight = energy_weight
        self.peak_weight = peak_weight
        
        # 定义观察空间
        # 每个区域: [温度, 湿度, CO2, 占用率, 功率]
        # 全局: [室外温度, 室外湿度, 太阳辐射, 电价, 时间编码(2)]
        obs_dim = num_zones * 5 + 6
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # 定义动作空间
        # 每个区域: [温度设定点调整, 通风率调整]
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(num_zones * 2,), dtype=np.float32
        )
        
        # 初始化状态
        self.reset()
        
        # 舒适度范围
        self.comfort_temp_range = (20, 26)
        self.comfort_humidity_range = (30, 60)
        self.comfort_co2_threshold = 1000
        
        # 能耗基准
        self.baseline_power = 50.0  # kW
        self.peak_power_limit = 100.0  # kW
        
        # 历史记录
        self.episode_history = []
        
    def reset(self) -> np.ndarray:
        """重置环境"""
        # 初始化区域状态
        self.zones = []
        for i in range(self.num_zones):
            zone = {
                'temperature': np.random.uniform(20, 25),
                'humidity': np.random.uniform(40, 60),
                'co2': np.random.uniform(400, 600),
                'occupancy': np.random.randint(0, 20),
                'power': np.random.uniform(2, 8),
                'temp_setpoint': 22.0,
                'ventilation_rate': 0.5
            }
            self.zones.append(zone)
            
        # 初始化天气
        self.weather = {
            'outdoor_temp': np.random.uniform(10, 30),
            'outdoor_humidity': np.random.uniform(30, 80),
            'solar_radiation': np.random.uniform(0, 800)
        }
        
        # 初始化时间和电价
        self.time_step = 0
        self.hour = 8  # 从早上8点开始
        self.energy_price = self._get_energy_price(self.hour)
        
        # 重置历史
        self.episode_history = []
        
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行动作"""
        # 解析动作
        temp_adjustments = action[:self.num_zones]
        vent_adjustments = action[self.num_zones:]
        
        # 应用控制
        old_power = sum(zone['power'] for zone in self.zones)
        self._apply_control(temp_adjustments, vent_adjustments)
        
        # 更新环境状态
        self._update_environment()
        
        # 计算奖励
        reward = self._calculate_reward(old_power)
        
        # 更新时间
        self.time_step += 1
        self.hour = (self.hour + 0.25) % 24  # 15分钟步长
        self.energy_price = self._get_energy_price(self.hour)
        
        # 检查是否结束
        done = self.time_step >= 96  # 24小时
        
        # 记录历史
        self.episode_history.append({
            'zones': copy.deepcopy(self.zones),
            'reward': reward,
            'action': action.copy()
        })
        
        info = self._get_info()
        
        return self._get_observation(), reward, done, info
    
    def _apply_control(self, temp_adjustments: np.ndarray, 
                      vent_adjustments: np.ndarray):
        """应用控制动作"""
        for i, zone in enumerate(self.zones):
            # 调整温度设定点 (±2°C)
            zone['temp_setpoint'] += temp_adjustments[i] * 2
            zone['temp_setpoint'] = np.clip(zone['temp_setpoint'], 18, 28)
            
            # 调整通风率 (0.2-2.0 次/小时)
            zone['ventilation_rate'] += vent_adjustments[i] * 0.5
            zone['ventilation_rate'] = np.clip(zone['ventilation_rate'], 0.2, 2.0)
            
    def _update_environment(self):
        """更新环境状态"""
        for zone in self.zones:
            # 温度变化模拟
            temp_error = zone['temp_setpoint'] - zone['temperature']
            heating_cooling = np.clip(temp_error * 2, -10, 10)  # kW
            
            # 热负荷计算
            outdoor_load = (self.weather['outdoor_temp'] - zone['temperature']) * 0.1
            solar_load = self.weather['solar_radiation'] * 0.001
            occupant_load = zone['occupancy'] * 0.1
            
            # 温度更新
            temp_change = (heating_cooling + outdoor_load + solar_load + occupant_load) * 0.01
            zone['temperature'] += temp_change
            zone['temperature'] = np.clip(zone['temperature'], 15, 35)
            
            # 湿度更新
            zone['humidity'] += np.random.normal(0, 1)
            zone['humidity'] = np.clip(zone['humidity'], 20, 80)
            
            # CO2更新
            co2_generation = zone['occupancy'] * 20
            co2_removal = zone['ventilation_rate'] * (zone['co2'] - 400)
            zone['co2'] += (co2_generation - co2_removal) * 0.01
            zone['co2'] = np.clip(zone['co2'], 400, 2000)
            
            # 功率计算
            zone['power'] = abs(heating_cooling) + zone['ventilation_rate'] * 2
            
            # 占用率更新（随机模拟）
            if 8 <= self.hour <= 18:
                zone['occupancy'] = np.random.randint(5, 20)
            else:
                zone['occupancy'] = np.random.randint(0, 5)
                
    def _calculate_reward(self, old_power: float) -> float:
        """计算奖励函数"""
        # 舒适度奖励
        comfort_reward = 0
        for zone in self.zones:
            # 温度舒适度
            if self.comfort_temp_range[0] <= zone['temperature'] <= self.comfort_temp_range[1]:
                temp_comfort = 1.0
            else:
                temp_deviation = min(
                    abs(zone['temperature'] - self.comfort_temp_range[0]),
                    abs(zone['temperature'] - self.comfort_temp_range[1])
                )
                temp_comfort = max(0, 1 - temp_deviation / 5)
                
            # 湿度舒适度
            if self.comfort_humidity_range[0] <= zone['humidity'] <= self.comfort_humidity_range[1]:
                humidity_comfort = 1.0
            else:
                humidity_deviation = min(
                    abs(zone['humidity'] - self.comfort_humidity_range[0]),
                    abs(zone['humidity'] - self.comfort_humidity_range[1])
                )
                humidity_comfort = max(0, 1 - humidity_deviation / 20)
                
            # CO2舒适度
            co2_comfort = max(0, 1 - (zone['co2'] - 400) / 1600)
            
            # 考虑占用率的加权舒适度
            zone_comfort = (temp_comfort * 0.5 + humidity_comfort * 0.3 + co2_comfort * 0.2)
            zone_comfort *= (1 + zone['occupancy'] / 20)  # 有人时舒适度更重要
            
            comfort_reward += zone_comfort
            
        comfort_reward /= self.num_zones
        
        # 能耗惩罚
        current_power = sum(zone['power'] for zone in self.zones)
        energy_penalty = current_power / self.baseline_power
        
        # 峰值惩罚
        peak_penalty = 0
        if current_power > self.peak_power_limit:
            peak_penalty = (current_power - self.peak_power_limit) / self.peak_power_limit
            
        # 电价调节
        price_factor = 1 + (self.energy_price - 0.1) * 5  # 基准电价0.1
        
        # 综合奖励
        reward = (
            self.comfort_weight * comfort_reward -
            self.energy_weight * energy_penalty * price_factor -
            self.peak_weight * peak_penalty
        )
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """获取观察"""
        obs = []
        
        # 区域状态
        for zone in self.zones:
            obs.extend([
                zone['temperature'],
                zone['humidity'],
                zone['co2'] / 1000,  # 归一化
                zone['occupancy'] / 20,  # 归一化
                zone['power'] / 10  # 归一化
            ])
            
        # 全局状态
        obs.extend([
            self.weather['outdoor_temp'],
            self.weather['outdoor_humidity'],
            self.weather['solar_radiation'] / 1000,
            self.energy_price,
            np.sin(2 * np.pi * self.hour / 24),  # 时间编码
            np.cos(2 * np.pi * self.hour / 24)
        ])
        
        return np.array(obs, dtype=np.float32)
    
    def _get_info(self) -> Dict:
        """获取额外信息"""
        total_power = sum(zone['power'] for zone in self.zones)
        avg_temp = np.mean([zone['temperature'] for zone in self.zones])
        comfort_violations = sum(
            1 for zone in self.zones
            if zone['temperature'] < self.comfort_temp_range[0] or
            zone['temperature'] > self.comfort_temp_range[1]
        )
        
        return {
            'total_power': total_power,
            'average_temperature': avg_temp,
            'comfort_violations': comfort_violations,
            'hour': self.hour,
            'energy_price': self.energy_price
        }
    
    def _get_energy_price(self, hour: float) -> float:
        """获取分时电价"""
        # 峰时: 10-15, 18-21
        if 10 <= hour < 15 or 18 <= hour < 21:
            return 0.15
        # 平时: 7-10, 15-18, 21-23
        elif 7 <= hour < 10 or 15 <= hour < 18 or 21 <= hour < 23:
            return 0.10
        # 谷时: 23-7
        else:
            return 0.05
            
    def render(self, mode='human'):
        """渲染环境状态"""
        if mode == 'human':
            print(f"\n=== Time: Hour {self.hour:.1f} ===")
            print(f"Energy Price: ${self.energy_price:.3f}/kWh")
            print(f"Outdoor: {self.weather['outdoor_temp']:.1f}°C")
            
            total_power = 0
            for i, zone in enumerate(self.zones):
                print(f"Zone {i}: T={zone['temperature']:.1f}°C, "
                      f"SP={zone['temp_setpoint']:.1f}°C, "
                      f"Occ={zone['occupancy']}, "
                      f"P={zone['power']:.1f}kW")
                total_power += zone['power']
                
            print(f"Total Power: {total_power:.1f}kW")


class MultiObjectiveHVACEnvironment(HVACEnvironment):
    """多目标HVAC控制环境"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 多目标权重可动态调整
        self.objective_weights = {
            'comfort': 0.3,
            'energy': 0.3,
            'peak_shaving': 0.2,
            'carbon': 0.2
        }
        
        # 碳排放因子
        self.carbon_factor = 0.5  # kg CO2/kWh
        
    def set_objective_weights(self, weights: Dict[str, float]):
        """设置目标权重"""
        self.objective_weights = weights
        
    def _calculate_reward(self, old_power: float) -> float:
        """计算多目标奖励"""
        rewards = {}
        
        # 舒适度目标
        comfort_score = self._calculate_comfort_score()
        rewards['comfort'] = comfort_score
        
        # 能耗目标
        energy_score = self._calculate_energy_score()
        rewards['energy'] = -energy_score  # 负值因为要最小化
        
        # 峰值削减目标
        peak_score = self._calculate_peak_score()
        rewards['peak_shaving'] = -peak_score
        
        # 碳排放目标
        carbon_score = self._calculate_carbon_score()
        rewards['carbon'] = -carbon_score
        
        # 加权综合
        total_reward = sum(
            self.objective_weights.get(obj, 0) * score
            for obj, score in rewards.items()
        )
        
        return total_reward
    
    def _calculate_comfort_score(self) -> float:
        """计算舒适度得分"""
        score = 0
        for zone in self.zones:
            if zone['occupancy'] > 0:  # 只考虑有人的区域
                temp_ok = self.comfort_temp_range[0] <= zone['temperature'] <= self.comfort_temp_range[1]
                humidity_ok = self.comfort_humidity_range[0] <= zone['humidity'] <= self.comfort_humidity_range[1]
                co2_ok = zone['co2'] < self.comfort_co2_threshold
                
                zone_score = (temp_ok + humidity_ok + co2_ok) / 3
                score += zone_score * zone['occupancy']
                
        total_occupancy = sum(zone['occupancy'] for zone in self.zones)
        return score / (total_occupancy + 1e-6)
    
    def _calculate_energy_score(self) -> float:
        """计算能耗得分"""
        total_power = sum(zone['power'] for zone in self.zones)
        return total_power / self.baseline_power
    
    def _calculate_peak_score(self) -> float:
        """计算峰值得分"""
        total_power = sum(zone['power'] for zone in self.zones)
        if total_power > self.peak_power_limit:
            return (total_power - self.peak_power_limit) / self.peak_power_limit
        return 0
    
    def _calculate_carbon_score(self) -> float:
        """计算碳排放得分"""
        total_power = sum(zone['power'] for zone in self.zones)
        carbon_emission = total_power * self.carbon_factor * 0.25  # 15分钟
        return carbon_emission / 50  # 归一化


if __name__ == "__main__":
    import copy
    
    # 测试环境
    env = HVACEnvironment(num_zones=5)
    
    print("Testing HVAC Environment...")
    obs = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    # 运行几个步骤
    for i in range(10):
        # 随机动作
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        
        print(f"\nStep {i+1}:")
        print(f"  Reward: {reward:.3f}")
        print(f"  Total Power: {info['total_power']:.1f}kW")
        print(f"  Comfort Violations: {info['comfort_violations']}")
        
        if done:
            break
            
    env.render()
    
    # 测试多目标环境
    print("\n\nTesting Multi-Objective Environment...")
    multi_env = MultiObjectiveHVACEnvironment(num_zones=3)
    multi_env.set_objective_weights({
        'comfort': 0.4,
        'energy': 0.3,
        'peak_shaving': 0.2,
        'carbon': 0.1
    })
    
    obs = multi_env.reset()
    for i in range(5):
        action = multi_env.action_space.sample()
        obs, reward, done, info = multi_env.step(action)
        print(f"Multi-objective Step {i+1}: Reward = {reward:.3f}")