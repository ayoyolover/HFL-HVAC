"""
HVAC设备仿真器 - 生成虚拟传感器数据和建筑环境模拟
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import random
from dataclasses import dataclass, asdict


@dataclass
class BuildingZone:
    """建筑区域配置"""
    zone_id: str
    area: float  # 平方米
    height: float  # 米
    max_occupancy: int
    window_ratio: float  # 窗户面积比例
    insulation_quality: float  # 0-1, 隔热质量
    
    
@dataclass
class WeatherCondition:
    """天气条件"""
    outdoor_temp: float  # 室外温度
    humidity: float  # 湿度
    solar_radiation: float  # 太阳辐射
    wind_speed: float  # 风速
    

class HVACDeviceSimulator:
    """HVAC设备仿真器"""
    
    def __init__(self, device_id: str, zone: BuildingZone, 
                 initial_temp: float = 22.0):
        self.device_id = device_id
        self.zone = zone
        self.current_temp = initial_temp
        self.current_humidity = 50.0
        self.current_co2 = 400.0  # ppm
        self.current_occupancy = 0
        self.power_consumption = 0.0  # kW
        
        # HVAC设备参数
        self.heating_capacity = 10.0  # kW
        self.cooling_capacity = 10.0  # kW
        self.ventilation_rate = 0.5  # 换气次数/小时
        
        # 控制设定值
        self.temp_setpoint = 22.0
        self.humidity_setpoint = 50.0
        
        # 历史数据缓存
        self.history = []
        
    def simulate_thermal_dynamics(self, weather: WeatherCondition, 
                                 control_signal: Dict, dt: float = 300) -> None:
        """
        模拟热力学动态
        dt: 时间步长（秒）
        """
        # 计算热负荷
        heat_load = self._calculate_heat_load(weather)
        
        # 应用HVAC控制
        heating_power, cooling_power = self._apply_hvac_control(control_signal, heat_load)
        
        # 更新温度（简化的热力学模型）
        volume = self.zone.area * self.zone.height
        air_mass = volume * 1.2  # kg (空气密度约1.2 kg/m³)
        specific_heat = 1.005  # kJ/(kg·K)
        
        # 温度变化 = (输入功率 - 热负荷) * 时间 / (质量 * 比热)
        temp_change = ((heating_power - cooling_power - heat_load) * dt) / (air_mass * specific_heat)
        self.current_temp += temp_change / 1000  # 转换单位
        
        # 添加噪声
        self.current_temp += np.random.normal(0, 0.1)
        
        # 更新功耗
        self.power_consumption = abs(heating_power) + abs(cooling_power) + 0.5  # 基础功耗
        
    def _calculate_heat_load(self, weather: WeatherCondition) -> float:
        """计算热负荷（kW）"""
        # 传导热负荷
        u_value = 2.0 / (1 + self.zone.insulation_quality * 4)  # W/(m²·K)
        wall_area = 2 * self.zone.area ** 0.5 * self.zone.height * 4  # 简化计算
        conduction_load = u_value * wall_area * (weather.outdoor_temp - self.current_temp) / 1000
        
        # 太阳辐射负荷
        window_area = wall_area * self.zone.window_ratio
        solar_load = weather.solar_radiation * window_area * 0.7 / 1000  # kW
        
        # 人员热负荷
        occupant_load = self.current_occupancy * 0.1  # 每人100W
        
        # 设备热负荷（随机）
        equipment_load = np.random.uniform(0.5, 2.0)
        
        return conduction_load + solar_load + occupant_load + equipment_load
    
    def _apply_hvac_control(self, control_signal: Dict, 
                           heat_load: float) -> Tuple[float, float]:
        """应用HVAC控制信号"""
        temp_error = control_signal.get('temperature_setpoint', self.temp_setpoint) - self.current_temp
        
        # PID控制（简化版）
        kp = 2.0
        control_output = kp * temp_error
        
        if control_output > 0:  # 需要加热
            heating_power = min(control_output, self.heating_capacity)
            cooling_power = 0
        else:  # 需要制冷
            heating_power = 0
            cooling_power = min(-control_output, self.cooling_capacity)
            
        return heating_power, cooling_power
    
    def simulate_occupancy(self, hour: int) -> None:
        """模拟占用率变化"""
        # 工作时间占用率模式
        if 8 <= hour < 18:  # 工作时间
            base_occupancy = 0.7 if hour not in [12, 13] else 0.3  # 午餐时间
        elif 18 <= hour < 20:  # 加班时间
            base_occupancy = 0.3
        elif 6 <= hour < 8 or 20 <= hour < 22:  # 早晚时段
            base_occupancy = 0.1
        else:
            base_occupancy = 0.05  # 夜间保持少量人员（如保安）
            
        # 添加随机波动
        self.current_occupancy = int(
            self.zone.max_occupancy * base_occupancy * np.random.uniform(0.8, 1.2)
        )
        
    def simulate_co2(self) -> None:
        """模拟CO2浓度"""
        # CO2生成（人员）
        co2_generation = self.current_occupancy * 20  # ppm/小时
        
        # 通风稀释
        ventilation_reduction = self.ventilation_rate * (self.current_co2 - 400)
        
        # 更新CO2浓度
        self.current_co2 += (co2_generation - ventilation_reduction) / 12  # 5分钟步长
        self.current_co2 = np.clip(self.current_co2, 400, 2000)
        
    def simulate_humidity(self, weather: WeatherCondition) -> None:
        """模拟湿度变化"""
        # 简化的湿度模型
        target_humidity = weather.humidity * 0.7 + self.current_occupancy * 0.5
        self.current_humidity += (target_humidity - self.current_humidity) * 0.1
        self.current_humidity = np.clip(self.current_humidity, 30, 70)
        
    def get_sensor_data(self) -> Dict:
        """获取传感器数据"""
        return {
            'device_id': self.device_id,
            'timestamp': datetime.now().isoformat(),
            'temperature': round(self.current_temp, 2),
            'humidity': round(self.current_humidity, 2),
            'co2': round(self.current_co2, 1),
            'occupancy': self.current_occupancy,
            'power_consumption': round(self.power_consumption, 2),
            'temp_setpoint': self.temp_setpoint,
            'zone_id': self.zone.zone_id
        }
        
    def add_noise(self, data: Dict, noise_level: float = 0.02) -> Dict:
        """添加传感器噪声"""
        noisy_data = data.copy()
        for key in ['temperature', 'humidity', 'co2']:
            if key in noisy_data:
                noise = np.random.normal(0, noisy_data[key] * noise_level)
                noisy_data[key] += noise
        return noisy_data


class WeatherSimulator:
    """天气条件仿真器"""
    
    def __init__(self, base_temp: float = 15.0, climate: str = 'temperate'):
        self.base_temp = base_temp
        self.climate = climate
        self.time = 0
        
    def get_weather(self, hour: int, day: int) -> WeatherCondition:
        """获取天气条件"""
        # 日变化
        daily_variation = 10 * np.sin((hour - 6) * np.pi / 12) if 6 <= hour <= 18 else -5
        
        # 季节变化
        seasonal_variation = 15 * np.sin(day * 2 * np.pi / 365)
        
        # 室外温度
        outdoor_temp = self.base_temp + daily_variation + seasonal_variation
        outdoor_temp += np.random.normal(0, 2)  # 随机波动
        
        # 湿度（反相关于温度）
        humidity = 60 - outdoor_temp * 0.5 + np.random.normal(0, 5)
        humidity = np.clip(humidity, 20, 90)
        
        # 太阳辐射
        if 6 <= hour <= 18:
            solar_radiation = 800 * np.sin((hour - 6) * np.pi / 12)
            solar_radiation *= np.random.uniform(0.3, 1.0)  # 云层影响
        else:
            solar_radiation = 0
            
        # 风速
        wind_speed = np.random.gamma(2, 2)  # km/h
        
        return WeatherCondition(
            outdoor_temp=outdoor_temp,
            humidity=humidity,
            solar_radiation=solar_radiation,
            wind_speed=wind_speed
        )


class BuildingSimulator:
    """建筑整体仿真器"""
    
    def __init__(self, num_zones: int = 10, start_hour: int = 8):
        self.zones = self._create_zones(num_zones)
        self.devices = self._create_devices()
        self.weather_sim = WeatherSimulator()
        # 设置仿真开始时间为指定的小时（默认早上8点）
        now = datetime.now()
        self.current_time = now.replace(hour=start_hour, minute=0, second=0, microsecond=0)
        
    def _create_zones(self, num_zones: int) -> List[BuildingZone]:
        """创建建筑区域"""
        zones = []
        zone_types = ['office', 'meeting', 'corridor', 'server', 'lobby']
        
        for i in range(num_zones):
            zone_type = zone_types[i % len(zone_types)]
            
            if zone_type == 'office':
                area, max_occ = np.random.uniform(50, 150), np.random.randint(5, 20)
            elif zone_type == 'meeting':
                area, max_occ = np.random.uniform(30, 80), np.random.randint(10, 30)
            elif zone_type == 'server':
                area, max_occ = np.random.uniform(20, 50), 0
            elif zone_type == 'corridor':
                area, max_occ = np.random.uniform(20, 100), 0
            else:  # lobby
                area, max_occ = np.random.uniform(100, 200), np.random.randint(20, 50)
                
            zone = BuildingZone(
                zone_id=f"zone_{i:03d}",
                area=area,
                height=3.0,
                max_occupancy=max_occ,
                window_ratio=np.random.uniform(0.2, 0.5),
                insulation_quality=np.random.uniform(0.5, 0.9)
            )
            zones.append(zone)
            
        return zones
    
    def _create_devices(self) -> List[HVACDeviceSimulator]:
        """为每个区域创建HVAC设备"""
        devices = []
        for i, zone in enumerate(self.zones):
            device = HVACDeviceSimulator(
                device_id=f"device_{i:03d}",
                zone=zone,
                initial_temp=np.random.uniform(20, 24)
            )
            devices.append(device)
        return devices
    
    def step(self, control_signals: Optional[Dict] = None) -> List[Dict]:
        """仿真步进"""
        hour = self.current_time.hour
        day = self.current_time.timetuple().tm_yday
        
        # 获取天气
        weather = self.weather_sim.get_weather(hour, day)
        
        # 收集所有设备数据
        all_data = []
        
        for device in self.devices:
            # 更新占用率
            device.simulate_occupancy(hour)
            
            # 应用控制信号
            control = control_signals.get(device.device_id, {}) if control_signals else {}
            
            # 模拟热力学
            device.simulate_thermal_dynamics(weather, control)
            
            # 模拟其他参数
            device.simulate_co2()
            device.simulate_humidity(weather)
            
            # 收集数据
            sensor_data = device.get_sensor_data()
            sensor_data['weather'] = asdict(weather)
            all_data.append(sensor_data)
            
        # 推进时间（5分钟）
        self.current_time += timedelta(minutes=5)
        
        return all_data
    
    def run_simulation(self, duration_hours: int = 24, 
                       save_data: bool = True) -> pd.DataFrame:
        """运行仿真"""
        steps = duration_hours * 12  # 5分钟步长
        all_data = []
        
        for _ in range(steps):
            step_data = self.step()
            all_data.extend(step_data)
            
        df = pd.DataFrame(all_data)
        
        if save_data:
            df.to_csv(f'simulation_data_{datetime.now():%Y%m%d_%H%M%S}.csv', index=False)
            
        return df
    
    def get_building_status(self) -> Dict:
        """获取建筑整体状态"""
        total_power = sum(d.power_consumption for d in self.devices)
        avg_temp = np.mean([d.current_temp for d in self.devices])
        total_occupancy = sum(d.current_occupancy for d in self.devices)
        
        return {
            'timestamp': self.current_time.isoformat(),
            'total_power_kw': round(total_power, 2),
            'average_temperature': round(avg_temp, 2),
            'total_occupancy': total_occupancy,
            'num_devices': len(self.devices),
            'comfort_violations': self._check_comfort_violations()
        }
        
    def _check_comfort_violations(self) -> int:
        """检查舒适度违规"""
        violations = 0
        for device in self.devices:
            if not (20 <= device.current_temp <= 26):
                violations += 1
            if not (30 <= device.current_humidity <= 60):
                violations += 1
            if device.current_co2 > 1000:
                violations += 1
        return violations


if __name__ == "__main__":
    # 测试仿真器
    building = BuildingSimulator(num_zones=5)
    
    print("Starting building simulation...")
    print(f"Created {len(building.devices)} HVAC devices")
    
    # 运行24小时仿真
    df = building.run_simulation(duration_hours=24)
    
    print(f"\nSimulation completed. Generated {len(df)} data points")
    print(f"Data saved to simulation_data_*.csv")
    
    # 显示统计信息
    print("\nSimulation Statistics:")
    print(f"Temperature range: {df['temperature'].min():.1f} - {df['temperature'].max():.1f}°C")
    print(f"Average power consumption: {df['power_consumption'].mean():.2f} kW")
    print(f"Total energy consumed: {df['power_consumption'].sum() * 5/60:.2f} kWh")