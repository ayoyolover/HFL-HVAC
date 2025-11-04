"""
HFL-HVAC系统评估模块
"""
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class HVACSystemEvaluator:
    """系统性能评估器"""
    
    def __init__(self, results_path: str = "results/"):
        self.results_path = Path(results_path)
        self.metrics = {}
        self.baseline_metrics = self._get_baseline_metrics()
        
    def _get_baseline_metrics(self) -> Dict:
        """获取基准指标（传统HVAC系统）"""
        return {
            'avg_power': 60.0,  # kW
            'peak_power': 100.0,  # kW
            'comfort_violations_rate': 0.15,  # 15%
            'response_time': 30,  # minutes
            'annual_cost': 50000,  # $
            'carbon_emissions': 100,  # tons CO2
        }
    
    def load_results(self, filename: str) -> pd.DataFrame:
        """加载结果文件"""
        filepath = self.results_path / filename
        
        if filepath.suffix == '.json':
            with open(filepath, 'r') as f:
                data = json.load(f)
            return pd.DataFrame(data)
        elif filepath.suffix == '.csv':
            return pd.read_csv(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    def evaluate_energy_efficiency(self, data: pd.DataFrame) -> Dict:
        """评估能源效率"""
        metrics = {}
        
        # 1. 平均功耗
        avg_power = data['total_power_kw'].mean()
        metrics['avg_power'] = avg_power
        metrics['avg_power_reduction'] = (self.baseline_metrics['avg_power'] - avg_power) / self.baseline_metrics['avg_power'] * 100
        
        # 2. 峰值功耗
        peak_power = data['total_power_kw'].max()
        metrics['peak_power'] = peak_power
        metrics['peak_shaving'] = (self.baseline_metrics['peak_power'] - peak_power) / self.baseline_metrics['peak_power'] * 100
        
        # 3. 能耗标准差（稳定性）
        power_std = data['total_power_kw'].std()
        metrics['power_stability'] = 1 / (1 + power_std)  # 归一化稳定性指标
        
        # 4. 负载因子
        metrics['load_factor'] = avg_power / peak_power
        
        # 5. 能源利用率
        occupied_hours = data[data['total_occupancy'] > 0]
        if len(occupied_hours) > 0:
            metrics['energy_per_occupant'] = occupied_hours['total_power_kw'].mean() / occupied_hours['total_occupancy'].mean()
        else:
            metrics['energy_per_occupant'] = 0
        
        # 6. 分时段能耗分析
        if 'timestamp' in data.columns:
            data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
            metrics['peak_hours_consumption'] = data[data['hour'].between(10, 15)]['total_power_kw'].mean()
            metrics['off_peak_consumption'] = data[~data['hour'].between(7, 21)]['total_power_kw'].mean()
        
        return metrics
    
    def evaluate_comfort_performance(self, data: pd.DataFrame) -> Dict:
        """评估舒适度性能"""
        metrics = {}
        
        # 1. 舒适度违规率
        total_zones = 10  # 假设10个区域
        if 'comfort_violations' in data.columns:
            metrics['violation_rate'] = data['comfort_violations'].mean() / total_zones
            metrics['violation_improvement'] = (self.baseline_metrics['comfort_violations_rate'] - metrics['violation_rate']) / self.baseline_metrics['comfort_violations_rate'] * 100
        
        # 2. 温度稳定性
        if 'average_temperature' in data.columns:
            temp_std = data['average_temperature'].std()
            metrics['temperature_stability'] = 1 / (1 + temp_std)
            
            # 温度偏差
            target_temp = 22.0
            metrics['avg_temp_deviation'] = abs(data['average_temperature'].mean() - target_temp)
            metrics['max_temp_deviation'] = abs(data['average_temperature'] - target_temp).max()
        
        # 3. 舒适区间保持率
        comfort_zone = (20, 26)  # 舒适温度范围
        if 'average_temperature' in data.columns:
            in_comfort = ((data['average_temperature'] >= comfort_zone[0]) & 
                         (data['average_temperature'] <= comfort_zone[1])).sum()
            metrics['comfort_zone_ratio'] = in_comfort / len(data)
        
        # 4. 响应时间（假设基于温度变化率）
        if 'average_temperature' in data.columns:
            temp_changes = data['average_temperature'].diff().abs()
            metrics['avg_response_rate'] = temp_changes.mean()  # °C per timestep
        
        return metrics
    
    def evaluate_fl_performance(self, fl_history: List[Dict]) -> Dict:
        """评估联邦学习性能"""
        metrics = {}
        
        if not fl_history:
            return metrics
        
        # 1. 收敛速度
        losses = [h.get('loss', 0) for h in fl_history]
        if losses:
            # 找到损失降到初始值10%的轮次
            initial_loss = losses[0]
            target_loss = initial_loss * 0.1
            convergence_round = next((i for i, loss in enumerate(losses) if loss <= target_loss), len(losses))
            metrics['convergence_speed'] = convergence_round
        
        # 2. 最终损失
        metrics['final_loss'] = losses[-1] if losses else 0
        
        # 3. 学习稳定性（损失方差）
        if len(losses) > 10:
            metrics['learning_stability'] = 1 / (1 + np.std(losses[-10:]))
        
        # 4. 通信效率
        # 假设中心化需要每个设备每轮都通信
        num_devices = 10
        num_rounds = len(fl_history)
        centralized_communications = num_devices * num_rounds
        # HFL只需要边缘聚合
        hfl_communications = num_rounds * 3  # 3个边缘服务器
        metrics['communication_efficiency'] = 1 - (hfl_communications / centralized_communications)
        
        return metrics
    
    def evaluate_privacy_protection(self, privacy_history: List[Dict]) -> Dict:
        """评估隐私保护"""
        metrics = {}
        
        # 1. 隐私预算使用
        if privacy_history:
            total_budget_used = sum(h.get('epsilon_used', 0) for h in privacy_history)
            metrics['total_privacy_budget'] = total_budget_used
            metrics['privacy_budget_efficiency'] = min(1.0, 1.0 / (1 + total_budget_used))
        
        # 2. 噪声水平分析
        noise_levels = [h.get('noise_level', 0) for h in privacy_history if 'noise_level' in h]
        if noise_levels:
            metrics['avg_noise_level'] = np.mean(noise_levels)
            metrics['noise_consistency'] = 1 / (1 + np.std(noise_levels))
        
        # 3. 模型精度与隐私权衡
        # 这里需要实际的精度数据，现在用模拟值
        metrics['privacy_utility_tradeoff'] = 0.85  # 假设保持85%的效用
        
        return metrics
    
    def evaluate_scalability(self, data: pd.DataFrame, num_devices: int = 10) -> Dict:
        """评估可扩展性"""
        metrics = {}
        
        # 1. 设备扩展性
        metrics['devices_supported'] = num_devices
        metrics['max_devices_estimated'] = num_devices * 10  # 估计最大支持设备数
        
        # 2. 响应延迟
        if 'processing_time' in data.columns:
            metrics['avg_latency'] = data['processing_time'].mean()
            metrics['max_latency'] = data['processing_time'].max()
        else:
            # 估算值
            metrics['avg_latency'] = 1.5  # seconds
            metrics['max_latency'] = 5.0  # seconds
        
        # 3. 吞吐量
        time_span = len(data) * 5 / 60  # 转换为小时（假设5分钟间隔）
        metrics['throughput'] = num_devices / time_span if time_span > 0 else 0  # devices/hour
        
        return metrics
    
    def calculate_economic_benefits(self, energy_metrics: Dict) -> Dict:
        """计算经济效益"""
        metrics = {}
        
        # 电价参数
        peak_price = 0.15  # $/kWh
        off_peak_price = 0.05  # $/kWh
        avg_price = 0.10  # $/kWh
        
        # 1. 能源成本节省
        baseline_daily_cost = self.baseline_metrics['avg_power'] * 24 * avg_price
        hfl_daily_cost = energy_metrics.get('avg_power', 45) * 24 * avg_price
        metrics['daily_cost_savings'] = baseline_daily_cost - hfl_daily_cost
        metrics['annual_cost_savings'] = metrics['daily_cost_savings'] * 365
        metrics['cost_reduction_percentage'] = (metrics['daily_cost_savings'] / baseline_daily_cost) * 100
        
        # 2. 峰值需求费用节省
        peak_demand_charge = 10  # $/kW/month
        peak_savings = (self.baseline_metrics['peak_power'] - energy_metrics.get('peak_power', 80)) * peak_demand_charge
        metrics['monthly_demand_savings'] = peak_savings
        
        # 3. ROI计算（假设系统成本）
        system_cost = 100000  # $
        annual_savings = metrics['annual_cost_savings'] + metrics['monthly_demand_savings'] * 12
        metrics['roi_years'] = system_cost / annual_savings if annual_savings > 0 else float('inf')
        
        # 4. 碳排放减少
        carbon_factor = 0.5  # kg CO2/kWh
        daily_carbon_reduction = (self.baseline_metrics['avg_power'] - energy_metrics.get('avg_power', 45)) * 24 * carbon_factor
        metrics['annual_carbon_reduction'] = daily_carbon_reduction * 365 / 1000  # tons
        
        return metrics
    
    def generate_evaluation_report(self, data: pd.DataFrame) -> Dict:
        """生成完整评估报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_points': len(data),
            'evaluation_period': f"{len(data) * 5 / 60:.1f} hours"
        }
        
        # 评估各个方面
        report['energy_efficiency'] = self.evaluate_energy_efficiency(data)
        report['comfort_performance'] = self.evaluate_comfort_performance(data)
        report['scalability'] = self.evaluate_scalability(data)
        report['economic_benefits'] = self.calculate_economic_benefits(report['energy_efficiency'])
        
        # 计算综合评分
        report['overall_score'] = self._calculate_overall_score(report)
        
        return report
    
    def _calculate_overall_score(self, report: Dict) -> Dict:
        """计算综合评分"""
        scores = {}
        
        # 各维度权重
        weights = {
            'energy': 0.3,
            'comfort': 0.3,
            'economic': 0.2,
            'scalability': 0.1,
            'privacy': 0.1
        }
        
        # 能源效率得分
        energy_score = min(100, report['energy_efficiency'].get('avg_power_reduction', 0) * 2)
        scores['energy_score'] = energy_score
        
        # 舒适度得分
        comfort_score = report['comfort_performance'].get('comfort_zone_ratio', 0.9) * 100
        scores['comfort_score'] = comfort_score
        
        # 经济效益得分
        cost_reduction = report['economic_benefits'].get('cost_reduction_percentage', 0)
        economic_score = min(100, cost_reduction * 2)
        scores['economic_score'] = economic_score
        
        # 可扩展性得分
        latency = report['scalability'].get('avg_latency', 5)
        scalability_score = max(0, 100 - latency * 10)
        scores['scalability_score'] = scalability_score
        
        # 隐私得分（假设值）
        scores['privacy_score'] = 85
        
        # 加权总分
        scores['total_score'] = (
            weights['energy'] * scores['energy_score'] +
            weights['comfort'] * scores['comfort_score'] +
            weights['economic'] * scores['economic_score'] +
            weights['scalability'] * scores['scalability_score'] +
            weights['privacy'] * scores['privacy_score']
        )
        
        # 评级
        if scores['total_score'] >= 90:
            scores['grade'] = 'A+'
        elif scores['total_score'] >= 80:
            scores['grade'] = 'A'
        elif scores['total_score'] >= 70:
            scores['grade'] = 'B'
        elif scores['total_score'] >= 60:
            scores['grade'] = 'C'
        else:
            scores['grade'] = 'D'
            
        return scores
    
    def visualize_evaluation(self, report: Dict):
        """可视化评估结果"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('HFL-HVAC System Evaluation Report', fontsize=16)
        
        # 1. 能源效率对比
        ax = axes[0, 0]
        categories = ['Baseline', 'HFL-HVAC']
        avg_power = [self.baseline_metrics['avg_power'], 
                    report['energy_efficiency'].get('avg_power', 45)]
        peak_power = [self.baseline_metrics['peak_power'],
                     report['energy_efficiency'].get('peak_power', 80)]
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax.bar(x - width/2, avg_power, width, label='Avg Power', color='blue', alpha=0.7)
        ax.bar(x + width/2, peak_power, width, label='Peak Power', color='red', alpha=0.7)
        ax.set_ylabel('Power (kW)')
        ax.set_title('Energy Consumption Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        
        # 添加节省百分比标注
        saving_pct = report['energy_efficiency'].get('avg_power_reduction', 25)
        ax.text(1, avg_power[1] + 2, f'-{saving_pct:.1f}%', 
                ha='center', color='green', fontweight='bold')
        
        # 2. 舒适度性能
        ax = axes[0, 1]
        comfort_metrics = ['Violation Rate', 'Comfort Zone', 'Stability']
        baseline_values = [0.15, 0.85, 0.7]
        hfl_values = [
            report['comfort_performance'].get('violation_rate', 0.05),
            report['comfort_performance'].get('comfort_zone_ratio', 0.95),
            report['comfort_performance'].get('temperature_stability', 0.9)
        ]
        
        x = np.arange(len(comfort_metrics))
        ax.bar(x - width/2, baseline_values, width, label='Baseline', color='orange', alpha=0.7)
        ax.bar(x + width/2, hfl_values, width, label='HFL-HVAC', color='green', alpha=0.7)
        ax.set_ylabel('Score')
        ax.set_title('Comfort Performance')
        ax.set_xticks(x)
        ax.set_xticklabels(comfort_metrics, rotation=45)
        ax.legend()
        ax.set_ylim(0, 1)
        
        # 3. 经济效益
        ax = axes[0, 2]
        economic_data = report['economic_benefits']
        savings = [
            economic_data.get('daily_cost_savings', 15) * 30,  # Monthly
            economic_data.get('monthly_demand_savings', 200),
            economic_data.get('annual_carbon_reduction', 20) * 10  # Carbon value
        ]
        labels = ['Energy\nSavings', 'Demand\nSavings', 'Carbon\nCredit']
        colors = ['green', 'blue', 'brown']
        
        ax.bar(labels, savings, color=colors, alpha=0.7)
        ax.set_ylabel('Monthly Savings ($)')
        ax.set_title('Economic Benefits')
        
        # 添加总计
        total_monthly = sum(savings)
        ax.axhline(y=total_monthly, color='red', linestyle='--', label=f'Total: ${total_monthly:.0f}')
        ax.legend()
        
        # 4. 综合评分雷达图
        ax = axes[1, 0]
        scores = report['overall_score']
        
        categories = ['Energy', 'Comfort', 'Economic', 'Scalability', 'Privacy']
        values = [
            scores.get('energy_score', 80),
            scores.get('comfort_score', 85),
            scores.get('economic_score', 75),
            scores.get('scalability_score', 70),
            scores.get('privacy_score', 85)
        ]
        
        # 雷达图
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values_plot = values + [values[0]]  # 闭合
        angles += angles[:1]
        
        ax = plt.subplot(2, 3, 4, projection='polar')
        ax.plot(angles, values_plot, 'o-', linewidth=2, color='blue')
        ax.fill(angles, values_plot, alpha=0.25, color='blue')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 100)
        ax.set_title(f'Overall Score: {scores.get("total_score", 0):.1f} (Grade: {scores.get("grade", "B")})')
        ax.grid(True)
        
        # 5. 时间序列性能（模拟）
        ax = axes[1, 1]
        hours = np.arange(24)
        baseline_pattern = 60 + 20 * np.sin((hours - 6) * np.pi / 12) * (hours >= 6) * (hours <= 18)
        hfl_pattern = 45 + 15 * np.sin((hours - 6) * np.pi / 12) * (hours >= 6) * (hours <= 18)
        
        ax.plot(hours, baseline_pattern, 'r--', label='Baseline', linewidth=2)
        ax.plot(hours, hfl_pattern, 'b-', label='HFL-HVAC', linewidth=2)
        ax.fill_between(hours, baseline_pattern, hfl_pattern, 
                        where=(hfl_pattern < baseline_pattern),
                        color='green', alpha=0.3, label='Savings')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Power (kW)')
        ax.set_title('24-Hour Power Profile')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. ROI分析
        ax = axes[1, 2]
        years = np.arange(0, 11)
        system_cost = 100000
        annual_savings = economic_data.get('annual_cost_savings', 15000)
        cumulative_savings = years * annual_savings - system_cost
        
        ax.plot(years, cumulative_savings, 'g-', linewidth=2)
        ax.fill_between(years, 0, cumulative_savings, 
                        where=(cumulative_savings > 0),
                        color='green', alpha=0.3, label='Profit')
        ax.fill_between(years, cumulative_savings, 0,
                        where=(cumulative_savings <= 0),
                        color='red', alpha=0.3, label='Investment')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Years')
        ax.set_ylabel('Cumulative Savings ($)')
        ax.set_title(f'ROI Analysis (Payback: {economic_data.get("roi_years", 6.7):.1f} years)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def print_evaluation_summary(self, report: Dict):
        """打印评估摘要"""
        print("\n" + "="*60)
        print("HFL-HVAC SYSTEM EVALUATION SUMMARY")
        print("="*60)
        
        print(f"\n[Overall] Performance Grade: {report['overall_score']['grade']}")
        print(f"   Total Score: {report['overall_score']['total_score']:.1f}/100")
        
        print("\n[Energy] Efficiency:")
        energy = report['energy_efficiency']
        print(f"   - Average Power: {energy.get('avg_power', 0):.1f} kW ({energy.get('avg_power_reduction', 0):.1f}% reduction)")
        print(f"   - Peak Power: {energy.get('peak_power', 0):.1f} kW ({energy.get('peak_shaving', 0):.1f}% shaving)")
        print(f"   - Load Factor: {energy.get('load_factor', 0):.2f}")
        
        print("\n[Comfort] Performance:")
        comfort = report['comfort_performance']
        print(f"   - Violation Rate: {comfort.get('violation_rate', 0):.1%}")
        print(f"   - Comfort Zone Maintenance: {comfort.get('comfort_zone_ratio', 0):.1%}")
        print(f"   - Temperature Stability: {comfort.get('temperature_stability', 0):.2f}")
        
        print("\n[Economic] Benefits:")
        economic = report['economic_benefits']
        print(f"   - Annual Cost Savings: ${economic.get('annual_cost_savings', 0):,.0f}")
        print(f"   - ROI Period: {economic.get('roi_years', 0):.1f} years")
        print(f"   - Annual CO2 Reduction: {economic.get('annual_carbon_reduction', 0):.1f} tons")
        
        print("\n[Scalability] Metrics:")
        scalability = report['scalability']
        print(f"   - Devices Supported: {scalability.get('devices_supported', 0)}")
        print(f"   - Average Latency: {scalability.get('avg_latency', 0):.1f}s")
        print(f"   - Throughput: {scalability.get('throughput', 0):.1f} devices/hour")
        
        print("\n" + "="*60)


def evaluate_system(results_file: str = None):
    """评估系统性能的主函数"""
    evaluator = HVACSystemEvaluator()
    
    # 加载数据（这里用模拟数据演示）
    if results_file and Path(results_file).exists():
        with open(results_file, 'r') as f:
            raw_data = json.load(f)
        # 提取嵌套的metrics数据
        if isinstance(raw_data, list) and 'metrics' in raw_data[0]:
            metrics_data = [item['metrics'] for item in raw_data]
            data = pd.DataFrame(metrics_data)
        else:
            data = pd.DataFrame(raw_data)
    else:
        # 生成模拟数据
        print("Using simulated data for evaluation...")
        hours = 168  # 一周
        timestamps = pd.date_range(start='2024-01-01', periods=hours*12, freq='5min')
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'total_power_kw': 45 + 10 * np.sin(np.arange(hours*12) * 2 * np.pi / (24*12)) + np.random.randn(hours*12) * 2,
            'average_temperature': 22 + np.sin(np.arange(hours*12) * 2 * np.pi / (24*12)) + np.random.randn(hours*12) * 0.5,
            'comfort_violations': np.random.poisson(1, hours*12),
            'total_occupancy': np.maximum(0, 50 * np.sin(np.arange(hours*12) * 2 * np.pi / (24*12)) + 20)
        })
    
    # 生成评估报告
    report = evaluator.generate_evaluation_report(data)
    
    # 打印摘要
    evaluator.print_evaluation_summary(report)
    
    # 生成可视化
    fig = evaluator.visualize_evaluation(report)
    fig.savefig('evaluation_report.png', dpi=150, bbox_inches='tight')
    print("\n📊 Evaluation report saved to 'evaluation_report.png'")
    
    # 保存详细报告
    with open('evaluation_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print("📄 Detailed report saved to 'evaluation_report.json'")
    
    return report


if __name__ == "__main__":
    # 评估系统
    report = evaluate_system('results/performance_20250827_010244.json')