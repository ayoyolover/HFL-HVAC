"""
系统可视化仪表板
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import seaborn as sns
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, List, Optional
import json

# 设置绘图风格
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class HVACDashboard:
    """HVAC系统监控仪表板"""
    
    def __init__(self, num_zones: int = 5, window_size: int = 100):
        """
        Args:
            num_zones: 监控区域数量
            window_size: 历史数据窗口大小
        """
        self.num_zones = num_zones
        self.window_size = window_size
        
        # 数据缓冲
        self.temperature_history = {f'zone_{i}': deque(maxlen=window_size) 
                                   for i in range(num_zones)}
        self.power_history = deque(maxlen=window_size)
        self.comfort_history = deque(maxlen=window_size)
        self.fl_loss_history = deque(maxlen=window_size)
        self.privacy_budget_history = deque(maxlen=window_size)
        self.reward_history = deque(maxlen=window_size)
        
        # 时间戳
        self.timestamps = deque(maxlen=window_size)
        
        # 初始化图形
        self.fig = None
        self.axes = {}
        self.lines = {}
        self.setup_dashboard()
        
    def setup_dashboard(self):
        """设置仪表板布局"""
        # 创建图形和网格布局
        self.fig = plt.figure(figsize=(20, 12))
        self.fig.suptitle('HFL-HVAC System Dashboard', fontsize=16, fontweight='bold')
        
        # 创建网格：4行3列
        gs = GridSpec(4, 3, figure=self.fig, hspace=0.3, wspace=0.3)
        
        # 1. 温度监控（左上，跨2列）
        self.axes['temperature'] = self.fig.add_subplot(gs[0, :2])
        self.axes['temperature'].set_title('Zone Temperatures')
        self.axes['temperature'].set_xlabel('Time Steps')
        self.axes['temperature'].set_ylabel('Temperature (°C)')
        self.axes['temperature'].set_ylim(18, 28)
        self.axes['temperature'].grid(True, alpha=0.3)
        
        # 2. 能耗监控（右上）
        self.axes['power'] = self.fig.add_subplot(gs[0, 2])
        self.axes['power'].set_title('Total Power Consumption')
        self.axes['power'].set_xlabel('Time Steps')
        self.axes['power'].set_ylabel('Power (kW)')
        self.axes['power'].grid(True, alpha=0.3)
        
        # 3. 舒适度违规（左中）
        self.axes['comfort'] = self.fig.add_subplot(gs[1, 0])
        self.axes['comfort'].set_title('Comfort Violations')
        self.axes['comfort'].set_xlabel('Time Steps')
        self.axes['comfort'].set_ylabel('Violations')
        self.axes['comfort'].grid(True, alpha=0.3)
        
        # 4. 联邦学习损失（中间）
        self.axes['fl_loss'] = self.fig.add_subplot(gs[1, 1])
        self.axes['fl_loss'].set_title('Federated Learning Loss')
        self.axes['fl_loss'].set_xlabel('FL Rounds')
        self.axes['fl_loss'].set_ylabel('Loss')
        self.axes['fl_loss'].grid(True, alpha=0.3)
        
        # 5. 隐私预算（右中）
        self.axes['privacy'] = self.fig.add_subplot(gs[1, 2])
        self.axes['privacy'].set_title('Privacy Budget Usage')
        self.axes['privacy'].set_xlabel('Time Steps')
        self.axes['privacy'].set_ylabel('ε Used')
        self.axes['privacy'].grid(True, alpha=0.3)
        
        # 6. DRL奖励（左下，跨2列）
        self.axes['reward'] = self.fig.add_subplot(gs[2, :2])
        self.axes['reward'].set_title('DRL Reward')
        self.axes['reward'].set_xlabel('Time Steps')
        self.axes['reward'].set_ylabel('Reward')
        self.axes['reward'].grid(True, alpha=0.3)
        
        # 7. 热力图：区域占用率（右下）
        self.axes['occupancy'] = self.fig.add_subplot(gs[2, 2])
        self.axes['occupancy'].set_title('Zone Occupancy Heatmap')
        
        # 8. 系统状态指标（底部，跨3列）
        self.axes['metrics'] = self.fig.add_subplot(gs[3, :])
        self.axes['metrics'].axis('off')
        
        # 初始化线条
        self.init_plots()
        
    def init_plots(self):
        """初始化绘图元素"""
        # 温度线条
        self.lines['temperature'] = {}
        for zone_id in self.temperature_history.keys():
            line, = self.axes['temperature'].plot([], [], label=zone_id, linewidth=2)
            self.lines['temperature'][zone_id] = line
        self.axes['temperature'].legend(loc='upper right', ncol=2)
        
        # 添加舒适区域
        self.axes['temperature'].axhspan(20, 26, alpha=0.2, color='green', label='Comfort Zone')
        
        # 功率线条
        self.lines['power'], = self.axes['power'].plot([], [], 'r-', linewidth=2)
        self.axes['power'].axhline(y=50, color='orange', linestyle='--', label='Baseline')
        self.axes['power'].axhline(y=100, color='red', linestyle='--', label='Peak Limit')
        self.axes['power'].legend()
        
        # 舒适度违规
        self.lines['comfort'], = self.axes['comfort'].plot([], [], 'b-', linewidth=2)
        
        # FL损失
        self.lines['fl_loss'], = self.axes['fl_loss'].plot([], [], 'g-', linewidth=2)
        
        # 隐私预算
        self.lines['privacy'], = self.axes['privacy'].plot([], [], 'm-', linewidth=2)
        self.axes['privacy'].axhline(y=1.0, color='red', linestyle='--', label='Budget Limit')
        self.axes['privacy'].legend()
        
        # DRL奖励
        self.lines['reward'], = self.axes['reward'].plot([], [], 'c-', linewidth=2)
        
    def update(self, data: Dict):
        """更新仪表板数据"""
        # 添加时间戳
        self.timestamps.append(len(self.timestamps))
        
        # 更新温度数据
        if 'zones' in data:
            for i, zone in enumerate(data['zones'][:self.num_zones]):
                zone_id = f'zone_{i}'
                self.temperature_history[zone_id].append(zone.get('temperature', 22))
                
        # 更新功率数据
        if 'total_power' in data:
            self.power_history.append(data['total_power'])
            
        # 更新舒适度违规
        if 'comfort_violations' in data:
            self.comfort_history.append(data['comfort_violations'])
            
        # 更新FL损失
        if 'fl_loss' in data:
            self.fl_loss_history.append(data['fl_loss'])
            
        # 更新隐私预算
        if 'privacy_budget' in data:
            self.privacy_budget_history.append(data['privacy_budget'])
            
        # 更新DRL奖励
        if 'drl_reward' in data:
            self.reward_history.append(data['drl_reward'])
            
        # 刷新图形
        self.refresh_plots(data)
        
    def refresh_plots(self, data: Dict):
        """刷新所有图表"""
        x_data = list(self.timestamps)
        
        # 更新温度图
        for zone_id, line in self.lines['temperature'].items():
            y_data = list(self.temperature_history[zone_id])
            if y_data:
                line.set_data(x_data[:len(y_data)], y_data)
        
        if x_data:
            self.axes['temperature'].set_xlim(max(0, x_data[-1]-100), x_data[-1]+5)
            
        # 更新功率图
        if self.power_history:
            self.lines['power'].set_data(x_data[:len(self.power_history)], 
                                        list(self.power_history))
            self.axes['power'].set_xlim(max(0, x_data[-1]-100), x_data[-1]+5)
            self.axes['power'].set_ylim(0, max(120, max(self.power_history)*1.2))
            
        # 更新舒适度违规图
        if self.comfort_history:
            self.lines['comfort'].set_data(x_data[:len(self.comfort_history)], 
                                          list(self.comfort_history))
            self.axes['comfort'].set_xlim(max(0, x_data[-1]-100), x_data[-1]+5)
            self.axes['comfort'].set_ylim(0, max(10, max(self.comfort_history)*1.2))
            
        # 更新FL损失图
        if self.fl_loss_history:
            fl_x = list(range(len(self.fl_loss_history)))
            self.lines['fl_loss'].set_data(fl_x, list(self.fl_loss_history))
            self.axes['fl_loss'].set_xlim(0, max(10, len(self.fl_loss_history)))
            self.axes['fl_loss'].set_ylim(0, max(1, max(self.fl_loss_history)*1.2))
            
        # 更新隐私预算图
        if self.privacy_budget_history:
            self.lines['privacy'].set_data(x_data[:len(self.privacy_budget_history)], 
                                          list(self.privacy_budget_history))
            self.axes['privacy'].set_xlim(max(0, x_data[-1]-100), x_data[-1]+5)
            self.axes['privacy'].set_ylim(0, 1.2)
            
        # 更新DRL奖励图
        if self.reward_history:
            self.lines['reward'].set_data(x_data[:len(self.reward_history)], 
                                        list(self.reward_history))
            self.axes['reward'].set_xlim(max(0, x_data[-1]-100), x_data[-1]+5)
            if self.reward_history:
                min_r = min(self.reward_history)
                max_r = max(self.reward_history)
                self.axes['reward'].set_ylim(min_r*1.2, max_r*1.2)
                
        # 更新占用率热力图
        if 'zones' in data:
            self.update_occupancy_heatmap(data['zones'])
            
        # 更新系统指标
        self.update_metrics_panel(data)
        
    def update_occupancy_heatmap(self, zones: List[Dict]):
        """更新占用率热力图"""
        self.axes['occupancy'].clear()
        
        # 创建占用率矩阵
        hours = 24
        occupancy_matrix = np.zeros((self.num_zones, hours))
        
        # 模拟24小时占用率模式
        current_hour = datetime.now().hour
        for i, zone in enumerate(zones[:self.num_zones]):
            base_occ = zone.get('occupancy', 0)
            for h in range(hours):
                if 8 <= h <= 18:  # 工作时间
                    occupancy_matrix[i, h] = base_occ * (1 + np.random.random() * 0.2)
                else:
                    occupancy_matrix[i, h] = base_occ * 0.1
                    
        # 绘制热力图
        sns.heatmap(occupancy_matrix, 
                   ax=self.axes['occupancy'],
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Occupancy'},
                   xticklabels=range(24),
                   yticklabels=[f'Zone {i}' for i in range(self.num_zones)])
        self.axes['occupancy'].set_xlabel('Hour')
        self.axes['occupancy'].set_ylabel('Zone')
        self.axes['occupancy'].axvline(x=current_hour, color='blue', linestyle='--', linewidth=2)
        
    def update_metrics_panel(self, data: Dict):
        """更新指标面板"""
        self.axes['metrics'].clear()
        self.axes['metrics'].axis('off')
        
        # 计算关键指标
        metrics = {
            'Current Time': datetime.now().strftime('%H:%M:%S'),
            'FL Round': data.get('fl_round', 0),
            'Active Devices': data.get('active_devices', 0),
            'Avg Temperature': f"{np.mean([z.get('temperature', 22) for z in data.get('zones', [])[:self.num_zones]]):.1f}°C",
            'Total Power': f"{data.get('total_power', 0):.1f} kW",
            'Energy Saved': f"{max(0, 50 - data.get('total_power', 50)):.1f} kW",
            'Comfort Score': f"{(1 - data.get('comfort_violations', 0)/10)*100:.1f}%",
            'Privacy Budget': f"{data.get('privacy_budget', 0):.3f}/1.0",
        }
        
        # 显示指标
        text = ' | '.join([f'{k}: {v}' for k, v in metrics.items()])
        self.axes['metrics'].text(0.5, 0.5, text, 
                                 horizontalalignment='center',
                                 verticalalignment='center',
                                 fontsize=12,
                                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
    def save_snapshot(self, filename: str = None):
        """保存当前仪表板快照"""
        if filename is None:
            filename = f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        self.fig.savefig(filename, dpi=100, bbox_inches='tight')
        print(f"Dashboard saved to {filename}")
        
    def show(self):
        """显示仪表板"""
        plt.show()
        

class SystemArchitecturePlot:
    """系统架构图生成器"""
    
    @staticmethod
    def create_architecture_diagram():
        """创建系统架构图"""
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # 标题
        ax.text(5, 9.5, 'HFL-HVAC System Architecture', 
                fontsize=20, fontweight='bold', ha='center')
        
        # 云层
        cloud_rect = plt.Rectangle((2, 7), 6, 1.5, 
                                  fill=True, facecolor='lightblue', 
                                  edgecolor='blue', linewidth=2)
        ax.add_patch(cloud_rect)
        ax.text(5, 7.75, 'Cloud Tier', fontsize=14, fontweight='bold', ha='center')
        ax.text(5, 7.4, 'Global Aggregation | DRL Agent | System Optimization', 
                fontsize=10, ha='center')
        
        # 边缘层
        edge_positions = [(1.5, 4.5), (5, 4.5), (8.5, 4.5)]
        for i, pos in enumerate(edge_positions):
            edge_rect = plt.Rectangle((pos[0]-0.8, pos[1]), 1.6, 1.2, 
                                     fill=True, facecolor='lightgreen', 
                                     edgecolor='green', linewidth=2)
            ax.add_patch(edge_rect)
            ax.text(pos[0], pos[1]+0.6, f'Edge {i+1}', 
                   fontsize=11, fontweight='bold', ha='center')
            
        ax.text(5, 4.0, 'Edge Tier: Federated Aggregation', 
                fontsize=12, ha='center', style='italic')
        
        # 设备层
        device_positions = [(x, 1.5) for x in np.linspace(1, 9, 10)]
        for i, pos in enumerate(device_positions):
            device_rect = plt.Rectangle((pos[0]-0.3, pos[1]), 0.6, 0.8, 
                                       fill=True, facecolor='lightyellow', 
                                       edgecolor='orange', linewidth=1.5)
            ax.add_patch(device_rect)
            ax.text(pos[0], pos[1]+0.4, f'D{i}', fontsize=8, ha='center')
            
        ax.text(5, 0.8, 'Device Tier: Local Training & Privacy Protection', 
                fontsize=12, ha='center', style='italic')
        
        # 连接线 - 设备到边缘
        for dev_pos in device_positions[:3]:
            ax.arrow(dev_pos[0], dev_pos[1]+0.8, 
                    edge_positions[0][0]-dev_pos[0], 
                    edge_positions[0][1]-dev_pos[1]-0.8,
                    head_width=0.1, head_length=0.1, 
                    fc='gray', ec='gray', alpha=0.3)
                    
        for dev_pos in device_positions[3:6]:
            ax.arrow(dev_pos[0], dev_pos[1]+0.8, 
                    edge_positions[1][0]-dev_pos[0], 
                    edge_positions[1][1]-dev_pos[1]-0.8,
                    head_width=0.1, head_length=0.1, 
                    fc='gray', ec='gray', alpha=0.3)
                    
        for dev_pos in device_positions[6:]:
            ax.arrow(dev_pos[0], dev_pos[1]+0.8, 
                    edge_positions[2][0]-dev_pos[0], 
                    edge_positions[2][1]-dev_pos[1]-0.8,
                    head_width=0.1, head_length=0.1, 
                    fc='gray', ec='gray', alpha=0.3)
                    
        # 连接线 - 边缘到云
        for edge_pos in edge_positions:
            ax.arrow(edge_pos[0], edge_pos[1]+1.2, 0, 1.3,
                    head_width=0.15, head_length=0.1, 
                    fc='blue', ec='blue', alpha=0.4)
                    
        # 数据流标注
        ax.annotate('Model Updates', xy=(2, 3), xytext=(0.5, 3),
                   arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                   fontsize=9, color='red')
                   
        ax.annotate('Aggregated Models', xy=(5, 6), xytext=(8.5, 6),
                   arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
                   fontsize=9, color='blue')
                   
        ax.annotate('Control Commands', xy=(7, 7), xytext=(9, 8),
                   arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                   fontsize=9, color='green')
                   
        # 图例
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, fc='lightblue', label='Cloud Layer'),
            plt.Rectangle((0, 0), 1, 1, fc='lightgreen', label='Edge Layer'),
            plt.Rectangle((0, 0), 1, 1, fc='lightyellow', label='Device Layer')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_fl_workflow():
        """创建联邦学习工作流程图"""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # 标题
        ax.text(5, 9.5, 'Federated Learning Workflow', 
                fontsize=18, fontweight='bold', ha='center')
        
        # 工作流步骤
        steps = [
            (2, 8, "1. Local\nData Collection"),
            (2, 6.5, "2. Feature\nEngineering"),
            (2, 5, "3. Local\nTraining"),
            (2, 3.5, "4. Privacy\nProtection"),
            (5, 3.5, "5. Edge\nAggregation"),
            (8, 3.5, "6. Global\nAggregation"),
            (8, 5, "7. Model\nDistribution"),
            (8, 6.5, "8. DRL\nOptimization"),
            (5, 8, "9. Control\nExecution")
        ]
        
        # 绘制步骤框
        for x, y, text in steps:
            if '1' in text or '9' in text:
                color = 'lightcoral'
            elif '5' in text or '6' in text:
                color = 'lightgreen'
            else:
                color = 'lightblue'
                
            rect = plt.Rectangle((x-0.6, y-0.4), 1.2, 0.8,
                                fill=True, facecolor=color,
                                edgecolor='black', linewidth=1.5)
            ax.add_patch(rect)
            ax.text(x, y, text, fontsize=10, ha='center', va='center')
            
        # 绘制箭头连接
        arrows = [
            ((2, 7.6), (2, 6.9)),  # 1->2
            ((2, 6.1), (2, 5.4)),  # 2->3
            ((2, 4.6), (2, 3.9)),  # 3->4
            ((2.6, 3.5), (4.4, 3.5)),  # 4->5
            ((5.6, 3.5), (7.4, 3.5)),  # 5->6
            ((8, 3.9), (8, 4.6)),  # 6->7
            ((8, 5.4), (8, 6.1)),  # 7->8
            ((7.4, 6.5), (5.6, 7.6)),  # 8->9
            ((4.4, 8), (2.6, 8)),  # 9->1 (循环)
        ]
        
        for start, end in arrows:
            ax.annotate('', xy=end, xytext=start,
                       arrowprops=dict(arrowstyle='->', lw=1.5, color='darkblue'))
                       
        # 添加时间标注
        ax.text(0.5, 6, '5 min', fontsize=9, color='gray', rotation=90)
        ax.text(3.5, 3.2, '30 min', fontsize=9, color='gray')
        ax.text(6.5, 3.2, '1 hour', fontsize=9, color='gray')
        
        plt.tight_layout()
        return fig


class PerformanceVisualizer:
    """性能可视化器"""
    
    @staticmethod
    def plot_convergence_curves(history_file: str = None):
        """绘制收敛曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('System Performance Convergence', fontsize=16)
        
        # 模拟数据（实际使用时从history_file读取）
        rounds = np.arange(1, 101)
        
        # FL Loss收敛
        fl_loss = 10 * np.exp(-rounds/20) + 0.5 + np.random.normal(0, 0.1, 100)
        axes[0, 0].plot(rounds, fl_loss, 'b-', linewidth=2)
        axes[0, 0].set_title('Federated Learning Loss')
        axes[0, 0].set_xlabel('FL Rounds')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 能耗优化
        energy = 60 - 15 * (1 - np.exp(-rounds/30)) + np.random.normal(0, 2, 100)
        axes[0, 1].plot(rounds, energy, 'r-', linewidth=2)
        axes[0, 1].axhline(y=50, color='green', linestyle='--', label='Target')
        axes[0, 1].set_title('Energy Consumption')
        axes[0, 1].set_xlabel('Time Steps')
        axes[0, 1].set_ylabel('Power (kW)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 舒适度提升
        comfort = 70 + 20 * (1 - np.exp(-rounds/25)) + np.random.normal(0, 1, 100)
        comfort = np.clip(comfort, 0, 100)
        axes[1, 0].plot(rounds, comfort, 'g-', linewidth=2)
        axes[1, 0].axhline(y=90, color='red', linestyle='--', label='Target')
        axes[1, 0].set_title('Comfort Score')
        axes[1, 0].set_xlabel('Time Steps')
        axes[1, 0].set_ylabel('Score (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # DRL奖励
        reward = -2 + 1.5 * (1 - np.exp(-rounds/20)) + np.random.normal(0, 0.2, 100)
        axes[1, 1].plot(rounds, reward, 'c-', linewidth=2)
        axes[1, 1].set_title('DRL Cumulative Reward')
        axes[1, 1].set_xlabel('Episodes')
        axes[1, 1].set_ylabel('Reward')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
        
    @staticmethod
    def plot_privacy_analysis():
        """隐私分析图表"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Privacy Protection Analysis', fontsize=16)
        
        # 隐私预算消耗
        steps = np.arange(100)
        budget = np.cumsum(np.random.exponential(0.01, 100))
        budget = np.minimum(budget, 1.0)
        
        axes[0].plot(steps, budget, 'r-', linewidth=2)
        axes[0].fill_between(steps, 0, budget, alpha=0.3, color='red')
        axes[0].axhline(y=1.0, color='black', linestyle='--', label='Budget Limit')
        axes[0].set_title('Privacy Budget Consumption')
        axes[0].set_xlabel('Training Rounds')
        axes[0].set_ylabel('ε (Epsilon)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 噪声级别分布
        noise_levels = np.random.gamma(2, 0.5, 1000)
        axes[1].hist(noise_levels, bins=50, color='blue', alpha=0.7, edgecolor='black')
        axes[1].set_title('Noise Level Distribution')
        axes[1].set_xlabel('Noise Magnitude')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True, alpha=0.3)
        
        # 模型精度 vs 隐私
        privacy_levels = np.linspace(0.1, 10, 50)
        accuracy = 95 * (1 - np.exp(-privacy_levels/2))
        
        axes[2].plot(privacy_levels, accuracy, 'g-', linewidth=2)
        axes[2].fill_between(privacy_levels, accuracy-5, accuracy+5, alpha=0.2, color='green')
        axes[2].set_title('Privacy-Accuracy Trade-off')
        axes[2].set_xlabel('Privacy Parameter (ε)')
        axes[2].set_ylabel('Model Accuracy (%)')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


if __name__ == "__main__":
    # 测试仪表板
    dashboard = HVACDashboard(num_zones=5)
    
    # 模拟数据更新
    for i in range(100):
        mock_data = {
            'zones': [
                {
                    'temperature': 22 + np.random.randn(),
                    'humidity': 50 + np.random.randn()*5,
                    'co2': 450 + np.random.randn()*50,
                    'occupancy': np.random.randint(0, 20),
                    'power': 5 + np.random.randn()
                }
                for _ in range(5)
            ],
            'total_power': 45 + np.random.randn()*5,
            'comfort_violations': np.random.randint(0, 5),
            'fl_loss': max(0.1, 1.0 - i*0.01 + np.random.randn()*0.1),
            'privacy_budget': min(1.0, i*0.01),
            'drl_reward': -1 + i*0.02 + np.random.randn()*0.1,
            'fl_round': i,
            'active_devices': 5
        }
        
        dashboard.update(mock_data)
        
    # 保存快照
    dashboard.save_snapshot("dashboard_test.png")
    
    # 创建架构图
    arch_fig = SystemArchitecturePlot.create_architecture_diagram()
    arch_fig.savefig("system_architecture.png", dpi=150, bbox_inches='tight')
    
    # 创建工作流图
    workflow_fig = SystemArchitecturePlot.create_fl_workflow()
    workflow_fig.savefig("fl_workflow.png", dpi=150, bbox_inches='tight')
    
    # 创建性能分析图
    perf_fig = PerformanceVisualizer.plot_convergence_curves()
    perf_fig.savefig("performance_convergence.png", dpi=150, bbox_inches='tight')
    
    privacy_fig = PerformanceVisualizer.plot_privacy_analysis()
    privacy_fig.savefig("privacy_analysis.png", dpi=150, bbox_inches='tight')
    
    print("All visualizations saved!")
    
    # 显示仪表板
    # dashboard.show()  # 交互式显示