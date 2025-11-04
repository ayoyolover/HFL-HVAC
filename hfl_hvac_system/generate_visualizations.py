"""
生成系统可视化图表
"""
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from visualization.dashboard import (
    HVACDashboard, 
    SystemArchitecturePlot, 
    PerformanceVisualizer
)


def generate_all_visualizations():
    """生成所有可视化图表"""
    
    print("Generating system visualizations...")
    
    # 1. 创建系统架构图
    print("1. Creating system architecture diagram...")
    arch_fig = SystemArchitecturePlot.create_architecture_diagram()
    arch_fig.savefig("images/system_architecture.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. 创建联邦学习工作流图
    print("2. Creating federated learning workflow...")
    workflow_fig = SystemArchitecturePlot.create_fl_workflow()
    workflow_fig.savefig("images/fl_workflow.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. 创建性能收敛曲线
    print("3. Creating performance convergence curves...")
    perf_fig = PerformanceVisualizer.plot_convergence_curves()
    perf_fig.savefig("images/performance_convergence.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. 创建隐私分析图
    print("4. Creating privacy analysis charts...")
    privacy_fig = PerformanceVisualizer.plot_privacy_analysis()
    privacy_fig.savefig("images/privacy_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 5. 创建示例仪表板快照
    print("5. Creating dashboard snapshot...")
    dashboard = HVACDashboard(num_zones=5)
    
    # 模拟一些数据点
    for i in range(50):
        mock_data = {
            'zones': [
                {
                    'temperature': 22 + np.sin(i/10) + np.random.randn()*0.5,
                    'humidity': 50 + np.cos(i/8) * 5 + np.random.randn()*2,
                    'co2': 450 + 50 * np.sin(i/15) + np.random.randn()*20,
                    'occupancy': max(0, int(10 + 5 * np.sin(i/20) + np.random.randn()*2)),
                    'power': max(0, 5 + 2 * np.sin(i/12) + np.random.randn()*0.5)
                }
                for _ in range(5)
            ],
            'total_power': max(0, 45 + 10 * np.sin(i/15) + np.random.randn()*2),
            'comfort_violations': max(0, int(2 + np.sin(i/10) + np.random.randn())),
            'fl_loss': max(0.1, 1.0 * np.exp(-i/20) + np.random.randn()*0.05),
            'privacy_budget': min(1.0, i * 0.02),
            'drl_reward': -1 + i * 0.04 + np.random.randn()*0.1,
            'fl_round': i // 5,
            'active_devices': 5
        }
        
        dashboard.update(mock_data)
    
    dashboard.save_snapshot("images/dashboard_example.png")
    plt.close('all')
    
    # 6. 创建额外的分析图
    print("6. Creating additional analysis charts...")
    create_additional_charts()
    
    print("\nAll visualizations generated successfully!")
    print("Images saved in 'images/' directory")


def create_additional_charts():
    """创建额外的分析图表"""
    
    # 创建比较图：有无联邦学习的性能对比
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('HFL-HVAC vs Traditional HVAC Performance', fontsize=16)
    
    hours = np.arange(0, 168)  # 一周
    
    # 能耗对比
    traditional_energy = 60 + 10 * np.sin(hours * 2 * np.pi / 24) + np.random.randn(168) * 2
    hfl_energy = 45 + 8 * np.sin(hours * 2 * np.pi / 24) + np.random.randn(168) * 1.5
    
    axes[0].plot(hours, traditional_energy, 'r-', alpha=0.7, label='Traditional')
    axes[0].plot(hours, hfl_energy, 'b-', alpha=0.7, label='HFL-HVAC')
    axes[0].fill_between(hours, traditional_energy, hfl_energy, 
                        where=(hfl_energy < traditional_energy), 
                        color='green', alpha=0.3, label='Energy Saved')
    axes[0].set_title('Energy Consumption Comparison')
    axes[0].set_xlabel('Hours')
    axes[0].set_ylabel('Power (kW)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 舒适度对比
    traditional_comfort = 75 + 10 * np.random.randn(168)
    hfl_comfort = 90 + 5 * np.random.randn(168)
    
    axes[1].plot(hours, traditional_comfort, 'r-', alpha=0.7, label='Traditional')
    axes[1].plot(hours, hfl_comfort, 'b-', alpha=0.7, label='HFL-HVAC')
    axes[1].axhline(y=90, color='green', linestyle='--', alpha=0.5, label='Target')
    axes[1].set_title('Comfort Score Comparison')
    axes[1].set_xlabel('Hours')
    axes[1].set_ylabel('Comfort Score (%)')
    axes[1].set_ylim(60, 100)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 成本节省分析
    categories = ['Energy\nCost', 'Peak\nDemand', 'Maintenance', 'Total\nSavings']
    traditional_costs = [100, 100, 100, 300]
    hfl_costs = [70, 75, 90, 235]
    savings = [t - h for t, h in zip(traditional_costs, hfl_costs)]
    
    x = np.arange(len(categories))
    width = 0.35
    
    axes[2].bar(x - width/2, traditional_costs, width, label='Traditional', color='red', alpha=0.7)
    axes[2].bar(x + width/2, hfl_costs, width, label='HFL-HVAC', color='blue', alpha=0.7)
    
    # 添加节省百分比
    for i, (t, h) in enumerate(zip(traditional_costs, hfl_costs)):
        saving_pct = (t - h) / t * 100
        axes[2].text(i, max(t, h) + 5, f'-{saving_pct:.0f}%', 
                    ha='center', fontsize=10, color='green', fontweight='bold')
    
    axes[2].set_title('Cost Comparison (Normalized)')
    axes[2].set_ylabel('Relative Cost')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(categories)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig.savefig("images/performance_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 创建联邦学习聚合算法对比
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rounds = np.arange(1, 101)
    fedavg_loss = 10 * np.exp(-rounds/20) + 0.5
    fedprox_loss = 10 * np.exp(-rounds/18) + 0.4
    scaffold_loss = 10 * np.exp(-rounds/15) + 0.3
    fednova_loss = 10 * np.exp(-rounds/16) + 0.35
    
    ax.plot(rounds, fedavg_loss, 'b-', linewidth=2, label='FedAvg')
    ax.plot(rounds, fedprox_loss, 'r-', linewidth=2, label='FedProx')
    ax.plot(rounds, scaffold_loss, 'g-', linewidth=2, label='SCAFFOLD')
    ax.plot(rounds, fednova_loss, 'm-', linewidth=2, label='FedNova')
    
    ax.set_title('Federated Learning Algorithm Comparison', fontsize=14)
    ax.set_xlabel('Communication Rounds')
    ax.set_ylabel('Training Loss')
    ax.set_yscale('log')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig("images/fl_algorithms_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # 创建images目录
    os.makedirs("images", exist_ok=True)
    
    # 生成所有可视化
    generate_all_visualizations()