# HFL-HVAC 项目结构说明

## 📁 完整文件列表及功能说明

```
hfl_hvac_system/
│
├── 📄 README.md                      # 项目主文档，包含系统介绍和使用说明
├── 📄 PROJECT_STRUCTURE.md           # 本文件，项目结构详细说明
├── 📄 requirements.txt               # Python依赖包列表
├── 📄 config.yaml                    # 系统配置文件
├── 📄 docker-compose.yml             # Docker部署配置
├── 📄 __init__.py                    # 包初始化文件
│
├── 📁 simulator/                     # 仿真模块
│   ├── 📄 __init__.py               
│   └── 📄 device_simulator.py        # 建筑和HVAC设备仿真器
│       ├── BuildingZone              # 建筑区域物理模型
│       ├── HVACDeviceSimulator       # HVAC设备仿真
│       ├── WeatherSimulator          # 天气条件生成
│       └── BuildingSimulator         # 建筑整体协调器
│
├── 📁 device_tier/                   # 端设备层
│   ├── 📄 __init__.py               
│   │
│   ├── 📁 data_collector/            # 数据采集模块
│   │   ├── 📄 __init__.py           
│   │   ├── 📄 sensor_interface.py    # 传感器接口和数据采集
│   │   │   ├── SensorInterface       # 传感器抽象基类
│   │   │   ├── SimulatedSensor       # 仿真传感器
│   │   │   ├── DataCollector         # 数据收集器
│   │   │   └── DataQualityChecker    # 数据质量检查
│   │   │
│   │   ├── 📄 data_preprocessor.py   # 特征工程和预处理
│   │   │   ├── DataPreprocessor      # 42维特征提取
│   │   │   └── FeatureEngineering    # 高级特征工程
│   │   │
│   │   └── 📄 privacy_guard.py       # 隐私保护机制
│   │       ├── DifferentialPrivacy   # 差分隐私实现
│   │       ├── DataAnonymizer        # 数据匿名化
│   │       ├── SecureAggregation     # 安全聚合
│   │       └── PrivacyAccountant     # 隐私预算管理
│   │
│   ├── 📁 local_trainer/             # 本地训练模块
│   │   ├── 📄 __init__.py           
│   │   ├── 📄 model_manager.py       # 模型管理
│   │   │   ├── HVACModel             # 神经网络架构(42→128→64→32→4)
│   │   │   └── ModelManager          # 版本控制和存储
│   │   │
│   │   └── 📄 trainer.py             # 训练逻辑
│   │       ├── LocalTrainer          # 基础训练器
│   │       ├── AdaptiveTrainer       # 自适应学习率
│   │       └── FederatedLocalTrainer # 联邦训练器
│   │
│   └── 📁 control/                   # 设备控制模块
│       ├── 📄 __init__.py           
│       └── 📄 hvac_controller.py      # HVAC控制接口
│           ├── SimulatedHVACController # 仿真控制器
│           ├── RealHVACController     # 真实设备接口
│           ├── ControlOptimizer       # 控制优化
│           └── SafetyController       # 安全控制
│
├── 📁 edge_tier/                     # 边缘计算层
│   ├── 📄 __init__.py               
│   │
│   └── 📁 aggregation/               # 聚合模块
│       ├── 📄 __init__.py           
│       └── 📄 federated_aggregator.py # 联邦聚合算法
│           ├── FederatedAggregator    # 基础聚合器
│           │   ├── federated_averaging # FedAvg
│           │   ├── fedprox_aggregation # FedProx
│           │   ├── scaffold_aggregation # SCAFFOLD
│           │   └── fednova_aggregation # FedNova
│           ├── PersonalizedFederatedAggregator # 个性化FL
│           └── ClusteredFederatedAggregator    # 聚类聚合
│
├── 📁 cloud_tier/                    # 云中心层
│   ├── 📄 __init__.py               
│   │
│   └── 📁 drl_agent/                 # DRL智能体
│       ├── 📄 __init__.py           
│       ├── 📄 environment.py         # 强化学习环境
│       │   ├── HVACEnvironment       # 基础环境(56维状态,20维动作)
│       │   └── MultiObjectiveHVACEnvironment # 多目标环境
│       │
│       └── 📄 sac_agent.py           # SAC算法实现
│           ├── ReplayBuffer          # 经验回放
│           ├── GaussianPolicy        # 高斯策略网络
│           ├── QNetwork              # Q值网络
│           ├── SACAgent              # SAC智能体
│           └── HVACController        # 控制接口
│
├── 📁 common/                        # 公共模块
│   ├── 📄 __init__.py               
│   └── 📄 protocol.proto             # gRPC通信协议定义
│       ├── DeviceRegistration        # 设备注册
│       ├── ModelUpdate               # 模型更新
│       ├── AggregatedModel           # 聚合模型
│       ├── ControlCommand            # 控制指令
│       └── DeviceStatus              # 设备状态
│
├── 📁 evaluation/                    # 评估模块
│   ├── 📄 __init__.py               
│   └── 📄 evaluator.py               # 系统评估器
│       └── HVACSystemEvaluator       # 性能评估
│           ├── evaluate_energy_efficiency    # 能效评估
│           ├── evaluate_comfort_performance  # 舒适度评估
│           ├── evaluate_fl_performance       # FL性能评估
│           ├── evaluate_privacy_protection   # 隐私评估
│           └── calculate_economic_benefits   # 经济效益
│
├── 📁 visualization/                 # 可视化模块
│   ├── 📄 __init__.py               
│   └── 📄 dashboard.py               # 监控仪表板
│       ├── HVACDashboard             # 实时监控面板
│       ├── SystemArchitecturePlot    # 架构图生成
│       └── PerformanceVisualizer     # 性能可视化
│
├── 📁 deployment/                    # 部署配置
│   ├── 📄 Dockerfile.device         # 设备层Docker镜像
│   ├── 📄 Dockerfile.edge           # 边缘层Docker镜像
│   ├── 📄 Dockerfile.cloud          # 云层Docker镜像
│   ├── 📄 kubernetes.yaml           # K8s部署配置
│   └── 📄 prometheus.yml            # 监控配置
│
├── 📁 tests/                        # 测试模块
│   ├── 📄 __init__.py              
│   ├── 📄 test_device_tier.py      # 设备层测试
│   ├── 📄 test_edge_tier.py        # 边缘层测试
│   ├── 📄 test_cloud_tier.py       # 云层测试
│   └── 📄 test_integration.py      # 集成测试
│
├── 📁 results/                      # 结果输出
│   ├── 📄 performance_*.json       # 性能数据
│   └── 📄 global_model_*.pt        # 模型文件
│
├── 📁 logs/                         # 日志文件
│   └── 📄 system.log               # 系统日志
│
├── 📁 models/                       # 模型存储
│   └── 📁 device_*/                # 各设备模型
│
└── 📁 images/                       # 生成的图表
    ├── 📄 system_architecture.png  # 系统架构图
    ├── 📄 fl_workflow.png          # FL工作流程
    ├── 📄 dashboard_example.png    # 仪表板示例
    └── 📄 evaluation_report.png    # 评估报告
```

## 🔧 核心执行文件

### 1. **main_system.py** - 主程序入口
- `DeviceTier`: 设备层管理器
- `EdgeTier`: 边缘层管理器
- `CloudTier`: 云层管理器
- `HFLHVACSystem`: 系统协调器

### 2. **generate_visualizations.py** - 生成可视化
- 生成系统架构图
- 创建性能分析图表
- 导出仪表板快照

### 3. **test_fixes.py** - 系统测试
- 测试占用率模拟
- 验证云层评估
- 检查系统修复

## 📊 数据流向

```
1. 数据采集流程：
   simulator → sensor_interface → data_preprocessor → privacy_guard → data_buffer

2. 联邦学习流程：
   local_trainer → edge_aggregator → cloud_aggregator → model_distribution

3. 控制流程：
   building_state → DRL_agent → control_command → hvac_controller → device_execution

4. 评估流程：
   results_data → evaluator → metrics → visualization → report
```

## 🚀 快速使用

### 运行完整系统
```bash
python main_system.py
```

### 生成可视化
```bash
python generate_visualizations.py
```

### 评估系统性能
```bash
python evaluation/evaluator.py
```

### 测试系统功能
```bash
python test_fixes.py
```

## 📈 关键指标

- **特征维度**: 42维输入 → 4维输出
- **网络架构**: [42, 128, 64, 32, 4]
- **FL轮次**: 100轮
- **隐私预算**: ε=1.0
- **通信间隔**: 5分钟数据，1小时FL，10小时DRL
- **性能目标**: 能耗降低30%，舒适度>90%

## 🔐 隐私保护机制

1. **本地差分隐私**: 梯度裁剪 + 高斯噪声
2. **安全聚合**: 掩码加密梯度
3. **数据匿名**: 设备ID哈希化
4. **K-匿名**: 占用率泛化

## 💡 扩展接口

系统预留了以下扩展接口：
- `RealSensor`: 真实传感器接入
- `RealHVACController`: 真实设备控制
- BACnet/Modbus协议支持
- 云平台部署接口

## 📝 配置说明

主要配置文件 `config.yaml` 包含：
- 系统参数配置
- FL训练参数
- DRL算法参数
- 隐私保护参数
- 通信协议配置
- 部署资源配置