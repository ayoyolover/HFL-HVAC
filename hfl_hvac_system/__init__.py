"""
HFL-HVAC: Hierarchical Federated Learning for HVAC Control System
"""

__version__ = "1.0.0"
__author__ = "HFL-HVAC Team"

# 导出主要类
from .main_system import HFLHVACSystem, SystemConfig
from .simulator.device_simulator import BuildingSimulator

__all__ = [
    'HFLHVACSystem',
    'SystemConfig',
    'BuildingSimulator'
]