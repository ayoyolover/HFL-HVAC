"""
Soft Actor-Critic (SAC) 算法实现
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from typing import Tuple, Dict, Optional
import copy
from collections import deque
import random
import logging

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int = 1000000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        """添加经验"""
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int):
        """采样批次"""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)


class GaussianPolicy(nn.Module):
    """高斯策略网络"""
    
    def __init__(self, state_dim: int, action_dim: int, 
                 hidden_dims: list = [256, 256],
                 log_std_min: float = -20, 
                 log_std_max: float = 2):
        super(GaussianPolicy, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # 构建网络
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
            
        self.shared_net = nn.Sequential(*layers)
        
        # 均值和标准差输出头
        self.mean_head = nn.Linear(prev_dim, action_dim)
        self.log_std_head = nn.Linear(prev_dim, action_dim)
        
    def forward(self, state):
        """前向传播"""
        x = self.shared_net(state)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state, epsilon: float = 1e-6):
        """采样动作"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # 重参数化技巧
        normal = Normal(mean, std)
        x_t = normal.rsample()  # 重参数化采样
        action = torch.tanh(x_t)
        
        # 计算对数概率
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob, torch.tanh(mean)
    
    def get_action(self, state):
        """获取确定性动作（用于评估）"""
        state = torch.FloatTensor(state).unsqueeze(0)
        mean, _ = self.forward(state)
        return torch.tanh(mean).cpu().data.numpy().flatten()


class QNetwork(nn.Module):
    """Q网络"""
    
    def __init__(self, state_dim: int, action_dim: int, 
                 hidden_dims: list = [256, 256]):
        super(QNetwork, self).__init__()
        
        layers = []
        prev_dim = state_dim + action_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, 1))
        
        self.q_net = nn.Sequential(*layers)
        
    def forward(self, state, action):
        """前向传播"""
        x = torch.cat([state, action], dim=1)
        return self.q_net(x)


class SACAgent:
    """SAC智能体"""
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: list = [256, 256],
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 alpha: float = 0.2,
                 auto_entropy_tuning: bool = True,
                 device: str = 'cpu'):
        """
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dims: 隐藏层维度
            lr: 学习率
            gamma: 折扣因子
            tau: 软更新系数
            alpha: 温度参数
            auto_entropy_tuning: 是否自动调节温度
            device: 计算设备
        """
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.action_dim = action_dim
        
        # 策略网络
        self.policy = GaussianPolicy(state_dim, action_dim, hidden_dims).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Q网络（双Q网络）
        self.q1 = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)
        
        # 目标Q网络
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        
        # 自动熵调节
        self.auto_entropy_tuning = auto_entropy_tuning
        if auto_entropy_tuning:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            
        # 经验回放
        self.replay_buffer = ReplayBuffer()
        
        # 训练统计
        self.training_step = 0
        self.losses = {
            'q1_loss': [],
            'q2_loss': [],
            'policy_loss': [],
            'alpha_loss': []
        }
        
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """选择动作"""
        if evaluate:
            # 评估模式：使用确定性策略
            return self.policy.get_action(state)
        else:
            # 训练模式：从策略分布采样
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, _, _ = self.policy.sample(state)
            return action.cpu().data.numpy().flatten()
            
    def update(self, batch_size: int = 256) -> Dict[str, float]:
        """更新网络"""
        if len(self.replay_buffer) < batch_size:
            return {}
            
        # 采样批次
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)
        
        # 更新Q网络
        with torch.no_grad():
            next_action, next_log_prob, _ = self.policy.sample(next_state)
            q1_next_target = self.q1_target(next_state, next_action)
            q2_next_target = self.q2_target(next_state, next_action)
            min_q_next_target = torch.min(q1_next_target, q2_next_target)
            
            # Bellman方程目标
            q_target = reward + (1 - done) * self.gamma * (min_q_next_target - self.alpha * next_log_prob)
            
        # Q1损失
        q1_pred = self.q1(state, action)
        q1_loss = F.mse_loss(q1_pred, q_target)
        
        # Q2损失
        q2_pred = self.q2(state, action)
        q2_loss = F.mse_loss(q2_pred, q_target)
        
        # 更新Q网络
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        
        # 更新策略网络
        action_new, log_prob, _ = self.policy.sample(state)
        q1_new = self.q1(state, action_new)
        q2_new = self.q2(state, action_new)
        min_q_new = torch.min(q1_new, q2_new)
        
        policy_loss = (self.alpha * log_prob - min_q_new).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # 更新温度参数
        alpha_loss = None
        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
            
        # 软更新目标网络
        self._soft_update(self.q1_target, self.q1)
        self._soft_update(self.q2_target, self.q2)
        
        # 记录损失
        losses = {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'policy_loss': policy_loss.item()
        }
        
        if alpha_loss is not None:
            losses['alpha_loss'] = alpha_loss.item()
            losses['alpha'] = self.alpha.item()
            
        self.training_step += 1
        
        return losses
    
    def _soft_update(self, target: nn.Module, source: nn.Module):
        """软更新目标网络"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
            
    def save(self, filepath: str):
        """保存模型"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'q1_state_dict': self.q1.state_dict(),
            'q2_state_dict': self.q2.state_dict(),
            'q1_target_state_dict': self.q1_target.state_dict(),
            'q2_target_state_dict': self.q2_target.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'q1_optimizer_state_dict': self.q1_optimizer.state_dict(),
            'q2_optimizer_state_dict': self.q2_optimizer.state_dict(),
            'training_step': self.training_step,
            'alpha': self.alpha
        }, filepath)
        logger.info(f"Model saved to {filepath}")
        
    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.q1.load_state_dict(checkpoint['q1_state_dict'])
        self.q2.load_state_dict(checkpoint['q2_state_dict'])
        self.q1_target.load_state_dict(checkpoint['q1_target_state_dict'])
        self.q2_target.load_state_dict(checkpoint['q2_target_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.q1_optimizer.load_state_dict(checkpoint['q1_optimizer_state_dict'])
        self.q2_optimizer.load_state_dict(checkpoint['q2_optimizer_state_dict'])
        self.training_step = checkpoint['training_step']
        self.alpha = checkpoint['alpha']
        
        logger.info(f"Model loaded from {filepath}")


class HVACController:
    """HVAC控制器（集成SAC智能体）"""
    
    def __init__(self, num_zones: int = 10, 
                 model_path: Optional[str] = None):
        """
        Args:
            num_zones: 控制区域数量
            model_path: 预训练模型路径
        """
        self.num_zones = num_zones
        
        # 状态和动作维度
        state_dim = num_zones * 5 + 6  # 每个区域5个特征 + 6个全局特征
        action_dim = num_zones * 2  # 每个区域2个控制变量
        
        # 创建SAC智能体
        self.agent = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=[256, 256],
            lr=3e-4,
            gamma=0.99,
            tau=0.005,
            alpha=0.2
        )
        
        # 加载预训练模型
        if model_path and os.path.exists(model_path):
            self.agent.load(model_path)
            
        # 控制历史
        self.control_history = deque(maxlen=100)
        
    def get_control_action(self, state: Dict) -> Dict:
        """获取控制动作"""
        # 将状态字典转换为向量
        state_vector = self._state_dict_to_vector(state)
        
        # 获取动作
        action = self.agent.select_action(state_vector, evaluate=True)
        
        # 将动作向量转换为控制指令
        control_commands = self._action_to_control_commands(action)
        
        # 记录历史
        self.control_history.append({
            'state': state,
            'action': action,
            'commands': control_commands
        })
        
        return control_commands
    
    def _state_dict_to_vector(self, state: Dict) -> np.ndarray:
        """将状态字典转换为向量"""
        vector = []
        
        # 区域状态
        for zone in state['zones']:
            vector.extend([
                zone['temperature'],
                zone['humidity'],
                zone['co2'] / 1000,
                zone['occupancy'] / 20,
                zone['power'] / 10
            ])
            
        # 全局状态
        vector.extend([
            state['weather']['outdoor_temp'],
            state['weather']['outdoor_humidity'],
            state['weather']['solar_radiation'] / 1000,
            state['energy_price'],
            np.sin(2 * np.pi * state['hour'] / 24),
            np.cos(2 * np.pi * state['hour'] / 24)
        ])
        
        return np.array(vector, dtype=np.float32)
    
    def _action_to_control_commands(self, action: np.ndarray) -> Dict:
        """将动作向量转换为控制指令"""
        commands = {
            'zones': []
        }
        
        # 解析每个区域的控制
        for i in range(self.num_zones):
            temp_adjustment = action[i] * 2  # ±2°C
            vent_adjustment = action[self.num_zones + i] * 0.5  # 通风率调整
            
            zone_command = {
                'zone_id': f"zone_{i:03d}",
                'temperature_adjustment': float(temp_adjustment),
                'ventilation_adjustment': float(vent_adjustment)
            }
            commands['zones'].append(zone_command)
            
        return commands
    
    def update_from_feedback(self, state: Dict, action: np.ndarray, 
                            reward: float, next_state: Dict, done: bool):
        """根据反馈更新模型"""
        # 转换状态
        state_vector = self._state_dict_to_vector(state)
        next_state_vector = self._state_dict_to_vector(next_state)
        
        # 添加到回放缓冲区
        self.agent.replay_buffer.push(
            state_vector, action, reward, next_state_vector, done
        )
        
        # 更新模型
        if len(self.agent.replay_buffer) > 1000:
            losses = self.agent.update()
            return losses
        
        return {}


if __name__ == "__main__":
    import os
    from environment import HVACEnvironment
    
    # 创建环境和智能体
    env = HVACEnvironment(num_zones=5)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device='cpu'
    )
    
    print(f"Created SAC agent with state_dim={state_dim}, action_dim={action_dim}")
    
    # 训练循环示例
    num_episodes = 10
    max_steps = 100
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # 选择动作
            action = agent.select_action(state)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 存储经验
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # 更新网络
            if len(agent.replay_buffer) > 256:
                losses = agent.update()
                
            episode_reward += reward
            state = next_state
            
            if done:
                break
                
        print(f"Episode {episode+1}: Reward = {episode_reward:.2f}")
        
    # 保存模型
    agent.save("sac_hvac_model.pt")
    print("Model saved!")