"""
轻量级神经网络模块
5层卷积网络 + Policy Head + Value Head
参数量约 30-50万，适合CPU训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class AlphaZeroNet(nn.Module):
    """
    轻量级 AlphaZero 网络
    
    输入: (batch, 4, 15, 15)
        - 通道0: 当前玩家的棋子
        - 通道1: 对手的棋子
        - 通道2: 当前玩家标识
        - 通道3: 最后一步落子位置
    
    输出:
        - policy: (batch, 225) - 每个位置的落子概率
        - value: (batch, 1) - 当前局面的胜率评估 [-1, 1]
    """
    
    def __init__(self, board_size=15, num_channels=64, num_res_blocks=3):
        super(AlphaZeroNet, self).__init__()
        
        self.board_size = board_size
        self.action_size = board_size * board_size
        
        # 初始卷积层
        self.conv_input = nn.Conv2d(4, num_channels, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_channels)
        
        # 残差块（轻量版使用3个）
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])
        
        # Policy Head
        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, self.action_size)
        
        # Value Head
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        """
        前向传播
        Args:
            x: (batch, 4, board_size, board_size)
        Returns:
            policy: (batch, action_size)
            value: (batch, 1)
        """
        # 主干网络
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Policy Head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.reshape(policy.size(0), -1)
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)
        
        # Value Head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.reshape(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value
    
    def predict(self, board_state):
        """
        预测单个棋盘状态
        Args:
            board_state: numpy array (board_size, board_size, 4)
        Returns:
            policy: numpy array (action_size,) - 概率分布
            value: float - 胜率评估
        """
        self.eval()
        with torch.no_grad():
            # 转换为 PyTorch tensor
            state_tensor = torch.FloatTensor(board_state).unsqueeze(0)
            # 调整维度: (1, 4, board_size, board_size)
            state_tensor = state_tensor.permute(0, 3, 1, 2)
            
            # 移到模型所在设备
            device = next(self.parameters()).device
            state_tensor = state_tensor.to(device)
            
            log_policy, value = self.forward(state_tensor)
            policy = torch.exp(log_policy).squeeze(0).cpu().numpy()
            value = value.item()
            
        return policy, value


class AlphaZeroLoss(nn.Module):
    """
    AlphaZero 损失函数
    Loss = (z - v)^2 - π^T * log(p) + c||θ||^2
    其中:
        z: 真实游戏结果 (1, -1, 0)
        v: 网络预测值
        π: MCTS搜索得到的策略
        p: 网络预测的策略
    """
    
    def __init__(self):
        super(AlphaZeroLoss, self).__init__()
        
    def forward(self, log_policy, value, target_policy, target_value):
        """
        Args:
            log_policy: (batch, action_size) - 网络预测的log概率
            value: (batch, 1) - 网络预测的价值
            target_policy: (batch, action_size) - MCTS搜索的策略
            target_value: (batch, 1) - 真实游戏结果
        """
        # Value loss (MSE)
        value_loss = F.mse_loss(value, target_value)
        
        # Policy loss (Cross Entropy)
        policy_loss = -torch.mean(torch.sum(target_policy * log_policy, dim=1))
        
        # Total loss
        total_loss = 0.7 *value_loss + policy_loss
        
        return total_loss, value_loss, policy_loss


def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 测试网络
    model = AlphaZeroNet(board_size=15, num_channels=64, num_res_blocks=3)
    print(f"模型参数量: {count_parameters(model):,}")
    
    # 测试前向传播
    batch_size = 4
    test_input = torch.randn(batch_size, 4, 15, 15)
    log_policy, value = model(test_input)
    
    print(f"输入形状: {test_input.shape}")
    print(f"Policy输出形状: {log_policy.shape}")
    print(f"Value输出形状: {value.shape}")
    print(f"Value范围: [{value.min().item():.3f}, {value.max().item():.3f}]")
