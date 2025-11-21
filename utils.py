"""
工具函数
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(stats, save_path=None):
    """
    绘制训练历史曲线
    Args:
        stats: 训练统计字典
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 总损失
    axes[0].plot(stats['losses'])
    axes[0].set_title('Total Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)
    
    # Value损失
    axes[1].plot(stats['value_losses'])
    axes[1].set_title('Value Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].grid(True)
    
    # Policy损失
    axes[2].plot(stats['policy_losses'])
    axes[2].set_title('Policy Loss')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"训练曲线已保存到: {save_path}")
    else:
        plt.show()


def save_training_stats(stats, filepath):
    """保存训练统计信息"""
    with open(filepath, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"统计信息已保存到: {filepath}")


def load_training_stats(filepath):
    """加载训练统计信息"""
    with open(filepath, 'r') as f:
        stats = json.load(f)
    return stats


def count_model_parameters(model):
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数量: {total:,}")
    print(f"可训练参数: {trainable:,}")
    
    return total, trainable


def get_latest_model(model_dir):
    """获取最新的模型文件"""
    if not os.path.exists(model_dir):
        return None
    
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    
    if not model_files:
        return None
    
    # 优先返回final模型
    if 'model_final.pth' in model_files:
        return os.path.join(model_dir, 'model_final.pth')
    
    # 否则返回最新的迭代模型
    iter_models = [f for f in model_files if 'iter' in f]
    if iter_models:
        iter_models.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        return os.path.join(model_dir, iter_models[-1])
    
    return None


def visualize_policy(policy, board_size=None, save_path=None):
    """
    可视化策略分布
    Args:
        policy: 1D策略向量 (225,)
        board_size: 棋盘大小
        save_path: 保存路径
    """
    if board_size is None:
        board_size = Config.BOARD_SIZE
    
    policy_map = policy.reshape(board_size, board_size)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(policy_map, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Probability')
    plt.title('MCTS Policy Distribution')
    plt.xlabel('Column')
    plt.ylabel('Row')
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"策略分布图已保存到: {save_path}")
    else:
        plt.show()


def print_board_with_policy(game, policy):
    """
    打印带有策略概率的棋盘
    Args:
        game: 游戏状态
        policy: 策略向量或字典
    """
    if isinstance(policy, dict):
        # 转换为向量格式
        policy_vec = np.zeros(game.board_size * game.board_size)
        for (r, c), p in policy.items():
            policy_vec[r * game.board_size + c] = p
        policy = policy_vec
    
    policy_map = policy.reshape(game.board_size, game.board_size)
    
    symbols = {0: '·', 1: '●', -1: '○'}
    
    print('\n   ', end='')
    for i in range(game.board_size):
        print(f'{i:2d}', end=' ')
    print()
    
    for i in range(game.board_size):
        print(f'{i:2d} ', end='')
        for j in range(game.board_size):
            if game.board[i, j] != 0:
                print(f' {symbols[game.board[i, j]]} ', end='')
            else:
                # 显示概率
                prob = policy_map[i, j]
                if prob > 0.01:
                    print(f'{prob:3.0%}', end='')
                else:
                    print(' · ', end='')
        print()
    print()


if __name__ == "__main__":
    # 测试工具函数
    from network import AlphaZeroNet
    
    model = AlphaZeroNet(board_size=Config.BOARD_SIZE, 
                        num_channels=Config.NUM_CHANNELS, 
                        num_res_blocks=Config.NUM_RES_BLOCKS)
    count_model_parameters(model)
    
    # 测试策略可视化
    policy = np.random.random(225)
    policy = policy / policy.sum()
    visualize_policy(policy)
