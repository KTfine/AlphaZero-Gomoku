"""
自对弈训练系统
实现AlphaZero的自对弈数据生成和模型训练流程
"""

import os
import time
import numpy as np
import torch
import torch.optim as optim
from collections import deque
from gomoku_game import GomokuGame
from network import AlphaZeroNet, AlphaZeroLoss
from mcts import MCTS


class SelfPlayBuffer:
    """自对弈数据缓冲区"""
    
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)
        
    def add(self, state, policy, value):
        """添加训练样本"""
        self.buffer.append((state, policy, value))
    
    def sample(self, batch_size):
        """随机采样一批数据（带数据增强）"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        states = np.array([x[0] for x in batch])
        policies = np.array([x[1] for x in batch])
        values = np.array([x[2] for x in batch])
        
        # 数据增强：随机旋转和翻转
        for i in range(len(states)):
            # 随机旋转 0°/90°/180°/270°
            k = np.random.randint(0, 4)
            if k > 0:
                # 旋转状态（所有通道）
                states[i] = np.rot90(states[i], k=k, axes=(0, 1))
                # 旋转策略
                policy_2d = policies[i].reshape(15, 15)
                policy_2d = np.rot90(policy_2d, k=k)
                policies[i] = policy_2d.flatten()
            
            # 随机水平翻转（50%概率）
            if np.random.random() < 0.5:
                # 翻转状态
                states[i] = np.flip(states[i], axis=1).copy()
                # 翻转策略
                policy_2d = policies[i].reshape(15, 15)
                policy_2d = np.flip(policy_2d, axis=1)
                policies[i] = policy_2d.flatten().copy()
        
        return states, policies, values
    
    def __len__(self):
        return len(self.buffer)


class AlphaZeroTrainer:
    """AlphaZero训练器"""
    
    def __init__(self, 
                 board_size=15,
                 num_channels=64,
                 num_res_blocks=3,
                 lr=0.001,
                 weight_decay=1e-4,
                 device='cpu',
                 buffer_size=10000):
        
        self.board_size = board_size
        self.device = device
        
        # 创建神经网络
        self.model = AlphaZeroNet(board_size, num_channels, num_res_blocks).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_fn = AlphaZeroLoss()
        
        # 训练数据缓冲区
        self.buffer = SelfPlayBuffer(max_size=buffer_size)
        
        # 训练统计
        self.training_stats = {
            'episodes': 0,
            'total_steps': 0,
            'losses': [],
            'value_losses': [],
            'policy_losses': []
        }
        
    def self_play(self, num_games=1, num_simulations=50, c_puct=1.0,
                  temperature=1.0, temperature_drop_step=None, verbose=False):
        """
        自对弈生成训练数据
        Args:
            num_games: 对弈局数
            num_simulations: 每步MCTS模拟次数
            c_puct: MCTS探索常数
            temperature: 初始温度参数（用于前若干步）
            temperature_drop_step: 在第多少步之后将温度降为0（贪心），None表示不降温
            verbose: 是否打印详细信息
        """
        
        for game_idx in range(num_games):
            game = GomokuGame(self.board_size)
            game_history = []  # [(state, policy, current_player)]
            
            # 为当前对局创建一个可重用的MCTS实例
            mcts = None
            
            step = 0
            while not game.is_game_over():
                # 获取当前状态
                state = game.get_board_state()
                current_player = game.current_player
                
                # 根据步数选择温度（模仿AlphaZero: 前若干步高温探索，之后温度=0）
                if temperature_drop_step is not None and step >= temperature_drop_step:
                    current_temperature = 0.0
                else:
                    current_temperature = temperature

                # 当温度发生变化时，为了简化实现，重新创建MCTS实例
                if mcts is None or mcts.temperature != current_temperature:
                    mcts = MCTS(self.model,
                                num_simulations=num_simulations,
                                c_puct=c_puct,
                                temperature=current_temperature,
                                root_game=game)

                # MCTS搜索
                move, policy = mcts.get_action(game, return_prob=True)
                
                if move is None:
                    break
                
                # 记录数据
                game_history.append((state, policy, current_player))
                
                # 执行移动
                game.make_move(move[0], move[1])

                # 沿真实落子更新MCTS根节点，实现树重用
                if mcts is not None:
                    mcts.move_root(move)
                step += 1
                
                # if verbose and step % 10 == 0:
                #     print(f"游戏 {game_idx+1}, 步数: {step}")
            
            # 游戏结束，计算每步的价值
            winner = game.get_winner()
            
            if verbose:
                result_str = {1: "黑棋胜", -1: "白棋胜", 0: "平局"}
                print(f"游戏 {game_idx+1} 结束: {result_str.get(winner, '未知')}, 总步数: {step}")
            
            # 将游戏历史加入缓冲区
            for state, policy, player in game_history:
                if winner == 0:
                    value = 0
                else:
                    # 从该步玩家的视角计算价值
                    value = 1 if winner == player else -1
                
                self.buffer.add(state, policy, value)
                self.training_stats['total_steps'] += 1
            
            self.training_stats['episodes'] += 1
        
        if verbose:
            print(f"自对弈完成: {num_games} 局, 缓冲区大小: {len(self.buffer)}")
    
    def train_step(self, batch_size=32):
        """
        训练一步
        Returns:
            loss, value_loss, policy_loss
        """
        if len(self.buffer) < batch_size:
            return None, None, None
        
        # 采样数据
        states, policies, values = self.buffer.sample(batch_size)
        
        # 转换为tensor
        states = torch.FloatTensor(states).to(self.device)
        states = states.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        
        policies = torch.FloatTensor(policies).to(self.device)
        values = torch.FloatTensor(values).unsqueeze(1).to(self.device)
        
        # 前向传播
        self.model.train()
        log_policy, value = self.model(states)
        
        # 计算损失
        loss, value_loss, policy_loss = self.loss_fn(log_policy, value, policies, values)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), value_loss.item(), policy_loss.item()
    
    def train(self, num_epochs=10, batch_size=32, verbose=True):
        """
        训练模型
        Args:
            num_epochs: 训练轮数
            batch_size: 批次大小
        """
        if len(self.buffer) < batch_size:
            print(f"缓冲区数据不足 ({len(self.buffer)} < {batch_size})")
            return
        
        for epoch in range(num_epochs):
            num_batches = len(self.buffer) // batch_size
            epoch_losses = []
            epoch_value_losses = []
            epoch_policy_losses = []
            
            for _ in range(num_batches):
                loss, value_loss, policy_loss = self.train_step(batch_size)
                if loss is not None:
                    epoch_losses.append(loss)
                    epoch_value_losses.append(value_loss)
                    epoch_policy_losses.append(policy_loss)
            
            if epoch_losses:
                avg_loss = np.mean(epoch_losses)
                avg_value_loss = np.mean(epoch_value_losses)
                avg_policy_loss = np.mean(epoch_policy_losses)
                
                self.training_stats['losses'].append(avg_loss)
                self.training_stats['value_losses'].append(avg_value_loss)
                self.training_stats['policy_losses'].append(avg_policy_loss)
                
                if verbose:
                    print(f"Epoch {epoch+1}/{num_epochs}: "
                          f"Loss={avg_loss:.4f}, "
                          f"Value Loss={avg_value_loss:.4f}, "
                          f"Policy Loss={avg_policy_loss:.4f}")
    
    def save_model(self, filepath):
        """保存模型"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats,
            'iteration': self.training_stats['episodes']  # 保存当前迭代次数
        }
        torch.save(checkpoint, filepath)
        # print(f"模型已保存到: {filepath}")  # 减少打印，避免刷屏
    
    def load_model(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint.get('training_stats', self.training_stats)
        print(f"模型已从 {filepath} 加载")
    
    def train_pipeline(self, 
                      num_iterations=10,
                      games_per_iteration=10,
                      train_epochs_per_iteration=5,
                      num_simulations=50,
                      c_puct=1.0,
                      temperature=1.0,
                      temperature_drop_step=None,
                      batch_size=32,
                      save_dir='./models',
                      save_interval=10):
        """
        完整的训练流程
        Args:
            num_iterations: 迭代次数
            games_per_iteration: 每次迭代的自对弈局数
            train_epochs_per_iteration: 每次迭代的训练轮数
            num_simulations: MCTS模拟次数
            batch_size: 批次大小
            save_dir: 模型保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print("=" * 60)
        print("开始 AlphaZero 训练")
        print("=" * 60)
        print(f"迭代次数: {num_iterations}")
        print(f"每次迭代对弈局数: {games_per_iteration}")
        print(f"每次迭代训练轮数: {train_epochs_per_iteration}")
        print(f"MCTS模拟次数: {num_simulations}")
        print(f"批次大小: {batch_size}")
        print("=" * 60)
        
        for iteration in range(num_iterations):
            print(f"\n【迭代 {iteration+1}/{num_iterations}】")
            
            # 自对弈
            print(f"正在进行自对弈 ({games_per_iteration} 局)...")
            start_time = time.time()
            self.self_play(num_games=games_per_iteration, 
                          num_simulations=num_simulations,
                          c_puct=c_puct,
                          temperature=temperature,
                          temperature_drop_step=temperature_drop_step,
                          verbose=True)
            self_play_time = time.time() - start_time
            print(f"自对弈耗时: {self_play_time:.1f}秒")
            
            # 训练
            print(f"\n正在训练模型 ({train_epochs_per_iteration} 轮)...")
            start_time = time.time()
            self.train(num_epochs=train_epochs_per_iteration, 
                      batch_size=batch_size,
                      verbose=True)
            train_time = time.time() - start_time
            print(f"训练耗时: {train_time:.1f}秒")
            
            # 保存模型
            # 1. 每次迭代都保存latest.pth（用于中断恢复）
            latest_path = os.path.join(save_dir, 'model_latest.pth')
            self.save_model(latest_path)
            
            # 2. 定期保存检查点（用于版本对比）
            if (iteration + 1) % save_interval == 0:
                checkpoint_path = os.path.join(save_dir, f'model_iter_{iteration+1}.pth')
                self.save_model(checkpoint_path)
                print(f"✓ 检查点已保存: {checkpoint_path}")
            
            # 打印统计信息
            print(f"\n统计: 总对弈局数={self.training_stats['episodes']}, "
                  f"总步数={self.training_stats['total_steps']}, "
                  f"缓冲区大小={len(self.buffer)}")
        
        # 保存最终模型
        final_path = os.path.join(save_dir, 'model_final.pth')
        self.save_model(final_path)
        
        print("\n" + "=" * 60)
        print("训练完成！")
        print("=" * 60)


if __name__ == "__main__":
    # 快速测试训练流程
    print("测试训练系统...")
    
    trainer = AlphaZeroTrainer(
        board_size=15,
        num_channels=64,
        num_res_blocks=3,
        lr=0.001
    )
    
    # 小规模测试
    trainer.train_pipeline(
        num_iterations=2,
        games_per_iteration=2,
        train_epochs_per_iteration=2,
        num_simulations=20,
        batch_size=16,
        save_dir='./test_models'
    )
