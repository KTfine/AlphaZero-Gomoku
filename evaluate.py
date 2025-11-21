"""
评估和对战工具
"""

import torch
import numpy as np
from gomoku_game import GomokuGame
from network import AlphaZeroNet
from mcts import MCTS, PureMCTS
from config import Config


class Evaluator:
    """模型评估器"""
    
    def __init__(self, model1, model2=None, num_simulations=100):
        """
        Args:
            model1: 第一个模型（被评估的模型）
            model2: 第二个模型（基准模型），如果为None则使用纯MCTS
            num_simulations: MCTS模拟次数
        """
        self.model1 = model1
        self.model2 = model2
        self.num_simulations = num_simulations
        
        self.agent1 = MCTS(model1, num_simulations=num_simulations, temperature=0)
        
        if model2 is not None:
            self.agent2 = MCTS(model2, num_simulations=num_simulations, temperature=0)
        else:
            self.agent2 = PureMCTS(num_simulations=num_simulations)
    
    def play_game(self, agent1_first=True, verbose=False):
        """
        两个AI对弈一局
        Args:
            agent1_first: agent1是否先手
            verbose: 是否打印详细信息
        Returns:
            winner: 1(agent1胜), -1(agent2胜), 0(平局)
        """
        game = GomokuGame(Config.BOARD_SIZE)
        
        # 确定哪个agent先手
        if agent1_first:
            agents = {1: self.agent1, -1: self.agent2}
        else:
            agents = {1: self.agent2, -1: self.agent1}
        
        step = 0
        while not game.is_game_over():
            current_agent = agents[game.current_player]
            move = current_agent.get_action(game)
            
            if move is None:
                break
            
            game.make_move(move[0], move[1])
            step += 1
            
            if verbose and step % 20 == 0:
                print(f"步数: {step}")
        
        winner = game.get_winner()
        
        # 转换为agent1的视角
        if agent1_first:
            return winner
        else:
            return -winner if winner != 0 else 0
    
    def evaluate(self, num_games=20, verbose=True):
        """
        评估模型
        Args:
            num_games: 对弈局数（agent1先手和后手各一半）
        Returns:
            win_rate: agent1的胜率
        """
        wins = 0
        losses = 0
        draws = 0
        
        # 一半先手，一半后手
        for i in range(num_games):
            agent1_first = (i % 2 == 0)
            
            if verbose:
                print(f"对局 {i+1}/{num_games}: ", end='')
                print(f"Agent1 {'先手' if agent1_first else '后手'} ... ", end='', flush=True)
            
            result = self.play_game(agent1_first=agent1_first, verbose=False)
            
            if result == 1:
                wins += 1
                if verbose:
                    print("胜")
            elif result == -1:
                losses += 1
                if verbose:
                    print("负")
            else:
                draws += 1
                if verbose:
                    print("平")
        
        win_rate = wins / num_games
        
        if verbose:
            print(f"\n评估结果:")
            print(f"  胜: {wins}/{num_games} ({wins/num_games*100:.1f}%)")
            print(f"  负: {losses}/{num_games} ({losses/num_games*100:.1f}%)")
            print(f"  平: {draws}/{num_games} ({draws/num_games*100:.1f}%)")
            print(f"  胜率: {win_rate*100:.1f}%")
        
        return win_rate


def compare_models(model_path1, model_path2=None, num_games=20, num_simulations=100):
    """
    比较两个模型的强度
    Args:
        model_path1: 第一个模型路径
        model_path2: 第二个模型路径（None表示使用纯MCTS）
        num_games: 对弈局数
        num_simulations: MCTS模拟次数
    """
    print("加载模型...")
    
    # 加载模型1
    model1 = AlphaZeroNet(board_size=Config.BOARD_SIZE,
                         num_channels=Config.NUM_CHANNELS,
                         num_res_blocks=Config.NUM_RES_BLOCKS)
    checkpoint1 = torch.load(model_path1, map_location='cpu', weights_only=False)
    model1.load_state_dict(checkpoint1['model_state_dict'])
    model1.eval()
    print(f"模型1已加载: {model_path1}")
    
    # 加载模型2
    model2 = None
    if model_path2 is not None:
        model2 = AlphaZeroNet(board_size=Config.BOARD_SIZE,
                             num_channels=Config.NUM_CHANNELS,
                             num_res_blocks=Config.NUM_RES_BLOCKS)
        checkpoint2 = torch.load(model_path2, map_location='cpu', weights_only=False)
        model2.load_state_dict(checkpoint2['model_state_dict'])
        model2.eval()
        print(f"模型2已加载: {model_path2}")
    else:
        print("模型2: 纯MCTS")
    
    # 评估
    print(f"\n开始评估（{num_games}局对弈）...")
    evaluator = Evaluator(model1, model2, num_simulations=num_simulations)
    win_rate = evaluator.evaluate(num_games=num_games, verbose=True)
    
    return win_rate


def play_against_human(model_path, human_first=True, num_simulations=100):
    """
    人机对战（命令行版本）
    Args:
        model_path: 模型路径
        human_first: 人类是否先手
        num_simulations: MCTS模拟次数
    """
    print("加载模型...")
    model = AlphaZeroNet(board_size=Config.BOARD_SIZE,
                        num_channels=Config.NUM_CHANNELS,
                        num_res_blocks=Config.NUM_RES_BLOCKS)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("模型加载成功！\n")
    
    ai = MCTS(model, num_simulations=num_simulations, temperature=0)
    game = GomokuGame(Config.BOARD_SIZE)
    
    human_player = 1 if human_first else -1
    
    print("=" * 50)
    print("AlphaZero 五子棋 - 人机对战")
    print("=" * 50)
    print(f"你执: {'黑棋 (先手)' if human_first else '白棋 (后手)'}")
    print("输入格式: 行 列 (例如: 7 7)")
    print("输入 'q' 退出")
    print("=" * 50)
    
    game.display()
    
    while not game.is_game_over():
        if game.current_player == human_player:
            # 人类回合
            while True:
                try:
                    user_input = input(f"\n你的回合 ({'●' if human_player == 1 else '○'}): ").strip()
                    
                    if user_input.lower() == 'q':
                        print("游戏结束")
                        return
                    
                    row, col = map(int, user_input.split())
                    
                    if game.make_move(row, col):
                        break
                    else:
                        print("无效落子，请重试")
                except (ValueError, IndexError):
                    print("输入格式错误，请输入: 行 列")
        else:
            # AI回合
            print(f"\nAI思考中 ({'●' if game.current_player == 1 else '○'})...")
            move = ai.get_action(game)
            game.make_move(move[0], move[1])
            print(f"AI落子: {move[0]} {move[1]}")
        
        game.display()
    
    # 游戏结束
    print("\n" + "=" * 50)
    winner = game.get_winner()
    if winner == human_player:
        print("你赢了！🎉")
    elif winner == -human_player:
        print("AI获胜！")
    else:
        print("平局！")
    print("=" * 50)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        
        if len(sys.argv) > 2:
            # 比较两个模型
            model_path2 = sys.argv[2]
            compare_models(model_path, model_path2, num_games=20)
        else:
            # 人机对战
            play_against_human(model_path, human_first=True)
    else:
        print("用法:")
        print("  人机对战: python evaluate.py <模型路径>")
        print("  模型对比: python evaluate.py <模型1路径> <模型2路径>")
