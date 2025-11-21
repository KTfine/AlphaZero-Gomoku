"""
蒙特卡洛树搜索 (MCTS) 模块
实现 AlphaZero 风格的 MCTS，结合神经网络进行搜索
"""

import numpy as np
import math
from collections import defaultdict


class MCTSNode:
    """MCTS 树节点"""
    
    def __init__(self, game, parent=None, move=None, prior=0):
        self.game = game  # 游戏状态
        self.parent = parent
        self.move = move  # 从父节点到此节点的移动
        self.prior = prior  # 先验概率（来自神经网络）
        
        self.children = {}  # {move: MCTSNode}
        self.visit_count = 0
        self.total_value = 0
        self.mean_value = 0
        
    def is_leaf(self):
        """是否为叶子节点"""
        return len(self.children) == 0
    
    def is_root(self):
        """是否为根节点"""
        return self.parent is None
    
    def get_value(self, c_puct=1.0):
        """
        计算节点的UCT值
        UCT = Q + U
        Q = mean_value (平均价值)
        U = c_puct * P * sqrt(parent_visits) / (1 + visit_count)
        """
        if self.visit_count == 0:
            q_value = 0
        else:
            q_value = self.mean_value
        
        # UCT公式
        u_value = c_puct * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        
        return q_value + u_value
    
    def select_child(self, c_puct=1.0):
        """选择UCT值最大的子节点"""
        return max(self.children.values(), key=lambda node: node.get_value(c_puct))
    
    def expand(self, policy):
        """
        扩展节点
        Args:
            policy: dict {move: probability}
        """
        for move, prob in policy.items():
            if move not in self.children:
                new_game = self.game.copy()
                new_game.make_move(move[0], move[1])
                self.children[move] = MCTSNode(new_game, parent=self, move=move, prior=prob)
    
    def update(self, value):
        """
        更新节点统计信息
        Args:
            value: 从当前玩家视角的价值 [-1, 1]
        """
        self.visit_count += 1
        self.total_value += value
        self.mean_value = self.total_value / self.visit_count
    
    def backup(self, value):
        """回溯更新从此节点到根节点的路径"""
        node = self
        while node is not None:
            node.update(value)
            value = -value  # 切换视角
            node = node.parent


class MCTS:
    """蒙特卡洛树搜索"""
    
    def __init__(self, model, num_simulations=50, c_puct=1.0, temperature=1.0, root_game=None):
        """
        Args:
            model: 神经网络模型
            num_simulations: 每次搜索的模拟次数
            c_puct: UCT公式中的探索常数
            temperature: 温度参数，控制策略的随机性
            root_game: 可选，用于树重用的根节点游戏状态。如果提供，将在此状态上构建/复用根节点。
        """
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature

        # 树重用相关：当前根节点；若不使用树重用，可以保持为 None
        self.root = None
        if root_game is not None:
            self.root = MCTSNode(root_game.copy())

    def move_root(self, move):
        """
        在真实对局落子后，沿对应子节点移动根节点，实现树重用。
        若该子节点不存在，则丢弃旧树并清空根节点。
        """
        if self.root is not None and move in self.root.children:
            self.root = self.root.children[move]
            self.root.parent = None
        else:
            self.root = None
        
    def search(self, game):
        """
        执行MCTS搜索
        Args:
            game: 当前游戏状态
        Returns:
            policy: dict {move: probability} 搜索得到的策略
        """
        # 若尚未有根节点（例如新对局），以当前游戏状态创建根
        if self.root is None:
            self.root = MCTSNode(game.copy())
        root = self.root
        
        # 评估根节点
        policy_probs, _ = self._evaluate(root.game)
        root.expand(policy_probs)
        
        # 执行多次模拟
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            
            # 1. Selection: 选择到叶子节点
            while not node.is_leaf() and not node.game.is_game_over():
                node = node.select_child(self.c_puct)
                search_path.append(node)
            
            # 2. Expansion & Evaluation
            value = 0
            if not node.game.is_game_over():
                # 扩展节点
                policy_probs, value = self._evaluate(node.game)
                node.expand(policy_probs)
            else:
                # 游戏结束，使用真实结果
                winner = node.game.get_winner()
                if winner == 0:
                    value = 0
                else:
                    # 从当前玩家视角
                    value = 1 if winner == node.game.current_player else -1
            
            # 3. Backup: 回溯更新
            for node in reversed(search_path):
                node.update(value)
                value = -value
        
        # 根据访问次数生成策略
        return self._get_policy(root)
    
    def _evaluate(self, game):
        """
        使用神经网络评估局面
        Returns:
            policy: dict {move: probability}
            value: float
        """
        board_state = game.get_board_state()
        policy_logits, value = self.model.predict(board_state)
        
        # 只保留合法移动
        legal_moves = game.get_legal_moves()
        policy = {}
        
        if len(legal_moves) > 0:
            # 提取合法移动的概率
            legal_probs = []
            for move in legal_moves:
                idx = move[0] * game.board_size + move[1]
                legal_probs.append(policy_logits[idx])
            
            # 归一化
            legal_probs = np.array(legal_probs)
            legal_probs = legal_probs / (legal_probs.sum() + 1e-8)
            
            for move, prob in zip(legal_moves, legal_probs):
                policy[move] = prob
        
        return policy, value
    
    def _get_policy(self, root):
        """
        根据访问次数生成策略
        使用温度参数控制随机性
        """
        moves = []
        visits = []
        
        for move, child in root.children.items():
            moves.append(move)
            visits.append(child.visit_count)
        
        if len(moves) == 0:
            return {}
        
        visits = np.array(visits, dtype=np.float32)
        
        if self.temperature == 0:
            # 贪心选择
            probs = np.zeros_like(visits)
            probs[np.argmax(visits)] = 1.0
        else:
            # 使用温度调节
            visits = visits ** (1.0 / self.temperature)
            probs = visits / visits.sum()
        
        policy = {}
        for move, prob in zip(moves, probs):
            policy[move] = prob
        
        return policy
    
    def get_action(self, game, return_prob=False):
        """
        获取MCTS推荐的行动
        Args:
            game: 当前游戏状态
            return_prob: 是否返回完整的策略分布
        Returns:
            move: (row, col) 推荐的落子位置
            policy: (可选) 完整的策略分布
        """
        policy = self.search(game)
        
        if len(policy) == 0:
            return None
        
        # 根据策略采样
        moves = list(policy.keys())
        probs = np.array(list(policy.values()))
        
        # 确保概率和为1（避免数值精度问题）
        probs = probs / probs.sum()
        
        move = moves[np.random.choice(len(moves), p=probs)]
        
        if return_prob:
            # 转换为完整的225维向量
            policy_vec = np.zeros(game.board_size * game.board_size)
            for m, p in policy.items():
                idx = m[0] * game.board_size + m[1]
                policy_vec[idx] = p
            return move, policy_vec
        
        return move


class PureMCTS:
    """纯MCTS（不使用神经网络，用于对比测试）"""
    
    def __init__(self, num_simulations=100, c_puct=1.4):
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        
    def get_action(self, game):
        """随机模拟的纯MCTS"""
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return None
        
        move_stats = defaultdict(lambda: {'wins': 0, 'visits': 0})
        
        for _ in range(self.num_simulations):
            move = legal_moves[np.random.choice(len(legal_moves))]
            
            # 模拟游戏
            sim_game = game.copy()
            sim_game.make_move(move[0], move[1])
            
            # 随机走到游戏结束
            while not sim_game.is_game_over():
                sim_moves = sim_game.get_legal_moves()
                if not sim_moves:
                    break
                sim_move = sim_moves[np.random.choice(len(sim_moves))]
                sim_game.make_move(sim_move[0], sim_move[1])
            
            # 更新统计
            winner = sim_game.get_winner()
            move_stats[move]['visits'] += 1
            if winner == game.current_player:
                move_stats[move]['wins'] += 1
        
        # 选择胜率最高的移动
        best_move = max(legal_moves, 
                       key=lambda m: move_stats[m]['wins'] / max(move_stats[m]['visits'], 1))
        
        return best_move


if __name__ == "__main__":
    from gomoku_game import GomokuGame
    
    # 测试纯MCTS
    game = GomokuGame()
    mcts = PureMCTS(num_simulations=50)
    
    print("测试纯MCTS...")
    for i in range(5):
        move = mcts.get_action(game)
        print(f"第{i+1}步: {move}")
        game.make_move(move[0], move[1])
        game.display()
