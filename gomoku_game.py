"""
五子棋游戏逻辑模块
实现15x15棋盘、落子规则、胜负判定等基础功能
"""

import numpy as np


class GomokuGame:
    """五子棋游戏类"""
    
    def __init__(self, board_size=15):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=np.int8)
        self.current_player = 1  # 1: 黑棋, -1: 白棋
        self.last_move = None
        self.winner = None
        self.game_over = False
        
    def reset(self):
        """重置游戏"""
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1
        self.last_move = None
        self.winner = None
        self.game_over = False
        
    def get_legal_moves(self):
        """获取所有合法落子位置"""
        if self.game_over:
            return []
        return list(zip(*np.where(self.board == 0)))
    
    def make_move(self, row, col):
        """
        落子
        Args:
            row, col: 落子位置
        Returns:
            bool: 是否成功落子
        """
        if self.game_over:
            return False
        
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            return False
        
        if self.board[row, col] != 0:
            return False
        
        self.board[row, col] = self.current_player
        self.last_move = (row, col)
        
        # 检查是否获胜
        if self._check_winner(row, col):
            self.winner = self.current_player
            self.game_over = True
        # 检查是否平局
        elif len(self.get_legal_moves()) == 0:
            self.winner = 0
            self.game_over = True
        else:
            self.current_player = -self.current_player
        
        return True
    
    def _check_winner(self, row, col):
        """
        检查当前落子是否导致胜利
        检查四个方向：横、竖、主对角线、副对角线
        """
        directions = [
            (0, 1),   # 横向
            (1, 0),   # 纵向
            (1, 1),   # 主对角线
            (1, -1)   # 副对角线
        ]
        
        player = self.board[row, col]
        
        for dr, dc in directions:
            count = 1  # 包含当前棋子
            
            # 正方向检查
            r, c = row + dr, col + dc
            while (0 <= r < self.board_size and 
                   0 <= c < self.board_size and 
                   self.board[r, c] == player):
                count += 1
                r += dr
                c += dc
            
            # 反方向检查
            r, c = row - dr, col - dc
            while (0 <= r < self.board_size and 
                   0 <= c < self.board_size and 
                   self.board[r, c] == player):
                count += 1
                r -= dr
                c -= dc
            
            if count >= 5:
                return True
        
        return False
    
    def get_board_state(self):
        """
        获取棋盘状态（用于神经网络输入）
        返回形状: (board_size, board_size, 4)
        通道0: 当前玩家的棋子
        通道1: 对手的棋子
        通道2: 当前玩家标识（全1或全0）
        通道3: 最后一步落子位置
        """
        state = np.zeros((self.board_size, self.board_size, 4), dtype=np.float32)
        
        # 通道0: 当前玩家的棋子
        state[:, :, 0] = (self.board == self.current_player).astype(np.float32)
        
        # 通道1: 对手的棋子
        state[:, :, 1] = (self.board == -self.current_player).astype(np.float32)
        
        # 通道2: 当前玩家标识
        if self.current_player == 1:
            state[:, :, 2] = 1.0
        
        # 通道3: 最后一步落子位置
        if self.last_move is not None:
            state[self.last_move[0], self.last_move[1], 3] = 1.0
        
        return state
    
    def copy(self):
        """复制当前游戏状态"""
        new_game = GomokuGame(self.board_size)
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.last_move = self.last_move
        new_game.winner = self.winner
        new_game.game_over = self.game_over
        return new_game
    
    def get_winner(self):
        """获取获胜者 (1: 黑棋, -1: 白棋, 0: 平局, None: 未结束)"""
        return self.winner
    
    def is_game_over(self):
        """游戏是否结束"""
        return self.game_over
    
    def display(self):
        """在终端显示棋盘"""
        symbols = {0: '·', 1: '●', -1: '○'}
        
        print('\n   ', end='')
        for i in range(self.board_size):
            print(f'{i:2d}', end=' ')
        print()
        
        for i in range(self.board_size):
            print(f'{i:2d} ', end='')
            for j in range(self.board_size):
                print(f' {symbols[self.board[i, j]]} ', end='')
            print()
        print()
