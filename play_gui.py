"""
对战GUI界面
使用pygame实现五子棋对战界面，支持人机对战和AI对战
"""

import sys
import pygame
import numpy as np
from gomoku_game import GomokuGame
from mcts import MCTS, PureMCTS
from config import Config


class GomokuGUI:
    """五子棋GUI界面"""
    
    def __init__(self, board_size=None, model=None):
        pygame.init()
        
        self.board_size = board_size if board_size is not None else Config.BOARD_SIZE
        self.cell_size = 40
        self.margin = 50
        self.board_width = self.cell_size * (board_size - 1)
        self.window_size = self.board_width + 2 * self.margin
        
        # 创建窗口
        self.screen = pygame.display.set_mode((self.window_size, self.window_size + 100))
        pygame.display.set_caption("AlphaZero 五子棋")
        
        # 颜色
        self.BG_COLOR = (220, 179, 92)
        self.LINE_COLOR = (0, 0, 0)
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.HIGHLIGHT = (255, 0, 0)
        self.TEXT_COLOR = (50, 50, 50)
        
        # 游戏状态
        self.game = GomokuGame(board_size)
        self.model = model
        
        # AI玩家
        if model is not None:
            self.ai = MCTS(model, num_simulations=50, temperature=0)
        else:
            self.ai = PureMCTS(num_simulations=100)
        
        # 游戏模式
        self.ai_vs_ai = False
        self.human_player = 1  # 1: 黑棋(先手), -1: 白棋(后手)
        
        # 字体
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
    def draw_board(self):
        """绘制棋盘"""
        self.screen.fill(self.BG_COLOR)
        
        # 绘制网格线
        for i in range(self.board_size):
            # 横线
            start_pos = (self.margin, self.margin + i * self.cell_size)
            end_pos = (self.margin + self.board_width, self.margin + i * self.cell_size)
            pygame.draw.line(self.screen, self.LINE_COLOR, start_pos, end_pos, 2)
            
            # 竖线
            start_pos = (self.margin + i * self.cell_size, self.margin)
            end_pos = (self.margin + i * self.cell_size, self.margin + self.board_width)
            pygame.draw.line(self.screen, self.LINE_COLOR, start_pos, end_pos, 2)
        
        # 绘制星位（天元和四个角的星）
        star_positions = [(3, 3), (3, 11), (11, 3), (11, 11), (7, 7)]
        for row, col in star_positions:
            x = self.margin + col * self.cell_size
            y = self.margin + row * self.cell_size
            pygame.draw.circle(self.screen, self.LINE_COLOR, (x, y), 5)
    
    def draw_pieces(self):
        """绘制棋子"""
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.game.board[row, col] != 0:
                    x = self.margin + col * self.cell_size
                    y = self.margin + row * self.cell_size
                    
                    color = self.BLACK if self.game.board[row, col] == 1 else self.WHITE
                    pygame.draw.circle(self.screen, color, (x, y), self.cell_size // 2 - 2)
                    
                    # 给白棋添加黑色边框
                    if color == self.WHITE:
                        pygame.draw.circle(self.screen, self.BLACK, (x, y), self.cell_size // 2 - 2, 2)
        
        # 高亮最后一步
        if self.game.last_move is not None:
            row, col = self.game.last_move
            x = self.margin + col * self.cell_size
            y = self.margin + row * self.cell_size
            pygame.draw.circle(self.screen, self.HIGHLIGHT, (x, y), 6)
    
    def draw_info(self):
        """绘制游戏信息"""
        y_offset = self.window_size - 80
        
        if self.game.is_game_over():
            if self.game.winner == 1:
                text = "黑棋获胜！"
            elif self.game.winner == -1:
                text = "白棋获胜！"
            else:
                text = "平局！"
            text += " 按R重新开始"
        else:
            if self.game.current_player == 1:
                text = "当前: 黑棋"
            else:
                text = "当前: 白棋"
            
            if not self.ai_vs_ai:
                if self.game.current_player == self.human_player:
                    text += " (您的回合)"
                else:
                    text += " (AI思考中...)"
        
        text_surface = self.font.render(text, True, self.TEXT_COLOR)
        text_rect = text_surface.get_rect(center=(self.window_size // 2, y_offset))
        self.screen.blit(text_surface, text_rect)
        
        # 控制说明
        help_text = "R: 重新开始 | A: AI对战 | H: 人机对战 | Q: 退出"
        help_surface = self.small_font.render(help_text, True, self.TEXT_COLOR)
        help_rect = help_surface.get_rect(center=(self.window_size // 2, y_offset + 35))
        self.screen.blit(help_surface, help_rect)
    
    def get_board_pos(self, mouse_pos):
        """将鼠标位置转换为棋盘坐标"""
        x, y = mouse_pos
        
        # 计算最近的交叉点
        col = round((x - self.margin) / self.cell_size)
        row = round((y - self.margin) / self.cell_size)
        
        # 检查是否在棋盘范围内
        if 0 <= row < self.board_size and 0 <= col < self.board_size:
            # 检查是否靠近交叉点
            actual_x = self.margin + col * self.cell_size
            actual_y = self.margin + row * self.cell_size
            
            distance = np.sqrt((x - actual_x)**2 + (y - actual_y)**2)
            
            if distance < self.cell_size / 2:
                return (row, col)
        
        return None
    
    def ai_move(self):
        """AI落子"""
        if self.game.is_game_over():
            return
        
        move = self.ai.get_action(self.game)
        if move is not None:
            self.game.make_move(move[0], move[1])
    
    def reset_game(self):
        """重置游戏"""
        self.game.reset()
    
    def run(self):
        """运行游戏主循环"""
        clock = pygame.time.Clock()
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.reset_game()
                    elif event.key == pygame.K_a:
                        self.ai_vs_ai = True
                        self.reset_game()
                        print("切换到AI对战模式")
                    elif event.key == pygame.K_h:
                        self.ai_vs_ai = False
                        self.reset_game()
                        print("切换到人机对战模式")
                    elif event.key == pygame.K_b:
                        self.human_player = -self.human_player
                        self.reset_game()
                        role = "黑棋(先手)" if self.human_player == 1 else "白棋(后手)"
                        print(f"切换为人类执{role}")
                    elif event.key == pygame.K_q:
                        running = False
                
                elif event.type == pygame.MOUSEBUTTONDOWN and not self.game.is_game_over():
                    if not self.ai_vs_ai and self.game.current_player == self.human_player:
                        pos = self.get_board_pos(pygame.mouse.get_pos())
                        if pos is not None:
                            row, col = pos
                            if self.game.make_move(row, col):
                                print(f"玩家落子: ({row}, {col})")
            
            # AI自动落子
            if not self.game.is_game_over():
                if self.ai_vs_ai:
                    # AI对战模式
                    pygame.time.delay(500)  # 延迟以便观看
                    self.ai_move()
                elif self.game.current_player != self.human_player:
                    # 人机对战，AI回合
                    self.ai_move()
            
            # 绘制
            self.draw_board()
            self.draw_pieces()
            self.draw_info()
            
            pygame.display.flip()
            clock.tick(30)
        
        pygame.quit()


def play_with_model(model_path=None):
    """
    使用训练好的模型进行对战
    Args:
        model_path: 模型文件路径，如果为None则使用纯MCTS
    """
    model = None
    
    if model_path is not None:
        import torch
        from network import AlphaZeroNet
        
        print(f"加载模型: {model_path}")
        model = AlphaZeroNet(board_size=Config.BOARD_SIZE, 
                            num_channels=Config.NUM_CHANNELS, 
                            num_res_blocks=Config.NUM_RES_BLOCKS)
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("模型加载成功！")
    else:
        print("使用纯MCTS（未加载神经网络）")
    
    # 启动GUI
    gui = GomokuGUI(board_size=Config.BOARD_SIZE, model=model)
    gui.run()


if __name__ == "__main__":
    # 可以传入模型路径
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        play_with_model(model_path)
    else:
        # 默认使用纯MCTS
        play_with_model(None)
