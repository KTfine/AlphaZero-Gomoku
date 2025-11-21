"""
配置文件
GPU版本的优化配置
"""

import torch


class Config:
    """AlphaZero五子棋配置"""
    
    # ========== 游戏设置 ==========
    BOARD_SIZE = 15
    
    # ========== 网络设置 ==========
    NUM_CHANNELS = 64  # GPU版增加到128通道
    NUM_RES_BLOCKS = 6  # GPU版增加到6个残差块
    
    # ========== MCTS设置 ==========
    NUM_SIMULATIONS = 100  # 增加模拟次数，提升数据质量
    C_PUCT = 1.5  # UCT探索常数
    TEMPERATURE = 1.0  # 温度参数
    TEMPERATURE_DROP_STEP = 15  # 前15步使用温度1.0，之后降为0
    
    # ========== 训练设置 ==========
    # 设备
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 学习率
    LEARNING_RATE = 0.0005  # 降低学习率，提升训练稳定性
    WEIGHT_DECAY = 1e-4
    
    # 批次大小
    BATCH_SIZE = 128  # GPU版大幅增加批次大小
    
    # 缓冲区
    BUFFER_SIZE = 20000  # 增加缓冲区大小
    
    # 训练迭代
    NUM_ITERATIONS = 150  # 总迭代次数
    GAMES_PER_ITERATION = 30  # 每次迭代对弈局数（GPU版增加）
    TRAIN_EPOCHS_PER_ITERATION = 8  # 每次迭代训练轮数
    
    # 模型保存
    SAVE_INTERVAL = 10  # 每10次迭代保存一次
    MODEL_DIR = './models'
    
    # ========== 评估设置 ==========
    EVAL_GAMES = 20  # 评估时对弈局数
    EVAL_SIMULATIONS = 100  # 评估时MCTS模拟次数
    
    # ========== GPU优化设置 ==========
    NUM_WORKERS = 4  # 数据加载线程数
    PIN_MEMORY = True if DEVICE == 'cuda' else False
    
    @classmethod
    def display(cls):
        """显示配置信息"""
        print("=" * 60)
        print("AlphaZero 五子棋配置")
        print("=" * 60)
        print(f"设备: {cls.DEVICE}")
        print(f"棋盘大小: {cls.BOARD_SIZE}x{cls.BOARD_SIZE}")
        print(f"\n网络:")
        print(f"  通道数: {cls.NUM_CHANNELS}")
        print(f"  残差块数: {cls.NUM_RES_BLOCKS}")
        print(f"\nMCTS:")
        print(f"  模拟次数: {cls.NUM_SIMULATIONS}")
        print(f"  探索常数: {cls.C_PUCT}")
        print(f"\n训练:")
        print(f"  迭代次数: {cls.NUM_ITERATIONS}")
        print(f"  每次迭代对弈局数: {cls.GAMES_PER_ITERATION}")
        print(f"  批次大小: {cls.BATCH_SIZE}")
        print(f"  学习率: {cls.LEARNING_RATE}")
        print(f"  缓冲区大小: {cls.BUFFER_SIZE}")
        print("=" * 60)


class LightConfig:
    """轻量级配置（适合CPU或快速测试）"""
    
    BOARD_SIZE = 15
    
    NUM_CHANNELS = 64
    NUM_RES_BLOCKS = 3
    
    NUM_SIMULATIONS = 50
    C_PUCT = 1.0
    TEMPERATURE = 1.0
    TEMPERATURE_DROP_STEP = 15
    
    DEVICE = 'cpu'
    
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    BATCH_SIZE = 32
    BUFFER_SIZE = 10000
    
    NUM_ITERATIONS = 50
    GAMES_PER_ITERATION = 20
    TRAIN_EPOCHS_PER_ITERATION = 5
    
    SAVE_INTERVAL = 5
    MODEL_DIR = './models_light'
    
    EVAL_GAMES = 10
    EVAL_SIMULATIONS = 50
    
    NUM_WORKERS = 2
    PIN_MEMORY = False
    
    @classmethod
    def display(cls):
        """显示配置信息"""
        print("=" * 60)
        print("AlphaZero 五子棋配置 (轻量版)")
        print("=" * 60)
        print(f"设备: {cls.DEVICE}")
        print(f"棋盘大小: {cls.BOARD_SIZE}x{cls.BOARD_SIZE}")
        print(f"\n网络:")
        print(f"  通道数: {cls.NUM_CHANNELS}")
        print(f"  残差块数: {cls.NUM_RES_BLOCKS}")
        print(f"\nMCTS:")
        print(f"  模拟次数: {cls.NUM_SIMULATIONS}")
        print(f"  探索常数: {cls.C_PUCT}")
        print(f"\n训练:")
        print(f"  迭代次数: {cls.NUM_ITERATIONS}")
        print(f"  每次迭代对弈局数: {cls.GAMES_PER_ITERATION}")
        print(f"  批次大小: {cls.BATCH_SIZE}")
        print(f"  学习率: {cls.LEARNING_RATE}")
        print("=" * 60)


if __name__ == "__main__":
    print("\nGPU配置:")
    Config.display()
    
    print("\n\n轻量级配置:")
    LightConfig.display()
    
    print(f"\n当前可用设备: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    if torch.cuda.is_available():
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
