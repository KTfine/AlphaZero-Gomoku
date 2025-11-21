"""
主训练脚本（GPU优化版本）

"""

import os
import argparse
from config import Config, LightConfig
from train import AlphaZeroTrainer


def main():
    parser = argparse.ArgumentParser(description='AlphaZero 五子棋训练')
    parser.add_argument('--mode', type=str, default='gpu', choices=['gpu', 'light'],
                       help='训练模式: gpu(完整) 或 light(轻量)')
    parser.add_argument('--iterations', type=int, default=None,
                       help='迭代次数（覆盖配置）')
    parser.add_argument('--games', type=int, default=None,
                       help='每次迭代对弈局数（覆盖配置）')
    parser.add_argument('--simulations', type=int, default=None,
                       help='MCTS模拟次数（覆盖配置）')
    parser.add_argument('--resume', type=str, default=None,
                       help='从检查点恢复训练的模型路径')
    
    args = parser.parse_args()
    
    # 选择配置
    if args.mode == 'gpu':
        cfg = Config
    else:
        cfg = LightConfig
    
    # 显示配置
    cfg.display()
    
    # 创建训练器
    print(f"\n正在创建训练器（设备: {cfg.DEVICE}）...")
    trainer = AlphaZeroTrainer(
        board_size=cfg.BOARD_SIZE,
        num_channels=cfg.NUM_CHANNELS,
        num_res_blocks=cfg.NUM_RES_BLOCKS,
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
        device=cfg.DEVICE,
        buffer_size=cfg.BUFFER_SIZE
    )
    
    # 如果需要恢复训练
    if args.resume:
        print(f"\n从检查点恢复: {args.resume}")
        trainer.load_model(args.resume)
    
    # 准备训练参数
    num_iterations = args.iterations if args.iterations else cfg.NUM_ITERATIONS
    games_per_iteration = args.games if args.games else cfg.GAMES_PER_ITERATION
    num_simulations = args.simulations if args.simulations else cfg.NUM_SIMULATIONS
    
    # 开始训练
    print("\n开始训练...")
    trainer.train_pipeline(
        num_iterations=num_iterations,
        games_per_iteration=games_per_iteration,
        train_epochs_per_iteration=cfg.TRAIN_EPOCHS_PER_ITERATION,
        num_simulations=num_simulations,
        c_puct=cfg.C_PUCT,
        temperature=cfg.TEMPERATURE,
        temperature_drop_step=cfg.TEMPERATURE_DROP_STEP,
        batch_size=cfg.BATCH_SIZE,
        save_dir=cfg.MODEL_DIR,
        save_interval=cfg.SAVE_INTERVAL
    )
    
    print("\n训练完成！")
    print(f"模型已保存到: {cfg.MODEL_DIR}")


if __name__ == "__main__":
    main()
