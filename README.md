# AlphaZero 五子棋 AI

基于 DeepMind AlphaZero 算法的五子棋 AI 实现，支持 GPU 加速训练。

> **不是玩具代码**：完整的神经网络 + MCTS + 自对弈训练系统，能真正下棋、学会战术、压制普通玩家。

---

## 快速开始

```bash
# 1. 安装依赖
pip install torch numpy pygame matplotlib

# 2. 快速测试（5次迭代，验证系统能跑）
python train_main.py --mode light --iterations 5 --games 5

# 3. 完整训练（GPU，n次迭代，若迭代100次每次30局耗时约60-80小时）
python train_main.py --mode gpu

# 4. 图形界面对战
python play_gui.py models/model_latest.pth
```

---

## 五子棋规则

- **棋盘**: 15×15 标准棋盘
- **获胜**: 横/竖/斜任意方向连成5子（长连也算）
- **规则**: 自由规则（无禁手），黑白对称
- **平局**: 棋盘下满无人获胜

---

## 项目架构

### 目录结构

### 核心模块

| 文件 | 功能 |
|------|------|
| `gomoku_game.py` | 游戏引擎：棋盘管理、落子检查、胜负判定 |
| `network.py` | 神经网络：ResNet架构，Policy+Value双输出头 |
| `mcts.py` | 蒙特卡洛树搜索：UCT选择、神经网络引导 |
| `train.py` | 训练系统：自对弈生成数据、网络更新 |
| **`config.py`** | **配置中心：两套预设配置（GPU/CPU）** |
| `train_main.py` | 主训练脚本 |
| `play_gui.py` | Pygame图形对战界面 |
| `evaluate.py` | 模型评估与命令行对战 |

---

## 配置说明（config.py）

### 两套预设配置

#### **Config** - GPU完整版

```python
NUM_CHANNELS = 128         # 卷积通道数
NUM_RES_BLOCKS = 6         # 残差块层数
NUM_SIMULATIONS = 100      # MCTS模拟次数
BATCH_SIZE = 128           # 训练批次大小
NUM_ITERATIONS = 100       # 总迭代次数
SAVE_INTERVAL = 10         # 每10次迭代保存检查点
```
- **参数量**: ~300万
- **训练时间**: 60-80小时（100次迭代）
- **适用**: 有GPU，追求最强性能

#### **LightConfig** - CPU轻量版

```python
NUM_CHANNELS = 64          # 减半
NUM_RES_BLOCKS = 3         # 减半
NUM_SIMULATIONS = 50       # 减半
BATCH_SIZE = 32            # 减小
NUM_ITERATIONS = 50        # 减少
SAVE_INTERVAL = 5          # 更频繁保存
```
- **参数量**: ~30万
- **训练时间**: 15-20小时（50次迭代）
- **适用**: CPU训练、快速测试

### 修改配置方法

#### 方法1：编辑 config.py（永久）

```python
class Config:
    SAVE_INTERVAL = 5      # 改为每5次保存
    BATCH_SIZE = 64        # 减小批次（GPU内存不足时）
    NUM_SIMULATIONS = 80   # 调整搜索深度
```

#### 方法2：命令行参数（临时）

```bash
python train_main.py --mode gpu \
    --iterations 30 \
    --games 25 \
    --simulations 80
```

---

## 训练使用

### 基础命令

```bash
# GPU完整训练（100次迭代）
python train_main.py --mode gpu

# 轻量训练（50次迭代）
python train_main.py --mode light

# 快速测试（5次迭代）
python train_main.py --mode light --iterations 5 --games 5
```

### 恢复训练

```bash
# 从检查点继续训练
python train_main.py --mode gpu \
    --resume models/2model_latest.pth \
    --iterations 20    # 再训练30次（总共50次）
```

### 训练流程

```
每次迭代 = 自对弈 + 训练 + 保存

1️⃣ 自对弈: AI自己下棋生成数据
   ├─ 每步用MCTS搜索（如100次模拟）
   └─ 记录：棋盘 + 策略 + 结果

2️⃣ 训练: 更新神经网络
   ├─ 从缓冲区采样批次
   ├─ 计算损失（value + policy）
   └─ 反向传播优化权重

3️⃣ 保存: 自动保存模型
   ├─ 每次迭代: model_latest.pth
   ├─ 每N次迭代: model_iter_N.pth
   └─ 训练完成: model_final.pth
```

---

## 对战测试

### GUI图形界面（人机对战）

```bash
python play_gui.py models/model_latest.pth

# 操作说明
# - 鼠标点击落子
# - R键: 重新开始
# - A键: AI对战模式（观看AI自己下棋）
# - H键: 人机对战模式
# - B键: 切换先手（黑棋先手 ↔ 白棋先手）
# - Q键: 退出
```

### 命令行对战（人机对战）

```bash
python evaluate.py models/model_final.pth

# 输入坐标格式: row col (如: 7 7)
```

### AI对战评估（模型对比）

```bash
# 比较两个模型的强度（20局对弈）
python evaluate.py models/model_iter_50.pth models/model_iter_30.pth



# 自动进行多局对弈，统计胜率
```

---

## 🔧 技术细节

### 神经网络输入

```python
输入张量: (4, 15, 15)
  通道0: 当前玩家的棋子位置
  通道1: 对手的棋子位置
  通道2: 当前玩家标识（全1或全0）
  通道3: 最后落子位置标记
```

### MCTS公式

```python
UCT = Q(s,a) + C_puct × P(s,a) × sqrt(N(s)) / (1 + N(s,a))

# Q: 平均价值
# P: 神经网络先验概率
# N: 访问次数
# C_puct: 探索常数
```

### 损失函数

```python
Loss = MSE(value_pred, value_true) + CrossEntropy(policy_pred, mcts_policy)
```

---

## 常见问题

**Q: GPU内存不足？**  
A: 减小 `BATCH_SIZE` 或 `NUM_CHANNELS`

**Q: 训练太慢？**  
A: 减少 `NUM_SIMULATIONS` 或使用 LightConfig

**Q: 模型不够强？**  
A: 增加迭代次数或提高 `NUM_SIMULATIONS`

**Q: 如何查看训练进度？**  
A: 查看 `models/` 目录下的 `model_iter_*.pth` 文件

---

## 开源协议

MIT License - 可自由使用、修改、分发

---

**开始你的 AlphaZero 之旅！** 

