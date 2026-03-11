# DiffusionGeometryLab

**扩散模型几何相变实验框架** — 研究不同环境维度 $d$ 下，Diffusion Model 在反向去噪过程中从"泛化"到"记忆"相变的几何性质。

## 研究目标

固定数据的**有效流形维度** $m$（如 $32 \times 32$ 灰度图，$m=1024$），通过零填充将数据嵌入到更高维度的环境空间 $d$（如 $64 \times 64$, $128 \times 128$），观察 $d/m$ 比率如何影响扩散模型的：

- **泛化状态** $M_{gen}$：模型学会了流形结构，能生成新颖样本
- **记忆状态** $M_{mem}$：模型背诵了训练集，发生模式坍塌
- **相变几何**：通过余弦相似度场、记忆比例等指标刻画相变的发生时机和几何性质

## 项目结构

```
DiffusionGeometryLab/
│
├── main.py                           # 顶层入口
├── setup.py                          # 安装配置
├── README.md                         # 本文件
│
├── configs/                          # 配置文件
│   ├── circle.yaml                   # 圆环测试配置
│   ├── default.yaml                  # 全局默认配置（CelebA）
│   ├── toy_test.yaml                 # 合成数据快速测试配置
│   ├── data/
│   │   ├── celeba_4096.yaml          # d=4096 覆盖
│   │   └── celeba_16384.yaml         # d=16384 覆盖
│   └── scheduler/
│       ├── of.yaml                   # OF 调度参数
│       ├── ve.yaml                   # VE 调度参数
│       └── vp.yaml                   # VP 调度参数
│
├── diffusion/                        # 数据存储等
│   └── CelebA/
│
├── outputs/                          # 输出结果目录
│
├── scripts/                          # 运行脚本
│   ├── demo_circle.py                # 2D 圆环可视化 demo
│   ├── evaluate.py                   # 全指标评估入口
│   ├── run_circle_experiments.py     # 圆环相关实验入口
│   ├── sweep_dimensions.py           # 多维度批量实验
│   ├── toy_full_pipeline.py          # ★ 合成数据端到端测试
│   ├── train.py                      # 正式训练入口
│   ├── verify_pipeline.py            # 管线单元验证
│   └── visualize_trajectories.py     # 轨迹可视化
│
└── src/                              # 核心源码
    ├── data/                         # 数据管线
    │   ├── dataset.py                #   Dataset/DataLoader
    │   ├── parquet_loader.py         #   Parquet 加载
    │   ├── preprocessing.py          #   灰度化 + resize
    │   ├── semantic_pairs.py         #   语义配对构建
    │   ├── synthetic.py              #   合成数据生成（双月/瑞士卷等）
    │   ├── synthetic_circle.py       #   合成圆环数据类
    │   └── zero_padding.py           #   零填充（核心维度控制）
    │
    ├── logging/                      # 日志与可视化
    │   ├── experiment_logger.py      #   WandB/TensorBoard 抽象层
    │   └── plot_utils.py             #   绘图工具
    │
    ├── metrics/                      # 评估指标
    │   ├── cosine_field.py           #   余弦相似度场（3种）
    │   ├── evaluator.py              #   统一评估门面
    │   ├── fid_evaluator.py          #   FID + 轨迹FID
    │   ├── lpips_ssim.py             #   LPIPS + SSIM
    │   ├── memory_ratio.py           #   动态记忆比例 f_mem(t)
    │   ├── nn_search.py              #   Faiss 最近邻
    │   └── reconstruction_gap.py     #   重建误差差距
    │
    ├── models/                       # 模型定义
    │   ├── mlp.py                    #   MLP 模型
    │   ├── score_network.py          #   得分网络封装
    │   └── unet.py                   #   自适应维度 UNet
    │
    ├── sampling/                     # 采样引擎
    │   ├── reverse_sampler.py        #   DDIM/Euler/Heun 采样器
    │   └── trajectory_store.py       #   轨迹存储
    │
    ├── schedulers/                   # 噪声调度器
    │   ├── base_scheduler.py         #   抽象基类
    │   ├── of_scheduler.py           #   OF (Optimal Flow)
    │   ├── ve_scheduler.py           #   VE (Variance Exploding)
    │   └── vp_scheduler.py           #   VP (Variance Preserving)
    │
    ├── training/                     # 训练引擎
    │   ├── checkpoint_manager.py     #   Checkpoint 管理
    │   ├── phase_detector.py         #   M_gen/M_mem 自动检测
    │   └── trainer.py                #   核心训练循环
    │
    └── utils/                        # 工具模块
        ├── config.py                 #   配置加载与验证
        ├── device.py                 #   GPU/CPU 设备管理
        ├── math_ops.py               #   批量数学运算
        └── seed.py                   #   随机种子管理
```

## 快速开始

### 1. 环境安装

```bash
# 克隆项目
cd /your/project/path

# 安装依赖
pip install torch torchvision omegaconf matplotlib numpy faiss-cpu

# 可选依赖（完整功能）
pip install wandb tensorboard torchmetrics lpips h5py pandas pyarrow

# 安装项目（开发模式）
pip install -e .
```

### 2. 合成数据快速测试（推荐首次运行）

```bash
# 用圆环数据走通完整管线（~3-5分钟）
python scripts/toy_full_pipeline.py

# 查看结果
ls outputs/toy_circle_test/
# → 01_synthetic_data.png  02_trajectory.png  03_reconstruction_gap.png
#   04_cosine_fields.png   05_memory_ratio.png  06_dimension_comparison.png
#   report.txt  checkpoints/
```

### 3. 2D 圆环可视化 Demo

```bash
# 纯 2D 扩散过程可视化（无需 UNet）
python scripts/toy_circle_demo.py

ls outputs/toy_circle_demo/
# → forward_process.png  score_field.png  reverse_process.png
#   generation_quality.png  dimension_experiment.png
```

### 4. CelebA 正式训练

```bash
# 默认配置（d=4096）
python scripts/train.py --config configs/default.yaml

# 自定义维度
python scripts/train.py --config configs/default.yaml \
    --overrides data.target_dim=16384 data.target_resolution=128

# 多维度批量实验
python scripts/sweep_dimensions.py --dimensions 4096 16384

# 评估（需要训练完成后的 checkpoint）
python scripts/evaluate.py \
    --config configs/default.yaml \
    --ckpt_dir outputs/phase_transition_d4096/.../checkpoints
```

## 核心指标说明

### A. 余弦相似度场 (Cosine Similarity Field)

| 指标 | 定义 | 物理意义 |
|------|------|---------|
| $S_{cos}(M_{gen}, M_{mem})$ | 同一 $x_t$ 在两个模型下输出的余弦相似度 | 泛化与记忆模型的决策差异 |
| $S_{cos}(\hat{s}_\theta, M_{emp})$ | 模型预测 vs 指向训练样本的理想方向 | 过拟合程度 |
| $S_{cos}(\text{Train}, \text{Test})$ | 语义相似对注入相同噪声后的输出对齐度 | 局部曲率崩溃检测 |

### B. 距离统计

| 指标 | 定义 |
|------|------|
| 轨迹 FID | 反向采样中间态 $x_t$ 与真实分布的 FID |
| $f_{mem}(t)$ | $R_t = \|x_t - a^{\mu_1}_t\| / \|x_t - a^{\mu_2}_t\| < 1/3$ 的样本比例 |
| 重建差距 | $(MSE_{test} - MSE_{train}) / MSE_{train}$ |

## 配置系统

基于 OmegaConf 的分层配置：

```bash
# 基础配置 + 维度覆盖 + CLI 覆盖
# 优先级: CLI > overlay > default

python scripts/train.py \
    --config configs/default.yaml \
    --overrides
    ```markdown
    --overrides \
    data.target_dim=16384 \
    scheduler.type=ve \
    training.max_steps=100000 \
    training.batch_size=32
```

### 关键配置项

| 配置路径 | 说明 | 默认值 |
|---------|------|--------|
| `data.grayscale_size` | 灰度图边长（决定 $m$） | 32 |
| `data.target_dim` | 环境维度 $d$（必须是完全平方数） | 4096 |
| `scheduler.type` | 噪声调度：`vp` / `ve` / `of` | vp |
| `model.output_type` | 输出模式：`noise` / `score` / `velocity` | noise |
| `training.max_steps` | 最大训练步数 | 500000 |
| `training.phase_detection.enabled` | 是否启用相变自动检测 | true |

## 噪声调度器

| 调度器 | 数学定义 | 推荐搭配 |
|--------|---------|---------|
| **VP** | $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon$ | `output_type=noise` |
| **VE** | $x_t = x_0 + \sigma(t)\epsilon$ | `output_type=score` |
| **OF** | $x_t = (1-t)x_0 + t\epsilon$ | `output_type=velocity` |

## 相变自动检测

训练过程中自动监控并保存两个关键 checkpoint：

### $M_{gen}$（泛化状态）
- **触发条件**: FID 达到全局最低后连续 `fid_patience` 次未改善
- **保存文件**: `checkpoints/model_gen.pt`
- **物理意义**: 学会了流形结构，能生成新颖样本

### $M_{mem}$（记忆状态）
- **触发条件**: 满足以下任一：
  - 重建误差差距 > `recon_gap_threshold`（默认 0.3）
  - LPIPS < `lpips_threshold`（默认 0.05）
  - 且训练步数 > `mem_step_multiplier` × $\tau_{gen}$
- **保存文件**: `checkpoints/model_mem.pt`
- **物理意义**: 背诵了训练集，发生模式坍塌

## 实验设计

### 核心实验：维度比率 $d/m$ 对相变的影响

```
固定 m = 1024（32×32 灰度图）

d = 4096  (64×64)   → d/m = 4
d = 16384 (128×128)  → d/m = 16
d = 65536 (256×256)  → d/m = 64

观察：
1. τ_gen 和 τ_mem 如何随 d/m 变化？
2. 余弦场曲线的形状变化？
3. f_mem(t) 的崩溃点如何移动？
```

### 合成数据实验（验证理论）

使用圆环等低维流形嵌入高维空间，在可控环境下验证：

```bash
# 修改 configs/toy_test.yaml 中的参数
python scripts/toy_full_pipeline.py
```

支持的合成数据类型（`configs/toy_test.yaml` 中设置）：
- `circle`: 单位圆（$m=1$）
- `swiss_roll`: Swiss Roll（$m=2$）
- `two_moons`: 双月形（$m=1$）

## 输出文件说明

### 合成数据测试输出

```
outputs/toy_circle_test/
├── 01_synthetic_data.png        # 原始数据 + 编码后图像
├── 02_trajectory.png            # 反向采样轨迹
├── 03_reconstruction_gap.png    # 训练/测试重建差距
├── 04_cosine_fields.png         # 余弦相似度场
├── 05_memory_ratio.png          # 动态记忆比例
├── 06_dimension_comparison.png  # 多维度 FID 对比
├── report.txt                   # 文本摘要报告
├── raw_points.pt                # 原始 2D 点数据
└── checkpoints/
    ├── latest.pt                # 最新 checkpoint
    └── final.pt                 # 最终 checkpoint
```

### 正式训练输出

```
outputs/phase_transition_d4096/
├── checkpoints/
│   ├── model_gen.pt             # ★ 泛化状态 checkpoint
│   ├── model_mem.pt             # ★ 记忆状态 checkpoint
│   ├── latest.pt
│   └── step_*.pt
├── evaluation/
│   ├── cosine_fields.png        # 三种余弦场对比
│   ├── cosine_train_test.png    # Train-Test 余弦场
│   └── recon_gap_*.png          # Gen/Mem 重建差距
├── tb_logs/                     # TensorBoard 日志
└── experiment.log               # 训练日志
```
