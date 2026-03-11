# MolCRAFT - Claude Code 项目配置

> ICML 2024: MolCRAFT: Structure-Based Drug Design in Continuous Parameter Space

## 项目概述

MolCRAFT 是一个基于贝叶斯流网络（Bayesian Flow Networks）的药物设计生成模型，专注于基于结构的药物设计（SBDD）。

### 核心技术
- **模型架构**: Bayesian Flow Networks (BFN) + Unified Transformer
- **任务**: 蛋白质-配体复合物的 3D 分子生成
- **框架**: PyTorch + PyTorch Geometric + PyTorch Lightning

---

## 项目结构

```
MolCRAFT/
├── configs/                    # 配置文件
│   └── default.yaml           # 默认配置（模型、数据、训练参数）
│
├── core/                       # 核心代码
│   ├── callbacks/             # 训练回调函数
│   │   ├── basic.py          # 基础回调（恢复、梯度裁剪、EMA等）
│   │   ├── validation_callback.py
│   │   └── validation_callback_for_sample.py
│   │
│   ├── config/                # 配置加载
│   │   └── config.py
│   │
│   ├── datasets/              # 数据集处理
│   │   ├── pl_pair_dataset.py  # 蛋白质-配体对数据集
│   │   ├── pl_data.py          # P-L 数据处理
│   │   ├── protein_ligand.py
│   │   ├── pdbbind.py
│   │   └── utils.py
│   │
│   ├── evaluation/            # 评估工具
│   │   └── utils/
│   │
│   ├── models/                # 模型定义
│   │   ├── bfn_base.py        # BFN 基类
│   │   ├── bfn4sbdd.py        # SBDD 专用 BFN
│   │   ├── uni_transformer.py # Unified Transformer 架构
│   │   ├── sbdd_train_loop.py # 训练循环
│   │   └── common.py          # 通用模型组件
│   │
│   └── utils/                 # 工具函数
│       ├── transforms_prop.py
│       └── misc.py
│
├── docker/                     # Docker 环境配置
│   ├── Dockerfile
│   ├── Makefile
│   ├── environment.yml        # Conda 环境配置
│   ├── asset/
│   │   ├── requirements.txt
│   │   └── apt_packages.txt
│   └── README.md
│
├── test/                       # 测试和评估脚本
│   ├── eval_dock_vina.py      # Vina 对接评估
│   ├── eval_posecheck.py      # PoseCheck 评估
│   ├── eval_rmsd.py           # RMSD 计算
│   └── eval_utils.py
│
├── train_bfn.py               # 主训练脚本
├── sample_for_pocket.py       # 单口袋采样脚本
├── scripts.mk                 # Makefile 任务定义
├── sample_for_pocket.py       # 口袋采样工具
├── data/                      # 数据目录（需自行下载）
└── checkpoints/               # 模型检查点
```

---

## 快速开始

### 1. 环境设置

**推荐方式 - Docker:**
```bash
cd docker
make
```

**Conda 方式:**
```bash
conda env create -f docker/environment.yml
conda activate molcraft
```

### 2. 数据准备

训练数据会自动下载，或手动下载：
```bash
make -f scripts.mk data
```

需要的数据文件：
- `crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb` - 训练 LMDB
- `crossdocked_pocket10_pose_split.pt` - 数据划分
- `test_set/` - 测试集 PDB 文件

### 3. 训练

```bash
# 使用默认参数训练
make -f scripts.mk run

# 或直接运行 python
python train_bfn.py --config_file configs/default.yaml --epochs 15
```

### 4. 评估/采样

```bash
# 下载预训练检查点后运行
make -f scripts.mk evaluate

# 或
python train_bfn.py --test_only --num_samples 10 --sample_steps 100 --ckpt_path ./checkpoints/last.ckpt
```

### 5. 单个口袋采样

```bash
python sample_for_pocket.py ${PDB_PATH} ${SDF_PATH}
```

---

## 核心配置参数

### 模型参数 (configs/default.yaml)
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `sigma1_coord` | 0.03 | 坐标噪声参数 |
| `beta1` | 1.5 | BFN beta 参数 |
| `time_emb_dim` | 1 | 时间嵌入维度 |
| `use_discrete_t` | True | 使用离散时间步 |
| `destination_prediction` | True | 目标预测 |
| `sampling_strategy` | end_back_pmf | 采样策略 |

### 训练参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lr` | 5e-4 | 学习率 |
| `epochs` | 15 | 训练轮数 |
| `batch_size` | 见配置 | 批大小 |
| `max_grad_norm` | Q | 梯度裁剪 |

### 模型架构 (uni_transformer)
| 参数 | 值 | 说明 |
|------|-----|------|
| `num_blocks` | 1 | Transformer 块数 |
| `num_layers` | 9 | 层数 |
| `hidden_dim` | 128 | 隐藏维度 |
| `n_heads` | 16 | 注意力头数 |
| `knn` | 32 | K-近邻 |

---

## 技术栈

### 核心依赖
- **Python**: 3.9
- **CUDA**: 11.6
- **PyTorch**: 1.12.0
- **PyTorch Geometric**: 2.1.0
- **PyTorch Lightning**: 训练框架
- **RDKit**: 2023.9.5 (化学信息学)

### 评估工具
- **vina**: 1.2.2 - 分子对接
- **posecheck**: 构象检查
- **spyrmsd**: RMSD 计算

### 其他
- **WandB**: 实验跟踪
- **absl-py**: 日志
- **scikit-learn**: 指标计算

---

## 关键文件说明

### [train_bfn.py](train_bfn.py)
主训练入口，包含：
- 数据加载器构建 (`get_dataloader`)
- 模型初始化
- PyTorch Lightning Trainer 设置
- WandB 日志配置

### [core/models/bfn4sbdd.py](core/models/bfn4sbdd.py)
SBDD 专用的贝叶斯流网络实现：
- 前向过程（加噪）
- 反向过程（去噪/采样）
- 损失计算

### [core/models/uni_transformer.py](core/models/uni_transformer.py)
Unified Transformer 架构：
- 蛋白质-配体联合建模
- 图注意力机制
- K-NN 边构建

### [sample_for_pocket.py](sample_for_pocket.py)
从 PDB 文件生成分子的独立脚本

---

## 数据格式

### 输入
- **蛋白质**: PDB 文件（口袋区域）
- **配体**: SDF 文件（用于定义口袋位置）

### 输出
- 生成的分子以 `.pt` 格式保存
- 包含坐标、原子类型、对接分数等

---

## 评估指标

1. **结合亲和力**: Vina Score / Min / Dock
2. **分子性质**: QED, SA
3. **构象质量**: PoseCheck (strain energy, clashes)
4. **几何性质**: 键长、键角、二面角、RMSD

---

## Make 任务 (scripts.mk)

| 命令 | 说明 |
|------|------|
| `make -f scripts.mk run` | 完整训练 |
| `make -f scripts.mk data` | 下载训练数据 |
| `make -f scripts.mk checkpoint` | 下载预训练检查点 |
| `make -f scripts.mk debug` | 调试模式（1 epoch） |
| `make -f scripts.mk evaluate` | 评估预训练模型 |

---

## 调试建议

1. **快速验证**: `make debug -f scripts.mk`
2. **日志级别**: 设置 `--logging_level DEBUG`
3. **禁用 WandB**: 使用 `--no_wandb`
4. **小数据集**: 在 `debug` 模式下使用子集

---

## 论文引用

```bibtex
@article{qu2024molcraft,
  title={MolCRAFT: Structure-Based Drug Design in Continuous Parameter Space},
  author={Qu, Yanru and Qiu, Keyue and Song, Yuxuan and Gong, Jingjing and Han, Jiawei and Zheng, Mingyue and Zhou, Hao and Ma, Wei-Ying},
  journal={ICML 2024},
  year={2024}
}
```

---

## 开发注意事项

1. **数据目录**: 确保 `data/` 目录包含必需文件
2. **GPU 内存**: 模型可能需要较大 GPU 内存（建议 >16GB）
3. **网络**: 数据下载需要稳定网络连接
4. **Docker**: 如果使用 Docker，确保 `nvidia-container-runtime` 已配置

---

## 相关链接

- 论文: https://arxiv.org/abs/2404.12141
- Demo: http://61.241.63.126:8000
- 数据: https://drive.google.com/drive/folders/1j21cc7-97TedKh_El5E34yI8o5ckI7eK
- 预训练模型: https://drive.google.com/file/d/1TcUQM7Lw1klH2wOVBu20cTsvBTcC1WKu
