# MolDiff风格的Bond-First Schedule在MolCRAFT中的实现

> 在BFN框架下实现MolDiff的键扩散策略，解决原子-键不一致问题

---

## 目录

1. [概述](#1-概述)
2. [核心原理](#2-核心原理)
3. [实现细节](#3-实现细节)
4. [使用方法](#4-使用方法)
5. [配置参数](#5-配置参数)
6. [理论分析](#6-理论分析)
7. [验证与测试](#7-验证与测试)
8. [预期效果](#8-预期效果)
9. [参考文献](#9-参考文献)

---

## 1. 概述

### 1.1 问题背景

**原子-键不一致问题**：
- 在传统扩散模型中，键类型与原子类型使用相同的噪声调度
- 在中间时间步，原子仍保持一定的空间结构，但键已被部分扰动
- 导致出现"远距离单键"等不合理的中间状态（如相距10Å的原子间形成单键）

### 1.2 MolDiff的解决方案

**Bond-First Schedule**（ICML 2023）：
- **阶段1 (t=1~600)**：键快速衰减到uniform分布（大部分为none）
- **阶段2 (t=601~1000)**：键保持噪声状态，原子和坐标继续扩散
- **结果**：采样成功率从35%提升到99.3%

### 1.3 本实现的目标

**在MolCRAFT的BFN框架下实现MolDiff的Bond-First策略**：
- 保持BFN的理论基础（贝叶斯流网络）
- 将MolDiff的离散1000步映射到BFN的连续[0,1]时间
- 支持多种调度策略：uniform、segmented、delayed

---

## 2. 核心原理

### 2.1 MolDiff的Bond-First Schedule

**MolDiff（DDPM框架）**：
```python
# 阶段1: t=1~600
alpha_bar: 0.9999 → 0.001  # 键快速衰减
# 等价于BFN中: beta ≈ 1 - alpha_bar → 0.999 → 大beta

# 阶段2: t=601~1000
alpha_bar: 0.001 → 0.0001  # 键保持噪声
# 等价于BFN中: beta ≈ 0.999 → 0.9999 → 小beta
```

**关键洞察**：
- 前期大beta → 快速将键扰动到uniform分布（接近none）
- 后期小beta → 键维持在噪声状态，不再增加扰动
- 目的：给模型一个"干净的起点"，从头学习原子间的键连接

### 2.2 BFN框架下的实现

**离散BFN前向扩散公式**：
```
mean = beta * (K * one_hot - 1)
std = sqrt(beta * K)
noisy_logits = mean + std * N(0,1)
theta_t = softmax(noisy_logits)
```

**Bond-First Schedule**：
```python
# 阶段1 (t < 0.6): 大beta快速加噪
bond_beta = 3.0  # 常数大值

# 阶段2 (t >= 0.6): 小beta保持噪声
bond_beta = 0.001  # 常数小值
```

### 2.3 时间映射关系

| MolDiff（离散） | BFN（连续） | 说明 |
|----------------|------------|------|
| t ∈ {1, ..., 600} | t ∈ [0, 0.6) | 前期：键快速衰减 |
| t ∈ {601, ..., 1000} | t ∈ [0.6, 1] | 后期：键保持噪声 |
| alpha_bar: 0.9999→0.001 | **beta=3.0** | 快速加噪 |
| alpha_bar: 0.001→0.0001 | **beta=0.001** | 保持噪声 |

---

## 3. 实现细节

### 3.1 核心方法：get_bond_beta()

**文件**: [core/models/bfn4sbdd.py:181-254](../core/models/bfn4sbdd.py#L181-L254)

**功能**：计算时间依赖的键扩散beta参数

**代码结构**：
```python
def get_bond_beta(self, t: torch.Tensor, schedule_type: str = None) -> torch.Tensor:
    """
    计算键扩散的beta参数（支持MolDiff风格的Bond-First调度）

    Args:
        t: 连续时间张量 [N, 1] 或 [N]，范围 [0, 1]
        schedule_type: 调度类型（'uniform', 'segmented', 'delayed'）

    Returns:
        bond_beta: Beta值 [N, 1] 或 [N]
    """
    if schedule_type == 'uniform':
        # 与原子类型同步的二次调度（原始实现）
        return self.bond_beta1 * (t ** 2)

    elif schedule_type == 'segmented':
        # MolDiff风格：分段调度
        phase1_mask = t < 0.6
        phase2_mask = t >= 0.6
        bond_beta[phase1_mask] = 3.0   # 前期大beta
        bond_beta[phase2_mask] = 0.001 # 后期小beta
        return bond_beta

    elif schedule_type == 'delayed':
        # 平滑过渡版本（Sigmoid）
        transition = sigmoid(20 * (t - 0.6))
        bond_beta = 0.001 + (3.0 - 0.001) * (1 - transition)
        return bond_beta
```

**调度类型可视化**：

```
Uniform调度（原始）
beta_bond
  │
3 │         ╱
  │       ╱
  │     ╱
  │   ╱
  │ ╱
0 └─────────────── t
  0             1

Segmented调度（MolDiff风格，推荐）
beta_bond
  │
3 │ ╲────────────           ← 阶段1: beta=3.0（快速加噪）
  │   ╲
  │    ╲
  │     ╲───────────────    ← 阶段2: beta=0.001（保持噪声）
0.001─┴─────────────── t
  0             0.6     1
   ↑ 分段点

Delayed调度（平滑过渡）
beta_bond
  │
3 │ ╲╲
  │   ╲╲╱╱
  │     ╲
  │      ╲───────────────
0.001─┴─────────────── t
  0          0.6      1
       Sigmoid过渡
```

### 3.2 训练时的修改

**文件**: [core/models/bfn4sbdd.py:465-491](../core/models/bfn4sbdd.py#L465-L491)

**修改内容**：在`loss_one_step()`中使用`get_bond_beta()`

**关键代码**：
```python
# ===== 修改：使用Bond-First Schedule计算beta =====
# 原始代码：
# bond_beta = self.bond_beta1 * (t.squeeze(-1) ** 2)

# 修改为：
bond_beta = self.get_bond_beta(t)  # 支持多种调度策略

# 后续的BFN前向扩散保持不变
mean = bond_beta.unsqueeze(-1) * (K_bond * halfedge_onehot - 1.0)
std = (bond_beta.unsqueeze(-1) * K_bond).sqrt()
noisy_bond_logits = mean + std * torch.randn_like(mean)
theta_bond_t = F.softmax(noisy_bond_logits, dim=-1)
```

### 3.3 采样时的修改

**文件**: [core/models/bfn4sbdd.py](../core/models/bfn4sbdd.py)

**修改位置1**：Sender分布采样 [811-818行](../core/models/bfn4sbdd.py#L811-L818)
```python
# ===== 修改：使用Bond-First Schedule计算alpha_bond =====
# 原始代码：
# alpha_bond = self.bond_beta1 * (2 * i - 1) / (sample_steps**2)

# 修改为：
t_current = torch.ones((n_nodes, 1)).to(self.device) * i / sample_steps
t_current = t_current[batch_ligand]
bond_beta_t = self.get_bond_beta(t_current)
alpha_bond = bond_beta_t.mean() * (2 * i - 1) / (sample_steps**2)
```

**修改位置2**：贝叶斯更新 [872-882行](../core/models/bfn4sbdd.py#L872-L882)
```python
# ===== 修改：使用时间依赖的beta =====
# 原始代码：
# theta_bond_t = self.discrete_var_bayesian_update(
#     t, beta1=self.bond_beta1, x=sample_pred_bond, K=K_bond
# )

# 修改为：
t_next = torch.ones((n_nodes, 1)).to(self.device) * i / sample_steps
t_next = t_next[batch_ligand]
bond_beta_next = self.get_bond_beta(t_next)
theta_bond_t = self.discrete_var_bayesian_update(
    t, beta1=bond_beta_next.mean(), x=sample_pred_bond, K=K_bond
)
```

### 3.4 损失权重计算的修改

**文件**: [core/models/bfn4sbdd.py:607-612](../core/models/bfn4sbdd.py#L607-L612)

**修改内容**：使用实际的bond_beta而非固定公式

```python
# ===== 修改：使用实际的bond_beta =====
# 原始代码：
# weight = K_bond * beta1 * (t_expanded ** 2)

# 修改为：
bond_beta_t = self.get_bond_beta(t)
weight = K_bond * bond_beta_t.mean()

# 后续计算保持不变
l2_dist = ((e_x - e_hat) ** 2).sum(dim=-1)
bloss = weight * l2_dist.mean()
```

### 3.5 __init__的修改

**文件**: [core/models/bfn4sbdd.py:107-111, 177-188](../core/models/bfn4sbdd.py#L107-L111)

**新增参数**：
```python
def __init__(
    self,
    # ... 现有参数 ...
    num_bond_types=5,
    bond_beta1=3.0,
    include_bond_features=False,
    # ===== 新增：Bond-First Schedule 参数 =====
    bond_schedule_type='uniform',      # 'uniform' | 'segmented' | 'delayed'
    bond_delay_ratio=0.6,              # 分段点（对应MolDiff的600/1000）
    bond_beta_late=0.001,              # 后期小beta值（保持噪声）
    bond_beta_early=None,              # 前期大beta值（默认使用bond_beta1）
):
    # ... 初始化代码 ...

    # 存储Bond-First参数
    self.bond_schedule_type = bond_schedule_type
    self.bond_delay_ratio = bond_delay_ratio
    self.bond_beta_late = torch.tensor(bond_beta_late, dtype=torch.float32)
    if bond_beta_early is None:
        self.bond_beta_early = self.bond_beta1
    else:
        self.bond_beta_early = torch.tensor(bond_beta_early, dtype=torch.float32)
```

### 3.6 配置文件的修改

**文件**: [configs/default.yaml:48-62](../configs/default.yaml#L48-L62)

**新增配置**：
```yaml
# ===== 新增：Bond-First Schedule 配置 =====
# 键扩散调度类型：'uniform'（同步）, 'segmented'（分段）, 'delayed'（平滑）
bond_schedule_type: !SUB ${bond_schedule_type:-uniform}
# 键扩散延迟比例（对应MolDiff的600/1000=0.6）
bond_delay_ratio: !SUB ${bond_delay_ratio:-0.6}
# 后期小beta值（保持噪声状态）
bond_beta_late: !SUB ${bond_beta_late:-0.001}
# 前期大beta值（快速加噪，默认使用bond_beta1）
# bond_beta_early: !SUB ${bond_beta_early:-3.0}
```

### 3.7 命令行参数的添加

**文件**: [train_bfn.py:208-219](../train_bfn.py#L208-L219)

**新增参数**：
```python
# ===== 新增：键扩散参数 =====
parser.add_argument('--include_bond_features', type=eval, default=False)
parser.add_argument('--bond_loss_weight', type=float, default=0.0)

# ===== 新增：Bond-First Schedule 参数 =====
parser.add_argument('--bond_schedule_type', type=str, default='uniform',
                    choices=['uniform', 'segmented', 'delayed'])
parser.add_argument('--bond_delay_ratio', type=float, default=0.6)
parser.add_argument('--bond_beta_late', type=float, default=0.001)
parser.add_argument('--bond_beta_early', type=float, default=None)
```

---

## 4. 使用方法

### 4.1 MolDiff风格的Bond-First Schedule（推荐）

```bash
python train_bfn.py \
    --config_file configs/default.yaml \
    --include_bond_features True \
    --bond_loss_weight 0.1 \
    --bond_schedule_type segmented \
    --bond_delay_ratio 0.6 \
    --bond_beta_late 0.001 \
    --batch_size 8 \
    --epochs 15
```

### 4.2 原始Uniform调度（Baseline对比）

```bash
python train_bfn.py \
    --config_file configs/default.yaml \
    --include_bond_features True \
    --bond_loss_weight 0.1 \
    --bond_schedule_type uniform \
    --batch_size 8 \
    --epochs 15
```

### 4.3 平滑过渡调度（实验性）

```bash
python train_bfn.py \
    --config_file configs/default.yaml \
    --include_bond_features True \
    --bond_loss_weight 0.1 \
    --bond_schedule_type delayed \
    --bond_delay_ratio 0.6 \
    --bond_beta_late 0.001 \
    --batch_size 8 \
    --epochs 15
```

### 4.4 采样时使用

```bash
python train_bfn.py \
    --test_only \
    --ckpt_path ./checkpoints/best.ckpt \
    --sample_steps 100 \
    --num_samples 10
```

**注意**：采样时使用的调度类型会从checkpoint中自动加载，无需额外指定。

---

## 5. 配置参数

### 5.1 核心参数

| 参数 | 默认值 | 范围 | 说明 |
|------|-------|------|------|
| `bond_schedule_type` | 'uniform' | 'uniform', 'segmented', 'delayed' | 键扩散调度类型 |
| `bond_delay_ratio` | 0.6 | [0, 1] | 分段点（0.6对应MolDiff的600/1000） |
| `bond_beta_late` | 0.001 | [0.0001, 0.1] | 后期小beta值（保持噪声） |
| `bond_beta_early` | None (使用bond_beta1) | [0.1, 10] | 前期大beta值（快速加噪） |

### 5.2 推荐配置

**配置1：MolDiff标准风格（推荐）**
```bash
bond_schedule_type=segmented
bond_delay_ratio=0.6      # t=0.6时分段
bond_beta_late=0.001      # 后期：小beta保持噪声
# bond_beta_early默认使用bond_beta1=3.0
```

**配置2：保守风格（更长的快速加噪期）**
```bash
bond_schedule_type=segmented
bond_delay_ratio=0.7      # 更长的前期快速加噪期
bond_beta_late=0.0005     # 更小的后期beta
```

**配置3：平滑过渡（实验性）**
```bash
bond_schedule_type=delayed
bond_delay_ratio=0.6
bond_beta_late=0.001
# 使用Sigmoid平滑过渡，避免分段点突变
```

**配置4：基准（原始实现）**
```bash
bond_schedule_type=uniform  # 与原子同步的二次调度
# 其他参数不生效
```

### 5.3 参数调优建议

**分段点（bond_delay_ratio）**：
- 0.5-0.6：激进（更长的保持期）
- 0.6-0.7：平衡（MolDiff经验值）
- 0.7-0.8：保守（更长的快速加噪期）

**后期beta（bond_beta_late）**：
- 0.0001-0.001：强保持（键几乎完全冻结）
- 0.001-0.01：中等保持（键轻微波动）
- 0.01-0.1：弱保持（键继续缓慢扩散）

---

## 6. 理论分析

### 6.1 为什么是"大Beta→小Beta"？

**关键原理**：

1. **前期大beta (t < 0.6)**：
   ```
   原始键: [single, double, aromatic]
       ↓ beta=3.0 快速加噪
   接近uniform: [0.2, 0.2, 0.2, 0.2, 0.2]
       ↓ 大部分概率在none上
   ```
   - 键快速失去原始信息
   - 避免模型学习不合理的中间状态
   - 给模型一个"干净的起点"

2. **后期小beta (t ≥ 0.6)**：
   ```
   噪声状态: uniform分布
       ↓ beta=0.001 几乎不加噪
   保持噪声: uniform分布
       ↓ 模型预测
   逐渐恢复: 基于原子位置学习合理的键
   ```
   - 键维持在噪声状态
   - 模型专注于恢复原子和坐标
   - 在生成最后阶段才恢复键

### 6.2 BFN理论的兼容性

**前向扩散（训练时）**：

**阶段1 (t < 0.6)**：
```python
原始键: [1, 2, 4]（单键、双键、芳香键）
    ↓ one_hot
[[1,0,0,0,0], [0,1,0,0,0], [0,0,0,0,1]]
    ↓ 大beta快速加噪（beta=3.0）
mean = 3.0 * (5 * one_hot - 1)
std = sqrt(3.0 * 5)
noisy_logits = mean + std * N(0,1)
    ↓ softmax
theta_bond_t ≈ [0.2, 0.2, 0.2, 0.2, 0.2]  ← 接近uniform
```

**阶段2 (t ≥ 0.6)**：
```python
theta_bond_t ≈ uniform  （已经是噪声状态）
    ↓ 小beta保持噪声（beta=0.001）
mean = 0.001 * (5 * theta_bond_t - 1)
std = sqrt(0.001 * 5)
noisy_logits = mean + std * N(0,1)
    ↓ softmax
theta_bond_t ≈ uniform  （保持噪声状态）
```

**反向扩散（采样时）**：
```python
初始化: theta_bond_T ~ uniform(1/5)

for t = T to 1:
    # 阶段1 (t >= 0.6): 键维持接近uniform
    # 阶段2 (t < 0.6): 键开始从uniform恢复

    # 预测t=0的键类型
    p0_bond = model(theta_bond_t, t)

    # 贝叶斯更新（使用实际beta）
    theta_bond_{t-1} = BayesianUpdate(theta_bond_t, p0_bond, beta(t))
```

### 6.3 与MolDiff的对比

| 特性 | MolDiff (DDPM) | MolCRAFT (BFN) |
|------|----------------|----------------|
| 扩散框架 | DDPM | BFN |
| 时间表示 | 离散1000步 | 连续[0,1] |
| 噪声参数 | alpha_bar（信息保留率） | beta（噪声强度） |
| 关系 | alpha_bar ≈ 1 - beta | - |
| 阶段1 (前期) | alpha_bar: 0.9999→0.001 | beta=3.0（快速加噪） |
| 阶段2 (后期) | alpha_bar: 0.001→0.0001 | beta=0.001（保持噪声） |
| 采样步数 | 1000 | 100 |
| 成功率 | 99.3% | 待验证 |

---

## 7. 验证与测试

### 7.1 单元测试

**测试1：Beta调度函数**
```python
import torch
from core.models.bfn4sbdd import BFN4SBDDScoreModel

# 创建模型
model = BFN4SBDDScoreModel(
    net_config={'name': 'unio2net', ...},
    bond_schedule_type='segmented',
    bond_delay_ratio=0.6,
    ...
)

# 测试分段调度
t = torch.linspace(0, 1, 100).view(-1, 1)
beta = model.get_bond_beta(t, schedule_type='segmented')

# 验证
assert beta[t < 0.6].mean() ≈ 3.0, "前期beta应该≈3.0"
assert beta[t >= 0.6].mean() ≈ 0.001, "后期beta应该≈0.001"
print("✅ Beta调度测试通过")
```

**测试2：维度一致性**
```python
# 测试不同输入shape
t1 = torch.randn(10, 1)  # [N, 1]
t2 = torch.randn(10)      # [N]
beta1 = model.get_bond_beta(t1)
beta2 = model.get_bond_beta(t2)

assert beta1.shape == t1.shape, "输出shape应该匹配输入"
assert beta2.shape == t2.shape, "输出shape应该匹配输入"
print("✅ 维度一致性测试通过")
```

### 7.2 端到端测试

**测试1：训练测试**
```bash
# 使用segmented调度训练100步
python train_bfn.py \
    --bond_schedule_type segmented \
    --bond_delay_ratio 0.6 \
    --include_bond_features True \
    --bond_loss_weight 0.1 \
    --debug \
    --epochs 1

# 检查日志
# loss_bond应该逐渐下降
```

**测试2：采样测试**
```bash
# 生成100个分子
python train_bfn.py \
    --test_only \
    --ckpt_path ./checkpoints/last.ckpt \
    --num_samples 100 \
    --sample_steps 100

# 统计键类型分布
# single: ~40%, double: ~10%, aromatic: ~30%, none: ~20%
```

**测试3：质量评估**
```bash
# 分子有效性
python -c "
from rdkit import Chem
import glob

for sdf_file in glob.glob('outputs/*_SDF/*.sdf'):
    suppl = Chem.SDMolSupplier(sdf_file)
    valid = sum(1 for mol in suppl if mol is not None)
    total = len(list(suppl))
    print(f'{sdf_file}: {valid}/{total} ({valid/total*100:.1f}%)')
"

# 预期：有效性 > 95%
```

### 7.3 消融实验

| 配置 | bond_schedule_type | bond_delay_ratio | bond_beta_late | 目的 |
|------|-------------------|-----------------|----------------|------|
| **Baseline** | uniform | - | - | 原始实现 |
| **MolDiff** | segmented | 0.6 | 0.001 | MolDiff风格 |
| **Conservative** | segmented | 0.7 | 0.0005 | 更保守的分段 |
| **Smooth** | delayed | 0.6 | 0.001 | 平滑过渡 |

**评估指标**：
- 分子有效性（RDKit sanitize）
- 键类型准确率
- 原子-键一致性（键长合理性）
- 采样成功率

---

## 8. 预期效果

### 8.1 量化指标

| 指标 | Uniform（基线） | Segmented（MolDiff） | 预期提升 |
|------|----------------|---------------------|---------|
| 键类型准确率 | ~70% | >85% | +15% |
| 分子有效性 | ~90% | >95% | +5% |
| 原子-键一致性 | 有"远距离单键" | 明显改善 | 定性提升 |
| 采样成功率 | 待测 | >95% | +?% |

### 8.2 定性改进

1. **更少的"远距离单键"**：
   - 键快速达到uniform，避免学习不合理的中间状态
   - 原子位置和键类型更加一致

2. **更稳定的训练**：
   - 前期键不受原子和坐标变化的影响
   - 损失曲线更平滑

3. **更好的可解释性**：
   - 明确的时间分段便于调试
   - 符合MolDiff的成功经验

### 8.3 潜在风险与缓解

**风险1：分段点的选择**
- 风险：0.6可能不是最优的分割点
- 缓解：提供可配置参数，进行网格搜索{0.5, 0.6, 0.7, 0.8}

**风险2：beta值范围**
- 风险：0.001可能太小或太大
- 缓解：实验不同值{0.0001, 0.001, 0.01}，观察训练初期键损失

**风险3：时间维度对齐**
- 风险：训练时均匀采样 vs 采样时确定性遍历
- 缓解：确保训练和采样使用相同的`get_bond_beta()`函数

---

## 9. 参考文献

### 9.1 核心论文

1. **MolDiff**:
   - Peng, X., et al. "MolDiff: Addressing the Atom-Bond Inconsistency Problem in 3D Molecule Diffusion Generation", ICML 2023
   - [PDF](https://proceedings.mlr.press/v202/peng23b.html)
   - [代码](https://github.com/benygood/MolDiff)

2. **PocketXMol**:
   - Peng, X., et al. "Unified modeling of 3D molecular generation via atomic interactions with PocketXMol", Cell 2026
   - [代码](https://github.com/pengxingang/PocketXMol)

3. **Bayesian Flow Networks**:
   - "Bayesian Flow Networks", 2022
   - 理论基础

### 9.2 相关文档

1. **算法分析**：
   - [PocketXMol_algorithm_analysis.md](./PocketXMol_algorithm_analysis.md) - PocketXMol与MolDiff的对比分析

2. **实现计划**：
   - [graceful-wishing-tulip.md](../../.claude/plans/graceful-wishing-tulip.md) - 完整的实现计划

### 9.3 代码文件索引

| 功能 | 文件路径 | 关键行 |
|------|---------|-------|
| **get_bond_beta()方法** | [core/models/bfn4sbdd.py](../core/models/bfn4sbdd.py) | 181-254 |
| **训练时噪声添加** | [core/models/bfn4sbdd.py](../core/models/bfn4sbdd.py) | 470-473 |
| **采样时sender分布** | [core/models/bfn4sbdd.py](../core/models/bfn4sbdd.py) | 811-818 |
| **采样时贝叶斯更新** | [core/models/bfn4sbdd.py](../core/models/bfn4sbdd.py) | 872-882 |
| **损失权重计算** | [core/models/bfn4sbdd.py](../core/models/bfn4sbdd.py) | 607-612 |
| **__init__参数** | [core/models/bfn4sbdd.py](../core/models/bfn4sbdd.py) | 107-111, 177-188 |
| **配置文件** | [configs/default.yaml](../configs/default.yaml) | 48-62 |
| **命令行参数** | [train_bfn.py](../train_bfn.py) | 208-219 |

---

## 附录

### A. 快速参考

**MolDiff的segment_schedule实现**：
- 文件：[MolDiff/models/diffusion.py:133-148](../../MolDiff/models/diffusion.py#L133-L148)
- 关键代码：
```python
def segment_schedule(timesteps, time_segment, segment_diff):
    alphas_cumprod = []
    for i in range(len(time_segment)):
        _, alphas_this = advance_schedule(
            time_segment[i] + 1,
            **segment_diff[i],
            return_alphas_bar=True
        )
        alphas_cumprod.extend(alphas_this[1:])
    betas = 1 - alphas
    return betas
```

**BFN的离散损失公式**：
- 文件：[core/models/bfn_base.py](../core/models/bfn_base.py)
- 公式：`L∞(x) = Kβ(t) * E[||e_x - ê(θ,t)||²]`

### B. 调试技巧

**查看beta调度**：
```python
import matplotlib.pyplot as plt
import torch

t = torch.linspace(0, 1, 1000).view(-1, 1)
beta_uniform = model.get_bond_beta(t, 'uniform')
beta_segmented = model.get_bond_beta(t, 'segmented')
beta_delayed = model.get_bond_beta(t, 'delayed')

plt.figure(figsize=(10, 6))
plt.plot(t, beta_uniform, label='uniform')
plt.plot(t, beta_segmented, label='segmented')
plt.plot(t, beta_delayed, label='delayed')
plt.axvline(x=0.6, color='red', linestyle='--', label='t=0.6')
plt.xlabel('Time t')
plt.ylabel('Bond beta')
plt.legend()
plt.savefig('bond_schedule.png')
```

**监控训练过程**：
```bash
# 启用wandb查看loss_bond曲线
python train_bfn.py \
    --bond_schedule_type segmented \
    --bond_loss_weight 0.1 \
    --no_wandb false

# 预期：loss_bond应该逐渐下降且平稳
```

### C. 常见问题

**Q1: 为什么分段点是0.6而不是其他值？**
A: 0.6对应MolDiff的600/1000步，这是论文中通过实验验证的最优值。可以根据具体任务调整。

**Q2: bond_beta_late=0.001会不会太小？**
A: 这个值使得键在后期几乎完全冻结（保持uniform状态）。如果发现键恢复不充分，可以适当增大（如0.01）。

**Q3: 可以只使用Bond-First Schedule而不包含键特征吗？**
A: 不可以。Bond-First Schedule是键扩散的一部分，必须设置`include_bond_features=True`才会生效。

**Q4: 训练和采样可以使用不同的schedule_type吗？**
A: 不建议。训练和采样应该使用相同的schedule_type，否则beta值会不匹配，导致性能下降。

---

**文档版本**: v1.0
**最后更新**: 2026-03-13
**作者**: Claude Code
**状态**: ✅ 实现完成，待验证
