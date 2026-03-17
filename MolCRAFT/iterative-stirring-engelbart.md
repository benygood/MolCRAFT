# MolCRAFT 分阶段扩散实现计划

## Context

在 MolCRAFT 中实现类似 MolDiff 的分阶段扩散机制：

**问题背景**：
- 当前 MolCRAFT 中原子和化学键使用相同的时间参数 t，同时进行贝叶斯更新
- 希望实现分阶段扩散，让生成的分子具有更好的原子-键一致性
- 参考论文: MolDiff (ICML 2023)

**重要：时间方向理解**

| 模型 | t=0 | t=1/T | 训练方向（加噪） | 采样方向（恢复） |
|------|-----|-------|------------------|------------------|
| **MolDiff** | 数据 | 噪声/先验 | 0→T（数据→先验） | T→0（先验→数据） |
| **MolCRAFT** | 先验 | 数据 | 1→0（数据→先验） | 0→1（先验→数据） |

两者都是从数据加噪到先验，采样都是从先验恢复到数据。只是 t 的数值表示相反。

**MolDiff 分阶段逻辑**（训练加噪 0→T）：
- 阶段一 (0→600)：键快速加噪到先验（t_bond 更快到达 T）
- 阶段二 (600→T)：原子继续加噪到先验

**对应到 MolCRAFT 训练**（加噪 1→0）：
- 阶段一 (1→0.6)：键快速加噪到先验（t_bond 更快到达 0）
- 阶段二 (0.6→0)：原子继续加噪到先验

**采样时的行为**（MolCRAFT 0→1，先验→数据）：
- 阶段一 (0→0.6)：原子从先验恢复，键保持先验（因为训练时键已先到先验）
- 阶段二 (0.6→1)：键从先验恢复，原子继续精细化

**预期效果**：
- 训练时：全程学习原子损失，只有阶段二学习键损失（键在阶段一保持先验，阶段二开始恢复）
- 采样时：先对原子信息去噪（阶段一），再对化学键去噪（阶段二）

---

## 实现方案

### 核心思想：时间映射

让键比原子**更晚到达数据端**（在更大的 t 时才到达数据）：

```
原子时间: t_atom = t ∈ [0, 1]
键的有效时间: t_bond_eff = max(0, (t - bond_time_ratio) / (1 - bond_time_ratio))

bond_time_ratio = 0.6 时:
- t=0.3: t_atom=0.3, t_bond_eff=0.0 (键保持先验)
- t=0.6: t_atom=0.6, t_bond_eff=0.0 (键开始恢复)
- t=0.8: t_atom=0.8, t_bond_eff=0.5 (键恢复中)
- t=1.0: t_atom=1.0, t_bond_eff=1.0 (键恢复完成)
```

**解释**：
- 在 BFN 中，t=0 是先验，t=1 是数据
- 键的有效时间延迟了，在 t < bond_time_ratio 时键保持先验状态
- 这样采样时先恢复原子（t: 0→bond_time_ratio），再恢复键（t: bond_time_ratio→1）

### 阶段划分（训练时，t 从 U(0,1) 随机采样）

| 阶段 | 原子时间 t | 键有效时间 t_bond_eff | 原子损失 | 键损失 |
|------|----------|---------------------|---------|--------|
| 阶段一 | [0, bond_time_ratio) | =0 (先验) | ✓ 学习 | ✗ 不学习 |
| 阶段二 | [bond_time_ratio, 1] | (0, 1] | ✓ 学习 | ✓ 学习 |

**与 MolDiff 的对应**：
- MolDiff 训练（0→T 加噪）：阶段一学习键，阶段二不学习键
- MolCRAFT 训练（随机 t）：阶段一（t 小）不学习键，阶段二（t 大）学习键

这是因为时间 t 的含义相反：
- MolDiff: t=0 是数据，t=T 是先验
- MolCRAFT: t=0 是先验，t=1 是数据

所以 MolDiff 的"早期"（t 小，数据端）对应 MolCRAFT 的"晚期"（t 大，数据端）。

### 采样时的逻辑

采样 t: 0→1：
- **阶段一 (0 → bond_time_ratio)**：
  - 原子时间: 0 → bond_time_ratio
  - 键有效时间: 保持 0（先验状态）
  - 行为：原子开始恢复，键不更新（保持先验）
- **阶段二 (bond_time_ratio → 1)**：
  - 原子时间: bond_time_ratio → 1
  - 键有效时间: 0 → 1
  - 行为：原子继续精细化，键开始恢复

---

## 需要修改的文件

### 1. 配置文件: `configs/default.yaml`

添加新参数:
```yaml
dynamics:
  # ... 现有参数 ...

  # 分阶段扩散参数
  bond_time_ratio: !SUB ${bond_time_ratio}       # 键到达先验的时间比例，默认 0.6
  bond_loss_cutoff: !SUB ${bond_loss_cutoff}     # 大于此时间不计算键损失，默认 0.6
```

### 2. 训练入口: `train_bfn.py`

添加命令行参数:
```python
parser.add_argument('--bond_time_ratio', type=float, default=0.6,
                    help='Bond time ratio for phased diffusion')
parser.add_argument('--bond_loss_cutoff', type=float, default=0.6,
                    help='Do not compute bond loss when t > this value')
```

### 3. 核心模型: `core/models/bfn4sbdd.py`

#### 3.1 修改 `__init__` 方法 (约第80-180行)

添加新参数:
```python
class BFN4SBDDScoreModel(BFNBase):
    def __init__(
        self,
        # ... 现有参数 ...
        # 分阶段扩散参数
        bond_time_ratio=0.6,
        bond_loss_cutoff=0.6,
    ):
        super(BFN4SBDDScoreModel, self).__init__()
        # ... 现有代码 ...

        # 分阶段扩散参数
        self.bond_time_ratio = bond_time_ratio
        self.bond_loss_cutoff = bond_loss_cutoff
```

#### 3.2 修改 `loss_one_step` 方法 (第361-533行)

**关键修改点**:

1. **键的贝叶斯更新使用延迟时间** (约第406-412行):
```python
# 原代码:
t_bond = (t[halfedge_index[0]] + t[halfedge_index[1]]) / 2

# 修改为: 计算键的有效时间（延迟开始）
t_bond_base = (t[halfedge_index[0]] + t[halfedge_index[1]]) / 2
# 键的有效时间：当 t < bond_time_ratio 时为 0，之后线性增长到 1
t_bond_effective = torch.clamp(
    (t_bond_base - self.bond_time_ratio) / (1 - self.bond_time_ratio),
    min=0.0, max=1.0
)

# 使用 t_bond_effective 进行贝叶斯更新
theta_bond = self.discrete_var_bayesian_update(
    t_bond_effective, beta1=self.beta1_bond, x=halfedge_type_onehot, K=K_bond
)
```

2. **键损失只在阶段二计算** (约第508-531行):
```python
# Bond BFN loss - 只在阶段二计算（t >= bond_loss_cutoff，即键开始恢复时）
if halfedge_type is not None and p0_bond is not None:
    K_bond = self.num_bond_types

    # 检查是否在阶段二（t_bond_base >= bond_loss_cutoff，即 t_bond_effective > 0）
    t_bond_base = (t[halfedge_index[0]] + t[halfedge_index[1]]) / 2
    in_phase_two = (t_bond_base >= self.bond_loss_cutoff).squeeze(-1)

    if in_phase_two.any():
        # 只计算阶段二的键损失
        if not self.use_discrete_t:
            bond_loss = self.ctime4discrete_loss(
                t=t_bond_effective[in_phase_two],
                beta1=self.beta1_bond,
                one_hot_x=halfedge_type_onehot[in_phase_two],
                p_0=p0_bond[in_phase_two],
                K=K_bond,
                segment_ids=batch_halfedge[in_phase_two],
            )
        else:
            i_bond = (t_bond_effective[in_phase_two] * self.discrete_steps).int() + 1
            bond_loss = self.dtime4discrete_loss_prob(
                i=i_bond,
                N=self.discrete_steps,
                beta1=self.beta1_bond,
                one_hot_x=halfedge_type_onehot[in_phase_two],
                p_0=p0_bond[in_phase_two],
                K=K_bond,
                segment_ids=batch_halfedge[in_phase_two],
            )
    else:
        bond_loss = torch.zeros_like(closs)
else:
    bond_loss = torch.zeros_like(closs)
```

#### 3.3 修改 `interdependency_modeling` 方法 (第181-340行)

添加 `t_bond_effective` 参数用于键的时间嵌入:
```python
def interdependency_modeling(
    self,
    time,
    # ... 其他参数 ...
    theta_bond_t=None,
    halfedge_index=None,
    t_bond_effective=None,  # 【新增】键的有效时间
    # ...
):
    # ... 现有代码 ...

    # 修改键时间嵌入计算 (约第253-260行)
    if theta_bond_t is not None and halfedge_index is not None:
        theta_bond_input = 2 * theta_bond_t - 1
        if self.time_emb_dim > 0:
            # 【修改】使用有效时间进行嵌入
            if t_bond_effective is not None:
                bond_time_emb = self.time_emb_layer(t_bond_effective)
            else:
                # 兼容旧代码
                bond_time = (time[halfedge_index[0]] + time[halfedge_index[1]]) / 2
                bond_time_emb = self.time_emb_layer(bond_time)
            theta_bond_input = torch.cat([theta_bond_input, bond_time_emb], -1)
        # ...
```

#### 3.4 修改 `sample` 方法 (第535-892行)

**关键修改点**:

1. **计算阶段分界步数** (在循环开始前):
```python
# 计算阶段分界步数（阶段一：0→phase_one_steps，阶段二：phase_one_steps→sample_steps）
phase_one_steps = int(sample_steps * self.bond_time_ratio)
```

2. **键使用有效时间** (在循环内，约第609-638行):
```python
# 计算键的有效时间（延迟开始）
t_bond_effective = None
if theta_bond_t is not None and halfedge_index is not None:
    t_bond_base = (t[halfedge_index[0]] + t[halfedge_index[1]]) / 2
    # 键的有效时间：当 t < bond_time_ratio 时为 0，之后线性增长到 1
    t_bond_effective = torch.clamp(
        (t_bond_base - self.bond_time_ratio) / (1 - self.bond_time_ratio),
        min=0.0, max=1.0
    )

# 调用 interdependency_modeling
coord_pred, p0_h_pred, k_hat, p0_bond_pred = self.interdependency_modeling(
    time=t,
    # ... 其他参数 ...
    theta_bond_t=theta_bond_t,
    halfedge_index=halfedge_index,
    t_bond_effective=t_bond_effective,  # 传入有效时间
)
```

3. **键更新只在阶段二** (贝叶斯更新部分，约第709-717行):
```python
# Bond BFN update - 只在阶段二（i > phase_one_steps，即 t > bond_time_ratio）
if i > phase_one_steps:
    if theta_bond_t is not None and p0_bond_pred is not None:
        # 计算键的有效时间值
        t_bond_eff_val = max(0.0, (i / sample_steps - self.bond_time_ratio) / (1 - self.bond_time_ratio))
        alpha_bond = self.beta1_bond * (2 * i - 1) / (sample_steps ** 2)

        k_bond = torch.distributions.Categorical(
            probs=p0_bond_pred.clamp(min=1e-6)
        ).sample()
        e_k_bond = F.one_hot(k_bond, num_classes=K_bond).float()
        mean_bond = alpha_bond * (K_bond * e_k_bond - 1)
        std_bond = torch.full_like(mean_bond, fill_value=alpha_bond * K_bond).sqrt()
        y_bond = mean_bond + std_bond * torch.randn_like(e_k_bond)
        theta_bond_prime = torch.exp(y_bond) * theta_bond_t
        theta_bond_t = theta_bond_prime / theta_bond_prime.sum(dim=-1, keepdim=True)
# 阶段一：键不更新，保持先验状态（theta_bond_t 保持 1/K 均匀分布）
```

4. **End-back 采样策略的修改** (约第736-748行):
```python
# Bond BFN end_back update - 只在阶段二更新
if theta_bond_t is not None and p0_bond_pred is not None:
    if i > phase_one_steps:
        t_bond_next = t_bond_effective  # 使用有效时间
        if self.sampling_strategy == "end_back_pmf":
            theta_bond_t = self.discrete_var_bayesian_update(
                t_bond_next, beta1=self.beta1_bond, x=p0_bond_pred, K=K_bond
            )
        else:
            bond_mode = torch.argmax(p0_bond_pred, dim=-1)
            bond_mode_pred = F.one_hot(bond_mode, num_classes=K_bond).float()
            theta_bond_t = self.discrete_var_bayesian_update(
                t_bond_next, beta1=self.beta1_bond, x=bond_mode_pred, K=K_bond
            )
# 阶段一： 键不更新，```

---

## 验证方案

### 1. 单元测试
```python
# 测试时间映射
def test_time_mapping():
    bond_time_ratio = 0.6
    t_values = [0.0, 0.3, 0.6, 0.8, 1.0]
    for t in t_values:
        t_eff = max(0.0, (t - bond_time_ratio) / (1 - bond_time_ratio))
        print(f"t={t} -> t_bond_eff={t_eff}")
# 期望输出:
# t=0.0 -> t_bond_eff=0.0 (键保持先验)
# t=0.3 -> t_bond_eff=0.0 (键保持先验)
# t=0.6 -> t_bond_eff=0.0 (键开始恢复)
# t=0.8 -> t_bond_eff=0.5 (键恢复中)
# t=1.0 -> t_bond_eff=1.0 (键恢复完成)
```

### 2. 训练验证
- 运行短训练 (1-2 epochs)
- 检查损失值: closs 和 dloss 应全程非零
- 检查 bond_loss:
  - t < 0.6 时接近零（阶段一，键保持先验，不学习）
  - t >= 0.6 时非零（阶段二，键开始恢复，学习键损失）
- 监控 wandb 日志中的损失曲线

### 3. 采样验证
- 使用修改后的模型进行采样
- 检查生成的分子质量 (有效性、键的合理性)
- 对比原始模型的生成质量

---

## 实现步骤

1. **Step 1**: 修改配置文件 `configs/default.yaml`，添加新参数
2. **Step 2**: 修改 `train_bfn.py` 添加命令行参数
3. **Step 3**: 修改 `bfn4sbdd.py` 的 `__init__` 方法
4. **Step 4**: 修改 `bfn4sbdd.py` 的 `loss_one_step` 方法
5. **Step 5**: 修改 `bfn4sbdd.py` 的 `interdependency_modeling` 方法
6. **Step 6**: 修改 `bfn4sbdd.py` 的 `sample` 方法
7. **Step 7**: 运行测试验证

---

## 关键文件列表

| 文件路径 | 修改内容 |
|----------|----------|
| `configs/default.yaml` | 添加 bond_time_ratio, bond_loss_cutoff 参数 |
| `train_bfn.py` | 添加命令行参数 |
| `core/models/bfn4sbdd.py` | 修改 __init__, loss_one_step, interdependency_modeling, sample 方法 |
