# SolarRHN

用太阳中微子探测器搜索MeV质量范围的右手中微子 (Right-Handed Neutrino)

> Search for right-handed neutrino in the MeV mass range with solar neutrino detector

---

## 📋 目录

- [项目概述](#项目概述)
- [快速开始](#快速开始)
- [主要功能](#主要功能)
- [参数扫描工作流](#参数扫描工作流)
- [从CSV文件计算电子谱](#从csv文件计算电子谱)
- [输出文件说明](#输出文件说明)
- [高级用法](#高级用法)
- [技术细节](#技术细节)
- [常见问题](#常见问题)
- [文件结构](#文件结构)

---

## 项目概述

本项目用于模拟太阳中产生的右手中微子(RHN)在飞向地球途中的衰变，以及衰变产生的左手中微子与地球探测器中电子的散射过程。

### 物理过程

1. **RHN产生**：太阳中的⁸B中微子通过混合产生RHN
2. **RHN衰变**：RHN在太阳-地球之间衰变为左手中微子和电子对
3. **中微子散射**：左手中微子与探测器中的电子发生散射
4. **信号探测**：探测散射产生的反冲电子

### 主要参数

- **U²**：RHN与标准模型中微子的混合参数平方（典型范围：10⁻⁶ - 10⁻¹）
- **MH**：RHN质量（单位：MeV，典型范围：2-12 MeV）

---

## 快速开始

### 环境要求

```bash
# Python 3.x with required packages:
pip install numpy pandas scipy matplotlib
```

### 1. 快速测试（推荐新用户）

测试单组参数以验证代码正常工作：

```bash
python run_quick_test.py
```

这将计算 U²=0.1, MH=4.0 MeV 的中微子和电子分布，并生成所有图表。

### 2. 完整参数扫描

编辑 `toymc.py` 设置参数范围：

```python
# 定义扫描的参数值
U2_values = [0.1, 0.01, 0.001]
MH_values = [2.0, 4.0, 6.0, 8.0, 10.0]
```

format
然后运行：

```bash
python toymc.py
```

### 3. 从已有数据计算电子谱

如果已有中微子分布CSV文件，可直接计算散射电子谱：

```bash
python compute_electron_from_csv.py ./data/simulation/diff_El_costheta_M4.0_U1.0e-01.csv
```

---

## 主要功能

### ✨ 核心计算功能

#### 1. RHN衰变产生的中微子分布

```python
from utils import get_and_save_nuL_El_costheta_decay_in_flight

diff_El, diff_costheta, diff_cosphi, diff_El_costheta = \
    get_and_save_nuL_El_costheta_decay_in_flight(
        spectrum_nuL_orig,  # 太阳中微子谱
        U2=0.01,            # 混合参数
        MH=4.0,             # RHN质量 (MeV)
        savepath='./data/'
    )
```

**输出**：

- 1D能量分布
- 1D角度分布（相对于太阳-地球方向）
- 2D能量-角度分布
- CSV数据文件

#### 2. 中微子-电子散射

```python
from utils import get_and_save_nuL_scatter_electron_El_costheta

electron_2d, e_bins, costheta_bins, _ = \
    get_and_save_nuL_scatter_electron_El_costheta(
        diff_El_costheta_nu,  # 中微子2D分布
        savepath='./data/',
        N_int_local=100000    # 蒙特卡罗样本数
    )
```

**特点**：

- 使用**改进的方位角采样**方法（12个φ采样点）
- 精确考虑 `cos(θ_lab) = cos(θ_in)·cos(θ_s) - sin(θ_in)·sin(θ_s)·cos(φ)`
- 对大角度入射显著提升精度

#### 3. 从CSV文件计算

```python
from utils import get_and_save_nuL_scatter_electron_El_costheta_from_csv

electron_2d, e_bins, costheta_bins, _ = \
    get_and_save_nuL_scatter_electron_El_costheta_from_csv(
        csv_path='./data/diff_El_costheta_M4.0_U1.0e-01.csv',
        savepath='./results/',
        N_int_local=100000,
        plot=True
    )
```

---

## 参数扫描工作流

### 自动化流程

`toymc.py` 脚本自动遍历 U² × MH 参数空间，对每组参数：

1. ✅ 计算RHN衰变产生的中微子分布
2. ✅ 生成中微子2D热图 + 1D能量谱 + 1D角度分布
3. ✅ 计算中微子-电子散射
4. ✅ 生成电子2D热图 + 1D能量谱 + 1D角度分布
5. ✅ 保存所有数据CSV文件
6. ✅ 生成汇总报告

### 配置选项

#### 参数网格设置

```python
# 对数间隔的U²值
U2_values = np.logspace(-5, -1, 20)  # 10⁻⁵ to 10⁻¹, 20个点

# 均匀间隔的MH值
MH_values = np.linspace(2.0, 12.0, 50)  # 2到12 MeV, 50个点

# 特定值
U2_values = [0.1, 0.05, 0.01, 0.005, 0.001]
MH_values = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0]
```

#### 能量网格设置

```python
# 粗网格（快速测试）
energy = np.arange(0.0, 16.0, step=0.5)

# 中等精度（推荐）
energy = np.arange(0.0, 16.0, step=0.2)

# 高精度（计算较慢）
energy = np.arange(0.0, 16.0, step=0.1)
```

#### 蒙特卡罗样本数

```python
# 快速测试
N_int_local = 10000

# 标准精度（推荐）
N_int_local = 100000

# 高精度
N_int_local = 500000
```

#### 并行处理

```python
# 顺序处理（默认，便于调试）
results = []
for args in args_list:
    result = process_single_parameter_set(args)
    results.append(result)

# 并行处理（更快）
import multiprocessing as mp
with mp.Pool(processes=4) as pool:  # 调整为CPU核心数
    results = pool.map(process_single_parameter_set, args_list)
```

### 输出结构

```txt
plots_grid_scan/                                   # 主输出目录
├── summary.txt                                    # 所有参数汇总
│
├── U2_1.00e-01_MH_2.0/                           # 每组参数一个子目录
│   ├── neutrino_2d_U2_1.00e-01_MH_2.0.pdf       # 中微子2D分布
│   ├── neutrino_energy_1d_U2_1.00e-01_MH_2.0.pdf # 中微子能量谱
│   ├── neutrino_angle_1d_U2_1.00e-01_MH_2.0.pdf  # 中微子角分布
│   ├── electron_2d_U2_1.00e-01_MH_2.0.pdf        # 电子2D分布
│   ├── electron_energy_1d_U2_1.00e-01_MH_2.0.pdf # 电子能量谱
│   ├── electron_angle_1d_U2_1.00e-01_MH_2.0.pdf  # 电子角分布
│   ├── diff_El_M2.0_U1.0e-01.csv                 # 能量分布数据
│   ├── diff_costheta_M2.0_U1.0e-01.csv           # 角度分布数据
│   ├── diff_El_costheta_M2.0_U1.0e-01.csv        # 2D分布数据
│   └── scattered_electrons_2d_lab.csv            # 电子2D分布数据
│
├── U2_1.00e-01_MH_4.0/
│   └── ...
└── ...
```

---

## 从CSV文件计算电子谱

### 单文件处理

```bash
# 基本用法
python compute_electron_from_csv.py <neutrino_csv_file>

# 示例
python compute_electron_from_csv.py ./data/simulation/diff_El_costheta_M4.0_U1.0e-01.csv

# 自定义输出目录
python compute_electron_from_csv.py input.csv --output-dir ./results/

# 调整蒙特卡罗样本数
python compute_electron_from_csv.py input.csv --samples 200000

# 只计算不绘图（加快速度）
python compute_electron_from_csv.py input.csv --no-plot
```

**优势**：

- 无需重新运行RHN衰变计算（节省时间）
- 可以使用不同的MC样本数重新计算
- 适合调试和参数优化

### 批量处理

```bash
# 处理目录中所有中微子CSV文件
python batch_compute_electrons.py [directory]

# 示例
python batch_compute_electrons.py ./plots_grid_scan/

# 先列出文件不处理
python batch_compute_electrons.py ./data/ --list-only

# 批量处理并调整参数
python batch_compute_electrons.py ./data/ --samples 50000 --no-plot
```

**输出**：

- 每个文件对应的电子谱CSV和图表
- `electron_batch_summary.csv` 汇总表格

### Python API

```python
# 列出所有中微子CSV文件
from utils import list_neutrino_csv_files
files = list_neutrino_csv_files('./data/simulation/')
for f in files:
    print(f"{f['filename']}: MH={f['MH']}, U2={f['U2']}")

# 批量处理
from utils import batch_compute_electrons_from_csv
results = batch_compute_electrons_from_csv(
    directory='./data/',
    N_int_local=100000,
    plot=True
)
```

---

## 输出文件说明

### 图表类型

1. **neutrino_2d** - 二维热图
   - X轴：中微子能量 (MeV)
   - Y轴：cos(θ)，θ是相对于太阳-地球方向的角度
   - 颜色：通量密度 (cm⁻² s⁻¹ MeV⁻¹ sr⁻¹)

2. **neutrino_energy_1d** - 一维能量谱
   - X轴：能量 (MeV)
   - Y轴：通量 (MeV⁻¹ cm⁻² s⁻¹)
   - 对2D分布在角度上积分的结果

3. **neutrino_angle_1d** - 一维角度分布
   - X轴：cos(θ)
   - Y轴：通量 (sr⁻¹ cm⁻² s⁻¹)
   - 对2D分布在能量上积分的结果

4. **electron_2d** - 二维热图
   - X轴：反冲电子能量 (MeV)
   - Y轴：cos(θ)，散射角度
   - 颜色：事例率密度

5. **electron_energy_1d** - 一维能量谱
   - X轴：能量 (MeV)
   - Y轴：事例率 (MeV⁻¹)

6. **electron_angle_1d** - 一维角度分布
   - X轴：cos(θ)
   - Y轴：事例率 (sr⁻¹)

### CSV数据格式

**1D分布** (如 `diff_El_M*.csv`, `diff_costheta_M*.csv`):

```csv
energy,value
0.0,0.0
0.2,1.234e-05
...
```

**2D分布** (如 `diff_El_costheta_M*.csv`):

```csv
energy,costheta,value
0.0,-1.0,0.0
0.0,-0.99,0.0
0.2,-1.0,1.234e-06
...
```

### 汇总文件

`summary.txt` 包含：

- **U2**: 混合参数平方
- **MH**: RHN质量 (MeV)
- **Neutrino_Flux**: 中微子总通量
- **Neutrino_2D_Integral**: 2D分布积分（验证归一化）
- **Electron_Success**: 电子散射计算是否成功
- **Output_Dir**: 输出目录路径

---

## 高级用法

### 工作流示例

#### 场景1：重新计算某个参数点的电子谱（高精度）

```bash
# 1. 找到对应的中微子CSV文件
ls ./plots_grid_scan/U2_1.00e-01_MH_4.0/diff_El_costheta_*.csv

# 2. 重新计算电子谱（使用更多样本提高精度）
python compute_electron_from_csv.py \
    ./plots_grid_scan/U2_1.00e-01_MH_4.0/diff_El_costheta_M4.0_U1.0e-01.csv \
    --samples 500000 \
    --output-dir ./high_precision/
```

#### 场景2：分离计算中微子和电子

```bash
# Step 1: 只计算中微子（修改toymc.py注释掉Step 5-8）
python toymc.py

# Step 2: 批量计算所有电子谱
python batch_compute_electrons.py ./plots_grid_scan/ --samples 100000

# Step 3: 检查汇总
cat ./plots_grid_scan/electron_batch_summary.csv
```

#### 场景3：探索MC样本数的影响

```bash
# 不同精度对比
python compute_electron_from_csv.py input.csv --samples 10000 --output-dir ./test_10k/
python compute_electron_from_csv.py input.csv --samples 100000 --output-dir ./test_100k/
python compute_electron_from_csv.py input.csv --samples 500000 --output-dir ./test_500k/

# 比较三个目录中的图表
```

### 只计算中微子分布（跳过电子散射）

修改 `process_single_parameter_set` 函数，注释掉 Step 5-8：

```python
# Step 5-8: 电子散射部分
# 注释掉以下代码以跳过电子计算
# electron_2d, ... = get_and_save_nuL_scatter_electron_El_costheta(...)
```

### 自定义能量阈值和探测器效率

在散射计算后添加筛选：

```python
# 应用能量阈值
E_threshold = 3.0  # MeV
mask = e_centers > E_threshold
electron_2d_filtered = electron_2d[mask, :]

# 应用角度接受度
costheta_min = 0.9  # 只接受前向事例
mask_angle = costheta_centers > costheta_min
electron_2d_filtered = electron_2d_filtered[:, mask_angle]
```

---

## 技术细节

### 物理模型

#### RHN衰变宽度

```python
Γ_total = Γ_νℓℓ + Γ_ννν
```

其中：

- `Γ_νℓℓ`: RHN → ν + e⁺ + e⁻ 衰变宽度
- `Γ_ννν`: RHN → ν + ν + ν̄ 衰变宽度
- 衰变宽度正比于 U² × M_H⁵

#### 洛伦兹变换

RHN在太阳中产生后以接近光速飞向地球。衰变产物的能量和角度分布需要从RHN静止系变换到实验室系：

```txt
E_lab = γ(E_cms + β·p_cms·cos(θ_cms))
cos(θ_lab) = (cos(θ_cms) + β) / (1 + β·cos(θ_cms))
```

#### 散射截面

中微子-电子散射使用标准模型弱相互作用截面，考虑：

- 荷电流贡献 (CC)
- 中性流贡献 (NC)
- 相对论运动学

#### 改进的角度映射

**完整公式**（含方位角依赖）：

```txt
cos(θ_lab) = cos(θ_in)·cos(θ_scatter) - sin(θ_in)·sin(θ_scatter)·cos(φ)
```

**实现**：

- 对每个散射角，采样12个方位角 φ ∈ [0, 2π]
- 精确考虑 sin 项的贡献
- 对 sin(θ_in) 较大时显著改善精度

**简化近似**（仅适用于小角度）：

```txt
<cos(θ_lab)> ≈ cos(θ_in)·cos(θ_scatter)
```

### 数值方法

#### 能量网格重采样

使用保守的bin-average方法，确保通量守恒：

```python
def resample_bin_average(x_src, y_src, x_tgt):
    # 对每个目标bin，积分源分布并除以bin宽度
    # 确保 ∫ y_src dx = ∫ y_tgt dx
```

#### 蒙特卡罗散射计算

`scatter_electron_spectrum` 使用重要性采样：

- 根据中微子通量对能量采样
- 根据散射截面对散射角采样
- N_int_local 控制样本数（默认100000）

#### 归一化

所有分布存储为**密度** (dN/dE/dΩ)：

- 积分时自动乘以bin宽度
- 确保 `∫∫ ρ(E,θ) dE dθ = N_total`

### 数值验证

每次计算后验证守恒性：

```txt
1D能量积分 ≈ 1D角度积分 ≈ 2D积分 ≈ N_total
```

典型误差 < 0.1%（数值精度限制）

---

## 常见问题

### Q: 为什么电子散射计算这么慢？

**A**: 每个入射角度都需要单独调用蒙特卡罗散射计算，这是物理上必需的：

- 不同角度的中微子能量分布不同（洛伦兹boost）
- 散射截面强烈依赖能量
- 不能简单地"先积分再散射"

**优化方法**：

- 减少 `--samples`（如降到50000）
- 减少角度bins（如从201降到51）
- 使用粗一点的能量网格
- 使用 `--no-plot` 跳过绘图
- 启用并行处理

### Q: 2D积分和1D积分不一致怎么办？

**A**: 检查归一化方法。当前版本使用：

```python
ρ[i,j] = N_total × dist[i,j] / Σ(dist)
```

确保：

- 分布存储为密度（不预乘bin宽度）
- 积分函数会自动推断并乘以bin宽度
- 所有积分应在0.1%误差内一致

### Q: 某些参数组合电子散射失败？

**A**: 常见原因：

- 中微子通量太低（小U²或大MH）→ 统计涨落
- 数值问题（极端参数值）→ 检查物理合理性
- 内存不足 → 减少样本数或网格点

检查 `summary.txt` 中的 `Electron_Success` 列。

### Q: 如何加快大规模参数扫描？

**A**:

1. **启用并行处理**（最有效）：

   ```python
   with mp.Pool(processes=8) as pool:
       results = pool.map(process_single_parameter_set, args_list)
   ```

2. **降低精度**（适当）：
   - 能量步长：0.5 MeV（代替0.2）
   - MC样本：50000（代替100000）
   - 角度bins：51（代替201）

3. **分离计算**：
   - 先算所有中微子分布（快）
   - 后用批处理算电子（可在后台）

### Q: 如何验证结果正确性？

**A**:

1. **物理检查**：
   - 前向峰化（中微子集中在 cosθ ≈ 1）
   - 能量范围合理（不超过 E_⁸B ≈ 15 MeV）
   - 通量随U²线性增长

2. **数值检查**：
   - 所有积分一致（误差 < 0.1%）
   - 增加MC样本数，结果收敛
   - 与已发表文献对比

3. **单元测试**：

   ```bash
   python run_quick_test.py  # 检查标准案例
   ```

### Q: CSV文件格式错误？

**A**: 确保：

- 3列：`energy,costheta,value`
- 逗号分隔
- 有header行（或无header都可以）
- 数值格式正确（科学计数法：`1.234e-05`）

由 `get_and_save_nuL_El_costheta_decay_in_flight` 自动生成的文件格式正确。

### Q: 绘图乱码或中文显示问题？

**A**: 配置matplotlib字体：

```python
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS
plt.rcParams['axes.unicode_minus'] = False
```

### Q: 内存不足？

**A**:

- 减少能量网格点数
- 减少角度bins
- 减少MC样本数
- 使用顺序处理（不并行）
- 分批处理参数点

---

## 文件结构

### 核心代码

```txt
SolarRHN/
├── core/                             # 核心物理计算包
│   ├── __init__.py                  # 包初始化
│   ├── constants.py                 # 物理常数
│   ├── rhn_physics.py               # RHN衰变物理
│   ├── transformations.py           # 洛伦兹变换
│   ├── decay_distributions.py       # 衰变运动学
│   ├── spectrum_utils.py            # 能谱工具
│   ├── neutrino_electron_scattering.py  # ν-e散射
│   ├── electron_scattering.py       # 电子散射计算
│   ├── sampling.py                  # 蒙特卡罗采样
│   └── tools.py                     # 辅助工具（计时器等）
│
├── workflows.py                      # 完整分析流程
├── plot_helpers.py                   # 绘图工具
└── toymc.py                          # 主参数扫描脚本
```

### 脚本工具

```txt
├── run_quick_test.py                 # 快速测试脚本
├── compute_electron_from_csv.py      # 单文件CSV处理
├── batch_compute_electrons.py        # 批量CSV处理
└── demo_csv_electron.py              # 演示脚本
```

### 数据和输出

```txt
├── data/
│   ├── 8BSpectrum.csv               # ⁸B太阳中微子能谱
│   └── simulation/                   # 模拟结果
│
└── plots_grid_scan/                  # 参数扫描输出
    ├── summary.txt
    └── U2_*_MH_*/
```

### 文档

```txt
└── README.md                         # 本文档（整合版）
```

---

## 引用和参考

如果使用本代码，请引用相关物理文献：

- RHN物理模型：[相关论文]
- 太阳中微子：[Bahcall et al.]
- 中微子-电子散射：[Standard Model]

---

## 更新日志

### 最新版本

- ✅ 改进的角度映射（方位角采样）
- ✅ CSV文件直接计算功能
- ✅ 批量处理工具
- ✅ 完善的错误处理
- ✅ 归一化修复（2D/1D一致性）

### 历史版本

- v1.0: 基础RHN衰变和散射计算
- v1.1: 添加参数扫描功能
- v1.2: Jacobian修正
- v2.0: 完整重构，添加CSV工具

---

## 贡献者

- 主要开发：Yutao Zhu, Zhicai Zhang

---

## 许可证

MIT

---

## 联系方式

- 问题反馈：[GitHub Issues]
- 邮箱：[zhu-yt24@mails.tsinghua.edu.cn]

---

**Happy hunting for RHNs! 🔬✨**
