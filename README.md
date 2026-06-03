# SolarRHN

太阳源右手中微子（RHN, MeV 标度）在探测器中的可见信号与排除灵敏度。

---

## 1. 物理流程

### S1（直接衰变）

$$
\nu_H \to \nu + e^+ + e^-
$$

RHN 在探测器内部衰变，直接观测 e⁺e⁻ 能谱。**JUNO 版同时拟合 B8（ES）+ B12（β⁻）+ C10（β⁺）三种本底**，使用 pyhf 做条件拟合。

### S2（衰变 + 散射）

$$
\nu_H \to \nu,\qquad \nu + e^- \to \nu + e^-
$$

RHN 在日地全程衰变产生中微子，中微子到达探测器后与电子散射。**JUNO 无方向信息**——将 2D (E_e, cosθ) 投影到 1D 能谱后做拟合。

### 统计量

Poisson 似然比（Asimov），pyhf 做条件拟合：

$$
\Delta\chi^2 = 2\left[-\ln L(\mu=1, \hat{\hat{\theta}}) + \ln L(\mu=0, \hat{\theta})\right]
$$

90% CL 阈值：Δχ² = 2.71（1 dof, one-sided）。

---

## 2. 全部脚本

### 排除曲线（核心输出）

| 脚本 | 探测器 | 本底 | 信号 | 方法 |
|---|---|---|---|---|
| `upper_limit_s1.py` | Borexino | B8 | e⁺e⁻ | pyhf, 1 normfactor |
| `upper_limit_s1_juno.py` | **JUNO** | B8 + B12 + C10 | e⁺e⁻ | pyhf, 3 normfactors |
| `upper_limit_s2_juno.py` | **JUNO** | B8 + B12 + C10 | ν→e⁻ 散射 | 读 toymc_s2 NPZ → pyhf |
| `borexino_upper_limit.py` | Borexino | B8 | e⁺e⁻ | Cowan 渐近公式, ±1σ/±2σ band |

### 信号生成

| 脚本 | 用途 |
|---|---|
| `toymc_s1.py` | S1 参数扫描：生成 e⁺e⁻ 能谱，计算 χ² 网格 |
| `toymc_s2.py` | **S2 前置**：对每个 (mH, u2) 计算 decayed ν → scattered e⁻ 2D，存为 NPZ |
| `compute_electron_from_csv.py` | 从已保存的 neutrino CSV 计算散射电子谱（演示/单点用） |

### Borexino 数据分析

| 脚本 | 用途 |
|---|---|
| `borexino_data_exclusion.py` | Borexino 2D 排除扫描（argparse，支持 MC 阈值校准） |
| `borexino_data_fit.py` | Borexino 数据拟合（iminuit，χ²） |
| `borexino_data_fit_pyhf.py` | Borexino 数据拟合（pyhf 版） |
| `reproduce_borexino_fit.py` | 复现 Borexino 发表结果：Asimov 排除 + toy MC ±1σ/±2σ band |

### 灵敏度和绘图

| 脚本 | 用途 |
|---|---|
| `plot_exclusion_csv.py` | **ROOT 综合图**：JUNO S1 + JUNO S2 + Borexino 参考线 |
| `eepair_exclusion_s1.py` | e⁺e⁻ 信号计数 contour（count = 1 近似灵敏度边界） |
| `plot_s2_exclusion.py` | S2 排除曲线（Borexino 2D 版，读 toymc_s2 NPZ，含角度） |
| `plot_s2_results.py` | 画单个或全部 S2 参数点的 signal/background 2D 图 |
| `ploter.py` | 通用绘图工具（2D map、1D 能谱/角分布） |

### 核心库

| 文件 | 内容 |
|---|---|
| `core/` | 物理计算（RHN 衰变、ν-e 散射、统计、常数） |
| `workflows.py` | S2 流程拼装：`process2_single_parameter_set` 等 |

---

## 3. 运行顺序（JUNO）

```bash
# S1：直接扫描（on-the-fly 信号计算 + pyhf）
python upper_limit_s1_juno.py

# S2：先生成 NPZ，再做排除
python toymc_s2.py                        # → plots_grid_scan_s2/*/electron_data.npz
python upper_limit_s2_juno.py             # 读 NPZ → pyhf → CSV

# 绘图
python plot_exclusion_csv.py              # S1 + S2 + Borexino 参考线
```

---

## 4. 本底数据（JUNO）

| 本底 | 文件 | 原始单位 | 原始曝光 | 实际曝光 |
|---|---|---|---|---|
| B8 ES | `data/juno/solar_juno_fv_5mev.root` | counts / 0.02 MeV | 2 yr | 2 yr |
| B12 β⁻ | `data/juno/b12_bkg.csv` | counts / 0.1 MeV | 10 yr | **×0.2 → 2 yr** |
| C10 β⁺ | `data/juno/c10_bkg.csv` | counts / 0.1 MeV | 10 yr | **×0.2 → 2 yr** |

pyhf workspace：三个本底各自独立 normfactor，B8 下界 0.3（不可消失），B12/C10 下界 0（可到零）。上界均为 5.0。

---

## 5. 拟合配置

| 参数 | JUNO | Borexino |
|---|---|---|
| 能量范围 | 5.0–12.8 MeV | 4.8–12.8 MeV |
| Bin 宽度 | 0.2 MeV | 0.2 MeV |
| 能量分辨率 | 3% | 5% |
| 探测器质量 | 16.2 kt（FV > 5 MeV） | 100 t |
| 曝光时间 | 2 yr | 446.2 d |
| 本底 | B8 + B12 + C10 | B8 |

### S2 特殊处理

- `scatter_electron_spectrum`（numba 版）硬编码 `_Ne = 1.673e32`（500t）
- `upper_limit_s2_juno.py` 内乘 `Ne / 1.673e32 ≈ 32.4` 修正到 JUNO
- NPZ 中的 2D 信号 sum over cosθ → 1D，再施加 3% 能量分辨率

---

## 6. 输出格式

CSV（`upper_limit_bands*.csv`）：

```
mH,u2_low,u2_high
2.000000,2.2236e-03,1.0000e-01
...
```

- 有效排除点：`u2_low < U2_MAX`（`U2_MAX = 1e-1`）
- 无排除能力：`u2_low = u2_high = U2_MAX`（sentinel，绘图时过滤）
- S1 为窗口形（同时有上下界），S2 低 mH 可能只有下界

---

## 7. 依赖

```bash
pip install numpy scipy matplotlib numba uproot pyhf tqdm iminuit
```

ROOT 绘图需本地 PyROOT 及 `pytools/rt_ploter.py`。

---

## 8. 目录结构

```
SolarRHN/
├── core/                          # 物理计算库
│   ├── constants.py               # 探测器参数、物理常数
│   ├── rhn_physics.py             # RHN 衰变宽度、寿命、谱
│   ├── decay_and_scattering.py    # 衰变+散射流程
│   ├── neutrino_electron_scattering.py  # ν-e 散射截面与谱
│   ├── stats.py                   # 能量/角度分辨率卷积、χ²
│   └── ...
├── workflows.py                   # S2 流程拼装与数据保存
├── data/
│   ├── 8BSpectrum.csv             # ⁸B 中微子能谱
│   ├── Borexino_exclusion.csv     # Borexino 发表排除曲线
│   └── juno/                      # JUNO 本底数据
│       ├── solar_juno_fv_5mev.root
│       ├── b12_bkg.csv
│       └── c10_bkg.csv
├── plots/juno/                    # 输出
│   ├── upper_limit_s1/            # S1 CSV + PDF
│   └── upper_limit_s2/            # S2 CSV + PDF
├── plots_grid_scan_s2/            # toymc_s2 输出（NPZ）
├── upper_limit_s1.py              # S1 排除（Borexino）
├── upper_limit_s1_juno.py         # S1 排除（JUNO）
├── upper_limit_s2_juno.py         # S2 排除（JUNO，读 NPZ）
├── plot_exclusion_csv.py          # ROOT 综合图
├── toymc_s1.py / toymc_s2.py      # 信号生成
├── borexino_*.py                  # Borexino 分析系列
└── README.md
```
