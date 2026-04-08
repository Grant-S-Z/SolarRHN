# SolarRHN

用太阳中微子探测器搜索 MeV 质量范围的右手中微子 (Right-Handed Neutrino)

> Search for right-handed neutrino in the MeV mass range with solar neutrino detector

---

## 📋 目录

- [项目概述](#项目概述)
- [快速开始](#快速开始)
- [两种分析方法](#两种分析方法)
- [核心功能](#核心功能)
- [使用指南](#使用指南)
- [输出文件](#输出文件)
- [文件结构](#文件结构)
- [联系方式](#联系方式)

---

## 项目概述

本项目模拟太阳中产生的右手中微子(RHN)在飞向地球途中的衰变，以及衰变产生的信号。提供两种分析方法：

1. **S1 方法（直接衰变分析）**：分析 RHN 直接衰变为电子-正电子对的信号
2. **S2 方法（完整衰变-散射链）**：分析 RHN 衰变为中微子，再经散射产生电子的完整物理链

### 主要参数
- **U²**：RHN 与标准模型中微子的混合参数平方（范围：10⁻⁷ - 10⁰）
- **MH**：RHN 质量（范围：2-14 MeV）

---

## 快速开始

### 安装依赖
```bash
pip install numpy pandas scipy matplotlib numba uproot tqdm
```

### 运行示例

#### S1 方法（直接衰变分析）
```bash
# 并行参数扫描（推荐）- Asimov χ²（似然比）
python toymc_s1_parallel.py

# 顺序版本（调试用）- Asimov χ²
python toymc_s1.py
```

#### S2 方法（完整衰变-散射链）
```bash
# 快速测试（单点）
python run_quick_test.py

# 自定义参数扫描
python toymc_s2.py
```

#### 从已有数据计算
```bash
# 从 CSV 计算电子谱
python compute_electron_from_csv.py ./data/simulation/diff_El_costheta_M4.0_U1.0e-01.csv

# 批量处理
python batch_compute_electrons.py ./data/simulation/
```

---

## 两种分析方法

### S1 方法：直接衰变分析
- 模拟 RHN → ν + e⁺ + e⁻ 衰变
- 直接分析衰变电子信号
- 计算 χ² 统计量进行排除分析
- 支持能量分辨率和角度分辨率卷积

### S2 方法：完整衰变-散射链
- 模拟 RHN → ν 衰变
- 计算 ν + e → ν + e 散射
- 生成完整的能量和角度分布
- 提供详细的 2D 可视化

---

## 核心功能

### 物理计算
- RHN 产生和衰变宽度计算
- 洛伦兹变换（CMS 到实验室系）
- 中微子-电子散射截面
- 蒙特卡罗抽样和重要性采样

### 数据分析
- 能量分辨率卷积（Gaussian smearing）
- 角度分辨率卷积
- χ² 统计量计算：支持两种方法
  - **Asimov χ²**（似然比检验）：`chi2_poisson_asimov()`
  - **Pearson χ²**：`pearson_chi2()`，χ² = Σ S²/B
- 排除区域生成（90% CL）

### 可视化
- 2D 热图（线性/对数尺度）
- 1D 能量和角度分布
- 自动坐标轴优化
- ROOT 和 Matplotlib 输出

---

## 使用指南

### 配置参数扫描
编辑相应的脚本文件调整参数：
```python
# toymc_s1_parallel.py 或 toymc_s2.py 中
U2_values = np.logspace(-7, -1, 7)    # 混合参数
MH_values = np.linspace(2.0, 14.0, 7) # RHN 质量
```

### 主要脚本说明
- `toymc_s1_parallel.py`：S1 方法的并行实现（推荐）
- `toymc_s1.py`：S1 方法的顺序实现
- `toymc_s2.py`：S2 方法的实现
- `run_quick_test.py`：快速测试脚本
- `compute_electron_from_csv.py`：从 CSV 文件计算电子谱
- `batch_compute_electrons.py`：批量处理工具

### 核心模块
```python
from core import *                    # 导入所有核心功能
from workflows import *               # 工作流函数
from core.stats import *              # 统计函数
```

---

## 输出文件

### 图表文件（每个参数点）
```
U2_1.00e-01_MH_4.0/
├── neutrino_2d_linear.pdf    # 中微子 2D 分布（线性）
├── neutrino_2d_log.pdf       # 中微子 2D 分布（对数）
├── neutrino_energy_1d.pdf    # 中微子能量谱
├── neutrino_angle_1d.pdf     # 中微子角度分布
├── electron_2d_linear.pdf    # 电子 2D 分布
├── electron_2d_log.pdf       # 电子 2D 分布（对数）
├── electron_energy_1d.pdf    # 电子能量谱
└── electron_angle_1d.pdf     # 电子角度分布
```

### 数据文件
- `diff_El_*.csv`：能量分布数据
- `diff_costheta_*.csv`：角度分布数据
- `diff_El_costheta_*.csv`：2D 分布数据
- `scattered_electrons_2d_lab.csv`：散射电子数据

### 分析结果
- `summary.txt`：参数扫描汇总
- `chi2_grid_s1.npz`：χ² 网格数据
- 排除区域图（90% CL）

---

## 文件结构

```
SolarRHN/
├── core/                    # 核心物理计算
│   ├── constants.py        # 物理常数
│   ├── rhn_physics.py      # RHN 物理
│   ├── decay_and_scattering.py # 衰变和散射
│   ├── stats.py           # 统计函数
│   └── ...
├── toymc_s1_parallel.py    # S1 并行主脚本
├── toymc_s1.py            # S1 顺序脚本
├── toymc_s2.py            # S2 主脚本
├── run_quick_test.py      # 快速测试
├── compute_electron_from_csv.py  # CSV 处理
├── batch_compute_electrons.py    # 批量处理
├── workflows.py           # 工作流函数
├── ploter.py             # 绘图工具
├── data/                 # 输入数据
│   ├── 8BSpectrum.csv   # 太阳中微子谱
│   └── Solar.root       # 背景数据
├── output/              # 输出目录
└── plots/               # 图表目录
```

---

## 联系方式

- 问题反馈：[GitHub Issues](https://github.com/your-repo/issues)
- 邮箱：zhu-yt24@mails.tsinghua.edu.cn

---

## 许可证

MIT License

---

**Happy hunting for RHNs! 🔬✨**
