# SolarRHN

用于研究太阳源右手中微子（RHN, MeV 标度）在探测器中的可见信号与排除灵敏度。

---

## 1. 物理流程（当前代码）

项目包含两条主分析链：

- **S1（直接衰变）**：
  \[
  \nu_H \to \nu + e^+ + e^-
  \]
  直接计算衰变电子（或 \(e^+e^-\)）信号，并与背景比较。

- **S2（衰变 + 散射）**：
  \[
  \nu_H \to \nu,\qquad \nu + e^- \to \nu + e^-
  \]
  先得到 RHN 衰变后的中微子分布，再计算电子散射谱（能量 \(E_e\) 与角度 \(\cos\theta\)）。

---

## 2. 统计量（Asimov）

当前排除主要基于 Poisson 似然比（Asimov χ²）：

\[
\chi^2 = 2\sum_k\left[(S_k+B_k)\ln\left(1+\frac{S_k}{B_k}\right)-S_k\right]
\]

其中 \(S_k\) 与 \(B_k\) 分别为第 \(k\) 个 bin 的信号与本底计数。

常用阈值（1 dof, one-sided）：
\[
\chi^2_{\mathrm{crit}} \approx 2.71 \quad (90\%\ \mathrm{CL})
\]

---

## 3. 环境与依赖

```bash
pip install numpy scipy matplotlib numba uproot tqdm
```

如需 ROOT 绘图，请保证本地可导入 `ROOT`（PyROOT）。

---

## 4. 最常用脚本

### S1 扫描
```bash
python toymc_s1.py
```

### S2 扫描（生成每个参数点的数据）
```bash
python toymc_s2.py
```

### S2 单点/批量结果绘图
```bash
python plot_s2_results.py plots_grid_scan_s2/U2_1.00e-01_MH_4.0
python plot_s2_results.py plots_grid_scan_s2 --all
```

### S2 排除曲线（读取已保存网格）
```bash
python plot_s2_exclusion.py plots_grid_scan_s2
```

（可选）只画某个参数点的 signal/background 2D 对比：
```bash
python plot_s2_exclusion.py --plot plots_grid_scan_s2/U2_1.00e-01_MH_4.0
```

---

## 5. 输出说明（S2）

每个参数点目录（如 `U2_1.00e-01_MH_4.0/`）通常包含：

- `electron_data.npz`：信号电子谱（含 2D/1D）
- `solar_nu_background.npz`：太阳中微子散射背景
- `neutrino_*` / `electron_*` 图像文件
- `signal_bg_ratio_2d_*` 等比较图

单位约定（代码中当前口径）：

- `counts_2d`（存盘）通常为 \(\mathrm{Counts}/(\mathrm{MeV}\cdot \Delta\cos\theta)\)
- 绘图时可按需要转换为 \(\mathrm{Counts}/\mathrm{MeV}\) 或 per-bin counts

---

## 6. 核心目录

- `core/`：物理计算与统计函数
- `workflows.py`：S2 流程拼装与数据保存
- `toymc_s1.py` / `toymc_s2.py`：主扫描入口
- `plot_s2_results.py`：S2 参数点结果图
- `plot_s2_exclusion.py`：S2 排除曲线
- `data/`：输入谱与实验数据

