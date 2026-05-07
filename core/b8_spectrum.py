import numpy as np
from typing import Tuple, Optional
import os

class B8Spectrum:
    """
    B8 太阳中微子能谱处理类
    
    功能:
    1. 从 originalTable.txt 加载最佳能谱、+3sigma 和 -3sigma 边界
    2. 将能谱插值到指定的能量网格
    3. 考虑到系统不确定度，生成包含统计涨落和系统变化的伪能谱 (Pseudo-data)
    """
    
    def __init__(self, data_path: str = "data/originalTable.txt", default_flux_norm: float = 5.69e6):
        """
        初始化 B8Spectrum
        
        参数:
        data_path: 包含 B8 能谱和不确定度信息的文件路径
        default_flux_norm: 默认的总通量归一化因子。如果不提供，默认为 5.69e6，以匹配 8BSpectrum.csv。
        """
        # 如果传入的是相对路径，尝试基于项目根目录解析
        if not os.path.isabs(data_path):
            # 假设当前脚本在 core 目录下，上一级是根目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            data_path = os.path.join(project_root, data_path)

        self.data_path = data_path
        self.default_flux_norm = default_flux_norm
        self.energy = None
        self.best_spectrum = None
        self.plus_3sigma = None
        self.minus_3sigma = None
        
        self._load_data()
        
    def _load_data(self):
        """内部方法：从文件加载数据"""
        try:
            # 跳过前 4 行 header (包括空行)
            data = np.loadtxt(self.data_path, delimiter=',', skiprows=4)
            
            self.energy = data[:, 0]
            self.best_spectrum = data[:, 1]
            self.plus_3sigma = data[:, 2]
            self.minus_3sigma = data[:, 3]
            
        except Exception as e:
            raise RuntimeError(f"Failed to load B8 spectrum data from {self.data_path}: {e}")
            
    def get_spectrum(self, target_energies: np.ndarray, shift_sigma: float = 0.0) -> np.ndarray:
        """
        获取插值后的能谱，支持施加整体的不确定度偏移。
        
        参数:
        target_energies: 目标能量数组 (MeV)
        shift_sigma: 系统误差的偏置 (单位：标准差)。例如 1.0 表示 +1 sigma 偏置
        
        返回:
        对应于 target_energies 的中微子通量密度数组
        """
        if shift_sigma == 0.0:
            source_spectrum = self.best_spectrum
        elif shift_sigma > 0:
            # 使用 +3sigma 边界计算 +1sigma 的偏移量
            sigma_plus = (self.plus_3sigma - self.best_spectrum) / 3.0
            source_spectrum = self.best_spectrum + shift_sigma * sigma_plus
        else:
            # shift_sigma < 0，使用 -3sigma 边界计算 -1sigma 的偏移量
            sigma_minus = (self.best_spectrum - self.minus_3sigma) / 3.0
            source_spectrum = self.best_spectrum + shift_sigma * sigma_minus # shift_sigma 本身是负数

        # 确保物理上的非负性
        source_spectrum = np.maximum(source_spectrum, 0.0)
        
        # 线性插值到目标能量网格
        interpolated = np.interp(target_energies, self.energy, source_spectrum, left=0.0, right=0.0)
        
        # 应用通量归一化
        interpolated *= self.default_flux_norm
        
        return interpolated
        
    def generate_pseudo_data(self, target_energies: np.ndarray, shift_sigma: float = 0.0, apply_poisson: bool = False, seed: Optional[int] = None) -> np.ndarray:
        """
        生成伪数据能谱 (Pseudo-data)
        
        可以包含系统不确定度的偏移 (shift_sigma)，以及可选的泊松统计涨落。
        注意：如果要应用泊松涨落，通常需要将能谱转换为事件计数（乘以上下文中的截面、探测器效率和曝光量）。
        纯能谱层面的泊松涨落物理意义有限，因此 apply_poisson 默认 False，仅用于特殊需求。
        
        参数:
        target_energies: 目标能量数组 (MeV)
        shift_sigma: 系统误差的偏置
        apply_poisson: 是否在每个 bin 应用泊松随机涨落
        seed: 随机数种子
        
        返回:
        伪数据能谱数组
        """
        if seed is not None:
            np.random.seed(seed)
            
        expected_spectrum = self.get_spectrum(target_energies, shift_sigma=shift_sigma)
        
        if not apply_poisson:
            return expected_spectrum
            
        # 注意：这里简单的泊松分布仅作演示，实际的伪事件生成应该在乘上 exposure 等因子后，
        # 变成具体的 expected_counts 再进行 np.random.poisson。
        pseudo_counts = np.random.poisson(expected_spectrum)
        return pseudo_counts.astype(float)
