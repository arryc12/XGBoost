# 时域与频域特征提取功能

## 功能概述
根据`data/feature extraction.py`和`data/waveleta_packet.py`的功能，为数据处理界面添加了时域特征提取和频域特征提取功能。

## 新增功能

### 1. 时域特征提取
**按钮位置**: 数据处理窗口 → 左侧面板 → "时域特征提取"

**提取的特征**:
- 均值 (mean)
- 标准差 (std)
- 最大值 (max)
- 最小值 (min)
- 峰峰值 (peak-to-peak)
- 均方根值 (RMS)
- 峰度 (kurtosis)
- 偏度 (skewness)
- 方差 (variance)
- 脉冲因子 (impulse factor)
- 裕度因子 (margin factor)
- 波形因子 (waveform factor)

**可视化**:
- 时域信号波形图

### 2. 频域特征提取
**按钮位置**: 数据处理窗口 → 左侧面板 → "频域特征提取"

**提取的特征**:
- 主频率 (dominant frequency)
- 总功率 (total power)
- 功率比 (power ratio)
- 平均频率 (mean frequency)
- 频率标准差 (frequency std)
- 频率重心 (frequency centroid)
- 小波包能量熵 (wavelet packet energy entropy)

**可视化**:
- 功率谱密度图
- 小波包能量分布柱状图

## 修改的文件

### software/main.py
1. **导入部分** (第13-25行)
   - 添加 `from scipy import stats`
   - 添加 `from scipy.fft import fft, fftfreq`
   - 添加 `import pywt`

2. **DataProcessWindow类**
   - 添加 "时域特征提取" 按钮 (第403-406行)
   - 添加 "频域特征提取" 按钮 (第408-411行)
   - 添加 `extract_time_domain_features()` 方法 (第512-547行)
   - 添加 `_calculate_time_domain_features()` 方法 (第549-564行)
   - 添加 `_plot_time_domain_features()` 方法 (第566-593行)
   - 添加 `extract_freq_domain_features()` 方法 (第595-630行)
   - 添加 `_calculate_freq_domain_features()` 方法 (第632-695行)
   - 添加 `_plot_freq_domain_features()` 方法 (第697-754行)
   - 添加 `_show_feature_results()` 方法 (第756-801行)

## 使用方法

### 前提条件
确保已安装必要的Python库：
```bash
pip install pandas numpy scipy pywt PyQt5 matplotlib seaborn
```

### 操作步骤
1. 启动程序：`python software/main.py`
2. 点击"选择文件"选择数据文件
3. 点击"数据处理"打开数据处理窗口
4. 选择要处理的文件和列
5. 点击"时域特征提取"或"频域特征提取"按钮
6. 查看特征提取结果表格和可视化图表

## 技术实现细节

### 时域特征计算
基于`data/feature extraction.py`中的`extract_basic_statistics()`函数，扩展了更多时域特征：
- 使用`scipy.stats`计算峰度和偏度
- 使用`numpy`计算均方根值、脉冲因子等

### 频域特征计算
结合了`data/feature extraction.py`和`data/waveleta_packet.py`的功能：
- 使用FFT计算功率谱密度
- 使用小波包分解计算能量熵
- 采样率默认设置为20000 Hz（可根据实际数据调整）

### 可视化
- 时域特征：绘制信号波形图
- 频域特征：绘制功率谱密度图和小波包能量分布图
- 结果显示：创建独立对话框以表格形式展示所有特征值

## 注意事项
1. 采样率(fs)默认设置为20000 Hz，如需修改请编辑`_calculate_freq_domain_features()`函数
2. 小波包分解使用db4小波，4层分解，可根据需要调整
3. 特征提取结果可通过对话框表格查看，暂未实现直接导出功能
4. 可视化图表会显示在数据处理窗口右侧的图表区域