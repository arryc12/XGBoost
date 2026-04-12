"""
特征提取与数据集构建模块，负责时域/频域特征计算和小波包分析。
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import fft, fftfreq


def calculate_time_domain_features(signal):
    """
    计算时域特征。

    参数:
        signal: 一维信号数组

    返回:
        包含时域特征的字典
    """
    features = {}
    features['均值'] = float(np.mean(signal))
    features['标准差'] = float(np.std(signal))
    features['最大值'] = float(np.max(signal))
    features['最小值'] = float(np.min(signal))
    features['峰峰值'] = float(np.max(signal) - np.min(signal))
    features['均方根值'] = float(np.sqrt(np.mean(signal ** 2)))
    features['峰度'] = float(stats.kurtosis(signal))
    features['偏度'] = float(stats.skew(signal))
    features['方差'] = float(np.var(signal))
    features['脉冲因子'] = float(np.max(np.abs(signal)) / np.mean(np.abs(signal)))
    features['裕度因子'] = float(np.max(np.abs(signal)) / np.mean(np.sqrt(np.abs(signal))) ** 2)
    features['波形因子'] = float(np.sqrt(np.mean(signal ** 2)) / np.mean(np.abs(signal)))
    return features


def calculate_freq_domain_features(signal, fs=20000):
    """
    计算频域特征。

    参数:
        signal: 一维信号数组
        fs: 采样频率

    返回:
        (特征字典, 频率数组, 功率谱数组)
    """
    n = len(signal)
    yf = fft(signal)
    xf = fftfreq(n, 1/fs)[:n//2]
    psd = np.abs(yf[:n//2])**2
    
    features = {}
    features['主频率'] = float(xf[np.argmax(psd)])
    features['总功率'] = float(np.sum(psd))
    features['功率比'] = float(np.max(psd) / np.sum(psd))
    features['平均频率'] = float(np.sum(xf * psd) / np.sum(psd))
    features['频率标准差'] = float(np.sqrt(np.sum((xf - features['平均频率'])**2 * psd) / np.sum(psd)))
    features['频率重心'] = float(np.sum(xf * psd) / np.sum(psd))
    
    try:
        import pywt
        wp = pywt.WaveletPacket(data=signal, wavelet='db4', mode='symmetric', maxlevel=4)
        nodes = [node.path for node in wp.get_level(wp.maxlevel, 'freq')]
        energy = np.array([np.sum(wp[n].data**2) for n in nodes])
        energy_norm = energy / energy.sum()
        energy_norm += 1e-16
        shannon_entropy = -np.sum(energy_norm * np.log2(energy_norm))
        features['小波包能量熵'] = float(shannon_entropy)
    except:
        features['小波包能量熵'] = 0.0
    
    return features, xf, psd


def build_feature_dataset(data_dict, selected_files, selected_cols, chunk_size=2000,
                          label_value=0, selected_pdf_features=None,
                          selected_time_features=None, selected_freq_features=None):
    """
    构建特征数据集。

    参数:
        data_dict: 文件路径到DataFrame的映射
        selected_files: 选中的文件路径列表
        selected_cols: 选中的列名列表
        chunk_size: 切片长度
        label_value: 标签值
        selected_pdf_features: 选中的PDF特征列表
        selected_time_features: 选中的时域特征列表
        selected_freq_features: 选中的频域特征列表

    返回:
        包含所有特征的 DataFrame
    """
    if selected_pdf_features is None:
        selected_pdf_features = []
    if selected_time_features is None:
        selected_time_features = []
    if selected_freq_features is None:
        selected_freq_features = []
    
    all_results = []
    
    for file_path in selected_files:
        if file_path not in data_dict:
            continue
        df = data_dict[file_path]
        file_name = os.path.basename(file_path)
        
        for col in selected_cols:
            if col in df.columns:
                signal = df[col].dropna().values
                
                for start in range(0, len(signal), chunk_size):
                    piece = signal[start:start + chunk_size]
                    if len(piece) < chunk_size // 2:
                        continue
                    
                    row = {'source_file': file_name, 'column': col}
                    
                    for feat in selected_pdf_features:
                        if feat == 'mean':
                            row['mean'] = float(np.mean(piece))
                        elif feat == 'std':
                            row['std'] = float(np.std(piece, ddof=0))
                        elif feat == 'var':
                            row['var'] = float(np.var(piece, ddof=0))
                        elif feat == 'median':
                            row['median'] = float(np.median(piece))
                        elif feat == 'mode':
                            vals, counts = np.unique(piece, return_counts=True)
                            row['mode'] = float(vals[np.argmax(counts)])
                        elif feat == 'skew':
                            row['skew'] = float(stats.skew(piece, bias=False))
                        elif feat == 'kurt':
                            row['kurt'] = float(stats.kurtosis(piece, bias=False))
                    
                    for feat in selected_time_features:
                        if feat == 'rms':
                            row['rms'] = float(np.sqrt(np.mean(piece ** 2)))
                        elif feat == 'peak':
                            row['peak'] = float(np.max(piece) - np.min(piece))
                        elif feat == 'impulse':
                            row['impulse'] = float(np.max(np.abs(piece)) / np.mean(np.abs(piece)))
                        elif feat == 'margin':
                            row['margin'] = float(np.max(np.abs(piece)) / (np.mean(np.sqrt(np.abs(piece))) ** 2))
                        elif feat == 'waveform':
                            row['waveform'] = float(np.sqrt(np.mean(piece ** 2)) / np.mean(np.abs(piece)))
                    
                    if selected_freq_features:
                        n = len(piece)
                        yf = fft(piece)
                        psd = np.abs(yf[:n//2])**2
                        xf = fftfreq(n, 1/20000)[:n//2]
                        
                        for feat in selected_freq_features:
                            if feat == 'dominant_freq':
                                row['dominant_freq'] = float(xf[np.argmax(psd)])
                            elif feat == 'total_power':
                                row['total_power'] = float(np.sum(psd))
                            elif feat == 'power_ratio':
                                row['power_ratio'] = float(np.max(psd) / np.sum(psd)) if np.sum(psd) > 0 else 0.0
                            elif feat == 'wavelet_entropy':
                                try:
                                    import pywt
                                    wp = pywt.WaveletPacket(data=piece, wavelet='db4', mode='symmetric', maxlevel=4)
                                    nodes = [node.path for node in wp.get_level(wp.maxlevel, 'freq')]
                                    energy = np.array([np.sum(wp[n].data**2) for n in nodes])
                                    energy_norm = energy / energy.sum() if energy.sum() > 0 else energy
                                    energy_norm = energy_norm + 1e-16
                                    row['wavelet_entropy'] = float(-np.sum(energy_norm * np.log2(energy_norm)))
                                except:
                                    row['wavelet_entropy'] = 0.0
                    
                    row['label'] = label_value
                    all_results.append(row)
    
    if not all_results:
        return pd.DataFrame()
    
    result_df = pd.DataFrame(all_results)
    
    feature_cols = []
    feature_cols.extend(selected_pdf_features)
    feature_cols.extend(selected_time_features)
    feature_cols.extend(selected_freq_features)
    feature_cols.append('label')
    
    final_cols = [c for c in feature_cols if c in result_df.columns]
    result_df = result_df[final_cols]
    
    return result_df
