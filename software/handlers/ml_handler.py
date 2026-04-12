"""
机器学习处理模块，负责特征提取、数据集构建和模型训练。
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
from scipy.fft import fft, fftfreq


def extract_time_domain_features(data, chunk_size=1000):
    """
    提取时域特征。

    参数:
        data: pandas DataFrame 或 Series
        chunk_size: 分块大小

    返回:
        包含时域特征的 DataFrame
    """
    if isinstance(data, pd.DataFrame):
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data = data[numeric_cols]
    
    n_samples = len(data)
    n_chunks = n_samples // chunk_size
    
    features = []
    for i in range(n_chunks):
        chunk = data.iloc[i * chunk_size:(i + 1) * chunk_size]
        if isinstance(chunk, pd.DataFrame):
            for col in chunk.columns:
                vals = chunk[col].values
                feat = {
                    'chunk': i,
                    'column': col,
                    'mean': np.mean(vals),
                    'std': np.std(vals),
                    'max': np.max(vals),
                    'min': np.min(vals),
                    'peak_to_peak': np.max(vals) - np.min(vals),
                    'rms': np.sqrt(np.mean(vals ** 2)),
                    'skewness': stats.skew(vals),
                    'kurtosis': stats.kurtosis(vals),
                }
                features.append(feat)
        else:
            vals = chunk.values
            feat = {
                'chunk': i,
                'mean': np.mean(vals),
                'std': np.std(vals),
                'max': np.max(vals),
                'min': np.min(vals),
                'peak_to_peak': np.max(vals) - np.min(vals),
                'rms': np.sqrt(np.mean(vals ** 2)),
                'skewness': stats.skew(vals),
                'kurtosis': stats.kurtosis(vals),
            }
            features.append(feat)
    
    return pd.DataFrame(features)


def extract_frequency_domain_features(data, chunk_size=1000, fs=1000):
    """
    提取频域特征。

    参数:
        data: pandas DataFrame 或 Series
        chunk_size: 分块大小
        fs: 采样频率

    返回:
        包含频域特征的 DataFrame
    """
    if isinstance(data, pd.DataFrame):
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data = data[numeric_cols]
    
    n_samples = len(data)
    n_chunks = n_samples // chunk_size
    
    features = []
    for i in range(n_chunks):
        chunk = data.iloc[i * chunk_size:(i + 1) * chunk_size]
        if isinstance(chunk, pd.DataFrame):
            for col in chunk.columns:
                vals = chunk[col].values
                feat = _compute_freq_features(vals, fs, i, col)
                features.append(feat)
        else:
            vals = chunk.values
            feat = _compute_freq_features(vals, fs, i)
            features.append(feat)
    
    return pd.DataFrame(features)


def _compute_freq_features(vals, fs, chunk_idx, col_name=None):
    """计算单个数据块的频域特征"""
    n = len(vals)
    yf = fft(vals)
    xf = fftfreq(n, 1 / fs)[:n // 2]
    magnitude = 2.0 / n * np.abs(yf[:n // 2])
    
    total_magnitude = np.sum(magnitude)
    if total_magnitude > 0:
        dominant_freq = xf[np.argmax(magnitude)]
        spectral_centroid = np.sum(xf * magnitude) / total_magnitude
        spectral_entropy = -np.sum((magnitude / total_magnitude) * np.log(magnitude / total_magnitude + 1e-10))
    else:
        dominant_freq = 0
        spectral_centroid = 0
        spectral_entropy = 0
    
    feat = {
        'chunk': chunk_idx,
        'dominant_freq': dominant_freq,
        'spectral_centroid': spectral_centroid,
        'spectral_entropy': spectral_entropy,
    }
    if col_name:
        feat['column'] = col_name
    return feat


def extract_pdf_features(data, chunk_size=1000, n_bins=50):
    """
    提取概率密度分布特征。

    参数:
        data: pandas DataFrame 或 Series
        chunk_size: 分块大小
        n_bins: 直方图分箱数

    返回:
        包含PDF特征的 DataFrame
    """
    if isinstance(data, pd.DataFrame):
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data = data[numeric_cols]
    
    n_samples = len(data)
    n_chunks = n_samples // chunk_size
    
    features = []
    for i in range(n_chunks):
        chunk = data.iloc[i * chunk_size:(i + 1) * chunk_size]
        if isinstance(chunk, pd.DataFrame):
            for col in chunk.columns:
                vals = chunk[col].values
                hist, bin_edges = np.histogram(vals, bins=n_bins, density=True)
                feat = {f'pdf_bin_{j}': hist[j] for j in range(n_bins)}
                feat['chunk'] = i
                feat['column'] = col
                features.append(feat)
        else:
            vals = chunk.values
            hist, bin_edges = np.histogram(vals, bins=n_bins, density=True)
            feat = {f'pdf_bin_{j}': hist[j] for j in range(n_bins)}
            feat['chunk'] = i
            features.append(feat)
    
    return pd.DataFrame(features)


def build_feature_dataset(file_paths, chunk_size=1000, label=0,
                          use_pdf=True, use_time=True, use_freq=True):
    """
    构建特征数据集，用于机器学习训练。

    参数:
        file_paths: 文件路径列表
        chunk_size: 分块大小
        label: 标签值
        use_pdf: 是否使用PDF特征
        use_time: 是否使用时域特征
        use_freq: 是否使用频域特征

    返回:
        包含所有特征的 DataFrame
    """
    from handlers.io_handler import load_data
    
    all_features = []
    for file_path in file_paths:
        data = load_data(file_path)
        features_list = []
        
        if use_time:
            time_feat = extract_time_domain_features(data, chunk_size)
            features_list.append(time_feat)
        
        if use_freq:
            freq_feat = extract_frequency_domain_features(data, chunk_size)
            features_list.append(freq_feat)
        
        if use_pdf:
            pdf_feat = extract_pdf_features(data, chunk_size)
            features_list.append(pdf_feat)
        
        if features_list:
            combined = pd.concat(features_list, axis=1)
            combined['Label'] = label
            all_features.append(combined)
    
    if all_features:
        return pd.concat(all_features, ignore_index=True)
    return pd.DataFrame()
