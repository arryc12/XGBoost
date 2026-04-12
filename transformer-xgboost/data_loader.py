#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List
import config


def load_signal_data(file_path: str, cols: List[int] = None) -> np.ndarray:
    """
    加载原始信号数据
    
    参数:
        file_path: CSV文件路径
        cols: 要提取的列索引列表，默认为[4, 5]（进口压力、出口压力）
    
    返回:
        合并后的信号数组，形状为 (n_samples, n_channels)
    """
    if cols is None:
        cols = config.INPUT_COLS
    
    df = pd.read_csv(file_path, skiprows=1, header=None)
    data = df.iloc[:, cols].values
    return data


def chunk_data(data: np.ndarray, chunk_size: int) -> List[np.ndarray]:
    """
    将信号数据分块
    
    参数:
        data: 原始信号数据，形状为 (n_samples, n_channels)
        chunk_size: 每个块的采样点数
    
    返回:
        分块后的数据列表，每个块形状为 (chunk_size, n_channels)
    """
    n_samples = len(data)
    chunks = []
    
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        chunk = data[start:end]
        
        if len(chunk) == chunk_size:
            chunks.append(chunk)
        elif len(chunk) >= chunk_size // 2:
            padded = np.zeros((chunk_size, chunk.shape[1]))
            padded[:len(chunk)] = chunk
            chunks.append(padded)
    
    return chunks


def load_dataset_from_folder(
    folder_path: str,
    label: int,
    chunk_size: int = None,
    cols: List[int] = None
) -> Tuple[List[np.ndarray], List[int]]:
    """
    从文件夹加载数据集
    
    参数:
        folder_path: 数据文件夹路径
        label: 流型标签
        chunk_size: 分块大小
        cols: 要提取的列
    
    返回:
        (数据块列表, 标签列表)
    """
    if chunk_size is None:
        chunk_size = config.CHUNK_SIZE
    if cols is None:
        cols = config.INPUT_COLS
    
    folder = Path(folder_path)
    all_chunks = []
    labels = []
    
    for csv_file in folder.rglob("*.csv"):
        try:
            data = load_signal_data(str(csv_file), cols)
            chunks = chunk_data(data, chunk_size)
            all_chunks.extend(chunks)
            labels.extend([label] * len(chunks))
        except Exception as e:
            print(f"跳过文件 {csv_file.name}: {e}")
    
    return all_chunks, labels


def load_all_datasets(
    data_root: str = None,
    chunk_size: int = None,
    cols: List[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载所有流型的数据集
    
    参数:
        data_root: 数据根目录
        chunk_size: 分块大小
        cols: 要提取的列
    
    返回:
        (特征数组, 标签数组)
    """
    if data_root is None:
        data_root = config.DATA_ROOT
    if chunk_size is None:
        chunk_size = config.CHUNK_SIZE
    if cols is None:
        cols = config.INPUT_COLS
    
    all_chunks = []
    all_labels = []
    
    for flow_type, label in config.FLOW_REGIMES.items():
        folder = Path(data_root) / flow_type
        if folder.exists():
            print(f"加载 {flow_type} 数据...")
            chunks, labels = load_dataset_from_folder(
                str(folder), label, chunk_size, cols
            )
            all_chunks.extend(chunks)
            all_labels.extend(labels)
            print(f"  获得 {len(chunks)} 个数据块")
    
    X = np.array(all_chunks)
    y = np.array(all_labels)
    
    return X, y


def normalize_data(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    标准化数据
    
    参数:
        X: 原始特征数据
    
    返回:
        (标准化后的数据, 均值, 标准差)
    """
    mean = X.mean(axis=(0, 1), keepdims=True)
    std = X.std(axis=(0, 1), keepdims=True)
    std[std == 0] = 1
    
    X_normalized = (X - mean) / std
    return X_normalized, mean, std


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = None,
    random_state: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    划分训练集和测试集
    
    参数:
        X: 特征数据
        y: 标签
        test_size: 测试集比例
        random_state: 随机种子
    
    返回:
        (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    if test_size is None:
        test_size = config.TEST_SIZE
    if random_state is None:
        random_state = config.RANDOM_STATE
    
    return train_test_split(
        X, y, test_size=test_size, 
        random_state=random_state, stratify=y
    )


if __name__ == "__main__":
    print("测试数据加载...")
    X, y = load_all_datasets()
    print(f"数据形状: X={X.shape}, y={y.shape}")
    print(f"标签分布: {np.bincount(y)}")