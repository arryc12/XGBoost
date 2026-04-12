#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import joblib
import json

import config
from data_loader import load_all_datasets, normalize_data, split_dataset
from transformer_model import get_transformer_model
from xgboost_classifier import XGBoostClassifier, train_and_evaluate, print_evaluation_results


class SignalDataset(Dataset):
    """信号数据集"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray = None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y) if y is not None else None
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


def train_transformer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
    epochs: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    model_type: str = "basic",
    device: str = "cpu"
):
    """
    训练Transformer模型
    
    参数:
        X_train: 训练数据
        y_train: 训练标签
        X_val: 验证数据
        y_val: 验证标签
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        model_type: 模型类型
        device: 设备
    
    返回:
        训练好的Transformer模型
    """
    if epochs is None:
        epochs = config.EPOCHS
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if learning_rate is None:
        learning_rate = config.LEARNING_RATE
    
    input_dim = X_train.shape[-1]
    
    model = get_transformer_model(
        model_type=model_type,
        input_dim=input_dim,
        d_model=config.D_MODEL,
        nhead=config.NHEAD,
        num_layers=config.NUM_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=config.DROPOUT,
        output_dim=config.D_MODEL
    ).to(device)
    
    train_dataset = SignalDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    if X_val is not None and y_val is not None:
        val_dataset = SignalDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        val_loader = None
    
    num_classes = len(np.unique(y_train))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        if val_loader is not None:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        else:
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model


def extract_features(
    model: nn.Module,
    X: np.ndarray,
    device: str = "cpu",
    batch_size: int = None
) -> np.ndarray:
    """
    使用Transformer提取特征
    
    参数:
        model: 训练好的Transformer模型
        X: 输入数据
        device: 设备
        batch_size: 批次大小
    
    返回:
        提取的特征
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    
    model.eval()
    dataset = SignalDataset(X)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    features = []
    with torch.no_grad():
        for batch_x in loader:
            batch_x = batch_x.to(device)
            feats = model(batch_x).cpu().numpy()
            features.append(feats)
    
    return np.vstack(features)


def train_pipeline(
    data_root: str = None,
    model_type: str = "basic",
    optimize_hyperparams: bool = None,
    save_dir: str = None,
    device: str = "cpu"
):
    """
    完整的训练流程
    
    参数:
        data_root: 数据根目录
        model_type: Transformer模型类型
        optimize_hyperparams: 是否优化超参数
        save_dir: 保存目录
        device: 设备
    """
    if data_root is None:
        data_root = config.DATA_ROOT
    if optimize_hyperparams is None:
        optimize_hyperparams = config.OPTIMIZE_HYPERPARAMS
    if save_dir is None:
        save_dir = config.OUTPUT_DIR
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 60)
    print("Transformer + XGBoost 混合模型训练")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n[1/4] 加载数据...")
    X, y = load_all_datasets(data_root)
    print(f"原始数据形状: X={X.shape}, y={y.shape}")
    print(f"标签分布: {np.bincount(y)}")
    
    # 2. 标准化
    print("\n[2/4] 数据预处理...")
    X_normalized, mean, std = normalize_data(X)
    joblib.dump({'mean': mean, 'std': std}, os.path.join(save_dir, "scaler.pkl"))
    print(f"标准化后数据形状: {X_normalized.shape}")
    
    # 3. 划分数据集
    print("\n[3/4] 划分数据集...")
    X_train, X_test, y_train, y_test = split_dataset(X_normalized, y)
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    
    # 4. 训练Transformer
    print("\n[4/4] 训练Transformer特征提取器...")
    device = device if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    transformer = train_transformer(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        model_type=model_type,
        device=device
    )
    
    # 保存Transformer模型
    torch.save(transformer.state_dict(), os.path.join(save_dir, "transformer.pth"))
    print(f"Transformer模型已保存")
    
    # 5. 提取特征
    print("\n提取特征...")
    X_train_features = extract_features(transformer, X_train, device)
    X_test_features = extract_features(transformer, X_test, device)
    print(f"特征形状: Train={X_train_features.shape}, Test={X_test_features.shape}")
    
    # 6. 训练XGBoost
    print("\n训练XGBoost分类器...")
    xgb_classifier = XGBoostClassifier(config.XGBOOST_PARAMS.copy())
    xgb_classifier.fit(X_train_features, y_train)
    
    # 保存XGBoost模型
    xgb_classifier.model.save_model(os.path.join(save_dir, "xgboost.json"))
    print(f"XGBoost模型已保存")
    
    # 7. 评估
    print("\n评估模型...")
    results = {}
    train_result = xgb_classifier.evaluate(X_train_features, y_train)
    test_result = xgb_classifier.evaluate(X_test_features, y_test)
    results['train'] = train_result
    results['test'] = test_result
    
    print_evaluation_results(results)
    
    # 保存结果
    with open(os.path.join(save_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {save_dir}")
    
    return transformer, xgb_classifier, results


if __name__ == "__main__":
    print("开始训练...")
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--model_type", type=str, default="basic", choices=["basic", "conv"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    train_pipeline(
        data_root=args.data_root,
        model_type=args.model_type,
        device=args.device
    )