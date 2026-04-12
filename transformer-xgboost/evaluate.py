#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import joblib
import json
import shap
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

import config
from data_loader import load_all_datasets, normalize_data, split_dataset
from transformer_model import get_transformer_model
from xgboost_classifier import XGBoostClassifier


def load_models(save_dir: str, device: str = "cpu"):
    """
    加载训练好的模型
    
    参数:
        save_dir: 模型保存目录
        device: 设备
    
    返回:
        (Transformer模型, XGBoost分类器, 标准化参数)
    """
    # 加载Transformer
    transformer = get_transformer_model(
        model_type="basic",
        input_dim=2,
        d_model=config.D_MODEL,
        nhead=config.NHEAD,
        num_layers=config.NUM_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=config.DROPOUT,
        output_dim=config.D_MODEL
    ).to(device)
    transformer.load_state_dict(
        torch.load(os.path.join(save_dir, "transformer.pth"), map_location=device)
    )
    transformer.eval()
    
    # 加载XGBoost
    xgb_classifier = XGBoostClassifier()
    xgb_classifier.model.load_model(os.path.join(save_dir, "xgboost.json"))
    
    # 加载标准化参数
    scaler = joblib.load(os.path.join(save_dir, "scaler.pkl"))
    
    return transformer, xgb_classifier, scaler


def extract_features(model, X: np.ndarray, device: str = "cpu", batch_size: int = 32) -> np.ndarray:
    """提取特征"""
    model.eval()
    from torch.utils.data import DataLoader, TensorDataset
    
    X_tensor = torch.FloatTensor(X)
    loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size, shuffle=False)
    
    features = []
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            feats = model(batch_x).cpu().numpy()
            features.append(feats)
    
    return np.vstack(features)


def evaluate_model(
    transformer,
    xgb_classifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: str = "cpu",
    save_dir: str = None
):
    """
    评估模型
    
    参数:
        transformer: Transformer模型
        xgb_classifier: XGBoost分类器
        X_test: 测试数据
        y_test: 测试标签
        device: 设备
        save_dir: 保存目录
    """
    print("提取测试集特征...")
    X_test_features = extract_features(transformer, X_test, device)
    
    print("\n模型评估...")
    results = xgb_classifier.evaluate(X_test_features, y_test)
    
    print("\n" + "=" * 50)
    print("测试集评估结果")
    print("=" * 50)
    print(f"准确率: {results['accuracy']:.4f}")
    print(f"精确率: {results['precision']:.4f}")
    print(f"召回率: {results['recall']:.4f}")
    print(f"F1分数: {results['f1']:.4f}")
    
    cm = np.array(results['confusion_matrix'])
    print(f"\n混淆矩阵:")
    print(cm)
    
    # 详细分类报告
    y_pred = xgb_classifier.predict(X_test_features)
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    
    # 保存结果
    if save_dir:
        with open(os.path.join(save_dir, "evaluation.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    return results


def shap_analysis(
    xgb_classifier,
    X_test_features: np.ndarray,
    feature_names: list = None,
    save_dir: str = None,
    n_samples: int = 100
):
    """
    SHAP解释分析
    
    参数:
        xgb_classifier: XGBoost分类器
        X_test_features: 测试集特征
        feature_names: 特征名称
        save_dir: 保存目录
        n_samples: 采样数量
    """
    print("\n" + "=" * 50)
    print("SHAP特征重要性分析")
    print("=" * 50)
    
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X_test_features.shape[1])]
    
    # 采样
    if len(X_test_features) > n_samples:
        idx = np.random.choice(len(X_test_features), n_samples, replace=False)
        X_sample = X_test_features[idx]
    else:
        X_sample = X_test_features
    
    # 创建SHAP解释器
    explainer = shap.TreeExplainer(xgb_classifier.model)
    shap_values = explainer.shap_values(X_sample)
    
    # 绘制SHAP Summary Plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "shap_summary.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 绘制SHAP Bar Plot
    plt.figure(figsize=(12, 6))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "shap_bar.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"SHAP分析图已保存到 {save_dir}")
    
    return shap_values


def main(evaluation_dir: str = None):
    """主函数"""
    if evaluation_dir is None:
        evaluation_dir = config.OUTPUT_DIR
    
    print("=" * 60)
    print("Transformer + XGBoost 模型评估")
    print("=" * 60)
    
    # 确定设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 加载模型
    print("\n加载模型...")
    transformer, xgb_classifier, scaler = load_models(evaluation_dir, device)
    
    # 加载数据
    print("\n加载数据...")
    X, y = load_all_datasets()
    X_normalized = (X - scaler['mean']) / scaler['std']
    
    # 划分数据集
    X_train, X_test, y_train, y_test = split_dataset(X_normalized, y)
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    
    # 评估
    results = evaluate_model(transformer, xgb_classifier, X_test, y_test, device, evaluation_dir)
    
    # SHAP分析
    print("\n提取测试集特征用于SHAP分析...")
    X_test_features = extract_features(transformer, X_test, device)
    shap_analysis(xgb_classifier, X_test_features, save_dir=evaluation_dir)
    
    print("\n评估完成!")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=None)
    args = parser.parse_args()
    
    model_dir = args.model_dir if args.model_dir else str(config.OUTPUT_DIR)
    main(evaluation_dir=model_dir)