#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from typing import Dict, Tuple, Optional
import config


class XGBoostClassifier:
    """XGBoost分类器封装"""
    
    def __init__(self, params: Dict = None):
        if params is None:
            params = config.XGBOOST_PARAMS.copy()
        self.params = params
        self.model = None
        self.classes_ = None
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "XGBoostClassifier":
        """训练模型"""
        self.classes_ = np.unique(y_train)
        
        # 类别加权
        if len(self.classes_) > 1:
            weight_per_class = np.bincount(y_train)
            if len(weight_per_class) == 2:
                scale_pos = weight_per_class[0] / weight_per_class[1]
                self.params['scale_pos_weight'] = scale_pos
        
        self.model = XGBClassifier(**self.params)
        self.model.fit(X_train, y_train)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        return self.model.predict_proba(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算准确率"""
        return self.model.score(X, y)
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> float:
        """交叉验证"""
        if self.model is None:
            self.fit(X, y)
        
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        return scores.mean()
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """评估模型"""
        y_pred = self.predict(X)
        
        result = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
        }
        
        return result
    
    def get_feature_importance(self) -> np.ndarray:
        """获取特征重要性"""
        if self.model is None:
            return None
        return self.model.feature_importances_


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    params: Dict = None
) -> Tuple[XGBClassifier, Dict]:
    """
    训练并评估模型
    
    参数:
        X_train: 训练特征
        y_train: 训练标签
        X_test: 测试特征
        y_test: 测试标签
        params: XGBoost参数
    
    返回:
        (训练好的模型, 评估结果)
    """
    classifier = XGBoostClassifier(params)
    classifier.fit(X_train, y_train)
    
    train_result = classifier.evaluate(X_train, y_train)
    test_result = classifier.evaluate(X_test, y_test)
    
    results = {
        'train': train_result,
        'test': test_result,
    }
    
    return classifier, results


def print_evaluation_results(results: Dict):
    """打印评估结果"""
    print("\n" + "=" * 50)
    print("模型评估结果")
    print("=" * 50)
    
    for split, result in results.items():
        print(f"\n{split.upper()}:")
        print(f"  准确率: {result['accuracy']:.4f}")
        print(f"  精确率: {result['precision']:.4f}")
        print(f"  召回率: {result['recall']:.4f}")
        print(f"  F1分数: {result['f1']:.4f}")
        print(f"  混淆矩阵:")
        cm = np.array(result['confusion_matrix'])
        print(f"    {cm}")


if __name__ == "__main__":
    print("测试XGBoost分类器...")
    
    # 模拟数据
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    classifier, results = train_and_evaluate(X_train, y_train, X_test, y_test)
    print_evaluation_results(results)