"""
机器学习处理模块，封装 XGBoost 训练、评估、SHAP 分析。
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import shap
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Callable
import joblib


@dataclass
class MLConfig:
    """XGBoost 配置"""
    n_estimators: int = 400
    max_depth: int = 10
    learning_rate: float = 0.05
    subsample: float = 0.8
    min_child_weight: int = 1
    scale_pos_weight: Optional[float] = None
    random_state: int = 42


class MLHandler:
    """机器学习处理类"""
    
    def __init__(self, model_dir: str = None):
        """
        初始化处理器
        
        Args:
            model_dir: 模型保存目录，默认为'software/model'
        """
        self.config: Optional[MLConfig] = None
        self.model: Optional[XGBClassifier] = None
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
        self.results: Dict = {}
        self.shap_values = None
        self.feature_names = []
        
        model_dir = model_dir or os.path.join(os.path.dirname(__file__), 'model')
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
    
    def load_and_preprocess(self, file_path: str, label_col: str,
                          test_size: float = 0.2, random_state: int = 42) -> Dict:
        """
        加载 CSV 数据并进行预处理
        
        Args:
            file_path: CSV 文件路径
            label_col: 标签列名
            test_size: 测试集比例
            random_state: 随机种子
        
        Returns:
            处理信息字典
        """
        df = pd.read_csv(file_path)
        self.feature_names = list(df.columns)
        
        if label_col not in df.columns:
            available = ', '.join(df.columns)
            raise ValueError(f"未找到标签列 '{label_col}'。可用列：{available}")
        
        X = df.drop(columns=[label_col])
        y = df[label_col]
        
        # 类别加权
        weight_per_class = y.value_counts().sort_index()
        if len(weight_per_class) == 2:
            scale_pos = float(weight_per_class[0]) / float(weight_per_class[1])
        else:
            scale_pos = 1.0
        
        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y)
        
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        
        return {
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'features': len(X_train.columns),
            'scale_pos_weight': scale_pos,
            'classes': dict(weight_per_class)
        }
    
    def train(self, config: MLConfig, progress_callback: Callable[[int], None] = None) -> None:
        """
        训练 XGBoost 模型
        
        Args:
            config: 配置对象
            progress_callback: 进度回调函数 (percent: int)
        """
        self.config = config
        if config.scale_pos_weight is None:
            scale_per_class = self.y_train.value_counts().sort_index()
            scale_pos = float(scale_per_class[0]) / float(scale_per_class[1])
        else:
            scale_pos = config.scale_pos_weight
        
        self.model = XGBClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            subsample=config.subsample,
            min_child_weight=config.min_child_weight,
            scale_pos_weight=scale_pos,
            random_state=config.random_state,
            eval_metric='logloss',
            verbose=False
        )
        
        class TrainingProgressCallback:
            """训练进度回调"""
            def __init__(self, callback):
                self.callback = callback
                self.last_percent = 0
            
            def __call__(self, epoch):
                # XGBoost 训练时估算进度
                est_percent = min(int(100 * epoch / max(config.n_estimators, 1)), 95)
                if est_percent > self.last_percent:
                    self.callback(est_percent)
                    self.last_percent = est_percent
        
        cb = TrainingProgressCallback(progress_callback) if progress_callback else None
        self.model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_test, self.y_test)],
            verbose=False
        )
        
        if progress_callback:
            progress_callback(100)
    
    def evaluate(self) -> Dict[str, float]:
        """
        评估模型性能
        
        Returns:
            评估指标字典
        """
        if not self.model:
            raise RuntimeError("模型未训练")
        
        y_pred = self.model.predict(self.X_test)
        y_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        mse = mean_squared_error(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        accuracy = self.model.score(self.X_test, self.y_test)
        
        self.results = {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'accuracy': accuracy,
            'num_features': len(self.feature_names)
        }
        
        return self.results
    
    def generate_shap_summary(self) -> Figure:
        """
        生成 SHAP 摘要图
        
        Returns:
            matplotlib Figure 对象
        """
        if not self.model:
            raise RuntimeError("模型未训练")
        
        explainer = shap.TreeExplainer(self.model)
        self.shap_values = explainer.shap_values(self.X_test)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        
        shap.summary_plot(
            self.shap_values, 
            self.X_test, 
            feature_names=self.feature_names,
            show=False,
            ax=ax
        )
        
        return fig
    
    def generate_shap_force_plot(self, idx: int = 0) -> list:
        """
        生成 SHAP force 图
        
        Args:
            idx: 样本索引
        
        Returns:
            matplotlib Figure 列表
        """
        if not self.model:
            raise RuntimeError("模型未训练")
        
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.X_test.iloc[[idx]])
        
        figs = []
        fig = plt.figure(figsize=(8, 4))
        shap.force_plot(
            explainer.expected_value,
            shap_values[0],
            self.X_test.iloc[idx],
            matplotlib=False,
            show=False
        )
        figs.append(fig)
        
        return figs
    
    def save_model(self, filename: str = 'model.joblib', 
                  save_plots: bool = True) -> Dict[str, str]:
        """
        保存模型和图表
        
        Args:
            filename: 模型文件名
            save_plots: 是否保存 SHAP 图表
        
        Returns:
            保存文件路径字典
        """
        if not self.model:
            raise RuntimeError("模型未训练")
        
        model_path = os.path.join(self.model_dir, filename)
        joblib.dump(self.model, model_path)
        
        saved_files = {'model': model_path}
        
        if save_plots and self.shap_values is not None:
            fig_summary = self.generate_shap_summary()
            summary_path = os.path.join(self.model_dir, 'shap_summary.png')
            fig_summary.savefig(summary_path, dpi=150, bbox_inches='tight')
            saved_files['summary_plot'] = summary_path
            
            fig_force = self.generate_shap_force_plot(0)
            force_path = os.path.join(self.model_dir, 'shap_force.png')
            fig_force[0].savefig(force_path, dpi=150, bbox_inches='tight')
            saved_files['force_plot'] = force_path
        
        return saved_files
    
    def get_status(self) -> Dict:
        """获取当前状态信息"""
        status = {
            'has_model': self.model is not None,
            'has_data': self.X_train is not None and self.y_train is not None,
            'config': vars(self.config) if self.config else None,
            'results': self.results,
            'feature_names': self.feature_names
        }
        return status
