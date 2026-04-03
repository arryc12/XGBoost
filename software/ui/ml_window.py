"""
机器学习窗口组件，提供 GUI 界面进行 XGBoost 模型训练和 SHAP 分析。
"""

import os
import sys
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QLineEdit, QSpinBox, QDoubleSpinBox,
                             QFileDialog, QTextEdit, QScrollArea, QProgressBar,
                             QGroupBox, QFormLayout, QSplitter, QTabWidget)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_handler import MLHandler, MLConfig


class ProgressThread(QThread):
    """训练线程，处理耗时任务"""
    progress = pyqtSignal(int)  # 进度信号
    finished = pyqtSignal(str)  # 完成信号
    error = pyqtSignal(str)     # 错误信号

    def __init__(self, handler, config, progress_callback):
        super().__init__()
        self.handler = handler
        self.config = config
        self.progress_callback = progress_callback

    def run(self):
        try:
            def update_progress(percent):
                self.progress.emit(percent)
            
            self.handler.train(self.config, update_progress)
            self.finished.emit("训练完成")
        except Exception as e:
            self.error.emit(str(e))


class SHAPFigureCanvas(FigureCanvas):
    """SHAP 图表画布"""
    
    def __init__(self, parent=None, figure=None):
        if figure is None:
            figure = Figure(figsize=(10, 8))
        self.figure = figure
        super().__init__(figure)
        self.setParent(parent)


class MLConfigWidget(QGroupBox):
    """机器学习参数配置组"""
    
    def __init__(self, parent=None):
        super().__init__("XGBoost 参数配置", parent)
        self._init_ui()

    def _init_ui(self):
        layout = QFormLayout(self)
        
        # n_estimators
        self.n_estimators_spin = QSpinBox()
        self.n_estimators_spin.setRange(50, 1000)
        self.n_estimators_spin.setValue(400)
        self.n_estimators_spin.setSuffix(" 次")
        layout.addRow("训练次数:", self.n_estimators_spin)
        
        # max_depth
        self.max_depth_spin = QSpinBox()
        self.max_depth_spin.setRange(3, 20)
        self.max_depth_spin.setValue(10)
        self.max_depth_spin.setSuffix(" 层")
        layout.addRow("最大深度:", self.max_depth_spin)
        
        # learning_rate
        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(0.001, 1.0)
        self.learning_rate_spin.setSingleStep(0.01)
        self.learning_rate_spin.setValue(0.05)
        self.learning_rate_spin.setSuffix(" (0.001-1.0)")
        layout.addRow("学习率:", self.learning_rate_spin)
        
        # subsample
        self.subsample_spin = QDoubleSpinBox()
        self.subsample_spin.setRange(0.5, 1.0)
        self.subsample_spin.setSingleStep(0.05)
        self.subsample_spin.setValue(0.8)
        layout.addRow("采样率:", self.subsample_spin)
        
        # min_child_weight
        self.min_child_weight_spin = QSpinBox()
        self.min_child_weight_spin.setRange(1, 10)
        self.min_child_weight_spin.setValue(1)
        layout.addRow("最小子节点权重:", self.min_child_weight_spin)
    
    def get_config(self) -> MLConfig:
        """获取当前配置"""
        return MLConfig(
            n_estimators=self.n_estimators_spin.value(),
            max_depth=self.max_depth_spin.value(),
            learning_rate=self.learning_rate_spin.value(),
            subsample=self.subsample_spin.value(),
            min_child_weight=self.min_child_weight_spin.value()
        )


class MLWindow(QDialog):
    """机器学习子窗口"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("机器学习 - XGBoost 模型训练")
        self.setMinimumSize(1000, 700)
        
        self.handler = MLHandler()
        self.thread = None
        
        self._init_ui()
        self._populate_label_cols()

    def _init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout(self)
        
        # 主拆分器
        splitter = QSplitter(Qt.Vertical)
        
        # 上部分：控制和参数配置
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        
        # 数据源选择
        data_layout = QHBoxLayout()
        self.data_path_label = QLabel("数据文件：未选择")
        self.data_path_label.setFont(QFont("", 10, QFont.Bold))
        self.select_data_btn = QPushButton("选择数据文件")
        self.select_data_btn.clicked.connect(self.select_data)
        data_layout.addWidget(self.data_path_label)
        data_layout.addWidget(self.select_data_btn)
        top_layout.addLayout(data_layout)
        
        # Label 列指定
        label_layout = QHBoxLayout()
        label_layout.addWidget(QLabel("标签列名:"))
        self.label_col_edit = QLineEdit("label")
        label_layout.addWidget(self.label_col_edit)
        top_layout.addLayout(label_layout)
        
        # 参数配置
        self.config_widget = MLConfigWidget()
        top_layout.addWidget(self.config_widget)
        
        # 训练按钮和进度
        control_layout = QHBoxLayout()
        self.train_btn = QPushButton("开始训练")
        self.train_btn.clicked.connect(self.start_training)
        self.train_btn.setEnabled(False)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        
        self.status_label = QLabel("就绪")
        
        control_layout.addWidget(self.train_btn)
        control_layout.addWidget(self.progress_bar, 1)
        control_layout.addWidget(self.status_label)
        top_layout.addLayout(control_layout)
        
        # 评估结果
        results_layout = QHBoxLayout()
        results_layout.addWidget(QLabel("评估结果:"))
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFixedHeight(100)
        results_layout.addWidget(self.results_text, 1)
        top_layout.addLayout(results_layout)
        
        splitter.addWidget(top_widget)
        
        # 下部分：SHAP 可视化
        shap_widget = QWidget()
        shap_layout = QVBoxLayout(shap_widget)
        
        tabs = QTabWidget()
        
        # SHAP Summary 标签
        summary_scroll = QScrollArea()
        summary_scroll.setWidgetResizable(True)
        self.summary_canvas = self._create_canvas()
        summary_scroll.setWidget(self.summary_canvas)
        
        summary_layout = QVBoxLayout()
        summary_layout.addWidget(QLabel("SHAP Summary 图（特征重要性）"))
        summary_layout.addWidget(summary_scroll)
        
        summary_tab = QWidget()
        summary_tab.setLayout(summary_layout)
        
        # SHAP Force 标签
        force_scroll = QScrollArea()
        force_scroll.setWidgetResizable(True)
        self.force_canvas = self._create_canvas(figsize=(8, 4))
        force_scroll.setWidget(self.force_canvas)
        
        force_layout = QVBoxLayout()
        force_layout.addWidget(QLabel("SHAP Force 图（单个样本解释）"))
        force_layout.addWidget(force_scroll)
        
        force_tab = QWidget()
        force_tab.setLayout(force_layout)
        
        tabs.addTab(summary_tab, "Summary 图")
        tabs.addTab(force_tab, "Force 图")
        
        shap_layout.addWidget(tabs)
        
        splitter.addWidget(shap_widget)
        
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        
        layout.addWidget(splitter)

    def _create_canvas(self, parent=None, figsize=(10, 8)):
        """创建 SHAP 图表画布"""
        canvas = SHAPFigureCanvas(parent, figure=Figure(figsize=figsize))
        return canvas

    def _populate_label_cols(self):
        """填充 label 列建议"""
        pass  # 稍后实现文件选择后自动推荐

    def select_data(self):
        """选择数据文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择数据文件", "", "CSV 文件 (*.csv)"
        )
        if file_path:
            self.data_path_label.setText(f"数据文件：{os.path.basename(file_path)}")
            try:
                # 读取文件以获取可用列
                df = __import__('pandas').read_csv(file_path)
                available_cols = ', '.join(df.columns)
                self.label_col_edit.setToolTip(f"可用列：{available_cols}")
                self.train_btn.setEnabled(True)
            except Exception as e:
                self.status_label.setText(f"读取失败：{e}")
                self.train_btn.setEnabled(False)

    def start_training(self):
        """开始训练模型"""
        if not self.handler.X_train is not None:
            file_path = self.data_path_label.text().replace("数据文件：", "").strip()
            if not file_path:
                self.status_label.setText("请先选择数据文件")
                return
            
            label_col = self.label_col_edit.text().strip()
            if not label_col:
                self.status_label.setText("请指定标签列名")
                return
            
            try:
                preprocess_info = self.handler.load_and_preprocess(
                    file_path, label_col
                )
                self.results_text.append(f"数据已加载：{preprocess_info['train_samples']} 训练样本，{preprocess_info['test_samples']} 测试样本")
                self.results_text.append(f"特征数：{preprocess_info['features']}, 类别权重：{preprocess_info['scale_pos_weight']:.2f}")
            except Exception as e:
                self.status_label.setText(f"数据加载失败：{e}")
                return
        
        config = self.config_widget.get_config()
        self.thread = ProgressThread(self.handler, config, self._update_progress)
        self.thread.finished.connect(self.on_training_finished)
        self.thread.error.connect(self.on_training_error)
        self.thread.start()
        
        self.train_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("训练进行中...")

    def _update_progress(self, percent):
        """更新进度"""
        self.progress_bar.setValue(percent)

    def on_training_finished(self, message):
        """训练完成"""
        self.status_label.setText(message)
        self.train_btn.setEnabled(True)
        self._evaluate_model()

    def on_training_error(self, error_msg):
        """训练错误"""
        self.status_label.setText(f"错误：{error_msg}")
        self.train_btn.setEnabled(True)
        self.progress_bar.setValue(0)

    def _evaluate_model(self):
        """评估模型"""
        try:
            results = self.handler.evaluate()
            
            result_str = f"""
MSE: {results['mse']:.4f}
MAE: {results['mae']:.4f}
R²:  {results['r2']:.4f}
准确率：{results['accuracy']:.4f}
特征数：{results['num_features']}
"""
            self.results_text.setPlainText(result_str)
            
            # 显示 SHAP 图
            fig = self.handler.generate_shap_summary()
            self.summary_canvas.figure = fig
            self.summary_canvas.draw()
            
        except Exception as e:
            self.status_label.setText(f"评估失败：{e}")
