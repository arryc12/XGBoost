"""
数据处理窗口模块，提供概率密度分析、时域/频域特征提取和特征集构建功能。
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pywt
from scipy.fft import fft, fftfreq
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QCheckBox,
                             QGroupBox, QDialog, QDialogButtonBox,
                             QTableWidget, QTableWidgetItem, QHeaderView,
                             QFileDialog, QMessageBox, QSpinBox)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from handlers.io_handler import load_data
from handlers.data_handler import compute_pdf_data, plot_pdf, process_and_save_data
from handlers.feature_handler import (calculate_time_domain_features, 
                                       calculate_freq_domain_features,
                                       build_feature_dataset)


class DataProcessWindow(QMainWindow):
    """数据处理窗口"""
    def __init__(self, file_paths, parent=None):
        super().__init__(parent)
        self.file_paths = file_paths
        self.data_dict = {}
        self.setWindowTitle("数据处理")
        self.setGeometry(200, 200, 900, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # 左侧控制面板（固定宽度）
        left_widget = QWidget()
        left_widget.setFixedWidth(250)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(5, 5, 5, 5)
        
        left_layout.addWidget(QLabel(f"已选择 {len(file_paths)} 个文件"))
        
        # 读取所有文件数据
        for file_path in file_paths:
            try:
                df = load_data(file_path)
                self.data_dict[file_path] = df
            except Exception as e:
                print(f"读取失败: {file_path}, {e}")
        
        # 文件选择
        left_layout.addWidget(QLabel("选择要绘制的文件:"))
        self.file_checkboxes = []
        for file_path in file_paths:
            cb = QCheckBox(os.path.basename(file_path))
            cb.setChecked(True)
            cb.file_path = file_path
            self.file_checkboxes.append(cb)
            left_layout.addWidget(cb)
        
        # 列选择
        if self.data_dict:
            first_file = list(self.data_dict.keys())[0]
            first_df = self.data_dict[first_file]
            left_layout.addWidget(QLabel(f"选择要处理的列 (共{len(first_df.columns)}列):"))
            self.col_checkboxes = []
            for col in first_df.columns:
                cb = QCheckBox(col)
                cb.setChecked(True)
                self.col_checkboxes.append(cb)
                left_layout.addWidget(cb)
        
        # 处理选项
        options_group = QGroupBox("处理选项")
        options_layout = QVBoxLayout()
        self.use_log_checkbox = QCheckBox("取对数")
        self.use_log_checkbox.setChecked(True)
        options_layout.addWidget(self.use_log_checkbox)
        self.abs_checkbox = QCheckBox("取绝对值")
        self.abs_checkbox.setChecked(True)
        options_layout.addWidget(self.abs_checkbox)
        options_group.setLayout(options_layout)
        left_layout.addWidget(options_group)
        
        # 功能按钮
        self.plot_btn = QPushButton("绘制概率密度图")
        self.plot_btn.clicked.connect(self.plot_pdf)
        left_layout.addWidget(self.plot_btn)
        
        self.time_domain_btn = QPushButton("时域特征提取")
        self.time_domain_btn.clicked.connect(self.extract_time_domain_features)
        left_layout.addWidget(self.time_domain_btn)
        
        self.freq_domain_btn = QPushButton("频域特征提取")
        self.freq_domain_btn.clicked.connect(self.extract_freq_domain_features)
        left_layout.addWidget(self.freq_domain_btn)
        
        self.build_dataset_btn = QPushButton("构建特征集")
        self.build_dataset_btn.clicked.connect(self.build_feature_dataset)
        left_layout.addWidget(self.build_dataset_btn)
        
        self.save_btn = QPushButton("保存处理后的数据")
        self.save_btn.clicked.connect(self.save_processed_data)
        left_layout.addWidget(self.save_btn)
        
        left_layout.addStretch()
        main_layout.addWidget(left_widget)

        # 右侧图表区域
        right_layout = QVBoxLayout()
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        right_layout.addWidget(self.canvas)
        main_layout.addLayout(right_layout)

    def _get_selected_files(self):
        """获取选中的文件路径列表"""
        return [cb.file_path for cb in self.file_checkboxes if cb.isChecked()]

    def _get_selected_cols(self):
        """获取选中的列名列表"""
        return [cb.text() for cb in self.col_checkboxes if cb.isChecked()]

    def plot_pdf(self):
        """绘制概率密度图"""
        if not self.data_dict:
            QMessageBox.warning(self, "警告", "没有数据！")
            return
        
        selected_files = self._get_selected_files()
        if not selected_files:
            QMessageBox.warning(self, "警告", "请至少选择一个文件！")
            return
        
        selected_cols = self._get_selected_cols()
        if not selected_cols:
            QMessageBox.warning(self, "警告", "请至少选择一列！")
            return
        
        use_log = self.use_log_checkbox.isChecked()
        use_abs = self.abs_checkbox.isChecked()
        
        pdf_data = compute_pdf_data(self.data_dict, selected_files, selected_cols, use_log, use_abs)
        if not pdf_data:
            QMessageBox.warning(self, "警告", "没有可绘制的数据！")
            return
        
        plot_pdf(self.figure, pdf_data)
        self.canvas.draw()

    def save_processed_data(self):
        """保存处理后的数据"""
        if not self.data_dict:
            QMessageBox.warning(self, "警告", "没有数据！")
            return
        
        selected_files = self._get_selected_files()
        if not selected_files:
            QMessageBox.warning(self, "警告", "请至少选择一个文件！")
            return
        
        selected_cols = self._get_selected_cols()
        if not selected_cols:
            QMessageBox.warning(self, "警告", "请至少选择一列！")
            return
        
        use_log = self.use_log_checkbox.isChecked()
        use_abs = self.abs_checkbox.isChecked()
        
        merged_df = process_and_save_data(self.data_dict, selected_files, selected_cols, use_log, use_abs)
        if merged_df.empty:
            QMessageBox.warning(self, "警告", "没有可保存的数据！")
            return
        
        default_name = "processed_data.csv"
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存处理后的数据", default_name, "CSV文件 (*.csv)"
        )
        if file_path:
            try:
                merged_df.to_csv(file_path, index=False, encoding='utf-8-sig')
                QMessageBox.information(self, "成功", "数据保存成功！")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败：{str(e)}")
    
    def build_feature_dataset(self):
        """构建特征集"""
        if not self.data_dict:
            QMessageBox.warning(self, "警告", "没有数据！")
            return
        
        selected_files = self._get_selected_files()
        if not selected_files:
            QMessageBox.warning(self, "警告", "请至少选择一个文件！")
            return
        
        selected_cols = self._get_selected_cols()
        if not selected_cols:
            QMessageBox.warning(self, "警告", "请至少选择一列！")
            return
        
        dialog = QDialog(self)
        dialog.setWindowTitle("构建特征集设置")
        dialog.setMinimumWidth(400)
        layout = QVBoxLayout(dialog)
        
        chunk_layout = QHBoxLayout()
        chunk_layout.addWidget(QLabel("切片长度:"))
        chunk_spin = QSpinBox()
        chunk_spin.setRange(100, 100000)
        chunk_spin.setValue(2000)
        chunk_spin.setSingleStep(100)
        chunk_layout.addWidget(chunk_spin)
        layout.addLayout(chunk_layout)
        
        pdf_group = QGroupBox("概率密度特征")
        pdf_layout = QVBoxLayout()
        pdf_features = {
            'mode': QCheckBox("众数 (mode)"),
            'mean': QCheckBox("均值 (mean)"),
            'median': QCheckBox("中位数 (median)"),
            'var': QCheckBox("方差 (var)"),
            'std': QCheckBox("标准差 (std)"),
            'skew': QCheckBox("偏度 (skew)"),
            'kurt': QCheckBox("峰度 (kurt)")
        }
        for cb in pdf_features.values():
            cb.setChecked(True)
            pdf_layout.addWidget(cb)
        pdf_group.setLayout(pdf_layout)
        layout.addWidget(pdf_group)
        
        time_group = QGroupBox("时域特征")
        time_layout = QVBoxLayout()
        time_features = {
            'rms': QCheckBox("均方根值 (RMS)"),
            'peak': QCheckBox("峰峰值 (peak-to-peak)"),
            'impulse': QCheckBox("脉冲因子 (impulse)"),
            'margin': QCheckBox("裕度因子 (margin)"),
            'waveform': QCheckBox("波形因子 (waveform)")
        }
        for cb in time_features.values():
            cb.setChecked(False)
            time_layout.addWidget(cb)
        time_group.setLayout(time_layout)
        layout.addWidget(time_group)
        
        freq_group = QGroupBox("频域特征")
        freq_layout = QVBoxLayout()
        freq_features = {
            'dominant_freq': QCheckBox("主频率 (dominant frequency)"),
            'total_power': QCheckBox("总功率 (total power)"),
            'power_ratio': QCheckBox("功率比 (power ratio)"),
            'wavelet_entropy': QCheckBox("小波包能量熵 (wavelet entropy)")
        }
        for cb in freq_features.values():
            cb.setChecked(False)
            freq_layout.addWidget(cb)
        freq_group.setLayout(freq_layout)
        layout.addWidget(freq_group)
        
        label_layout = QHBoxLayout()
        label_layout.addWidget(QLabel("Label值(应用到所有文件):"))
        label_spin = QSpinBox()
        label_spin.setRange(0, 100)
        label_spin.setValue(0)
        label_layout.addWidget(label_spin)
        layout.addLayout(label_layout)
        
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(dialog.accept)
        btn_box.rejected.connect(dialog.reject)
        layout.addWidget(btn_box)
        
        if dialog.exec_() != QDialog.Accepted:
            return
        
        chunk_size = chunk_spin.value()
        label_value = label_spin.value()
        
        selected_pdf = [k for k, cb in pdf_features.items() if cb.isChecked()]
        selected_time = [k for k, cb in time_features.items() if cb.isChecked()]
        selected_freq = [k for k, cb in freq_features.items() if cb.isChecked()]
        
        try:
            result_df = build_feature_dataset(
                self.data_dict, selected_files, selected_cols, chunk_size,
                label_value, selected_pdf, selected_time, selected_freq
            )
            
            if result_df.empty:
                QMessageBox.warning(self, "警告", "未生成任何特征数据！")
                return
            
            default_name = "feature_dataset.csv"
            save_path, _ = QFileDialog.getSaveFileName(
                self, "保存特征集", default_name, "CSV文件 (*.csv)"
            )
            
            if save_path:
                result_df.to_csv(save_path, index=False, float_format='%.6f')
                QMessageBox.information(self, "成功", 
                    f"特征集构建完成！\n共 {len(result_df)} 条记录\n保存至: {save_path}")
        
        except Exception as e:
            QMessageBox.critical(self, "错误", f"构建特征集失败: {str(e)}")
    
    def extract_time_domain_features(self):
        """提取时域特征"""
        if not self.data_dict:
            QMessageBox.warning(self, "警告", "没有数据！")
            return
        
        selected_files = self._get_selected_files()
        if not selected_files:
            QMessageBox.warning(self, "警告", "请至少选择一个文件！")
            return
        
        selected_cols = self._get_selected_cols()
        if not selected_cols:
            QMessageBox.warning(self, "警告", "请至少选择一列！")
            return
        
        all_features = []
        for file_path in selected_files:
            if file_path not in self.data_dict:
                continue
            df = self.data_dict[file_path]
            file_name = os.path.basename(file_path)
            
            for col in selected_cols:
                if col in df.columns:
                    signal_data = df[col].dropna().values
                    features = calculate_time_domain_features(signal_data)
                    features['file'] = file_name
                    features['column'] = col
                    all_features.append(features)
        
        self._show_feature_results(all_features, "时域特征")
        self._plot_time_domain_features(selected_files, selected_cols)
    
    def _plot_time_domain_features(self, selected_files, selected_cols):
        """绘制时域特征可视化"""
        self.figure.clear()
        n_files = len(selected_files)
        if n_files == 0:
            return
        
        axes = self.figure.subplots(n_files, 1, squeeze=False)
        colors = plt.cm.tab10.colors
        
        for idx, file_path in enumerate(selected_files):
            if file_path not in self.data_dict:
                continue
            df = self.data_dict[file_path]
            file_name = os.path.basename(file_path)
            ax = axes[idx, 0]
            
            for col_idx, col in enumerate(selected_cols):
                if col in df.columns:
                    signal_data = df[col].dropna().values
                    color = colors[col_idx % len(colors)]
                    ax.plot(signal_data, label=f"{col}", color=color, alpha=0.7)
                    
                    mean_val = np.mean(signal_data)
                    std_val = np.std(signal_data)
                    max_val = np.max(signal_data)
                    min_val = np.min(signal_data)
                    
                    ax.axhline(y=mean_val, color='r', linestyle='--', alpha=0.5,
                              label=f'{col} 均值 ({mean_val:.2f})')
                    ax.axhline(y=mean_val + std_val, color='g', linestyle='--', alpha=0.5,
                              label=f'{col} 均值+标准差 ({mean_val + std_val:.2f})')
                    ax.axhline(y=mean_val - std_val, color='g', linestyle='--', alpha=0.5,
                              label=f'{col} 均值-标准差 ({mean_val - std_val:.2f})')
                    ax.axhline(y=max_val, color='b', linestyle='--', alpha=0.5,
                              label=f'{col} 最大值 ({max_val:.2f})')
                    ax.axhline(y=min_val, color='m', linestyle='--', alpha=0.5,
                              label=f'{col} 最小值 ({min_val:.2f})')
            
            ax.set_xlabel('样本点')
            ax.set_ylabel('幅值')
            ax.set_title(f'{file_name} - 时域信号及特征标注')
            ax.legend(loc='best', fontsize='small')
            ax.grid(True)
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def extract_freq_domain_features(self):
        """提取频域特征"""
        if not self.data_dict:
            QMessageBox.warning(self, "警告", "没有数据！")
            return
        
        selected_files = self._get_selected_files()
        if not selected_files:
            QMessageBox.warning(self, "警告", "请至少选择一个文件！")
            return
        
        selected_cols = self._get_selected_cols()
        if not selected_cols:
            QMessageBox.warning(self, "警告", "请至少选择一列！")
            return
        
        all_features = []
        for file_path in selected_files:
            if file_path not in self.data_dict:
                continue
            df = self.data_dict[file_path]
            file_name = os.path.basename(file_path)
            
            for col in selected_cols:
                if col in df.columns:
                    signal_data = df[col].dropna().values
                    features, xf, psd = calculate_freq_domain_features(signal_data)
                    features['file'] = file_name
                    features['column'] = col
                    all_features.append(features)
        
        self._show_feature_results(all_features, "频域特征")
        self._plot_freq_domain_features(selected_files, selected_cols)
    
    def _plot_freq_domain_features(self, selected_files, selected_cols):
        """绘制频域特征可视化"""
        self.figure.clear()
        n_files = len(selected_files)
        if n_files == 0:
            return
        
        axes = self.figure.subplots(n_files, 2, squeeze=False)
        colors = plt.cm.tab10.colors
        
        for idx, file_path in enumerate(selected_files):
            if file_path not in self.data_dict:
                continue
            df = self.data_dict[file_path]
            file_name = os.path.basename(file_path)
            color = colors[idx % len(colors)]
            
            for col_idx, col in enumerate(selected_cols):
                if col in df.columns:
                    signal_data = df[col].dropna().values
                    features, xf, psd = calculate_freq_domain_features(signal_data)
                    
                    ax1 = axes[idx, 0]
                    ax1.plot(xf, psd, label=f"{col}", color=color, alpha=0.7)
                    ax1.axvline(x=features['主频率'], color='r', linestyle='--', 
                               label=f"主频率: {features['主频率']:.2f} Hz")
                    ax1.set_xlabel('频率 (Hz)')
                    ax1.set_ylabel('功率谱密度')
                    ax1.set_title(f'{file_name} - 功率谱')
                    ax1.legend()
                    ax1.grid(True)
                    
                    ax2 = axes[idx, 1]
                    try:
                        import pywt
                        wp = pywt.WaveletPacket(data=signal_data, wavelet='db4', mode='symmetric', maxlevel=4)
                        nodes = [node.path for node in wp.get_level(wp.maxlevel, 'freq')]
                        energy = np.array([np.sum(wp[n].data**2) for n in nodes])
                        energy_norm = energy / energy.sum()
                        
                        x = np.arange(len(nodes))
                        ax2.bar(x, energy_norm, color=color, alpha=0.7)
                        ax2.set_xlabel('小波包节点')
                        ax2.set_ylabel('归一化能量')
                        ax2.set_title(f'{file_name} - 小波包能量分布')
                        ax2.set_xticks(x)
                        ax2.set_xticklabels(nodes, rotation=45, ha='right')
                        ax2.grid(True)
                    except:
                        ax2.text(0.5, 0.5, '小波包分解失败', 
                                ha='center', va='center', transform=ax2.transAxes)
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def _show_feature_results(self, features_list, feature_type):
        """显示特征提取结果"""
        if not features_list:
            return
        
        result_dialog = QDialog(self)
        result_dialog.setWindowTitle(f"{feature_type}提取结果")
        result_dialog.setGeometry(300, 300, 800, 600)
        
        layout = QVBoxLayout(result_dialog)
        table = QTableWidget()
        
        if features_list:
            first_feature = features_list[0]
            columns = ['文件', '列名'] + [k for k in first_feature.keys() if k not in ['file', 'column']]
            table.setColumnCount(len(columns))
            table.setHorizontalHeaderLabels(columns)
            table.setRowCount(len(features_list))
            
            for row_idx, features in enumerate(features_list):
                table.setItem(row_idx, 0, QTableWidgetItem(features.get('file', '')))
                table.setItem(row_idx, 1, QTableWidgetItem(features.get('column', '')))
                
                col_idx = 2
                for key, value in features.items():
                    if key not in ['file', 'column']:
                        if isinstance(value, float):
                            table.setItem(row_idx, col_idx, QTableWidgetItem(f"{value:.6f}"))
                        else:
                            table.setItem(row_idx, col_idx, QTableWidgetItem(str(value)))
                        col_idx += 1
            
            table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        layout.addWidget(table)
        
        button_box = QDialogButtonBox(QDialogButtonBox.Ok)
        button_box.accepted.connect(result_dialog.accept)
        layout.addWidget(button_box)
        
        result_dialog.exec_()
