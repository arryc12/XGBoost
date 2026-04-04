"""
数据处理窗口模块，提供概率密度分析、时域/频域特征提取和特征集构建功能。
"""
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.fft import fft, fftfreq
import pywt
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QSpinBox, QCheckBox,
                             QGroupBox, QDialog, QDialogButtonBox,
                             QTableWidget, QTableWidgetItem, QHeaderView,
                             QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import functions


class DataProcessWindow(QMainWindow):
    """数据处理窗口，使用概率密度进行数据处理"""
    def __init__(self, file_paths, parent=None):
        super().__init__(parent)
        self.file_paths = file_paths
        self.data_dict = {}
        self.setWindowTitle("数据处理 - 概率密度分析")
        self.setGeometry(200, 200, 900, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # 左侧控制面板（固定宽度）
        left_widget = QWidget()
        left_widget.setFixedWidth(250)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(5, 5, 5, 5)
        
        # 文件列表信息
        left_layout.addWidget(QLabel(f"已选择 {len(file_paths)} 个文件"))
        
        # 读取所有文件数据
        for file_path in file_paths:
            try:
                df = functions.load_data(file_path)
                self.data_dict[file_path] = df
            except Exception as e:
                print(f"读取失败: {file_path}, {e}")
        
        # 显示文件列表并允许选择
        left_layout.addWidget(QLabel("选择要绘制的文件:"))
        self.file_checkboxes = []
        for file_path in file_paths:
            cb = QCheckBox(os.path.basename(file_path))
            cb.setChecked(True)
            cb.file_path = file_path
            self.file_checkboxes.append(cb)
            left_layout.addWidget(cb)
        
        # 列选择 - 基于第一个文件
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
        
        # 绘制按钮
        self.plot_btn = QPushButton("绘制概率密度图")
        self.plot_btn.clicked.connect(self.plot_pdf)
        left_layout.addWidget(self.plot_btn)
        
        # 时域特征提取按钮
        self.time_domain_btn = QPushButton("时域特征提取")
        self.time_domain_btn.clicked.connect(self.extract_time_domain_features)
        left_layout.addWidget(self.time_domain_btn)
        
        # 频域特征提取按钮
        self.freq_domain_btn = QPushButton("频域特征提取")
        self.freq_domain_btn.clicked.connect(self.extract_freq_domain_features)
        left_layout.addWidget(self.freq_domain_btn)
        
        # 构建特征集按钮
        self.build_dataset_btn = QPushButton("构建特征集")
        self.build_dataset_btn.clicked.connect(self.build_feature_dataset)
        left_layout.addWidget(self.build_dataset_btn)
        
        # 保存结果按钮
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

    def plot_pdf(self):
        """绘制概率密度图"""
        if not self.data_dict:
            QMessageBox.warning(self, "警告", "没有数据！")
            return
        
        selected_files = [cb.file_path for cb in self.file_checkboxes if cb.isChecked()]
        if not selected_files:
            QMessageBox.warning(self, "警告", "请至少选择一个文件！")
            return
        
        selected_cols = [cb.text() for cb in self.col_checkboxes if cb.isChecked()]
        if not selected_cols:
            QMessageBox.warning(self, "警告", "请至少选择一列！")
            return
        
        use_log = self.use_log_checkbox.isChecked()
        use_abs = self.abs_checkbox.isChecked()
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        colors = plt.cm.tab10.colors
        
        for idx, file_path in enumerate(selected_files):
            df = self.data_dict[file_path]
            file_name = os.path.basename(file_path)
            color = colors[idx % len(colors)]
            
            for col in selected_cols:
                if col in df.columns:
                    plot_data = df[col].dropna()
                    if use_abs:
                        plot_data = plot_data.abs()
                    if use_log:
                        plot_data = np.log(np.abs(plot_data))
                    
                    label = f"{file_name} - {col}"
                    sns.kdeplot(plot_data, ax=ax, label=label, linewidth=1.5, color=color)
        
        ax.set_xlabel('Value')
        ax.set_ylabel('Probability Density')
        ax.set_title('概率密度分布')
        ax.legend()
        ax.grid(True)
        self.figure.tight_layout()
        self.canvas.draw()

    def save_processed_data(self):
        """保存处理后的数据"""
        if not self.data_dict:
            QMessageBox.warning(self, "警告", "没有数据！")
            return
        
        selected_files = [cb.file_path for cb in self.file_checkboxes if cb.isChecked()]
        if not selected_files:
            QMessageBox.warning(self, "警告", "请至少选择一个文件！")
            return
        
        selected_cols = [cb.text() for cb in self.col_checkboxes if cb.isChecked()]
        if not selected_cols:
            QMessageBox.warning(self, "警告", "请至少选择一列！")
            return
        
        use_log = self.use_log_checkbox.isChecked()
        use_abs = self.abs_checkbox.isChecked()
        
        all_data = []
        for file_path in selected_files:
            df = self.data_dict[file_path]
            save_df = df[selected_cols].copy()
            
            if use_abs:
                save_df = save_df.abs()
            if use_log:
                save_df = np.log(np.abs(save_df))
            
            save_df['source_file'] = os.path.basename(file_path)
            all_data.append(save_df)
        
        merged_df = pd.concat(all_data, ignore_index=True)
        
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
        """构建特征集，参照datasets_switch.py的逻辑"""
        if not self.data_dict:
            QMessageBox.warning(self, "警告", "没有数据！")
            return
        
        selected_files = [cb.file_path for cb in self.file_checkboxes if cb.isChecked()]
        if not selected_files:
            QMessageBox.warning(self, "警告", "请至少选择一个文件！")
            return
        
        selected_cols = [cb.text() for cb in self.col_checkboxes if cb.isChecked()]
        if not selected_cols:
            QMessageBox.warning(self, "警告", "请至少选择一列！")
            return
        
        # 创建设置对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("构建特征集设置")
        dialog.setMinimumWidth(400)
        layout = QVBoxLayout(dialog)
        
        # 切片长度设置
        chunk_layout = QHBoxLayout()
        chunk_layout.addWidget(QLabel("切片长度:"))
        chunk_spin = QSpinBox()
        chunk_spin.setRange(100, 100000)
        chunk_spin.setValue(2000)
        chunk_spin.setSingleStep(100)
        chunk_layout.addWidget(chunk_spin)
        layout.addLayout(chunk_layout)
        
        # 概率密度特征选择
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
        
        # 时域特征选择
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
        
        # 频域特征选择
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
        
        # Label设置
        label_layout = QHBoxLayout()
        label_layout.addWidget(QLabel("Label值(应用到所有文件):"))
        label_spin = QSpinBox()
        label_spin.setRange(0, 100)
        label_spin.setValue(0)
        label_layout.addWidget(label_spin)
        layout.addLayout(label_layout)
        
        # 按钮
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(dialog.accept)
        btn_box.rejected.connect(dialog.reject)
        layout.addWidget(btn_box)
        
        if dialog.exec_() != QDialog.Accepted:
            return
        
        # 获取设置
        chunk_size = chunk_spin.value()
        label_value = label_spin.value()
        
        selected_pdf_features = [k for k, cb in pdf_features.items() if cb.isChecked()]
        selected_time_features = [k for k, cb in time_features.items() if cb.isChecked()]
        selected_freq_features = [k for k, cb in freq_features.items() if cb.isChecked()]
        
        # 构建特征
        try:
            all_results = []
            
            for file_path in selected_files:
                df = self.data_dict[file_path]
                file_name = os.path.basename(file_path)
                
                for col in selected_cols:
                    if col in df.columns:
                        signal = df[col].dropna().values
                        
                        # 按切片长度分块处理
                        for start in range(0, len(signal), chunk_size):
                            piece = signal[start:start + chunk_size]
                            if len(piece) < chunk_size // 2:  # 跳过过短的切片
                                continue
                            
                            row = {'source_file': file_name, 'column': col}
                            
                            # 概率密度特征
                            if 'mean' in selected_pdf_features:
                                row['mean'] = np.mean(piece)
                            if 'std' in selected_pdf_features:
                                row['std'] = np.std(piece, ddof=0)
                            if 'var' in selected_pdf_features:
                                row['var'] = np.var(piece, ddof=0)
                            if 'median' in selected_pdf_features:
                                row['median'] = np.median(piece)
                            if 'mode' in selected_pdf_features:
                                # 众数计算（取出现次数最多的值）
                                vals, counts = np.unique(piece, return_counts=True)
                                row['mode'] = vals[np.argmax(counts)]
                            if 'skew' in selected_pdf_features:
                                row['skew'] = stats.skew(piece, bias=False)
                            if 'kurt' in selected_pdf_features:
                                row['kurt'] = stats.kurtosis(piece, bias=False)
                            
                            # 时域特征
                            if 'rms' in selected_time_features:
                                row['rms'] = np.sqrt(np.mean(piece ** 2))
                            if 'peak' in selected_time_features:
                                row['peak'] = np.max(piece) - np.min(piece)
                            if 'impulse' in selected_time_features:
                                row['impulse'] = np.max(np.abs(piece)) / np.mean(np.abs(piece))
                            if 'margin' in selected_time_features:
                                row['margin'] = np.max(np.abs(piece)) / (np.mean(np.sqrt(np.abs(piece))) ** 2)
                            if 'waveform' in selected_time_features:
                                row['waveform'] = np.sqrt(np.mean(piece ** 2)) / np.mean(np.abs(piece))
                            
                            # 频域特征
                            if selected_freq_features:
                                n = len(piece)
                                yf = fft(piece)
                                psd = np.abs(yf[:n//2])**2
                                xf = fftfreq(n, 1/20000)[:n//2]  # 假设采样率20000
                                
                                if 'dominant_freq' in selected_freq_features:
                                    row['dominant_freq'] = xf[np.argmax(psd)]
                                if 'total_power' in selected_freq_features:
                                    row['total_power'] = np.sum(psd)
                                if 'power_ratio' in selected_freq_features:
                                    row['power_ratio'] = np.max(psd) / np.sum(psd) if np.sum(psd) > 0 else 0
                                if 'wavelet_entropy' in selected_freq_features:
                                    try:
                                        wp = pywt.WaveletPacket(data=piece, wavelet='db4', mode='symmetric', maxlevel=4)
                                        nodes = [node.path for node in wp.get_level(wp.maxlevel, 'freq')]
                                        energy = np.array([np.sum(wp[n].data**2) for n in nodes])
                                        energy_norm = energy / energy.sum() if energy.sum() > 0 else energy
                                        energy_norm = energy_norm + 1e-16
                                        row['wavelet_entropy'] = -np.sum(energy_norm * np.log2(energy_norm))
                                    except:
                                        row['wavelet_entropy'] = 0
                            
                            row['label'] = label_value
                            all_results.append(row)
            
            if not all_results:
                QMessageBox.warning(self, "警告", "未生成任何特征数据！")
                return
            
            # 保存结果
            result_df = pd.DataFrame(all_results)
            
            # 重新排列列顺序（与annular.csv格式保持一致）
            feature_cols = []
            feature_cols.extend(selected_pdf_features)
            feature_cols.extend(selected_time_features)
            feature_cols.extend(selected_freq_features)
            feature_cols.append('label')
            
            # 只保留存在的列
            final_cols = [c for c in feature_cols if c in result_df.columns]
            result_df = result_df[final_cols]
            
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
        
        selected_files = [cb.file_path for cb in self.file_checkboxes if cb.isChecked()]
        if not selected_files:
            QMessageBox.warning(self, "警告", "请至少选择一个文件！")
            return
        
        selected_cols = [cb.text() for cb in self.col_checkboxes if cb.isChecked()]
        if not selected_cols:
            QMessageBox.warning(self, "警告", "请至少选择一列！")
            return
        
        # 存储所有特征结果
        all_features = []
        
        for file_path in selected_files:
            df = self.data_dict[file_path]
            file_name = os.path.basename(file_path)
            
            for col in selected_cols:
                if col in df.columns:
                    signal_data = df[col].dropna().values
                    
                    # 提取时域特征
                    features = self._calculate_time_domain_features(signal_data)
                    features['file'] = file_name
                    features['column'] = col
                    all_features.append(features)
        
        # 显示特征结果
        self._show_feature_results(all_features, "时域特征")
        
        # 可视化时域特征
        self._plot_time_domain_features(selected_files, selected_cols)
    
    def _calculate_time_domain_features(self, signal):
        """计算时域特征"""
        features = {}
        features['均值'] = np.mean(signal)
        features['标准差'] = np.std(signal)
        features['最大值'] = np.max(signal)
        features['最小值'] = np.min(signal)
        features['峰峰值'] = np.max(signal) - np.min(signal)
        features['均方根值'] = np.sqrt(np.mean(signal ** 2))
        features['峰度'] = stats.kurtosis(signal)
        features['偏度'] = stats.skew(signal)
        features['方差'] = np.var(signal)
        features['脉冲因子'] = np.max(np.abs(signal)) / np.mean(np.abs(signal))
        features['裕度因子'] = np.max(np.abs(signal)) / np.mean(np.sqrt(np.abs(signal))) ** 2
        features['波形因子'] = np.sqrt(np.mean(signal ** 2)) / np.mean(np.abs(signal))
        return features
    
    def _plot_time_domain_features(self, selected_files, selected_cols):
        """绘制时域特征可视化，参照feature extraction.py的标注方式"""
        self.figure.clear()
        
        n_files = len(selected_files)
        n_cols = len(selected_cols)
        
        if n_files == 0 or n_cols == 0:
            return
        
        # 创建子图
        axes = self.figure.subplots(n_files, 1, squeeze=False)
        
        colors = plt.cm.tab10.colors
        
        for idx, file_path in enumerate(selected_files):
            df = self.data_dict[file_path]
            file_name = os.path.basename(file_path)
            
            ax = axes[idx, 0]
            
            for col_idx, col in enumerate(selected_cols):
                if col in df.columns:
                    signal_data = df[col].dropna().values
                    color = colors[col_idx % len(colors)]
                    
                    # 绘制信号
                    ax.plot(signal_data, label=f"{col}", color=color, alpha=0.7)
                    
                    # 计算时域特征
                    mean_val = np.mean(signal_data)
                    std_val = np.std(signal_data)
                    max_val = np.max(signal_data)
                    min_val = np.min(signal_data)
                    
                    # 标注特征线（参照feature extraction.py的方式）
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
        
        selected_files = [cb.file_path for cb in self.file_checkboxes if cb.isChecked()]
        if not selected_files:
            QMessageBox.warning(self, "警告", "请至少选择一个文件！")
            return
        
        selected_cols = [cb.text() for cb in self.col_checkboxes if cb.isChecked()]
        if not selected_cols:
            QMessageBox.warning(self, "警告", "请至少选择一列！")
            return
        
        # 存储所有特征结果
        all_features = []
        
        for file_path in selected_files:
            df = self.data_dict[file_path]
            file_name = os.path.basename(file_path)
            
            for col in selected_cols:
                if col in df.columns:
                    signal_data = df[col].dropna().values
                    
                    # 提取频域特征
                    features, xf, psd = self._calculate_freq_domain_features(signal_data)
                    features['file'] = file_name
                    features['column'] = col
                    all_features.append(features)
        
        # 显示特征结果
        self._show_feature_results(all_features, "频域特征")
        
        # 可视化频域特征
        self._plot_freq_domain_features(selected_files, selected_cols)
    
    def _calculate_freq_domain_features(self, signal, fs=20000):
        """计算频域特征"""
        n = len(signal)
        yf = fft(signal)
        xf = fftfreq(n, 1/fs)[:n//2]
        psd = np.abs(yf[:n//2])**2
        
        features = {}
        features['主频率'] = xf[np.argmax(psd)]
        features['总功率'] = np.sum(psd)
        features['功率比'] = np.max(psd) / np.sum(psd)
        features['平均频率'] = np.sum(xf * psd) / np.sum(psd)
        features['频率标准差'] = np.sqrt(np.sum((xf - features['平均频率'])**2 * psd) / np.sum(psd))
        features['频率重心'] = np.sum(xf * psd) / np.sum(psd)
        
        # 小波包能量特征
        try:
            wp = pywt.WaveletPacket(data=signal, wavelet='db4', mode='symmetric', maxlevel=4)
            nodes = [node.path for node in wp.get_level(wp.maxlevel, 'freq')]
            energy = np.array([np.sum(wp[n].data**2) for n in nodes])
            energy_norm = energy / energy.sum()
            energy_norm += 1e-16
            shannon_entropy = -np.sum(energy_norm * np.log2(energy_norm))
            features['小波包能量熵'] = shannon_entropy
        except:
            features['小波包能量熵'] = 0
        
        return features, xf, psd
    
    def _plot_freq_domain_features(self, selected_files, selected_cols):
        """绘制频域特征可视化"""
        self.figure.clear()
        
        n_files = len(selected_files)
        n_cols = len(selected_cols)
        
        if n_files == 0 or n_cols == 0:
            return
        
        # 创建子图
        axes = self.figure.subplots(n_files, 2, squeeze=False)
        
        colors = plt.cm.tab10.colors
        
        for idx, file_path in enumerate(selected_files):
            df = self.data_dict[file_path]
            file_name = os.path.basename(file_path)
            color = colors[idx % len(colors)]
            
            for col_idx, col in enumerate(selected_cols):
                if col in df.columns:
                    signal_data = df[col].dropna().values
                    
                    # 计算频域特征
                    features, xf, psd = self._calculate_freq_domain_features(signal_data)
                    
                    # 绘制功率谱
                    ax1 = axes[idx, 0]
                    ax1.plot(xf, psd, label=f"{col}", color=color, alpha=0.7)
                    ax1.axvline(x=features['主频率'], color='r', linestyle='--', 
                               label=f"主频率: {features['主频率']:.2f} Hz")
                    ax1.set_xlabel('频率 (Hz)')
                    ax1.set_ylabel('功率谱密度')
                    ax1.set_title(f'{file_name} - 功率谱')
                    ax1.legend()
                    ax1.grid(True)
                    
                    # 绘制小波包能量分布
                    ax2 = axes[idx, 1]
                    try:
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
        
        # 创建结果显示窗口
        result_dialog = QDialog(self)
        result_dialog.setWindowTitle(f"{feature_type}提取结果")
        result_dialog.setGeometry(300, 300, 800, 600)
        
        layout = QVBoxLayout(result_dialog)
        
        # 创建表格显示结果
        table = QTableWidget()
        
        # 设置表格列
        if features_list:
            first_feature = features_list[0]
            columns = ['文件', '列名'] + [k for k in first_feature.keys() if k not in ['file', 'column']]
            table.setColumnCount(len(columns))
            table.setHorizontalHeaderLabels(columns)
            table.setRowCount(len(features_list))
            
            # 填充数据
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
            
            # 调整列宽
            table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        layout.addWidget(table)
        
        # 添加关闭按钮
        button_box = QDialogButtonBox(QDialogButtonBox.Ok)
        button_box.accepted.connect(result_dialog.accept)
        layout.addWidget(button_box)
        
        result_dialog.exec_()
