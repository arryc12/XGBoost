"""
主窗口模块，构建GUI界面并响应用户操作。
"""
import os
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QListWidget, QTextEdit,
                             QFileDialog, QMessageBox,QTableWidget, QTableWidgetItem, 
                             QHeaderView, QLabel, QSpinBox, QDialog, 
                             QDialogButtonBox, QCheckBox, QComboBox, QGroupBox,
                             QFormLayout, QRadioButton)
from PyQt5.QtCore import Qt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# 导入功能函数模块
import functions


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("多格式文件处理器")
        self.setGeometry(100, 100, 600, 400)

        # 创建中央部件和主布局（左右布局）
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # 左侧按钮布局（垂直）
        left_layout = QVBoxLayout()
        self.select_btn = QPushButton("选择文件")
        self.process_btn = QPushButton("处理文件")
        self.clear_btn = QPushButton("清空列表")
        self.show_btn = QPushButton("显示数据")
        self.data_process_btn = QPushButton("数据处理")
        self.show_btn.clicked.connect(self.show_data)
        self.data_process_btn.clicked.connect(self.open_data_process)
        left_layout.addWidget(self.select_btn)
        left_layout.addWidget(self.process_btn)
        left_layout.addWidget(self.clear_btn)
        left_layout.addWidget(self.show_btn)
        left_layout.addWidget(self.data_process_btn)
        left_layout.addStretch()
        main_layout.addLayout(left_layout)

        # 右侧布局（上下）
        right_layout = QVBoxLayout()
        
        # 文件列表部件
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.ExtendedSelection)
        right_layout.addWidget(self.file_list)

        # 结果显示区域
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        right_layout.addWidget(self.result_text)
        
        main_layout.addLayout(right_layout)

        # 连接信号与槽
        self.select_btn.clicked.connect(self.select_files)
        self.process_btn.clicked.connect(self.process_files)
        self.clear_btn.clicked.connect(self.clear_files)

        # 存储已选文件路径
        self.file_paths = []

    def select_files(self):
        """打开文件对话框，选择多个文件"""
        # 定义文件过滤器
        filter = "数据文件 (*.tdms *.csv *.xls *.xlsx);;所有文件 (*.*)"
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择文件", "", filter
        )
        if files:
            self.file_paths.extend(files)  # 追加到列表
            self.update_file_list()

    def update_file_list(self):
        """更新列表控件显示所有已选文件"""
        self.file_list.clear()
        for path in self.file_paths:
            self.file_list.addItem(path)

    def clear_files(self):
        """清空文件列表"""
        self.file_paths.clear()
        self.file_list.clear()
        self.result_text.clear()

    def process_files(self):
        """调用功能函数处理选中的文件（或全部）"""
        # 如果列表中有选中的项，则只处理选中的文件；否则处理全部
        selected_items = self.file_list.selectedItems()
        if selected_items:
            # 获取选中项的路径
            paths = [item.text() for item in selected_items]
        else:
            # 没有选中项则处理所有文件
            paths = self.file_paths

        if not paths:
            QMessageBox.warning(self, "警告", "请先选择文件！")
            return

        # 调用功能函数处理文件
        try:
            results = functions.process_files(paths)
            # 显示结果
            self.result_text.clear()
            self.result_text.append("处理结果：\n" + "\n".join(results))
        except Exception as e:
            QMessageBox.critical(self, "错误", f"处理过程中发生异常：{str(e)}")
    
    def show_data(self):
        """打开新窗口显示选中文件的数据"""
        # 确定要显示的文件：优先选中的文件，否则全部
        selected_items = self.file_list.selectedItems()
        if selected_items:
            paths = [item.text() for item in selected_items]
        else:
            paths = self.file_paths

        if not paths:
            QMessageBox.warning(self, "警告", "请先选择文件！")
            return

        # 每次只显示第一个文件的数据（可扩展为多标签页）
        file_path = paths[0]
        try:
            data = functions.load_data(file_path)  # 调用新功能函数
        except Exception as e:
            QMessageBox.critical(self, "错误", f"读取数据失败：{str(e)}")
            return

        # 创建并显示新窗口
        self.data_viewer = DataViewer(data, file_path, self)
        self.data_viewer.show()

    def open_data_process(self):
        """打开数据处理窗口"""
        selected_items = self.file_list.selectedItems()
        if selected_items:
            paths = [item.text() for item in selected_items]
        else:
            paths = self.file_paths

        if not paths:
            QMessageBox.warning(self, "警告", "请先选择文件！")
            return

        self.data_process_window = DataProcessWindow(paths, self)
        self.data_process_window.show()



class DataViewer(QMainWindow):
    """数据显示窗口，使用表格展示 DataFrame（作为主窗口）"""
    def __init__(self, data, file_name, parent=None):
        super().__init__(parent)
        self.data = data
        self.file_name = file_name
        self.setWindowTitle(f"数据预览 - {os.path.basename(file_name)}")
        self.setGeometry(200, 200, 800, 600)

        # 创建中央部件和布局（左右布局）
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # 左侧按钮布局（垂直）
        left_layout = QVBoxLayout()
        self.save_csv_btn = QPushButton("保存为CSV")
        self.save_excel_btn = QPushButton("保存为Excel")
        self.plot_btn = QPushButton("绘制图表")
        self.save_plot_btn = QPushButton("保存图表")
        self.save_csv_btn.clicked.connect(self.save_as_csv)
        self.save_excel_btn.clicked.connect(self.save_as_excel)
        self.plot_btn.clicked.connect(self.plot_data)
        self.save_plot_btn.clicked.connect(self.save_plot)
        left_layout.addWidget(self.save_csv_btn)
        left_layout.addWidget(self.save_excel_btn)
        left_layout.addWidget(self.plot_btn)
        left_layout.addWidget(self.save_plot_btn)
        
        # 数据区间设置
        left_layout.addWidget(QLabel("起始行:"))
        self.start_spin = QSpinBox()
        self.start_spin.setRange(0, max(0, data.shape[0] - 1) if data is not None else 0)
        self.start_spin.setValue(0)
        left_layout.addWidget(self.start_spin)
        
        left_layout.addWidget(QLabel("结束行:"))
        self.end_spin = QSpinBox()
        self.end_spin.setRange(0, max(0, data.shape[0] - 1) if data is not None else 0)
        self.end_spin.setValue(data.shape[0] - 1 if data is not None else 0)
        left_layout.addWidget(self.end_spin)
        
        left_layout.addStretch()
        main_layout.addLayout(left_layout)

        # 右侧布局（上下：表格 + 图表）
        right_layout = QVBoxLayout()
        self.table = QTableWidget()
        right_layout.addWidget(self.table)
        
        # 图表画布
        self.figure = Figure(figsize=(8, 4))
        self.canvas = FigureCanvas(self.figure)
        right_layout.addWidget(self.canvas)
        
        main_layout.addLayout(right_layout)

        # 填充数据
        if data is not None and not data.empty:
            self.table.setRowCount(data.shape[0])
            self.table.setColumnCount(data.shape[1])
            self.table.setHorizontalHeaderLabels(data.columns.astype(str))

            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    item = QTableWidgetItem(str(data.iloc[i, j]))
                    self.table.setItem(i, j, item)

            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        else:
            self.table.setRowCount(1)
            self.table.setColumnCount(1)
            self.table.setItem(0, 0, QTableWidgetItem("无数据或数据为空"))

    def save_as_csv(self):
        """保存数据为CSV文件"""
        default_name = os.path.splitext(self.file_name)[0] + "_export.csv"
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存为CSV", default_name, "CSV文件 (*.csv)"
        )
        if file_path:
            try:
                functions.save_data(self.data, file_path)
                QMessageBox.information(self, "成功", "CSV文件保存成功！")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败：{str(e)}")

    def save_as_excel(self):
        """保存数据为Excel文件"""
        default_name = os.path.splitext(self.file_name)[0] + "_export.xlsx"
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存为Excel", default_name, "Excel文件 (*.xlsx)"
        )
        if file_path:
            try:
                functions.save_data(self.data, file_path)
                QMessageBox.information(self, "成功", "Excel文件保存成功！")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败：{str(e)}")

    def plot_data(self):
        """绘制数据图表"""
        if self.data is None or self.data.empty:
            QMessageBox.warning(self, "警告", "没有数据可绘制！")
            return
        
        start_idx = self.start_spin.value()
        end_idx = self.end_spin.value()
        
        if start_idx > end_idx:
            QMessageBox.warning(self, "警告", "起始行不能大于结束行！")
            return
        
        if start_idx >= self.data.shape[0] or end_idx >= self.data.shape[0]:
            QMessageBox.warning(self, "警告", "行索引超出数据范围！")
            return
        
        dialog = QDialog(self)
        dialog.setWindowTitle("选择要绘制的列")
        dialog_layout = QVBoxLayout(dialog)
        
        checkboxes = []
        for col in self.data.columns:
            cb = QCheckBox(col)
            cb.setChecked(True)
            checkboxes.append(cb)
            dialog_layout.addWidget(cb)
        
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(dialog.accept)
        btn_box.rejected.connect(dialog.reject)
        dialog_layout.addWidget(btn_box)
        
        if dialog.exec_() != QDialog.Accepted:
            return
        
        selected_cols = [cb.text() for cb in checkboxes if cb.isChecked()]
        if not selected_cols:
            QMessageBox.warning(self, "警告", "请至少选择一个要绘制的列！")
            return
        
        plot_data = self.data.iloc[start_idx:end_idx + 1]
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        for col in selected_cols:
            ax.plot(plot_data[col].values, label=col)
        ax.set_xlabel('Index')
        ax.set_ylabel('Value')
        ax.set_title(f'{os.path.basename(self.file_name)} - 行 {start_idx} 至 {end_idx}')
        ax.legend()
        ax.grid(True)
        self.figure.tight_layout()
        self.canvas.draw()

    def save_plot(self):
        """保存图表"""
        default_name = os.path.splitext(self.file_name)[0] + "_plot.png"
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存图表", default_name, "PNG图片 (*.png)"
        )
        if file_path:
            self.figure.savefig(file_path)
            QMessageBox.information(self, "成功", "图表保存成功！")



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

        # 左侧控制面板
        left_layout = QVBoxLayout()
        
        # 文件列表信息
        left_layout.addWidget(QLabel(f"已选择 {len(file_paths)} 个文件"))
        
        # 读取所有文件数据
        for file_path in file_paths:
            try:
                df = pd.read_csv(file_path)
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
        
        # 保存结果按钮
        self.save_btn = QPushButton("保存处理后的数据")
        self.save_btn.clicked.connect(self.save_processed_data)
        left_layout.addWidget(self.save_btn)
        
        left_layout.addStretch()
        main_layout.addLayout(left_layout)

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



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())