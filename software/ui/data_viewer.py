"""
数据预览窗口模块，用于展示和绘制数据图表。
"""
import os
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QSpinBox, QCheckBox,
                             QTableWidget, QTableWidgetItem, QHeaderView,
                             QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class DataViewer(QMainWindow):
    """数据显示窗口，使用表格展示 DataFrame"""
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
        self.plot_btn = QPushButton("绘制图表")
        self.save_plot_btn = QPushButton("保存图表")
        self.plot_btn.clicked.connect(self.plot_data)
        self.save_plot_btn.clicked.connect(self.save_plot)
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
        
        # 列选择
        if data is not None and not data.empty:
            left_layout.addWidget(QLabel("选择要绘制的列:"))
            self.col_checkboxes = []
            for col in data.columns:
                cb = QCheckBox(col)
                cb.setChecked(True)
                self.col_checkboxes.append(cb)
                left_layout.addWidget(cb)
        
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
        
        # 获取选中的列
        selected_cols = [cb.text() for cb in self.col_checkboxes if cb.isChecked()]
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
