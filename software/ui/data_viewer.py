"""
数据预览窗口模块，用于展示和绘制数据图表。
支持同时显示多个文件，通过下拉框切换。
"""

import os
from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QSpinBox,
    QCheckBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QFileDialog,
    QMessageBox,
    QComboBox,
)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class DataViewer(QMainWindow):
    """数据显示窗口，支持多文件切换显示"""

    def __init__(self, data_dict, parent=None):
        super().__init__(parent)
        self.data_dict = data_dict
        self.current_file = list(data_dict.keys())[0] if data_dict else None
        self.current_data = (
            data_dict.get(self.current_file) if self.current_file else None
        )

        file_count = len(data_dict)
        if file_count == 0:
            self.setWindowTitle("数据预览 - 无文件")
        elif file_count == 1:
            self.setWindowTitle(f"数据预览 - {os.path.basename(self.current_file)}")
        else:
            self.setWindowTitle(f"数据预览 - 共 {file_count} 个文件")
        self.setGeometry(200, 200, 800, 600)

        # 创建中央部件和布局（左右布局）
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # 左侧按钮布局（垂直）
        left_layout = QVBoxLayout()

        # 文件选择下拉框（多文件时显示）
        if file_count > 1:
            left_layout.addWidget(QLabel("选择文件:"))
            self.file_combo = QComboBox()
            for file_path in data_dict.keys():
                self.file_combo.addItem(os.path.basename(file_path), file_path)
            self.file_combo.currentIndexChanged.connect(self.on_file_changed)
            left_layout.addWidget(self.file_combo)
            left_layout.addWidget(QLabel("---"))

        self.plot_btn = QPushButton("绘制图表")
        self.save_plot_btn = QPushButton("保存图表")
        self.save_slice_btn = QPushButton("保存切片数据")
        self.plot_btn.clicked.connect(self.plot_data)
        self.save_plot_btn.clicked.connect(self.save_plot)
        self.save_slice_btn.clicked.connect(self.save_slice_data)
        left_layout.addWidget(self.plot_btn)
        left_layout.addWidget(self.save_plot_btn)
        left_layout.addWidget(self.save_slice_btn)

        # 数据区间设置
        left_layout.addWidget(QLabel("起始行:"))
        self.start_spin = QSpinBox()
        self.start_spin.setRange(
            0,
            max(0, self.current_data.shape[0] - 1)
            if self.current_data is not None
            else 0,
        )
        self.start_spin.setValue(0)
        left_layout.addWidget(self.start_spin)

        left_layout.addWidget(QLabel("结束行:"))
        self.end_spin = QSpinBox()
        max_val = self.current_data.shape[0] - 1 if self.current_data is not None else 0
        self.end_spin.setRange(0, max(0, max_val))
        self.end_spin.setValue(
            self.current_data.shape[0] - 1 if self.current_data is not None else 0
        )
        left_layout.addWidget(self.end_spin)

        # 列选择
        if self.current_data is not None and not self.current_data.empty:
            left_layout.addWidget(QLabel("选择要绘制的列:"))
            self.col_checkboxes = []
            for col in self.current_data.columns:
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
        self.populate_table()

    def on_file_changed(self, index):
        """切换文件时更新数据"""
        if index < 0:
            return
        file_path = self.file_combo.itemData(index)
        self.current_file = file_path
        self.current_data = self.data_dict.get(file_path)

        # 更新窗口标题
        self.setWindowTitle(f"数据预览 - {os.path.basename(file_path)}")

        # 更新行号范围
        if self.current_data is not None:
            max_row = self.current_data.shape[0] - 1
            self.start_spin.setRange(0, max(0, max_row))
            self.start_spin.setValue(0)
            self.end_spin.setRange(0, max(0, max_row))
            self.end_spin.setValue(max_row)

            # 更新列选择
            self.update_column_checkboxes()

            # 填充表格
            self.populate_table()
        else:
            self.table.setRowCount(1)
            self.table.setColumnCount(1)
            self.table.setItem(0, 0, QTableWidgetItem("无数据或数据为空"))

    def update_column_checkboxes(self):
        """更新列选择复选框"""
        # 清除现有的复选框
        if hasattr(self, "col_checkboxes"):
            for cb in self.col_checkboxes:
                cb.deleteLater()

        self.col_checkboxes = []

        # 重新创建复选框
        if self.current_data is not None and not self.current_data.empty:
            layout = self.start_spin.parentWidget().layout()
            # 找到"选择要绘制的列:"标签的位置
            for i in range(layout.count()):
                widget = layout.itemAt(i).widget()
                if isinstance(widget, QLabel) and widget.text() == "选择要绘制的列:":
                    # 从该位置之后清除旧复选框
                    while layout.count() > i + 1:
                        item = layout.takeAt(i + 1)
                        if item.widget():
                            item.widget().deleteLater()
                        else:
                            break
                    # 添加新复选框
                    for col in self.current_data.columns:
                        cb = QCheckBox(col)
                        cb.setChecked(True)
                        self.col_checkboxes.append(cb)
                        layout.addWidget(cb)
                    break

    def populate_table(self):
        """填充表格数据"""
        if self.current_data is not None and not self.current_data.empty:
            self.table.setRowCount(self.current_data.shape[0])
            self.table.setColumnCount(self.current_data.shape[1])
            self.table.setHorizontalHeaderLabels(self.current_data.columns.astype(str))

            for i in range(self.current_data.shape[0]):
                for j in range(self.current_data.shape[1]):
                    item = QTableWidgetItem(str(self.current_data.iloc[i, j]))
                    self.table.setItem(i, j, item)

            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        else:
            self.table.setRowCount(1)
            self.table.setColumnCount(1)
            self.table.setItem(0, 0, QTableWidgetItem("无数据或数据为空"))

    def plot_data(self):
        """绘制数据图表"""
        if self.current_data is None or self.current_data.empty:
            QMessageBox.warning(self, "警告", "没有数据可绘制！")
            return

        start_idx = self.start_spin.value()
        end_idx = self.end_spin.value()

        if start_idx > end_idx:
            QMessageBox.warning(self, "警告", "起始行不能大于结束行！")
            return

        if (
            start_idx >= self.current_data.shape[0]
            or end_idx >= self.current_data.shape[0]
        ):
            QMessageBox.warning(self, "警告", "行索引超出数据范围！")
            return

        # 获取选中的列
        selected_cols = [cb.text() for cb in self.col_checkboxes if cb.isChecked()]
        if not selected_cols:
            QMessageBox.warning(self, "警告", "请至少选择一个要绘制的列！")
            return

        plot_data = self.current_data.iloc[start_idx : end_idx + 1]

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        for col in selected_cols:
            ax.plot(plot_data[col].values, label=col)
        ax.set_xlabel("Index")
        ax.set_ylabel("Value")
        ax.set_title(
            f"{os.path.basename(self.current_file)} - 行 {start_idx} 至 {end_idx}"
        )
        ax.legend()
        ax.grid(True)
        self.figure.tight_layout()
        self.canvas.draw()

    def save_plot(self):
        """保存图表"""
        default_name = os.path.splitext(self.current_file)[0] + "_plot.png"
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存图表", default_name, "PNG图片 (*.png)"
        )
        if file_path:
            self.figure.savefig(file_path)
            QMessageBox.information(self, "成功", "图表保存成功！")

    def save_slice_data(self):
        """保存切片数据"""
        if self.current_data is None or self.current_data.empty:
            QMessageBox.warning(self, "警告", "没有数据可保存！")
            return

        start_idx = self.start_spin.value()
        end_idx = self.end_spin.value()

        if start_idx > end_idx:
            QMessageBox.warning(self, "警告", "起始行不能大于结束行！")
            return

        if (
            start_idx >= self.current_data.shape[0]
            or end_idx >= self.current_data.shape[0]
        ):
            QMessageBox.warning(self, "警告", "行索引超出数据范围！")
            return

        selected_cols = [cb.text() for cb in self.col_checkboxes if cb.isChecked()]
        if not selected_cols:
            QMessageBox.warning(self, "警告", "请至少选择一个要保存的列！")
            return

        slice_data = self.current_data.iloc[start_idx : end_idx + 1][selected_cols]

        default_name = (
            os.path.splitext(self.current_file)[0] + f"_slice_{start_idx}_{end_idx}.csv"
        )
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存切片数据", default_name, "CSV文件 (*.csv);;Excel文件 (*.xlsx)"
        )
        if file_path:
            try:
                ext = os.path.splitext(file_path)[1].lower()
                if ext == ".csv":
                    slice_data.to_csv(file_path, index=False, encoding="utf-8-sig")
                elif ext in (".xls", ".xlsx"):
                    slice_data.to_excel(file_path, index=False, sheet_name="Sheet1")
                QMessageBox.information(self, "成功", "切片数据保存成功！")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")
