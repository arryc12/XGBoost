"""
主窗口模块，构建GUI界面并响应用户操作。
"""

import os
import sys
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QListWidget,
    QTextEdit,
    QFileDialog,
    QMessageBox,
)
from PyQt5.QtCore import Qt
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 导入处理器模块
from handlers.io_handler import load_data, save_data, get_file_summary

# 导入UI子模块
from ui.data_viewer import DataViewer
from ui.data_process import DataProcessWindow


class MainWindow(QMainWindow):
    """主窗口类"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("气液两相流流型分析")
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
        self.show_btn = QPushButton("数据预览")
        self.data_process_btn = QPushButton("数据处理")
        self.save_csv_btn = QPushButton("保存为CSV")
        self.save_excel_btn = QPushButton("保存为Excel")
        self.show_btn.clicked.connect(self.show_data)
        self.data_process_btn.clicked.connect(self.open_data_process)
        self.save_csv_btn.clicked.connect(self.save_as_csv)
        self.save_excel_btn.clicked.connect(self.save_as_excel)
        left_layout.addWidget(self.select_btn)
        left_layout.addWidget(self.process_btn)
        left_layout.addWidget(self.clear_btn)
        left_layout.addWidget(self.show_btn)
        left_layout.addWidget(self.data_process_btn)
        left_layout.addWidget(self.save_csv_btn)
        left_layout.addWidget(self.save_excel_btn)
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
        filter = "数据文件 (*.tdms *.csv *.xls *.xlsx);;所有文件 (*.*)"
        files, _ = QFileDialog.getOpenFileNames(self, "选择文件", "", filter)
        if files:
            self.file_paths.extend(files)
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
        selected_items = self.file_list.selectedItems()
        if selected_items:
            paths = [item.text() for item in selected_items]
        else:
            paths = self.file_paths

        if not paths:
            QMessageBox.warning(self, "警告", "请先选择文件！")
            return

        try:
            results = [get_file_summary(p) for p in paths]
            self.result_text.clear()
            self.result_text.append("处理结果：\n" + "\n".join(results))
        except Exception as e:
            QMessageBox.critical(self, "错误", f"处理过程中发生异常：{str(e)}")

    def show_data(self):
        """打开新窗口显示选中文件的数据（支持多文件）"""
        selected_items = self.file_list.selectedItems()
        if selected_items:
            paths = [item.text() for item in selected_items]
        else:
            paths = self.file_paths

        if not paths:
            QMessageBox.warning(self, "警告", "请先选择文件！")
            return

        # 加载所有选中文件的数据
        data_dict = {}
        for file_path in paths:
            try:
                data = load_data(file_path)
                data_dict[file_path] = data
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "警告",
                    f"读取文件失败: {os.path.basename(file_path)}\n{str(e)}",
                )

        if not data_dict:
            QMessageBox.critical(self, "错误", "所有文件读取失败！")
            return

        self.data_viewer = DataViewer(data_dict, self)
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

    def save_as_csv(self):
        """保存选中文件为CSV"""
        selected_items = self.file_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "警告", "请先选择文件！")
            return

        file_path = selected_items[0].text()
        try:
            data = load_data(file_path)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"读取数据失败：{str(e)}")
            return

        default_name = os.path.splitext(os.path.basename(file_path))[0] + "_export.csv"
        save_path, _ = QFileDialog.getSaveFileName(
            self, "保存为CSV", default_name, "CSV文件 (*.csv)"
        )
        if save_path:
            try:
                save_data(data, save_path)
                QMessageBox.information(self, "成功", "CSV文件保存成功！")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败：{str(e)}")

    def save_as_excel(self):
        """保存选中文件为Excel"""
        selected_items = self.file_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "警告", "请先选择文件！")
            return

        file_path = selected_items[0].text()
        try:
            data = load_data(file_path)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"读取数据失败：{str(e)}")
            return

        default_name = os.path.splitext(os.path.basename(file_path))[0] + "_export.xlsx"
        save_path, _ = QFileDialog.getSaveFileName(
            self, "保存为Excel", default_name, "Excel文件 (*.xlsx)"
        )
        if save_path:
            try:
                save_data(data, save_path)
                QMessageBox.information(self, "成功", "Excel文件保存成功！")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败：{str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
