"""
主窗口模块，构建 GUI 界面并响应用户操作。
"""

import os
import sys
import pandas as pd

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QListWidget, QTextEdit,
                             QFileDialog, QMessageBox)


def process_files(file_paths):
    """处理文件"""
    results = []
    for file_path in file_paths:
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == '.tdms':
                info = read_tdms(file_path)
            elif ext == '.csv':
                info = read_csv(file_path)
            elif ext in ('.xls', '.xlsx'):
                info = read_excel(file_path)
            else:
                info = f"不支持的文件格式：{ext}"
        except Exception as e:
            info = f"读取失败：{str(e)}"
        results.append(f"{os.path.basename(file_path)}: {info}")
    return results


def read_tdms(file_path):
    try:
        from nptdms import TdmsFile
        with TdmsFile.open(file_path) as tdms_file:
            groups = tdms_file.groups()
            return f"TDMS 文件，包含 {len(groups)} 个组"
    except ImportError:
        return "需要安装 npTDMS 库"


def read_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        return f"CSV 文件，形状：" + str(df.shape)
    except Exception as e:
        return f"CSV 读取错误：" + str(e)


def read_excel(file_path):
    try:
        df = pd.read_excel(file_path, sheet_name=0)
        return f"Excel 文件，形状：" + str(df.shape)
    except Exception as e:
        return f"Excel 读取错误：" + str(e)


def load_data(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.tdms':
            return load_tdms_as_dataframe(file_path)
        elif ext == '.csv':
            return pd.read_csv(file_path)
        elif ext in ('.xls', '.xlsx'):
            return pd.read_excel(file_path, sheet_name=0)
        else:
            raise ValueError(f"不支持的文件格式：{ext}")
    except Exception as e:
        raise RuntimeError(f"加载文件失败：" + str(e))


def load_tdms_as_dataframe(file_path):
    from nptdms import TdmsFile
    with TdmsFile.open(file_path) as tdms_file:
        groups = tdms_file.groups()
        if not groups:
            return pd.DataFrame()
        group = groups[0]
        data_dict = {}
        for channel in group.channels():
            data_dict[channel.name] = channel[:]
        return pd.DataFrame(data_dict)


def save_data(data, file_path):
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.csv':
            data.to_csv(file_path, index=False, encoding='utf-8-sig')
            return f"CSV 文件保存成功：" + file_path
        elif ext in ('.xls', '.xlsx'):
            data.to_excel(file_path, index=False, sheet_name='Sheet1')
            return f"Excel 文件保存成功：" + file_path
        else:
            raise ValueError(f"不支持的保存格式：{ext}")
    except Exception as e:
        raise RuntimeError(f"保存文件失败：" + str(e))



class MainWindow(QMainWindow):
    """主窗口类"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("气液两相流流型分析")
        self.setGeometry(100, 100, 800, 600)
        self._init_ui()
        self.file_paths = []

    def _init_ui(self):
        """初始化用户界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        left_layout = QVBoxLayout()
        self.select_btn = QPushButton("选择文件")
        self.process_btn = QPushButton("处理文件")
        self.clear_btn = QPushButton("清空列表")
        self.show_btn = QPushButton("数据预览")
        self.data_process_btn = QPushButton("数据处理")
        self.save_csv_btn = QPushButton("保存为 CSV")
        self.save_excel_btn = QPushButton("保存为 Excel")
        self.ml_btn = QPushButton("机器学习")
        self.ml_btn.clicked.connect(self.open_ml_window)
        left_layout.addWidget(self.ml_btn)

        self._connect_signals()
        
        left_layout.addWidget(self.select_btn)
        left_layout.addWidget(self.process_btn)
        left_layout.addWidget(self.clear_btn)
        left_layout.addWidget(self.show_btn)
        left_layout.addWidget(self.data_process_btn)
        left_layout.addWidget(self.save_csv_btn)
        left_layout.addWidget(self.save_excel_btn)
        left_layout.addStretch()
        main_layout.addLayout(left_layout)

        right_layout = QVBoxLayout()
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.ExtendedSelection)
        right_layout.addWidget(self.file_list)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        right_layout.addWidget(self.result_text)
        main_layout.addLayout(right_layout)

    def _connect_signals(self):
        """连接信号与槽"""
        self.show_btn.clicked.connect(self.show_data)
        self.data_process_btn.clicked.connect(self.open_data_process)
        self.save_csv_btn.clicked.connect(self.save_as_csv)
        self.save_excel_btn.clicked.connect(self.save_as_excel)
        self.select_btn.clicked.connect(self.select_files)
        self.process_btn.clicked.connect(self.process_files)
        self.clear_btn.clicked.connect(self.clear_files)

    def select_files(self):
        """打开文件对话框，选择多个文件"""
        filter = "数据文件 (*.tdms *.csv *.xls *.xlsx);;所有文件 (*.*)"
        files, _ = QFileDialog.getOpenFileNames(self, "选择文件", "", filter)
        if files:
            self.file_paths.extend(files)
            self._update_file_list()

    def _update_file_list(self):
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
        """调用 IO 处理函数处理选中的文件"""
        selected_items = self.file_list.selectedItems()
        paths = [item.text() for item in selected_items] if selected_items else self.file_paths

        if not paths:
            QMessageBox.warning(self, "警告", "请先选择文件！")
            return

        try:
            results = process_files(paths)
            self.result_text.clear()
            self.result_text.append("处理结果：\n" + "\n".join(results))
        except Exception as e:
            QMessageBox.critical(self, "错误", f"处理过程中发生异常：{str(e)}")

    def show_data(self):
        """打开新窗口显示选中文件的数据"""
        from .data_viewer import DataViewer
        selected_items = self.file_list.selectedItems()
        paths = [item.text() for item in selected_items] if selected_items else self.file_paths

        if not paths:
            QMessageBox.warning(self, "警告", "请先选择文件！")
            return

        file_path = paths[0]
        try:
            data = load_data(file_path)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"读取数据失败：{str(e)}")
            return

        self.data_viewer = DataViewer(data, file_path, self)
        self.data_viewer.show()

    def open_data_process(self):
        """打开数据处理窗口"""
        from .data_process import DataProcessWindow
        selected_items = self.file_list.selectedItems()
        paths = [item.text() for item in selected_items] if selected_items else self.file_paths

        if not paths:
            QMessageBox.warning(self, "警告", "请先选择文件！")
            return

        self.data_process_window = DataProcessWindow(paths, self)
        self.data_process_window.show()

    def save_as_csv(self):
        """保存选中文件为 CSV"""
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
            self, "保存为 CSV", default_name, "CSV 文件 (*.csv)"
        )
        if save_path:
            try:
                save_data(data, save_path)
                QMessageBox.information(self, "成功", "CSV 文件保存成功！")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败：{str(e)}")

    def save_as_excel(self):
        """保存选中文件为 Excel"""
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
            self, "保存为 Excel", default_name, "Excel 文件 (*.xlsx)"
        )
        if save_path:
            try:
                save_data(data, save_path)
                QMessageBox.information(self, "成功", "Excel 文件保存成功！")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败：{str(e)}")
    
    def open_ml_window(self):
        """打开机器学习窗口"""
        from .ml_window import MLWindow
        self.ml_window = MLWindow(self)
        self.ml_window.show()

