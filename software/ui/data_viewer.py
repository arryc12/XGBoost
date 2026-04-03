"""
数据预览子窗口，显示加载的数据内容。
"""

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QTextEdit, QPushButton,
                             QHBoxLayout, QLabel)
from PyQt5.QtCore import Qt


class DataViewer(QDialog):
    """数据预览对话框"""
    
    def __init__(self, data, file_path, parent=None):
        super().__init__(parent)
        self.data = data
        self.file_path = file_path
        self._init_ui()

    def _init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("数据预览 - " + self.file_path)
        self.setGeometry(100, 100, 800, 600)
        
        layout = QVBoxLayout(self)
        
        info_label = QLabel(f"文件：{self.file_path}\n形状：{self.data.shape}\n")
        layout.addWidget(info_label)
        
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setPlainText(str(self.data.head(100)))
        layout.addWidget(self.text_edit)
        
        btn_layout = QHBoxLayout()
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.close)
        btn_layout.addWidget(close_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
