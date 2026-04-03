"""
数据处理子窗口，提供数据处理功能。
"""

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QTextEdit, QPushButton,
                             QHBoxLayout, QLabel, QProgressBar)


class DataProcessWindow(QDialog):
    """数据处理对话框"""
    
    def __init__(self, file_paths, parent=None):
        super().__init__(parent)
        self.file_paths = file_paths
        self._init_ui()

    def _init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("数据处理")
        self.setGeometry(100, 100, 500, 400)
        
        layout = QVBoxLayout(self)
        
        info_label = QLabel(f"准备处理 {len(self.file_paths)} 个文件")
        layout.addWidget(info_label)
        
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setPlainText("等待处理...\n")
        layout.addWidget(self.status_text)
        
        progress = QProgressBar()
        layout.addWidget(progress)
        
        btn_layout = QHBoxLayout()
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.close)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)
