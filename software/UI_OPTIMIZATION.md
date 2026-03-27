# 界面优化修改说明

## 修改概述
根据用户需求，对数据处理界面和数据预览界面进行了以下优化：
1. 数据处理界面左侧按钮布局固定宽度
2. 数据预览界面添加列选择功能，绘制图表时不弹出对话框

## 详细修改内容

### 1. DataProcessWindow类（数据处理界面）

**修改位置**: `software/main.py` 第347-430行

**修改内容**:
- 将左侧控制面板从`QVBoxLayout`改为`QWidget`包装
- 设置左侧控件固定宽度为250像素
- 使用`setContentsMargins`设置边距

**修改前**:
```python
# 左侧控制面板
left_layout = QVBoxLayout()
...
main_layout.addLayout(left_layout)
```

**修改后**:
```python
# 左侧控制面板（固定宽度）
left_widget = QWidget()
left_widget.setFixedWidth(250)
left_layout = QVBoxLayout(left_widget)
left_layout.setContentsMargins(5, 5, 5, 5)
...
main_layout.addWidget(left_widget)
```

### 2. DataViewer类（数据预览界面）

**修改位置**: `software/main.py` 第185-214行和第271-324行

#### 2.1 添加列选择checkbox

在`__init__`方法中添加列选择功能：

```python
# 列选择
if data is not None and not data.empty:
    left_layout.addWidget(QLabel("选择要绘制的列:"))
    self.col_checkboxes = []
    for col in data.columns:
        cb = QCheckBox(col)
        cb.setChecked(True)
        self.col_checkboxes.append(cb)
        left_layout.addWidget(cb)
```

#### 2.2 修改plot_data方法

**修改前**: 弹出对话框选择列
```python
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
```

**修改后**: 使用左侧checkbox
```python
# 获取选中的列
selected_cols = [cb.text() for cb in self.col_checkboxes if cb.isChecked()]
```

## 界面效果

### DataProcessWindow（数据处理界面）
- 左侧控制面板固定宽度250像素
- 窗口拉伸时，左侧控件不会随之拉伸
- 右侧图表区域会随窗口拉伸

### DataViewer（数据预览界面）
- 左侧新增"选择要绘制的列"区域
- 显示所有列的checkbox，默认全选
- 点击"绘制图表"按钮直接绘制，无需弹窗确认

## 使用方法

### 数据预览界面
1. 选择文件后点击"显示数据"
2. 在左侧"选择要绘制的列"区域勾选需要绘制的列
3. 设置起始行和结束行
4. 点击"绘制图表"按钮直接绘制

### 数据处理界面
1. 选择文件后点击"数据处理"
2. 左侧控制面板固定宽度，不会随窗口拉伸
3. 选择文件和列后点击相应功能按钮

## 注意事项
1. 数据预览界面的列选择checkbox在窗口初始化时创建，基于数据的列名
2. 所有列默认选中，用户可根据需要取消勾选
3. 数据处理界面的固定宽度设置为250像素，可根据需要调整