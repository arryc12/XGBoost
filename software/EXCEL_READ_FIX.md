# Excel文件读取问题修复

## 问题描述
当点击"数据处理"按钮时，读取.xlsx文件失败，错误信息：
```
读取失败: C:/Users/31726/Desktop/graduation_project/data/第三次实验数据/0.1MPa-l500-g200-01.xlsx, 'utf-8' codec can't decode byte 0xa7 in position 14: invalid start byte
```

## 问题原因
在`software/main.py`的`DataProcessWindow`类中，错误地使用`pd.read_csv()`读取所有文件，包括.xlsx格式的Excel文件。Excel文件是二进制格式，不能用文本方式读取。

## 修改内容
**文件**: `software/main.py`

**修改位置**: 第354-361行

**原代码**:
```python
# 读取所有文件数据
for file_path in file_paths:
    try:
        df = pd.read_csv(file_path)
        self.data_dict[file_path] = df
    except Exception as e:
        print(f"读取失败: {file_path}, {e}")
```

**修改后代码**:
```python
# 读取所有文件数据
for file_path in file_paths:
    try:
        df = functions.load_data(file_path)
        self.data_dict[file_path] = df
    except Exception as e:
        print(f"读取失败: {file_path}, {e}")
```

## 修改说明
1. 使用`functions.load_data()`替代`pd.read_csv()`
2. `functions.load_data()`会根据文件扩展名自动选择正确的读取方法：
   - `.csv`文件：使用`pd.read_csv()`
   - `.xlsx`/`.xls`文件：使用`pd.read_excel()`
   - `.tdms`文件：使用专门的TDMS读取函数

## 测试方法
### Windows系统
1. 确保已安装必要的Python库：
   ```bash
   pip install pandas openpyxl PyQt5 numpy matplotlib seaborn
   ```

2. 运行测试脚本：
   ```bash
   cd software
   python test/test_excel_read_windows.py
   ```
   
   或者双击运行：`software/test/run_test.bat`

### 手动测试
1. 启动GUI程序：
   ```bash
   cd software
   python main.py
   ```

2. 选择.xlsx文件，点击"数据处理"按钮，验证是否能正常读取数据

## 相关文件
- `software/main.py` - 主窗口GUI代码（已修改）
- `software/functions.py` - 文件读取功能函数（未修改，已存在正确的读取逻辑）
- `software/test/test_excel_read_windows.py` - 测试脚本（新增）
- `software/test/run_test.bat` - Windows批处理测试脚本（新增）

## 注意事项
1. 确保系统中已安装`openpyxl`库，用于读取.xlsx文件
2. 如果仍然遇到编码问题，检查文件是否损坏或路径是否正确
3. 对于中文路径，Python 3.x通常能正确处理，但建议使用原始字符串（r"path"）