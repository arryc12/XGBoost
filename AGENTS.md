# 全局规则

- 始终使用中文回答

---

# 项目概述

本项目是一个数据处理与分析工具，主要包含以下模块：

- **software/** - PyQt5 GUI应用程序，用于处理多格式文件（.tdms, .csv, .xls, .xlsx）
- **data/** - 数据处理脚本，包含PDF分析、EMD分解、样条拟合、数据合并等功能
- **XGBoost_SHAP/** - 机器学习模块，使用XGBoost和SHAP进行分类和特征重要性分析

项目语言：Python 3.x

---

# 构建与测试命令

## 环境配置

### 安装依赖

```bash
# 使用 pip 安装所有依赖
pip install -r requirements.txt

# 如果没有 requirements.txt，手动安装常用依赖
pip install pandas numpy matplotlib seaborn scipy xgboost shap PyQt5 nptdms openpyxl
```

### 虚拟环境（推荐）

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

## 运行应用程序

### GUI 主程序

```bash
# 运行 PyQt5 GUI 应用
cd software
python main.py
```

### 数据处理脚本

```bash
# 运行数据处理脚本（根据需要选择）
cd data
python pdf.py
python emd.py
python merge_data.py
python test.py
```

### 机器学习模块

```bash
cd XGBoost_SHAP
python test.py
```

## 测试命令

### 运行单个测试文件

```bash
# 使用 pytest 运行单个测试文件
pytest data/test.py -v

# 运行特定测试函数
pytest data/test.py::test_function_name -v

# 使用 unittest 运行
python -m unittest data.test

# 直接运行 Python 脚本
python data/test.py
```

### 运行所有测试

```bash
# 使用 pytest
pytest . -v

# 使用 pytest（指定目录）
pytest data/ -v
pytest XGBoost_SHAP/ -v

# 使用 unittest
python -m unittest discover -s . -p "test*.py"
```

### 常用测试选项

```bash
# 显示详细输出
pytest -v

# 显示 print 输出
pytest -s

# 在第一个失败处停止
pytest -x

# 显示本地变量
pytest -l

# 生成 HTML 报告（需要安装 pytest-html）
pytest --html=report.html
```

## 代码质量工具（推荐配置）

### 安装工具

```bash
# 安装代码检查和格式化工具
pip install pylint flake8 black isort mypy

# 安装测试相关工具
pip install pytest pytest-cov pytest-html
```

### 运行代码检查

```bash
# pylint 检查
pylint software/functions.py

# flake8 检查
flake8 software/ --max-line-length=120

# mypy 类型检查
mypy software/functions.py --ignore-missing-imports
```

### 代码格式化

```bash
# Black 格式化（推荐）
black software/
black data/
black XGBoost_SHAP/

# isort 排序导入
isort software/
```

### 一键检查（项目根目录）

```bash
# 运行所有检查
pylint software/ data/ XGBoost_SHAP/
flake8 . --max-line-length=120 --ignore=E501,W503
black --check .
isort --check .
```

---

# 代码风格指南

## 通用规则

### 文件头注释

每个 Python 文件应包含模块文档字符串，说明文件功能：

```python
"""
功能函数模块，负责读取不同格式的文件并返回摘要信息。
支持格式：.tdms, .csv, .xls, .xlsx
"""
```

### 编码声明

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
```

## 导入规范

### 导入顺序（建议）

1. 标准库
2. 第三方库
3. 本地模块

```python
# 标准库
import os
import sys
import re
from collections import Counter

# 第三方库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# 本地模块
from . import functions
from .utils import helper
```

### 导入原则

- 使用 `import xxx` 或 `from xxx import yyy`
- 避免使用 `from xxx import *`
- 保持导入简洁，必要时使用别名

```python
# 推荐
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow

# 避免
from PyQt5.QtWidgets import *
```

## 命名规范

### 变量和函数

- 使用小写字母和下划线（snake_case）
- 变量名应具有描述性

```python
# 推荐
file_path = "data.csv"
max_value = 100
input_data = pd.DataFrame()

def process_files(file_paths):
    """处理多个文件"""
    pass

def load_data(file_path):
    """加载数据文件"""
    pass
```

### 类名

- 使用大驼峰命名（PascalCase）

```python
class MainWindow(QMainWindow):
    """主窗口类"""

class DataViewer(QMainWindow):
    """数据显示窗口类"""
```

### 常量

- 使用大写字母和下划线

```python
MAX_ITERATIONS = 1000
DEFAULT_DPI = 150
```

### 文件名

- 使用小写字母和下划线（snake_case）
- 避免使用特殊字符和中文

```python
# 推荐
data_processor.py
merge_data.py
main_window.py

# 避免
dataProcessor.py
数据处理.py
```

## 格式化规范

### 行长度

- 建议每行不超过 120 个字符
- 使用反斜杠或括号进行换行

```python
# 使用括号换行
result = some_function(
    arg1="value1",
    arg2="value2"
)

# 使用反斜杠（不推荐在表达式中使用）
if condition1 and condition2 \
   and condition3:
    pass
```

### 缩进

- 使用 4 个空格进行缩进
- 不要使用 Tab

### 空格

- 运算符前后使用空格

```python
# 推荐
a = b + c
x = y * z

# 避免
a=b+c
x = y*z
```

- 函数调用时，关键字参数后不加空格

```python
# 推荐
function(arg1, arg2, key=value)

# 避免
function( arg1, arg2, key = value )
```

### 空行

- 模块内函数定义之间使用两个空行
- 类内方法定义之间使用一个空行
- 相关逻辑组之间使用空行分隔

## 类型注解（建议）

为函数添加类型注解，提高代码可读性：

```python
from typing import List, Optional, Dict, Any

def process_files(file_paths: List[str]) -> List[str]:
    """处理多个文件"""
    ...

def load_data(file_path: str) -> Optional[pd.DataFrame]:
    """加载数据文件"""
    ...

def compute_stats(data: np.ndarray) -> Dict[str, float]:
    """计算统计信息"""
    ...
```

## 文档字符串

### 函数文档字符串

使用 Google 或 NumPy 风格的文档字符串：

```python
def process_files(file_paths: List[str]) -> List[str]:
    """
    处理多个文件，返回每个文件的摘要信息列表。

    Args:
        file_paths: 文件路径列表

    Returns:
        每个文件对应的摘要字符串列表
    """
    results = []
    for file_path in file_paths:
        ...
    return results
```

### 类文档字符串

```python
class DataViewer(QMainWindow):
    """
    数据显示窗口，使用表格展示 DataFrame。

    Attributes:
        data: 要展示的 pandas DataFrame
        file_name: 数据文件名
    """
```

## 错误处理

### 异常处理原则

- 使用具体的异常类型
- 提供有意义的错误信息
- 避免裸露的 except 子句

```python
# 推荐
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"文件不存在: {file_path}")
except pd.errors.EmptyDataError:
    print(f"文件为空: {file_path}")
except Exception as e:
    raise RuntimeError(f"读取文件失败: {str(e)}")

# 避免
try:
    df = pd.read_csv(file_path)
except:
    pass
```

### 异常传播

在必要时向上传播异常，提供清晰的错误上下文：

```python
def load_data(file_path):
    """加载数据文件"""
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise RuntimeError(f"加载文件失败: {str(e)}")
    return df
```

## 注释规范

### 行内注释

- 使用中文注释
- 保持简洁，说明"为什么"而非"是什么"

```python
# 计算差值
df['diff'] = df['col5'] - df['col6']

# 使用自然对数，想换 10 底用 np.log10
df['col5'] = np.log(np.abs(df['col5']))
```

### 块注释

- 用于解释复杂的代码逻辑
- 保持与代码一致的缩进

```python
# ========== 用户只改这里 ==========
root_dir = r'data\erect\data0\Bullet flow'
recursive = True
# ==================================
```

## 测试规范

### 测试文件命名

```python
# 推荐
test_processor.py
test_functions.py
test_main.py

# 测试函数命名
def test_process_files():
    ...

def test_load_data():
    ...
```

### 测试结构

```python
import unittest
import pytest

class TestDataProcessor(unittest.TestCase):
    """数据处理测试类"""

    def setUp(self):
        """测试前准备工作"""
        self.test_data = "test.csv"

    def test_process_files(self):
        """测试文件处理功能"""
        result = process_files([self.test_data])
        self.assertIsNotNone(result)

    def tearDown(self):
        """测试后清理工作"""
        pass

if __name__ == '__main__':
    unittest.main()
```

## 最佳实践

### 性能优化

- 避免在循环中重复创建对象
- 使用向量化操作代替循环
- 合理使用缓存

```python
# 推荐：向量化操作
df['result'] = df['col1'] + df['col2']

# 避免：逐行循环
results = []
for i in range(len(df)):
    results.append(df.iloc[i]['col1'] + df.iloc[i]['col2'])
df['result'] = results
```

### 路径处理

使用 `pathlib` 处理跨平台路径：

```python
from pathlib import Path

# 推荐
data_dir = Path('./data')
output_file = data_dir / 'output.csv'

# 避免
data_dir = './data'
output_file = data_dir + '/output.csv'
```

### 图表绘制

设置中文字体支持：

```python
# 解决中文显示
plt.rcParams["font.family"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False
```

---

# 目录结构

```
project/
├── AGENTS.md                 # 本文件
├── software/                 # GUI 应用程序
│   ├── main.py              # 主窗口
│   └── functions.py          # 功能函数
├── data/                    # 数据处理脚本
│   ├── pdf.py               # PDF 分析
│   ├── emd.py               # EMD 分解
│   ├── merge_data.py        # 数据合并
│   ├── test.py              # 测试脚本
│   └── *.csv                # 数据文件
└── XGBoost_SHAP/            # 机器学习模块
    ├── test.py              # 测试脚本
    └── datasets/            # 数据集
```

---

# 常用命令速查

| 操作 | 命令 |
|------|------|
| 运行 GUI | `python software/main.py` |
| 运行测试 | `pytest data/test.py -v` |
| 代码检查 | `pylint software/` |
| 格式化代码 | `black software/` |
| 安装依赖 | `pip install pandas numpy matplotlib` |

---

# 注意事项

1. **始终使用中文回答**（已在全局规则中定义）
2. 修改代码前先阅读理解现有逻辑
3. 新增功能建议添加文档字符串和注释
4. 提交代码前运行基本测试确保功能正常
5. 遇到问题优先查看报错信息，再搜索解决方案
