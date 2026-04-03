# 全局规则

- 始终使用中文回答
- 避免使用 Emoji，除非用户明确要求

---

# 项目概述

数据处理与分析工具，包含：
- **software/** - PyQt5 GUI 应用，处理 .tdms, .csv, .xls, .xlsx 文件
- **data/** - 数据处理脚本（PDF 分析、EMD 分解、样条拟合）
- **XGBoost_SHAP/** - 机器学习模块，使用 XGBoost 和 SHAP

语言：Python 3.x

---

# 运行与测试

## 环境配置

```bash
pip install pandas numpy matplotlib seaborn scipy xgboost shap PyQt5 nptdms openpyxl
```

## 运行程序

```bash
# GUI 应用
python software/app.py

# 数据处理脚本
python data/pdf.py
python data/emd.py
python data/merge_data.py

# 机器学习模块
cd XGBoost_SHAP && python test.py
```

## 测试命令

```bash
# 运行单个测试文件（推荐）
pytest data/test.py -v

# 运行特定测试函数
pytest data/test.py::test_function_name -v

# 运行所有测试
pytest data/ -v
pytest XGBoost_SHAP/ -v

# 直接运行测试脚本
python data/test.py
python -m unittest discover -s . -p "test*.py"
```

常见测试选项：`pytest -v -s -x -l`（详细/打印输出/失败停止/显示变量）

---

# 代码风格

## 文件结构

每个 Python 文件应包含：
```python
"""模块功能描述文档字符串。"""
import os
import pandas as pd
from . import local_module

def process_data(data: pd.DataFrame) -> pd.DataFrame:
    """处理数据。
    
    Args:
        data: 输入数据
    
    Returns:
        处理后的数据
    """
    pass
```

## 命名规范

- **变量/函数**: snake_case（如 `file_path`, `max_value`）
- **类名**: PascalCase（如 `MainWindow`, `DataViewer`）
- **常量**: 全大写下划线（如 `MAX_ITERATIONS`, `DEFAULT_DPI`）

## 格式化

- 每行不超过 120 字符
- 4 空格缩进，不使用 Tab
- 运算符前后加空格
- 函数/类之间 2 个空行，方法之间 1 个空行

```python
# 推荐
result = some_function(arg1="value1", arg2="value2")

# 使用 Path 处理路径
from pathlib import Path
output_file = Path('./data') / 'output.csv'
```

## 错误处理

使用具体异常类型，提供清晰错误信息：

```python
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"文件不存在：{file_path}")
except Exception as e:
    raise RuntimeError(f"读取失败：{e}")
```

## 注释

- 使用中文
- 说明"为什么"而非"是什么"
- 块注释解释复杂逻辑

```python
# 使用自然对数，10 底改用 np.log10
df['col'] = np.log(np.abs(df['col']))
```

---

# 质量工具

```bash
# 安装
pip install pylint flake8 black isort mypy pytest pytest-cov

# 代码检查
pylint software/ data/
flake8 . --max-line-length=120

# 格式化
black .
isort .
```

---

# 目录结构

```
project/
├── AGENTS.md
├── software/          # PyQt5 GUI 应用
│   ├── __init__.py
│   ├── app.py             # 应用入口
│   ├── io_handler.py      # 数据 IO 处理
│   ├── ml_handler.py      # 机器学习处理
│   ├── ui/                # UI 组件
│   │   ├── __init__.py
│   │   ├── main_window.py # 主窗口
│   │   ├── data_viewer.py # 数据预览窗口
│   │   ├── data_process.py# 数据处理窗口 (备用)
│   │   └── ml_window.py   # 机器学习窗口
│   ├── resource/          # 资源文件
│   ├── config/            # 配置文件
│   └── tests/             # 单元测试
├── data/              # 数据处理
│   ├── pdf.py
│   ├── emd.py
│   ├── merge_data.py
│   ├── test.py
│   └── *.csv
└── XGBoost_SHAP/      # 机器学习模块
    └── test.py
```

---

# 注意事项

1. 修改前先理解现有逻辑
2. 新增功能添加文档字符串和注释
3. 提交前运行测试验证功能
4. 优先查看报错信息定位问题
