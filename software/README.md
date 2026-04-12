# 气液两相流流型分析系统 - Software 模块

## 项目结构

```
software/
├── __init__.py           # 模块入口
├── app.py                # 应用入口
├── config/               # 配置文件目录
├── resource/             # 资源文件目录
├── handlers/             # 处理器模块（业务逻辑）
│   ├── __init__.py
│   ├── io_handler.py     # 数据 IO（读取/保存文件）
│   ├── data_handler.py   # 数据处理（概率密度分析/保存）
│   ├── feature_handler.py # 特征提取（时域/频域/数据集构建）
│   └── ml_handler.py    # 机器学习处理
├── ui/                   # UI 组件（界面逻辑）
│   ├── __init__.py
│   ├── main_window.py    # 主窗口 MainWindow
│   ├── data_viewer.py   # 数据查看器 DataViewer
│   └── data_process.py  # 数据处理窗口 DataProcessWindow
├── test/                 # 测试文件
└── README.md             # 项目文档
```

## 模块说明

### handlers/ - 处理器模块

负责所有业务逻辑，与界面解耦。

| 模块 | 功能 |
|------|------|
| `io_handler.py` | 数据读写，支持 .tdms, .csv, .xls, .xlsx 格式 |
| `data_handler.py` | 概率密度分析、数据保存 |
| `feature_handler.py` | 时域/频域特征提取、特征集构建 |
| `ml_handler.py` | 机器学习相关处理 |

### ui/ - UI 组件

负责界面展示和用户交互。

| 组件 | 功能 |
|------|------|
| `main_window.py` | 主窗口，文件选择、操作按钮 |
| `data_viewer.py` | 数据预览，表格展示、图表绘制 |
| `data_process.py` | 数据处理，特征提取、特征集构建 |

## 功能说明

### 主界面 (MainWindow)
- 选择文件、处理文件、清空列表
- 数据预览、数据处理
- 保存为CSV/Excel

### 数据预览 (DataViewer)
- 表格展示数据
- 绘制图表（支持列选择、行范围）
- 保存图表

### 数据处理 (DataProcessWindow)
- 概率密度图绘制
- 时域特征提取（均值、标准差、峰度、偏度等）
- 频域特征提取（主频率、功率谱、小波包能量熵）
- 构建特征集（可选特征、切片长度、label值）
- 保存处理结果

## 运行方式

```bash
cd software
python app.py
```

## 依赖库

```bash
pip install pandas numpy scipy pywt PyQt5 matplotlib seaborn openpyxl nptdms
```

## 更新记录

### 架构重构 (2026)
- 采用模块化架构，将业务逻辑与界面分离
- handlers/ 目录：包含所有数据处理逻辑
- ui/ 目录：包含所有界面组件
- 统一的 app.py 入口

### 特征提取功能
- 可选概率密度特征：众数、均值、中位数、方差、标准差、偏度、峰度
- 可选时域特征：均方根值、峰峰值、脉冲因子、裕度因子、波形因子
- 可选频域特征：主频率、总功率、功率比、小波包能量熵
- 输出格式与 XGBoost_SHAP/datasets/annular.csv 一致