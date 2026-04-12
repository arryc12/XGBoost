# 全局规则

- 始终使用中文回答
- 本项目是气液两相流流型分析系统，专注于信号处理与机器学习分类

---

# 项目概述

本项目是一个气液两相流流型分析系统，基于压力/压差信号进行流型识别。主要包含以下模块：

- **software/** - PyQt5 GUI应用程序，用于处理多格式文件（.tdms, .csv, .xls, .xlsx），提供数据可视化、特征提取和数据集构建功能
- **data/** - 数据处理脚本，包含PDF分析、EMD分解、样条拟合、小波包分解、时域/频域特征提取等功能
- **XGBoost_SHAP/** - 机器学习模块，使用XGBoost和SHAP进行流型分类和特征重要性分析
- **transformer-xgboost/** - Transformer与XGBoost混合模型，用于时序特征提取与分类

项目语言：Python 3.x

# 目录结构

```
project/
├── AGENTS.md                 # 本文件
├── software/                 # GUI 应用程序
├── data/                    # 数据处理脚本
├── XGBoost_SHAP/            # 机器学习模块
└── transformer-xgboost/    # Transformer+XGBoost混合模型
```

---

# 常用命令速查

| 操作 | 命令 |
|------|------|
| 运行 GUI | `python software/app.py` |
| 运行测试 | `pytest data/test.py -v` |
| 代码检查 | `pylint software/` |
| 格式化代码 | `black software/` |
| 安装依赖 | `pip install pandas numpy matplotlib` |

---

# 重要注意事项

1. **始终使用中文回答**（已在全局规则中定义）
2. **只能修改 software/ 目录下的文件**，不要修改 data/ 和 XGBoost_SHAP/ 目录
3. 修改代码前先阅读理解现有逻辑
4. 新增功能建议添加文档字符串和注释
5. 提交代码前运行基本测试确保功能正常
6. 遇到问题优先查看报错信息，再搜索解决方案
7. 读取 Excel 文件必须使用 `pd.read_excel()`，不能使用 `pd.read_csv()`
8. 特征提取实现参考 `data/feature extraction.py` 和 `data/waveleta_packet.py`
9. 输出特征集格式参考 `XGBoost_SHAP/datasets/annular.csv`
10. Label值应用于所有选中文件（用户只需输入一次）
