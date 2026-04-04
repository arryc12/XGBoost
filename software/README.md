# Software 模块说明文档

## 项目结构

```
software/
├── main.py           # 应用程序入口 + MainWindow类
├── data_viewer.py    # 数据预览窗口类
├── data_process.py   # 数据处理窗口类
├── functions.py      # 功能函数模块
└── test/             # 测试目录
```

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
python main.py
```

## 依赖库

```bash
pip install pandas numpy scipy pywt PyQt5 matplotlib seaborn openpyxl nptdms
```

## 更新记录

### Excel读取修复
- 使用`functions.load_data()`自动识别文件格式
- 支持.csv、.xlsx、.xls、.tdms格式

### 界面优化
- 数据处理界面左侧固定宽度250px
- 数据预览界面列选择直接显示，无需弹窗

### 时域特征标注
- 参照feature extraction.py方式
- 均值（红）、标准差（绿）、最大值（蓝）、最小值（紫）

### 构建特征集
- 可选概率密度特征：众数、均值、中位数、方差、标准差、偏度、峰度
- 可选时域特征：均方根值、峰峰值、脉冲因子、裕度因子、波形因子
- 可选频域特征：主频率、总功率、功率比、小波包能量熵
- 输出格式与XGBoost_SHAP/datasets/annular.csv一致
