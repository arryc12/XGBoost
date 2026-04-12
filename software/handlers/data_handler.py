"""
数据处理模块，负责概率密度分析、数据保存等基础处理功能。
"""
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def compute_pdf_data(data_dict, selected_files, selected_cols, use_log=True, use_abs=True):
    """
    计算概率密度数据，供绘图使用。

    参数:
        data_dict: 文件路径到DataFrame的映射
        selected_files: 选中的文件路径列表
        selected_cols: 选中的列名列表
        use_log: 是否取对数
        use_abs: 是否取绝对值

    返回:
        包含PDF数据的列表，每个元素为 (file_name, col, processed_data)
    """
    results = []
    for file_path in selected_files:
        if file_path not in data_dict:
            continue
        df = data_dict[file_path]
        file_name = os.path.basename(file_path)
        for col in selected_cols:
            if col in df.columns:
                plot_data = df[col].dropna()
                if use_abs:
                    plot_data = plot_data.abs()
                if use_log:
                    plot_data = np.log(np.abs(plot_data))
                results.append((file_name, col, plot_data))
    return results


def plot_pdf(figure, pdf_data_list):
    """
    在给定Figure上绘制概率密度图。

    参数:
        figure: matplotlib Figure对象
        pdf_data_list: compute_pdf_data返回的PDF数据列表
    """
    figure.clear()
    ax = figure.add_subplot(111)
    colors = plt.cm.tab10.colors
    
    for idx, (file_name, col, plot_data) in enumerate(pdf_data_list):
        color = colors[idx % len(colors)]
        label = f"{file_name} - {col}"
        sns.kdeplot(plot_data, ax=ax, label=label, linewidth=1.5, color=color)
    
    ax.set_xlabel('Value')
    ax.set_ylabel('Probability Density')
    ax.set_title('概率密度分布')
    ax.legend()
    ax.grid(True)
    figure.tight_layout()


def process_and_save_data(data_dict, selected_files, selected_cols, use_log=True, use_abs=True):
    """
    处理数据并返回合并后的DataFrame。

    参数:
        data_dict: 文件路径到DataFrame的映射
        selected_files: 选中的文件路径列表
        selected_cols: 选中的列名列表
        use_log: 是否取对数
        use_abs: 是否取绝对值

    返回:
        处理后的合并DataFrame
    """
    all_data = []
    for file_path in selected_files:
        if file_path not in data_dict:
            continue
        df = data_dict[file_path]
        save_df = df[selected_cols].copy()
        
        if use_abs:
            save_df = save_df.abs()
        if use_log:
            save_df = np.log(np.abs(save_df))
        
        save_df['source_file'] = os.path.basename(file_path)
        all_data.append(save_df)
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()
