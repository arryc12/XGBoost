#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Excel文件读取功能（Windows版本）
"""
import os
import sys
import pandas as pd

# 添加当前目录到路径，以便导入functions模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import functions
    print("成功导入functions模块")
except ImportError as e:
    print(f"导入functions模块失败: {e}")
    sys.exit(1)

def test_excel_read():
    """测试Excel文件读取功能"""
    # 测试文件路径 - 使用原始字符串处理Windows路径
    test_file = r"C:\Users\31726\Desktop\graduation_project\data\第三次实验数据\0.1MPa-l500-g200-01.xlsx"
    
    # 检查文件是否存在
    if not os.path.exists(test_file):
        print(f"测试文件不存在: {test_file}")
        print("请确保文件路径正确，或创建一个测试Excel文件")
        return False
    
    print(f"测试文件: {test_file}")
    
    try:
        # 使用functions.load_data读取Excel文件
        df = functions.load_data(test_file)
        print(f"成功读取Excel文件，数据形状: {df.shape}")
        print(f"列名: {list(df.columns)}")
        print(f"前5行数据:")
        print(df.head())
        return True
    except Exception as e:
        print(f"读取Excel文件失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_csv_read():
    """测试CSV文件读取功能"""
    # 测试文件路径
    test_file = r"C:\Users\31726\Desktop\graduation_project\data\第三次实验数据\0.1MPa-l500-g200-01.csv"
    
    # 检查文件是否存在
    if not os.path.exists(test_file):
        print(f"测试文件不存在: {test_file}")
        print("请确保文件路径正确，或创建一个测试CSV文件")
        return False
    
    print(f"测试文件: {test_file}")
    
    try:
        # 使用functions.load_data读取CSV文件
        df = functions.load_data(test_file)
        print(f"成功读取CSV文件，数据形状: {df.shape}")
        print(f"列名: {list(df.columns)}")
        print(f"前5行数据:")
        print(df.head())
        return True
    except Exception as e:
        print(f"读取CSV文件失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始测试文件读取功能...")
    print("=" * 50)
    
    # 测试Excel文件读取
    print("测试1: Excel文件读取")
    excel_success = test_excel_read()
    print()
    
    # 测试CSV文件读取
    print("测试2: CSV文件读取")
    csv_success = test_csv_read()
    print()
    
    # 总结测试结果
    print("=" * 50)
    print("测试结果总结:")
    print(f"Excel文件读取: {'成功' if excel_success else '失败'}")
    print(f"CSV文件读取: {'成功' if csv_success else '失败'}")
    
    if excel_success and csv_success:
        print("所有测试通过！")
        sys.exit(0)
    else:
        print("部分测试失败，请检查错误信息")
        sys.exit(1)