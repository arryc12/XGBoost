@echo off
echo 测试Excel文件读取功能...
echo.

REM 检查Python是否可用
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未找到Python，请确保Python已安装并添加到PATH
    pause
    exit /b 1
)

REM 检查pandas是否已安装
python -c "import pandas" >nul 2>&1
if %errorlevel% neq 0 (
    echo 警告: 未找到pandas库，尝试安装...
    pip install pandas openpyxl
    if %errorlevel% neq 0 (
        echo 错误: 安装pandas失败
        pause
        exit /b 1
    )
)

REM 运行测试脚本
python test\test_excel_read_windows.py
if %errorlevel% neq 0 (
    echo 测试失败
) else (
    echo 测试完成
)

echo.
pause