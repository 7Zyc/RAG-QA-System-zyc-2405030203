@echo off
REM pack.bat
REM 任务5：使用 PyInstaller 打包 Streamlit 应用为独立 exe

setlocal

echo ============================================
echo   打包 RAG-QA-System
echo ============================================

where pyinstaller >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到 pyinstaller，请先执行：pip install pyinstaller
    pause
    exit /b 1
)

pyinstaller --noconfirm --clean RAG-QA-System.spec

if errorlevel 1 (
    echo [失败] 打包过程中出现错误
    pause
    exit /b 1
)

echo.
echo [完成] 已生成 dist\RAG-QA-System\RAG-QA-System.exe
echo 使用说明：
echo   1) 在目标机器上安装并启动 Ollama：https://ollama.com
echo   2) 下载所需模型：ollama pull deepseek-r1:7b
echo   3) 双击运行 RAG-QA-System.exe 即可
echo.
pause
endlocal
