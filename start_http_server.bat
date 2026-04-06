@echo off
setlocal
set PY_EXE=d:\桌面\18343\2026春\智能图像分析\实验一\.venv\Scripts\python.exe

if not exist "%PY_EXE%" (
  echo [ERROR] Python not found: %PY_EXE%
  pause
  exit /b 1
)

echo Starting local server...
echo Open this URL in browser: http://127.0.0.1:8000/1.html
"%PY_EXE%" -m http.server 8000 --bind 127.0.0.1
