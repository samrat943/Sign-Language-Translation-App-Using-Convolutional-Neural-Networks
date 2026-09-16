@echo off
title Sign Language Translator AI
cd /d "%~dp0"

echo ========================================================
echo        Starting Sign Language Translator App...
echo ========================================================
echo.

if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment 'venv' not found!
    echo Please create the virtual environment first by running:
    echo python -m venv venv
    echo.
    pause
    exit /b 1
)

call .\venv\Scripts\activate.bat

echo Virtual environment activated.
echo Launching Streamlit interface...
echo.

streamlit run app/main.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Application exited with code %ERRORLEVEL%.
    pause
)
