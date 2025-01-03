@echo off
chcp 65001 > nul
setlocal EnableDelayedExpansion

:: Application path in Documents
set "APP_DIR=%USERPROFILE%\Documents\VoiceScribePro\app"
set "PYTHON_PATH=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python311\python.exe"
set "PIP_PATH=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python311\Scripts\pip3.11.exe"

:: Check Python 3.11
if not exist "%PYTHON_PATH%" (
    echo Python 3.11 is not installed!
    echo Please install Python 3.11 from https://www.python.org/downloads/
    echo Make sure to select "Add Python 3.11 to PATH" during installation
    pause
    exit /b 1
)

:: Check application directory
if not exist "%APP_DIR%" (
    echo Creating application directory...
    mkdir "%APP_DIR%"
)

:: Check requirements.txt
if not exist "%APP_DIR%\requirements.txt" (
    echo Creating requirements.txt...
    (
        echo customtkinter>=5.2.0
        echo sounddevice>=0.4.6
        echo PyAudio>=0.2.13
        echo numpy>=1.24.0
        echo scipy>=1.11.3
        echo faster-whisper>=0.9.0
        echo tqdm>=4.66.1
        echo mutagen>=1.47.0
        echo torch>=2.1.1
        echo torchvision>=0.16.1
        echo torchaudio>=2.1.1
        echo av>=14.0.1
        echo ctranslate2>=4.5.0
        echo huggingface-hub>=0.27.0
    ) > "%APP_DIR%\requirements.txt"
)

:: Check installed packages
echo Checking installed packages...
set "MISSING_PACKAGES="
for %%p in (customtkinter sounddevice numpy scipy torch faster-whisper) do (
    "%PIP_PATH%" show "%%p" >nul 2>&1
    if errorlevel 1 (
        set "MISSING_PACKAGES=!MISSING_PACKAGES! %%p"
    )
)

:: Install missing packages
if not "!MISSING_PACKAGES!"=="" (
    echo Missing packages:!MISSING_PACKAGES!
    echo Installing dependencies...
    "%PIP_PATH%" install -r "%APP_DIR%\requirements.txt"
    if errorlevel 1 (
        echo Error installing dependencies!
        pause
        exit /b 1
    )
) else (
    echo All required packages are already installed
)

:: Copy application files if they don't exist
if not exist "%APP_DIR%\audio_to_text.py" (
    echo Copying application files...
    copy "%~dp0audio_to_text.py" "%APP_DIR%\"
    copy "%~dp0app.ico" "%APP_DIR%\"
)

:: Launch application
echo Launching VoiceScribePro...
cd /d "%APP_DIR%"
"%PYTHON_PATH%" audio_to_text.py
if errorlevel 1 (
    echo An error occurred while running the application!
    pause
)

endlocal 
