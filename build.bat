@echo off
REM Build script for Krira Augment Rust library

echo ============================================
echo Building Krira Augment (Rust -> Python)
echo ============================================
echo.

REM Initialize Visual Studio environment
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -no_logo

REM Ensure Cargo is in PATH
set PATH=%USERPROFILE%\.cargo\bin;%PATH%

REM Check Rust version
echo Rust version:
rustc --version
echo.

REM Build the library
echo Building in release mode...
cargo build --release

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================
    echo Build successful!
    echo ============================================
    echo.
    echo To install in Python:
    echo   pip install maturin
    echo   maturin develop --release
) else (
    echo.
    echo ============================================
    echo Build failed!
    echo ============================================
    exit /b 1
)
