@echo off
setlocal
title genimg - local image generation
cd /d "%~dp0"

rem ===========================================================================
rem  Double-click this file to start the local image-generation web UI.
rem  Backend + page start together, and the default model gear is preloaded in
rem  the background, so the very first click on "generate" is already fast.
rem
rem  Optional arguments:
rem    web.bat -Port 8080        use another port
rem    web.bat -NoPreload        skip preloading the default gear
rem    web.bat -NoOpen           do not auto-open the browser
rem    web.bat -Help             show server options
rem
rem  Keep this window open while using the UI; closing it stops the server.
rem  NOTE: this file is ASCII-only on purpose - cmd.exe mis-parses UTF-8 batch
rem        files (verified), so all Chinese output comes from Python/PowerShell.
rem ===========================================================================

if "%~1"=="" (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0web.ps1" -Open
) else (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0web.ps1" %*
)

if errorlevel 1 (
    echo.
    echo [ERROR] Failed to start. See the message above.
    echo         If the conda env is missing, run setup.ps1 first.
    echo.
    pause
)
endlocal
