@echo off
REM ===========================================================================
REM  AIC26 Console - chay he thong.
REM
REM  Double-click file nay. Khong can cai gi truoc: neu may chua co Docker
REM  Desktop / WSL2, script se tu cai (co hoi quyen Administrator mot lan).
REM
REM  File .bat co tinh ASCII-only: cmd.exe lam vo dau tieng Viet. Toan bo phan
REM  hien thi co dau nam trong scripts\aic26.ps1, chay duoi PowerShell (UTF-8).
REM ===========================================================================
title AIC26 Console
cd /d "%~dp0"

if not exist "scripts\aic26.ps1" (
  echo.
  echo   [X] Khong tim thay scripts\aic26.ps1
  echo.
  echo   Hay giai nen lai file .zip va GIU NGUYEN toan bo cac file canh nhau,
  echo   roi chay CHAY-APP.bat tu trong thu muc vua giai nen.
  echo.
  pause
  exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\aic26.ps1"
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
  echo.
  echo   Script ket thuc voi ma loi %RC%.
  pause
)
exit /b %RC%
