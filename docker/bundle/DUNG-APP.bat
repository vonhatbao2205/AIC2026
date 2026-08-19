@echo off
REM Dung AIC26 Console. Cau hinh trong config\ va lich su submit trong data\
REM van duoc giu nguyen.
title AIC26 Console - Dung
cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\stop.ps1"
pause
