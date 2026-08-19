@echo off
REM Stop the console. config\ and data\ are left untouched.
setlocal
cd /d "%~dp0"
set AIC26_UID=0
set AIC26_GID=0
docker compose down
echo.
echo Da dung. Cau hinh trong config\ va lich su submit trong data\ van duoc giu.
echo.
pause
