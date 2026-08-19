@echo off
REM Start the AIC26 console on Windows (Docker Desktop).
REM Deliberately ASCII-only: cmd.exe mangles Vietnamese diacritics.
setlocal enabledelayedexpansion
cd /d "%~dp0"

for /f "tokens=2 delims==" %%i in ('findstr /b "AIC26_IMAGE=" bundle.env') do set "IMAGE=%%i"
for /f "tokens=2 delims==" %%i in ('findstr /b "AIC26_IMAGE_ARCHIVE=" bundle.env') do set "ARCHIVE=%%i"

where docker >nul 2>&1
if errorlevel 1 (
  echo.
  echo Chua cai Docker Desktop.
  echo   Tai tai: https://www.docker.com/products/docker-desktop/
  echo   Cai xong, mo Docker Desktop, doi den khi no bao "Running",
  echo   roi chay lai file nay.
  echo.
  pause
  exit /b 1
)

docker info >nul 2>&1
if errorlevel 1 (
  echo.
  echo Docker Desktop chua chay.
  echo   Mo Docker Desktop, doi den khi no bao "Running", roi chay lai file nay.
  echo.
  pause
  exit /b 1
)

docker image inspect "%IMAGE%" >nul 2>&1
if errorlevel 1 (
  if not exist "%ARCHIVE%" (
    echo Khong tim thay %ARCHIVE% trong thu muc nay.
    echo Giai nen lai file zip, giu nguyen tat ca cac file trong cung mot thu muc.
    pause
    exit /b 1
  )
  echo ==^> Nap image lan dau ^(%ARCHIVE%^) - mat 1-2 phut...
  docker load -i "%ARCHIVE%"
  if errorlevel 1 (
    echo Nap image that bai.
    pause
    exit /b 1
  )
)

if not exist config mkdir config
if not exist data mkdir data

REM Docker Desktop maps Windows folders without a host uid, so a non-root
REM container user can fail to write into config/. Root is the norm there.
set AIC26_UID=0
set AIC26_GID=0

echo ==^> Khoi dong...
docker compose up -d
if errorlevel 1 (
  echo Khoi dong that bai. Xem log:  docker compose logs
  pause
  exit /b 1
)

echo ==^> Cho backend san sang...
set READY=0
for /l %%i in (1,1,60) do (
  if !READY!==0 (
    curl -fsS http://localhost:8000/api/health >nul 2>&1
    if !errorlevel!==0 (set READY=1) else (timeout /t 1 /nobreak >nul)
  )
)

if !READY!==0 (
  echo Backend khong phan hoi sau 60s. Xem log:  docker compose logs
  pause
  exit /b 1
)

echo ==^> San sang: http://localhost:8000
if not exist config\.env (
  echo.
  echo Chua co cau hinh - app se tu mo man hinh Settings.
  echo Keo tha file .env cua nhom vao do roi bam "Import ^& ap dung".
  echo Mau cac bien can dien: .env.example trong thu muc nay.
  echo.
)
start "" http://localhost:8000
echo.
echo   Dung app:  double-click stop.bat
echo   Xem log:   docker compose logs -f
echo.
pause
