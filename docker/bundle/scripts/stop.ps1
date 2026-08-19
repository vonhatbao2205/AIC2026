<#
    Dừng AIC26 Console.

    Chỉ gỡ container. Thư mục config/ (file .env) và data/ (lịch sử submit) nằm
    ngoài container nên không bị đụng tới — chạy lại CHAY-APP.bat là mọi thứ
    trở lại đúng như cũ.
#>
$ErrorActionPreference = 'Stop'
try {
    [Console]::OutputEncoding = [Text.Encoding]::UTF8
    $OutputEncoding = [Text.Encoding]::UTF8
} catch { }

$BundleDir = Split-Path -Parent $PSScriptRoot

function Get-DockerExe {
    $machine = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $user    = [Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = (@($machine, $user) | Where-Object { $_ }) -join ';'

    $cmd = Get-Command docker.exe -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    foreach ($candidate in @(
        (Join-Path $env:ProgramFiles 'Docker\Docker\resources\bin\docker.exe'),
        (Join-Path ${env:ProgramFiles(x86)} 'Docker\Docker\resources\bin\docker.exe')
    )) {
        if ($candidate -and (Test-Path $candidate)) { return $candidate }
    }
    return $null
}

$docker = Get-DockerExe
if (-not $docker) {
    Write-Host "`nKhông tìm thấy Docker — có lẽ app cũng chưa chạy." -ForegroundColor Yellow
    return
}

# Compose thay biến này vào khoá `user:`; thiếu nó thì lệnh down sẽ cảnh báo.
$env:AIC26_UID = '0'
$env:AIC26_GID = '0'

Push-Location $BundleDir
try {
    & $docker compose down
} finally {
    Pop-Location
}

Write-Host ''
Write-Host 'Đã dừng. Cấu hình trong config\ và lịch sử submit trong data\ vẫn được giữ.' -ForegroundColor Green
Write-Host ''
