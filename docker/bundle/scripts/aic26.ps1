<#
    AIC26 Console — trình cài đặt và khởi động cho Windows.

    Một file duy nhất lo trọn vòng đời trên máy người dùng:

      1. Kiểm tra máy có đủ điều kiện chạy Docker không (phiên bản Windows,
         ảo hoá, dung lượng đĩa).
      2. Bật WSL2 và cài Docker Desktop nếu chưa có — tự xin quyền Administrator
         đúng lúc cần, không sớm hơn.
      3. Khởi động Docker Desktop và chờ engine sẵn sàng.
      4. Nạp image AIC26 từ file .tar.gz đi kèm (chỉ lần đầu).
      5. `docker compose up -d`, chờ /api/health, rồi mở trình duyệt.

    Chạy lại được nhiều lần: mỗi bước tự bỏ qua nếu đã xong từ trước.

    Đừng gọi trực tiếp — hãy double-click CHAY-APP.bat ở thư mục cha.
#>
[CmdletBinding()]
param(
    # Chế độ nội bộ: tiến trình con chạy với quyền Administrator, chỉ cài đặt
    # rồi thoát. Cửa sổ gốc (quyền thường) chờ xong và tự chạy tiếp phần app,
    # nên bản thân app không bao giờ chạy dưới quyền Administrator.
    [switch]$InstallOnly,

    # Tài khoản Windows thật sự dùng app. Phải truyền tường minh: khi UAC nâng
    # quyền bằng một tài khoản khác thì $env:USERNAME trong tiến trình con là
    # tài khoản đó, không phải người đang ngồi trước máy.
    [string]$TargetUser = $env:USERNAME
)

$ErrorActionPreference = 'Stop'
try {
    [Console]::OutputEncoding = [Text.Encoding]::UTF8
    $OutputEncoding = [Text.Encoding]::UTF8
} catch { }

# --------------------------------------------------------------------------
# Hằng số
# --------------------------------------------------------------------------

$DockerInstallerUrl = 'https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe'
$WslKernelUrl       = 'https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi'
$AppUrl             = 'http://localhost:8000'

# Docker Desktop yêu cầu Windows 10 22H2 (build 19045) trở lên. WSL2 chạy được
# từ 19041, nên dưới 19045 vẫn thử — chỉ cảnh báo, không chặn.
$MinBuildSupported  = 19045
$MinBuildPossible   = 19041

# Lần đầu bật máy ảo WSL của Docker rất lâu trên ổ cứng cơ.
$EngineTimeoutSec   = 300
$HealthTimeoutSec   = 180

$BundleDir = Split-Path -Parent $PSScriptRoot
$StateDir  = Join-Path $BundleDir '.state'

# --------------------------------------------------------------------------
# Hiển thị
# --------------------------------------------------------------------------

function Write-Step { param([string]$Text) Write-Host "`n==> $Text" -ForegroundColor Cyan }
function Write-Ok   { param([string]$Text) Write-Host "    OK  $Text" -ForegroundColor Green }
function Write-Note { param([string]$Text) Write-Host "    $Text" -ForegroundColor DarkGray }
function Write-Warn { param([string]$Text) Write-Host "`n[!] $Text" -ForegroundColor Yellow }
function Write-Fail { param([string]$Text) Write-Host "`n[X] $Text" -ForegroundColor Red }

function Write-Banner {
    Write-Host ''
    Write-Host '  ╔════════════════════════════════════════════════╗' -ForegroundColor Cyan
    Write-Host '  ║          AIC26 CONSOLE — KHỞI ĐỘNG             ║' -ForegroundColor Cyan
    Write-Host '  ╚════════════════════════════════════════════════╝' -ForegroundColor Cyan
}

function Read-YesNo {
    param([string]$Question, [bool]$Default = $true)
    $hint = if ($Default) { '[Y/n]' } else { '[y/N]' }
    while ($true) {
        $answer = (Read-Host "$Question $hint").Trim().ToLowerInvariant()
        if ($answer -eq '')  { return $Default }
        if ($answer -in @('y','yes','c','co')) { return $true }
        if ($answer -in @('n','no','k','khong')) { return $false }
    }
}

function Stop-WithError {
    param([string]$Message, [string[]]$Hints = @())
    Write-Fail $Message
    foreach ($hint in $Hints) { Write-Host "    $hint" -ForegroundColor Yellow }
    Write-Host ''
    if (-not $InstallOnly) { Read-Host 'Nhấn Enter để đóng' | Out-Null }
    exit 1
}

# --------------------------------------------------------------------------
# Tiện ích môi trường
# --------------------------------------------------------------------------

# Trình cài đặt sửa PATH trong registry, nhưng tiến trình đang chạy giữ bản sao
# PATH từ lúc khởi động. Không nạp lại thì `docker` vẫn "không tồn tại" ngay sau
# khi vừa cài xong.
function Update-PathFromRegistry {
    $machine = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $user    = [Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = (@($machine, $user) | Where-Object { $_ }) -join ';'
}

function Get-DockerExe {
    Update-PathFromRegistry
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

function Get-DockerDesktopExe {
    foreach ($candidate in @(
        (Join-Path $env:ProgramFiles 'Docker\Docker\Docker Desktop.exe'),
        (Join-Path ${env:ProgramFiles(x86)} 'Docker\Docker\Docker Desktop.exe')
    )) {
        if ($candidate -and (Test-Path $candidate)) { return $candidate }
    }
    return $null
}

function Test-DockerEngine {
    param([string]$DockerExe)
    if (-not $DockerExe) { return $false }
    & $DockerExe info 2>&1 | Out-Null
    return ($LASTEXITCODE -eq 0)
}

function Test-IsAdmin {
    $identity  = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Get-PowerShellExe {
    # PowerShell 7 đặt tên pwsh.exe; 5.1 mặc định của Windows là powershell.exe.
    $self = (Get-Process -Id $PID).Path
    if ($self) { return $self }
    return (Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe')
}

function Get-FreeSpaceGB {
    # DriveInfo rather than Get-PSDrive: `Get-Item 'C:'` resolves to the
    # provider's *current directory* on that drive, which is not always a drive
    # root and has surprised people before.
    $drive = New-Object System.IO.DriveInfo($env:SystemDrive)
    return [math]::Round($drive.AvailableFreeSpace / 1GB, 1)
}

function Save-File {
    param([string]$Url, [string]$Destination)
    # TLS 1.2 không bật mặc định trên PowerShell 5.1/Windows 10 đời đầu.
    try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch { }
    # BITS/Invoke-WebRequest có thanh tiến trình rất chậm với file lớn; tắt đi
    # thì tải nhanh hơn nhiều lần.
    $previous = $ProgressPreference
    $ProgressPreference = 'SilentlyContinue'
    try {
        Invoke-WebRequest -Uri $Url -OutFile $Destination -UseBasicParsing
    } finally {
        $ProgressPreference = $previous
    }
}

# --------------------------------------------------------------------------
# Giai đoạn cài đặt (chạy với quyền Administrator)
# --------------------------------------------------------------------------

function Test-Prerequisites {
    Write-Step 'Kiểm tra máy'

    $os    = Get-CimInstance Win32_OperatingSystem
    $build = [int](Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion').CurrentBuildNumber

    if ([Environment]::Is64BitOperatingSystem -ne $true) {
        Stop-WithError 'Docker Desktop chỉ chạy trên Windows 64-bit.'
    }

    if ($build -lt $MinBuildPossible) {
        Stop-WithError "Windows quá cũ (build $build). WSL2 cần build $MinBuildPossible trở lên." @(
            'Chạy Windows Update cho tới khi lên Windows 10 22H2 hoặc Windows 11, rồi thử lại.'
        )
    }
    if ($build -lt $MinBuildSupported) {
        Write-Warn "Windows build $build thấp hơn mức Docker Desktop hỗ trợ ($MinBuildSupported). Vẫn thử cài, nhưng nếu lỗi thì hãy chạy Windows Update trước."
    } else {
        Write-Ok "Windows build $build ($($os.Caption))"
    }

    $free = Get-FreeSpaceGB
    if ($free -lt 12) {
        Write-Warn "Ổ $env:SystemDrive chỉ còn $free GB trống. Docker Desktop + image AIC26 cần khoảng 12 GB. Dọn bớt rồi chạy lại nếu cài lỗi."
    } else {
        Write-Ok "Còn $free GB trống trên $env:SystemDrive"
    }

    # Ảo hoá: nếu firmware tắt VT-x/AMD-V thì WSL2 không thể khởi động và lỗi chỉ
    # lộ ra rất muộn, lúc Docker Desktop đã cài xong — nên báo ngay từ đây.
    $system = Get-CimInstance Win32_ComputerSystem
    if (-not $system.HypervisorPresent) {
        $cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
        if ($cpu.VirtualizationFirmwareEnabled -eq $false) {
            Write-Warn @'
Ảo hoá (VT-x / AMD-V) đang TẮT trong BIOS/UEFI.
WSL2 và Docker sẽ không chạy được cho tới khi bật nó.
  - Khởi động lại máy, vào BIOS/UEFI (thường là F2 / DEL / F10 lúc mới bật máy)
  - Bật "Intel VT-x" / "Intel Virtualization Technology" / "AMD-V" / "SVM Mode"
  - Lưu lại, khởi động vào Windows rồi chạy lại file này.
'@
            if (-not (Read-YesNo 'Vẫn muốn thử cài tiếp?' $false)) { exit 1 }
        }
    } else {
        Write-Ok 'Ảo hoá đã bật'
    }
}

function Enable-Wsl2 {
    <# Trả về $true nếu cần khởi động lại máy. #>
    Write-Step 'Cài đặt WSL2'

    $rebootNeeded = $false
    $features = @('Microsoft-Windows-Subsystem-Linux', 'VirtualMachinePlatform')

    foreach ($feature in $features) {
        $state = (Get-WindowsOptionalFeature -Online -FeatureName $feature -ErrorAction SilentlyContinue).State
        if ($state -eq 'Enabled') {
            Write-Ok "$feature đã bật"
            continue
        }
        Write-Note "Đang bật $feature ..."
        $result = Enable-WindowsOptionalFeature -Online -FeatureName $feature -All -NoRestart -ErrorAction Stop
        Write-Ok "$feature đã bật"
        if ($result.RestartNeeded) { $rebootNeeded = $true }
    }

    # Nhân WSL nằm ngoài tính năng Windows. `wsl --update` là đường chính thức và
    # cũng nâng cấp nhân cũ; máy chưa reboot sau khi bật tính năng thì lệnh này
    # sẽ lỗi — lúc đó rơi xuống gói MSI, hoặc để lần chạy sau lo.
    if (-not $rebootNeeded) {
        Write-Note 'Đang cập nhật nhân WSL ...'
        & wsl.exe --update 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Write-Note 'wsl --update không chạy được, chuyển sang gói cập nhật nhân ...'
            $msi = Join-Path $env:TEMP 'wsl_update_x64.msi'
            try {
                Save-File -Url $WslKernelUrl -Destination $msi
                Start-Process msiexec.exe -ArgumentList "/i `"$msi`" /qn /norestart" -Wait -NoNewWindow
                Write-Ok 'Đã cài nhân WSL2'
            } catch {
                Write-Warn "Không cài được nhân WSL2 tự động: $($_.Exception.Message)"
                Write-Note 'Docker Desktop thường tự xử lý phần này khi khởi động lần đầu.'
            }
        } else {
            Write-Ok 'Nhân WSL đã cập nhật'
        }
        & wsl.exe --set-default-version 2 2>&1 | Out-Null
    }

    if ($rebootNeeded) { Write-Ok 'Đã bật tính năng Windows — cần khởi động lại' }
    return $rebootNeeded
}

function Install-DockerDesktop {
    Write-Step 'Cài đặt Docker Desktop'

    if (Get-DockerDesktopExe) {
        Write-Ok 'Docker Desktop đã được cài'
        return
    }

    $installer = Join-Path $env:TEMP 'DockerDesktopInstaller.exe'
    Write-Note 'Đang tải Docker Desktop (~600 MB, tuỳ mạng có thể mất vài phút) ...'
    try {
        Save-File -Url $DockerInstallerUrl -Destination $installer
    } catch {
        # winget có sẵn trên Windows 11 và Windows 10 đã cập nhật; dùng làm
        # đường lui khi tải trực tiếp bị chặn (proxy công ty, DNS trường học).
        Write-Note "Tải trực tiếp thất bại ($($_.Exception.Message)). Thử qua winget ..."
        if (-not (Get-Command winget.exe -ErrorAction SilentlyContinue)) {
            Stop-WithError 'Không tải được Docker Desktop và máy cũng không có winget.' @(
                'Hãy tải thủ công tại https://www.docker.com/products/docker-desktop/',
                'cài xong rồi chạy lại file CHAY-APP.bat này.'
            )
        }
        & winget.exe install --id Docker.DockerDesktop --exact --silent `
            --accept-package-agreements --accept-source-agreements
        if (-not (Get-DockerDesktopExe)) {
            Stop-WithError 'winget chạy xong nhưng không thấy Docker Desktop.' @(
                'Hãy tải thủ công tại https://www.docker.com/products/docker-desktop/'
            )
        }
        Write-Ok 'Đã cài Docker Desktop (winget)'
        return
    }

    Write-Note 'Đang cài (im lặng, không hiện cửa sổ — vui lòng đợi) ...'
    $proc = Start-Process -FilePath $installer -Wait -PassThru -NoNewWindow -ArgumentList @(
        'install', '--quiet', '--accept-license', '--backend=wsl-2', '--always-run-service'
    )
    # 3010 = cài xong, cần khởi động lại. Không phải lỗi.
    if ($proc.ExitCode -ne 0 -and $proc.ExitCode -ne 3010) {
        Stop-WithError "Trình cài Docker Desktop kết thúc với mã $($proc.ExitCode)." @(
            'Thử cài thủ công tại https://www.docker.com/products/docker-desktop/',
            'rồi chạy lại file CHAY-APP.bat này.'
        )
    }
    Remove-Item $installer -Force -ErrorAction SilentlyContinue
    Write-Ok 'Đã cài Docker Desktop'
}

function Grant-DockerAccess {
    # Chỉ thành viên nhóm docker-users mới dùng được Docker Desktop. Trình cài
    # thêm sẵn tài khoản đang cài; nếu UAC được nâng quyền bằng tài khoản khác
    # thì người dùng thật vẫn đứng ngoài nhóm.
    try {
        $members = @(Get-LocalGroupMember -Group 'docker-users' -ErrorAction Stop |
                     ForEach-Object { ($_.Name -split '\\')[-1] })
        if ($members -notcontains $TargetUser) {
            Add-LocalGroupMember -Group 'docker-users' -Member $TargetUser -ErrorAction Stop
            Write-Ok "Đã thêm '$TargetUser' vào nhóm docker-users"
        }
    } catch {
        Write-Note "Bỏ qua bước nhóm docker-users ($($_.Exception.Message))."
    }
}

function Invoke-Installation {
    Write-Banner
    Write-Host '  Chế độ cài đặt (quyền Administrator)' -ForegroundColor DarkGray

    Test-Prerequisites
    $rebootNeeded = Enable-Wsl2
    Install-DockerDesktop
    Grant-DockerAccess

    if ($rebootNeeded) {
        Write-Host ''
        Write-Ok 'Cài đặt xong. Cần khởi động lại Windows để WSL2 có hiệu lực.'
        exit 3010
    }
    Write-Host ''
    Write-Ok 'Cài đặt xong.'
    exit 0
}

# --------------------------------------------------------------------------
# Giai đoạn chạy app (quyền người dùng thường)
# --------------------------------------------------------------------------

function Request-Installation {
    Write-Warn 'Máy này chưa có Docker. Cần cài Docker Desktop + WSL2 (một lần duy nhất).'
    Write-Host '    Windows sẽ hỏi quyền Administrator ở cửa sổ tiếp theo.' -ForegroundColor DarkGray
    Write-Host '    Quá trình tải và cài mất khoảng 5–15 phút tuỳ tốc độ mạng.' -ForegroundColor DarkGray
    Write-Host ''
    if (-not (Read-YesNo 'Tiến hành cài đặt bây giờ?' $true)) {
        Stop-WithError 'Đã huỷ. Không có Docker thì app không chạy được.'
    }

    $arguments = @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass',
        '-File', "`"$PSCommandPath`"",
        '-InstallOnly',
        '-TargetUser', "`"$TargetUser`""
    )

    try {
        $proc = Start-Process -FilePath (Get-PowerShellExe) -Verb RunAs -Wait -PassThru -ArgumentList $arguments
    } catch {
        Stop-WithError 'Không nâng được quyền Administrator (bạn đã bấm "No" ở hộp thoại UAC?).' @(
            'Chuột phải vào CHAY-APP.bat → "Run as administrator", rồi thử lại.'
        )
    }

    if ($proc.ExitCode -eq 3010) {
        # RunOnce để lần đăng nhập sau tự chạy tiếp — người dùng không phải nhớ
        # quay lại thư mục này. Vẫn nói rõ cách làm thủ công phòng khi bị chặn.
        try {
            $command = '"{0}" -NoProfile -ExecutionPolicy Bypass -File "{1}"' -f (Get-PowerShellExe), $PSCommandPath
            Set-ItemProperty -Path 'HKCU:\Software\Microsoft\Windows\CurrentVersion\RunOnce' `
                             -Name 'AIC26Console' -Value $command
        } catch { }

        Write-Host ''
        Write-Warn 'Phải khởi động lại Windows để hoàn tất WSL2.'
        Write-Host '    Sau khi khởi động lại, app sẽ TỰ chạy tiếp.' -ForegroundColor Yellow
        Write-Host '    (Nếu không tự chạy: double-click lại CHAY-APP.bat.)' -ForegroundColor DarkGray
        Write-Host ''
        if (Read-YesNo 'Khởi động lại ngay bây giờ?' $true) {
            Restart-Computer -Force
            exit 0
        }
        Write-Host 'Hãy khởi động lại máy rồi chạy lại CHAY-APP.bat.' -ForegroundColor Yellow
        Read-Host 'Nhấn Enter để đóng' | Out-Null
        exit 0
    }

    if ($proc.ExitCode -ne 0) {
        Stop-WithError "Cài đặt thất bại (mã $($proc.ExitCode)). Xem thông báo ở cửa sổ Administrator vừa rồi."
    }
    Write-Ok 'Đã cài xong Docker Desktop.'
}

function Start-DockerEngine {
    param([string]$DockerExe)

    Write-Step 'Khởi động Docker Desktop'
    $desktop = Get-DockerDesktopExe
    if (-not $desktop) {
        Stop-WithError 'Không tìm thấy Docker Desktop dù đã cài.' @(
            'Mở Start menu, gõ "Docker Desktop", chạy nó, đợi báo "Running", rồi chạy lại file này.'
        )
    }

    if (-not (Get-Process 'Docker Desktop' -ErrorAction SilentlyContinue)) {
        Start-Process -FilePath $desktop | Out-Null
    }

    Write-Note "Chờ engine sẵn sàng (lần đầu có thể tới $([int]($EngineTimeoutSec/60)) phút) ..."
    $deadline = (Get-Date).AddSeconds($EngineTimeoutSec)
    while ((Get-Date) -lt $deadline) {
        if (-not $DockerExe) { $DockerExe = Get-DockerExe }
        if (Test-DockerEngine -DockerExe $DockerExe) {
            Write-Host ''
            Write-Ok 'Docker engine đã sẵn sàng'
            return $DockerExe
        }
        Write-Host '.' -NoNewline -ForegroundColor DarkGray
        Start-Sleep -Seconds 3
    }

    Write-Host ''
    Stop-WithError 'Docker Desktop không khởi động được trong thời gian chờ.' @(
        'Mở Docker Desktop bằng tay và xem nó báo lỗi gì.',
        'Hay gặp nhất: chưa bật ảo hoá trong BIOS, hoặc máy cần khởi động lại sau khi cài.',
        'Khởi động lại máy rồi chạy lại CHAY-APP.bat thường là đủ.'
    )
}

function Import-AppImage {
    param([string]$DockerExe, [string]$Image, [string]$Archive)

    & $DockerExe image inspect $Image 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Ok "Image $Image đã có sẵn"
        return
    }

    Write-Step 'Nạp image AIC26 (chỉ lần đầu)'
    $archivePath = Join-Path $BundleDir $Archive
    if (-not (Test-Path $archivePath)) {
        Stop-WithError "Không tìm thấy $Archive trong thư mục này." @(
            'Hãy giải nén lại file .zip và GIỮ NGUYÊN tất cả các file cạnh nhau,',
            'rồi chạy CHAY-APP.bat từ trong thư mục vừa giải nén.'
        )
    }

    $sizeGb = [math]::Round((Get-Item $archivePath).Length / 1GB, 2)
    Write-Note "$Archive — $sizeGb GB, mất khoảng 1–3 phút ..."
    & $DockerExe load -i $archivePath
    if ($LASTEXITCODE -ne 0) {
        Stop-WithError 'Nạp image thất bại — file .tar.gz có thể bị hỏng lúc tải về.' @(
            'Tải lại file zip và giải nén lại.'
        )
    }
    Write-Ok 'Đã nạp image'
}

function Start-Stack {
    param([string]$DockerExe)

    Write-Step 'Khởi động hệ thống'
    foreach ($dir in @('config', 'data')) {
        $path = Join-Path $BundleDir $dir
        if (-not (Test-Path $path)) { New-Item -ItemType Directory -Path $path | Out-Null }
    }

    # Docker Desktop gắn thư mục Windows qua một lớp dịch không có uid, nên một
    # user không phải root trong container có thể không ghi được vào config/.
    # Trên Windows, root mới là mặc định đúng.
    $env:AIC26_UID = '0'
    $env:AIC26_GID = '0'

    Push-Location $BundleDir
    try {
        & $DockerExe compose up -d
        if ($LASTEXITCODE -ne 0) {
            Stop-WithError 'docker compose up thất bại.' @(
                "Xem log:  docker compose logs   (chạy trong $BundleDir)"
            )
        }
    } finally {
        Pop-Location
    }
    Write-Ok 'Container đang chạy'
}

function Wait-AppReady {
    Write-Step 'Chờ backend trả lời'
    $deadline = (Get-Date).AddSeconds($HealthTimeoutSec)
    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-WebRequest -Uri "$AppUrl/api/health" -UseBasicParsing -TimeoutSec 4
            if ($response.StatusCode -eq 200) {
                Write-Host ''
                Write-Ok "Sẵn sàng: $AppUrl"
                return $true
            }
        } catch { }
        Write-Host '.' -NoNewline -ForegroundColor DarkGray
        Start-Sleep -Seconds 2
    }
    Write-Host ''
    return $false
}

function Show-Ready {
    $envFile = Join-Path $BundleDir 'config\.env'
    Write-Host ''
    Write-Host '  ────────────────────────────────────────────────' -ForegroundColor Green
    Write-Host "   AIC26 Console đang chạy tại  $AppUrl" -ForegroundColor Green
    Write-Host '  ────────────────────────────────────────────────' -ForegroundColor Green

    if (-not (Test-Path $envFile)) {
        Write-Host ''
        Write-Warn 'Chưa có cấu hình — app sẽ tự mở màn hình Cấu hình.'
        Write-Host '    Kéo thả file .env của nhóm vào đó rồi bấm "Import & áp dụng".' -ForegroundColor Yellow
        Write-Host '    (Mẫu các biến cần điền: .env.example trong thư mục này.)' -ForegroundColor DarkGray
    }

    Write-Host ''
    Write-Host '    Dừng app:  double-click DUNG-APP.bat' -ForegroundColor DarkGray
    Write-Host "    Xem log:   docker compose logs -f   (trong $BundleDir)" -ForegroundColor DarkGray
    Write-Host ''
    Start-Process $AppUrl | Out-Null
}

# --------------------------------------------------------------------------
# Điều phối
# --------------------------------------------------------------------------

function Get-BundleSettings {
    $file = Join-Path $BundleDir 'bundle.env'
    if (-not (Test-Path $file)) {
        Stop-WithError 'Thiếu file bundle.env.' @(
            'Hãy giải nén lại file .zip và giữ nguyên toàn bộ các file cạnh nhau.'
        )
    }
    $settings = @{}
    foreach ($line in Get-Content $file) {
        if ($line -match '^\s*([A-Za-z0-9_]+)\s*=\s*(.*?)\s*$') {
            $settings[$matches[1]] = $matches[2]
        }
    }
    foreach ($key in @('AIC26_IMAGE', 'AIC26_IMAGE_ARCHIVE')) {
        if (-not $settings.ContainsKey($key)) {
            Stop-WithError "bundle.env thiếu khoá $key."
        }
    }
    return $settings
}

function Invoke-Main {
    Write-Banner
    $settings = Get-BundleSettings
    Write-Host "  $($settings['AIC26_IMAGE'])" -ForegroundColor DarkGray

    $dockerExe = Get-DockerExe
    if (-not (Test-DockerEngine -DockerExe $dockerExe)) {
        if (-not (Get-DockerDesktopExe)) {
            Request-Installation
            $dockerExe = Get-DockerExe
        }
        $dockerExe = Start-DockerEngine -DockerExe $dockerExe
    } else {
        Write-Step 'Kiểm tra Docker'
        Write-Ok 'Docker engine đang chạy'
    }

    Import-AppImage -DockerExe $dockerExe `
                    -Image $settings['AIC26_IMAGE'] `
                    -Archive $settings['AIC26_IMAGE_ARCHIVE']
    Start-Stack -DockerExe $dockerExe

    if (-not (Wait-AppReady)) {
        Stop-WithError "Backend không phản hồi sau $HealthTimeoutSec giây." @(
            "Xem log:  docker compose logs   (chạy trong $BundleDir)"
        )
    }

    # Đánh dấu lần chạy thành công, để lần sau bỏ qua phần giới thiệu dài dòng.
    if (-not (Test-Path $StateDir)) { New-Item -ItemType Directory -Path $StateDir -Force | Out-Null }
    Set-Content -Path (Join-Path $StateDir 'last-run.txt') -Value (Get-Date -Format 'o')

    Show-Ready
    Read-Host 'Nhấn Enter để đóng cửa sổ này (app vẫn chạy nền)' | Out-Null
}

if ($InstallOnly) {
    try {
        Invoke-Installation
    } catch {
        Write-Fail $_.Exception.Message
        Write-Host ''
        Read-Host 'Nhấn Enter để đóng' | Out-Null
        exit 1
    }
} else {
    try {
        Invoke-Main
    } catch {
        Write-Fail $_.Exception.Message
        Write-Host $_.ScriptStackTrace -ForegroundColor DarkGray
        Write-Host ''
        Read-Host 'Nhấn Enter để đóng' | Out-Null
        exit 1
    }
}
