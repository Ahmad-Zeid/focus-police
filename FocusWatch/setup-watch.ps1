# Windows version of setup-watch.sh: one-time setup of the Galaxy Watch, then a 15-second live test.
#   powershell -ExecutionPolicy Bypass -File setup-watch.ps1              # real watch over Wi-Fi (asks for the pairing code)
#   powershell -ExecutionPolicy Bypass -File setup-watch.ps1 -Uninstall   # remove the app from the watch
param([switch]$Uninstall)
$ErrorActionPreference = "Continue"

$Here     = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root     = Split-Path -Parent $Here
$Pkg      = "edu.aub.focuspolice.watch"
$Apk      = Join-Path $Here "FocusPolice-watch.apk"
$Receiver = Join-Path $Root "WristStream\receiver.py"
$Port     = 8000
$Log      = Join-Path $env:TEMP "focus-receiver.log"

function Say($m) { Write-Host "`n$m" -ForegroundColor Cyan }
function Die($m) { Write-Host "`nPROBLEM: $m" -ForegroundColor Red; exit 1 }

# adb: on PATH, or where winget / Android Studio put it
$Adb = (Get-Command adb -ErrorAction SilentlyContinue).Source
if (-not $Adb) {
    $Adb = @(
        (Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Recurse -Filter adb.exe -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName),
        "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe"
    ) | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
}
if (-not $Adb) { Die "adb not found. Install it with: winget install Google.PlatformTools" }

# Python: the project's virtual environment first
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { $Py = (Get-Command python -ErrorAction SilentlyContinue).Source }
if (-not $Py) { Die "Python not found. Create the venv first (see README)." }
if (-not (Test-Path $Apk)) { Die "App file missing: $Apk" }

# ------------------------------------------------------------ connect
$Dev = & $Adb devices | Select-Object -Skip 1 | Where-Object { $_ -match "^\S+\s+device$" -and $_ -notmatch "^emulator-" } |
       ForEach-Object { ($_ -split "\s+")[0] } | Select-Object -First 1
if (-not $Dev) {
    Say "Watch: Settings > Developer options > Wireless debugging > 'Pair new device'."
    $Pair = Read-Host "Pairing IP:port shown on the watch (e.g. 192.168.1.50:41234)"
    $Code = Read-Host "Pairing code"
    & $Adb pair $Pair $Code
    if ($LASTEXITCODE -ne 0) { Die "Pairing failed. Same Wi-Fi as the laptop? Code still on screen?" }
    Say "Now the IP address & port on the MAIN Wireless debugging screen (NOT the pairing one)."
    $Conn = Read-Host "IP:port"
    $out = & $Adb connect $Conn
    Write-Host $out
    if ($out -notmatch "connected") { Die "Could not connect to $Conn" }
    $Dev = $Conn
}

# The laptop's Wi-Fi address (the watch posts to it)
$Ip = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
      Where-Object { $_.InterfaceAlias -match "Wi-?Fi|Wireless|WLAN" -and $_.IPAddress -notlike "169.254.*" } |
      Select-Object -First 1 -ExpandProperty IPAddress
if (-not $Ip) { Die "This laptop has no Wi-Fi IP address. Is Wi-Fi connected?" }
$Server = "http://${Ip}:$Port"

function A { & $Adb -s $Dev @args }
$model = (A shell getprop ro.product.model) -join ""; $rel = (A shell getprop ro.build.version.release) -join ""
Say "Device: $Dev  ($model, Android $rel)"

if ($Uninstall) { A uninstall $Pkg; Say "Removed. The watch is as it was."; exit 0 }

# ------------------------------------------------------------ install + keep alive
Say "Installing the app..."
$res = (A install -r -g $Apk) -join "`n"
if ($res -notmatch "Success") { Write-Host $res; Die "Install failed." }
A shell pm grant $Pkg android.permission.POST_NOTIFICATIONS 2>$null | Out-Null
A shell dumpsys deviceidle whitelist +$Pkg | Out-Null
A shell cmd appops set $Pkg RUN_IN_BACKGROUND allow 2>$null | Out-Null
A shell cmd appops set $Pkg RUN_ANY_IN_BACKGROUND allow 2>$null | Out-Null
A shell am set-standby-bucket $Pkg active 2>$null | Out-Null
if (((A shell dumpsys deviceidle whitelist) -join "`n") -match [regex]::Escape($Pkg)) { "battery-saver exemption: ON" } else { "WARNING: exemption not confirmed" }

# ------------------------------------------------------------ receiver
# 127.0.0.1, not localhost: on Windows "localhost" tries IPv6 first and wastes ~2 s before falling back
function Status { try { Invoke-RestMethod "http://127.0.0.1:$Port/status" -TimeoutSec 5 } catch { $null } }
if (-not (Status)) {
    Say "Starting the laptop receiver in the background (log: $Log)..."
    Start-Process -FilePath $Py -ArgumentList "`"$Receiver`"", $Port -WorkingDirectory (Split-Path $Receiver) `
        -RedirectStandardOutput $Log -RedirectStandardError "$Log.err" -WindowStyle Hidden
    $ok = $false
    for ($i = 0; $i -lt 20 -and -not $ok; $i++) { Start-Sleep 1; $ok = [bool](Status) }   # first start loads numpy/scikit-learn
    if (-not $ok) { Die "Receiver did not start; see $Log.err" }
}

# ------------------------------------------------------------ live test
Say "Setting the laptop address on the watch to $Server and running a 15 s test..."
A shell am start -n "$Pkg/.MainActivity" --es server $Server | Out-Null
Start-Sleep 2
$before = (Status).total
A shell am start -n "$Pkg/.MainActivity" --ez autostart true | Out-Null
Start-Sleep 15
A shell am start -n "$Pkg/.MainActivity" --ez autostop true | Out-Null
Start-Sleep 4
$st = Status
$got = $st.total - $before
$st | ConvertTo-Json -Compress
if ($got -ge 500) {
    Say "PASS: received $got samples in ~15 s (expect ~750). Ready."
    Say "Live page: http://localhost:$Port    Runner: http://localhost:$Port/runner"
} else {
    Write-Host "`nFAIL: only $got samples arrived." -ForegroundColor Red
    Write-Host "Check: watch and laptop on the same Wi-Fi? Windows Firewall must allow Python on this network"
    Write-Host "(set the Wi-Fi network to Private, and click Allow if Windows asks about python.exe)."
    Write-Host "The watch screen shows the reason under the button. Laptop address on watch: $Server"
    exit 1
}
