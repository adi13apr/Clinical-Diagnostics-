<#
.SYNOPSIS
    Run the API on this PC and publish it with a Cloudflare quick tunnel (free, no account).

.DESCRIPTION
    Use this after a reboot, or after changing the API. It:
      1. (-Rebuild) rebuilds the clinical-api:local image from a clean snapshot of -Ref, using
         deploy_space.ps1 -DockerTest (which also checks /health);
      2. (re)creates the `clinical-api` container: localhost only, restarts with Docker Desktop,
         your .streamlit/secrets.toml mounted read-only, FRONTEND_URL set for CORS;
      3. restarts the tunnel (cloudflared, downloaded to -ToolsDir if missing) and reads its
         new https://<random>.trycloudflare.com URL;
      4. if that URL differs from Vercel's VITE_API_URL, updates it and redeploys web/ to
         production with the Vercel CLI (must be logged in: npx vercel login).
    A quick tunnel's URL changes every time it starts and stops when this PC sleeps or shuts down.

.EXAMPLE
    .\scripts\serve_local_api.ps1

.EXAMPLE
    .\scripts\serve_local_api.ps1 -Rebuild -Ref main
#>
param(
    [string]$FrontendUrl = "https://clinical-diagnostics.vercel.app",
    [string]$ToolsDir = "D:\tools\cloudflared",
    [switch]$Rebuild,               # rebuild the image from -Ref first
    [string]$Ref = "main",
    [string]$WorkRoot = $env:TEMP,  # snapshot folder for -Rebuild
    [switch]$SkipVercel             # don't touch Vercel even if the URL changed
)

$ErrorActionPreference = "Stop"
# Text piped to native commands (vercel env add) must not start with a UTF-8 BOM: an invisible
# U+FEFF in VITE_API_URL breaks every API call, and in a header value it makes fetch() throw
$OutputEncoding = New-Object System.Text.UTF8Encoding $false
$Repo = Split-Path -Parent $PSScriptRoot
$Image = "clinical-api:local"
$Container = "clinical-api"
$Port = 7860

function Step($text) { Write-Host ""; Write-Host "==> $text" -ForegroundColor Cyan }

# Native commands are judged by exit code; Windows PowerShell 5.1 turns their stderr into errors
function Native {
    param([string]$What, [scriptblock]$Block)
    $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    try { & $Block } finally { $ErrorActionPreference = $old }
    if ($LASTEXITCODE -ne 0) { throw "$What failed (exit code $LASTEXITCODE)" }
}

function Quiet {
    param([scriptblock]$Block)
    $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    try { & $Block 2>$null } finally { $ErrorActionPreference = $old }
}

function Wait-Health($url, $tries) {
    for ($i = 0; $i -lt $tries; $i++) {
        try { return Invoke-RestMethod "$url/health" -TimeoutSec 15 } catch { Start-Sleep -Seconds 3 }
    }
    return $null
}

# --- 1. Docker and the image ----------------------------------------------------------------
Step "Checking Docker"
$Engine = Quiet { docker version --format "{{.Server.Version}}" }
if ($LASTEXITCODE -ne 0 -or -not "$Engine".Trim()) {
    throw "Docker's engine isn't running: start Docker Desktop and wait for 'Engine running'"
}
if ($Rebuild) {
    Step "Rebuilding $Image from $Ref"
    & (Join-Path $PSScriptRoot "deploy_space.ps1") -Space "local/clinical-diagnostics-api" -Ref $Ref -DockerTest -WorkRoot $WorkRoot
}
Quiet { docker image inspect $Image } | Out-Null
if ($LASTEXITCODE -ne 0) { throw "No $Image image yet: run again with -Rebuild" }

# --- 2. Container ---------------------------------------------------------------------------
Step "Starting the $Container container"
Quiet { docker rm -f $Container } | Out-Null
$RunArgs = @("run", "-d", "--name", $Container, "--restart", "unless-stopped",
             "-p", "127.0.0.1:${Port}:7860", "-e", "FRONTEND_URL=$FrontendUrl")
$Secrets = Join-Path $Repo ".streamlit\secrets.toml"
if (Test-Path $Secrets) { $RunArgs += @("-v", "${Secrets}:/app/.streamlit/secrets.toml:ro") }
else { Write-Warning "No .streamlit/secrets.toml: the API will start without Supabase and Groq settings" }
Native "docker run" { docker @RunArgs $Image | Out-Null }
$Health = Wait-Health "http://localhost:$Port" 40
if (-not $Health) { Quiet { docker logs --tail 40 $Container } | Out-Host; throw "The API didn't answer /health" }
Write-Host ("model_loaded={0} calibrated={1} auth_configured={2} llm_configured={3}" -f `
    $Health.model_loaded, $Health.calibrated, $Health.auth_configured, $Health.llm_configured)

# --- 3. Tunnel ------------------------------------------------------------------------------
$Cloudflared = Join-Path $ToolsDir "cloudflared.exe"
if (-not (Test-Path $Cloudflared)) {
    Step "Downloading cloudflared to $ToolsDir"
    New-Item -ItemType Directory -Force -Path $ToolsDir | Out-Null
    $ProgressPreference = "SilentlyContinue"
    Invoke-WebRequest "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe" -OutFile $Cloudflared
}
$Sig = Get-AuthenticodeSignature $Cloudflared
if ($Sig.Status -ne "Valid" -or $Sig.SignerCertificate.Subject -notmatch 'O="?Cloudflare, Inc\.') {
    throw "cloudflared.exe isn't validly signed by Cloudflare; delete it and run again"
}

Step "Restarting the Cloudflare quick tunnel"
Get-Process cloudflared -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq $Cloudflared } | Stop-Process -Force
$Log = Join-Path $ToolsDir "tunnel.log"
Remove-Item $Log -ErrorAction SilentlyContinue
Start-Process -FilePath $Cloudflared -WindowStyle Hidden `
    -ArgumentList "tunnel", "--no-autoupdate", "--url", "http://localhost:$Port", "--logfile", $Log | Out-Null
$ApiUrl = $null
for ($i = 0; $i -lt 60 -and -not $ApiUrl; $i++) {
    Start-Sleep -Seconds 2
    $m = [regex]::Match("$(Get-Content $Log -Raw -ErrorAction SilentlyContinue)", 'https://[a-z0-9-]+\.trycloudflare\.com')
    if ($m.Success) { $ApiUrl = $m.Value }
}
if (-not $ApiUrl) { throw "The tunnel didn't report a URL; see $Log" }
if (-not (Wait-Health $ApiUrl 40)) { throw "$ApiUrl/health doesn't answer through the tunnel; see $Log" }
Write-Host "API is public at $ApiUrl" -ForegroundColor Green

# --- 4. Vercel ------------------------------------------------------------------------------
if ($SkipVercel) { Write-Host "Skipped Vercel. Set VITE_API_URL=$ApiUrl there and redeploy."; return }

Step "Checking the API URL the live site uses"
$Web = Join-Path $Repo "web"
$Current = $null
try {
    $Html = (Invoke-WebRequest $FrontendUrl -UseBasicParsing -TimeoutSec 30).Content
    foreach ($Js in ([regex]::Matches($Html, '/assets/[A-Za-z0-9._-]+\.js') | ForEach-Object { $_.Value } | Sort-Object -Unique)) {
        $Code = (Invoke-WebRequest "$FrontendUrl$Js" -UseBasicParsing -TimeoutSec 60).Content
        $m = [regex]::Match($Code, 'https://[a-z0-9-]+\.trycloudflare\.com')
        if ($m.Success) { $Current = $m.Value; break }
    }
} catch { Write-Warning "Couldn't read the live site: $_" }

if ($Current -eq $ApiUrl) {
    Write-Host "The site already uses $ApiUrl; nothing to redeploy."
} else {
    Step "Pointing Vercel at $ApiUrl and redeploying (~1 min)"
    Push-Location $Web
    try {
        Quiet { npx --yes vercel@latest env rm VITE_API_URL production --yes } | Out-Null
        Native "vercel env add" { $ApiUrl | npx --yes vercel@latest env add VITE_API_URL production | Out-Null }
        Native "vercel deploy" { npx --yes vercel@latest deploy --prod --yes | Out-Null }
    } finally { Pop-Location }
}

Step "Done"
Write-Host "Web app:  $FrontendUrl"
Write-Host "API:      $ApiUrl  (docs: $ApiUrl/docs)"
Write-Host "Both stay up while this PC is awake and Docker Desktop is running."
