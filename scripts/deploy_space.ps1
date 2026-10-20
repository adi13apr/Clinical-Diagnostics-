<#
.SYNOPSIS
    Deploy the FastAPI backend (api/) to a Hugging Face Docker Space.

.DESCRIPTION
    The GitHub history can't be pushed to the Space: it has the 16 MB checkpoint as a plain git
    blob (Hugging Face needs files over 10 MB in LFS), and the GitHub README has no Space front
    matter. So this script:
      1. exports a clean snapshot of -Ref (default main) with `git archive`: committed files only,
         no history, no untracked or gitignored files (so never .streamlit/secrets.toml);
      2. keeps only what the Docker image needs, and puts deploy/space/README.md (front matter:
         sdk docker, app_port 7860) in place of the GitHub README;
      3. refuses to continue if a secrets file or an unexpected large file is in the snapshot;
      4. makes a new one-commit git repo there, with *.pth tracked by Git LFS;
      5. -DockerTest: builds the image from that snapshot and checks GET /health
         (model_loaded and calibrated must be true), then removes the container;
      6. -Push: force-pushes the snapshot to the Space (replacing whatever is there).
    Without -Push nothing leaves your machine.

.EXAMPLE
    .\scripts\deploy_space.ps1 -Space adi13apr/clinical-diagnostics-api -DockerTest

.EXAMPLE
    .\scripts\deploy_space.ps1 -Space adi13apr/clinical-diagnostics-api -Push
#>
param(
    [Parameter(Mandatory = $true)]
    [string]$Space,                 # "<hf-username>/<space-name>"
    [string]$Ref = "main",          # git branch, tag or commit to deploy
    [switch]$DockerTest,            # build + run the image locally and check /health
    [switch]$Push,                  # force-push the snapshot to the Space
    [int]$Port = 7860
)

$ErrorActionPreference = "Stop"

# What the Docker image needs (see Dockerfile / .dockerignore). git archive fails if one is missing.
$Paths = @(
    "Dockerfile", ".dockerignore", "requirements.txt", "requirements-api.txt",
    "api", "services", "agent", "config", "outputs/calibration.json",
    "best_student_model_aligned_final.pth", "deploy/space/README.md"
)
$LfsPatterns = @("*.pth")
$MaxPlainFileBytes = 10MB   # Hugging Face rejects larger files that aren't in LFS

function Step($text) { Write-Host ""; Write-Host "==> $text" -ForegroundColor Cyan }

# Native commands are judged by exit code. Windows PowerShell 5.1 turns their stderr (git and docker
# write progress there) into errors, which "Stop" would make fatal, so it's relaxed around them.
function Native {
    # Run a native command; stop on a non-zero exit code
    param([string]$What, [scriptblock]$Block)
    $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    try { & $Block } finally { $ErrorActionPreference = $old }
    if ($LASTEXITCODE -ne 0) { throw "$What failed (exit code $LASTEXITCODE)" }
}

function Quiet {
    # Run a native command, hiding stderr; the caller checks $LASTEXITCODE if it matters
    param([scriptblock]$Block)
    $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    try { & $Block 2>$null } finally { $ErrorActionPreference = $old }
}

function Show-ContainerLogs {
    # uvicorn logs to stderr, so merge it in and print it as plain text
    param([string]$Container)
    $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    try { docker logs --tail 60 $Container 2>&1 | ForEach-Object { "$_" } | Out-Host } finally { $ErrorActionPreference = $old }
}

if ($Space -notmatch '^[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*$') {
    throw "-Space must look like <hf-username>/<space-name>, got '$Space'"
}
$SpaceUrl = "https://huggingface.co/spaces/$Space"
$AppUrl = "https://" + ($Space.ToLower() -replace '[/_.]', '-') + ".hf.space"

# --- 1. Which commit -----------------------------------------------------------------------
Step "Resolving $Ref"
$Repo = (& git -C $PSScriptRoot rev-parse --show-toplevel)
if ($LASTEXITCODE -ne 0) { throw "Not inside the git repo" }
$Commit = (& git -C $Repo rev-parse --verify "$Ref^{commit}")
if ($LASTEXITCODE -ne 0) { throw "Unknown ref '$Ref'" }
$Short = $Commit.Substring(0, 7)
$Subject = (& git -C $Repo log -1 --format=%s $Commit)
Write-Host "Deploying $Short  $Subject"

Quiet { git -C $Repo fetch --quiet origin }
$Remote = (Quiet { git -C $Repo rev-parse --verify --quiet "origin/$Ref" })
if ($Remote -and $Remote -ne $Commit) {
    Write-Warning "$Ref ($Short) differs from origin/$Ref ($($Remote.Substring(0, 7))). The Space should match GitHub: push or pull first."
}

# --- 2. Snapshot ---------------------------------------------------------------------------
$Name = ($Space -split '/')[1]
$Work = Join-Path $env:TEMP ("hf-space-{0}-{1}" -f $Name, (Get-Date -Format "yyyyMMdd-HHmmss"))
Step "Exporting a clean snapshot to $Work"
New-Item -ItemType Directory -Path $Work | Out-Null
$Tar = "$Work.tar"
Native "git archive" { git -C $Repo archive --format=tar -o $Tar $Commit -- $Paths }
Native "tar" { tar -xf $Tar -C $Work }
Remove-Item $Tar

# The Space README (front matter) replaces the GitHub README
Move-Item (Join-Path $Work "deploy/space/README.md") (Join-Path $Work "README.md")
Remove-Item -Recurse (Join-Path $Work "deploy")

# --- 3. Safety checks ----------------------------------------------------------------------
Step "Checking the snapshot for secrets and large files"
$Forbidden = Get-ChildItem -Path $Work -Recurse -Force -File |
    Where-Object { $_.Name -eq "secrets.toml" -or $_.Name -like ".env*" -or $_.Name -like "*.pem" }
if ($Forbidden) { throw "Refusing to deploy, secret-like files in the snapshot: $($Forbidden.FullName -join ', ')" }

$Large = Get-ChildItem -Path $Work -Recurse -Force -File | Where-Object {
    $file = $_
    $_.Length -gt $MaxPlainFileBytes -and -not ($LfsPatterns | Where-Object { $file.Name -like $_ })
}
if ($Large) { throw "Files over 10 MB that aren't LFS-tracked: $($Large.FullName -join ', ')" }
Get-ChildItem -Path $Work -Force | Format-Table Name, Length -AutoSize | Out-String | Write-Host

# --- 4. One-commit repo with LFS -----------------------------------------------------------
Step "Creating a one-commit git repo with Git LFS"
Native "git lfs" { git lfs version | Out-Null }
Native "git init" { git -C $Work init --quiet --initial-branch=main }
Native "git lfs install" { git -C $Work lfs install --local | Out-Null }
foreach ($pattern in $LfsPatterns) { Native "git lfs track" { git -C $Work lfs track $pattern | Out-Null } }
Native "git add" { git -C $Work add --all }
Native "git commit" { git -C $Work commit --quiet -m "Deploy $Short from GitHub ($Ref): $Subject" }
$LfsFiles = (& git -C $Work lfs ls-files --name-only)
if (-not ($LfsFiles -match '\.pth$')) { throw "The checkpoint is not tracked by LFS" }
Write-Host "LFS files: $($LfsFiles -join ', ')"

# --- 5. Local Docker test ------------------------------------------------------------------
if ($DockerTest) {
    Step "Building the Docker image from the snapshot (first build downloads PyTorch, ~5-10 min)"
    Quiet { docker info --format "{{.ServerVersion}}" } | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Docker isn't running: start Docker Desktop and wait until it says 'Engine running'" }
    Native "docker build" { docker build -t clinical-api:local $Work }

    $Container = "clinical-api-test"
    Quiet { docker rm -f $Container } | Out-Null
    $RunArgs = @("run", "-d", "--name", $Container, "-p", "${Port}:7860")
    # Local only: mount your secrets read-only so auth/LLM show as configured. Never copied into the image.
    $Secrets = Join-Path $Repo ".streamlit/secrets.toml"
    if (Test-Path $Secrets) { $RunArgs += @("-v", "${Secrets}:/app/.streamlit/secrets.toml:ro") }
    Native "docker run" { docker @RunArgs clinical-api:local | Out-Null }

    try {
        Step "Waiting for http://localhost:$Port/health (model load takes ~10-30 s)"
        $Health = $null
        for ($i = 0; $i -lt 60 -and -not $Health; $i++) {
            Start-Sleep -Seconds 3
            try { $Health = Invoke-RestMethod "http://localhost:$Port/health" -TimeoutSec 5 } catch { }
        }
        if (-not $Health) {
            Show-ContainerLogs $Container
            throw "/health never answered"
        }
        $Health | Format-List | Out-String | Write-Host
        if (-not ($Health.model_loaded -and $Health.calibrated)) {
            Show-ContainerLogs $Container
            throw "/health: the model must be loaded and calibrated"
        }
        Write-Host "Local image OK. While it runs you can also open http://localhost:$Port/docs" -ForegroundColor Green
    }
    finally {
        Quiet { docker rm -f $Container } | Out-Null
    }
}

# --- 6. Push -------------------------------------------------------------------------------
if (-not $Push) {
    Step "Not pushed (add -Push to deploy). Snapshot kept at $Work"
    return
}

Step "Force-pushing to $SpaceUrl"
Write-Host "If Git asks to sign in: username = your Hugging Face username, password = a Hugging Face access token with write access."
Native "git remote add" { git -C $Work remote add space $SpaceUrl }
Native "git push" { git -C $Work push --force space main }

Step "Pushed $Short. The Space now rebuilds (~5-10 min)"
Write-Host "Build logs:  $SpaceUrl  (Logs button at the top)"
Write-Host "Health:      $AppUrl/health"
Write-Host "API docs:    $AppUrl/docs"
