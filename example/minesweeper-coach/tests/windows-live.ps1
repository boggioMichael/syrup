# The coach live on Windows, end to end: a game's positions are shown on the
# screen (show-frames.ps1) while `mines-coach live` watches, and what it said
# must match the game — a new game, advice, and how it ended. What it saw
# and said is left in dump/ (the CI keeps it).
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
$exe = 'target\release\mines-coach.exe'
foreach ($d in 'frames', 'dump') { if (Test-Path $d) { Remove-Item -Recurse -Force $d } }

# The game, played in simulation first: the lines to expect.
& $exe demo 4 --frames frames | Tee-Object -FilePath expected.txt | Out-Host

$coach = Start-Process -FilePath $exe -ArgumentList 'live', '--dump', 'dump' -PassThru `
    -RedirectStandardOutput live.out -RedirectStandardError live.err
Start-Sleep -Seconds 2
$viewer = Start-Process -FilePath powershell -PassThru -ArgumentList '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $PSScriptRoot 'show-frames.ps1'), '-Dir', 'frames'
$viewer | Wait-Process -Timeout 180
Start-Sleep -Seconds 2
if (-not $coach.HasExited) { Stop-Process -Id $coach.Id }

Write-Host '--- the coach, live:'
Get-Content live.out
Write-Host '--- its errors:'
Get-Content live.err
$said = if (Test-Path dump\said.txt) { Get-Content dump\said.txt -Raw } else { '' }
$frames = @(Get-ChildItem dump -Filter 'frame-*.png' -ErrorAction SilentlyContinue).Count
Write-Host "--- $frames frames saved"
if ($said -notmatch 'New game') { throw 'the coach never saw the new game' }
if ($said -notmatch 'safe|No sure move|Nothing is sure') { throw 'the coach never gave advice' }
if ($said -notmatch 'Cleared|Boom') { throw 'the coach never saw the game end' }
if ($frames -lt 5) { throw "only $frames positions seen" }
Write-Host 'the coach saw the game through and said so'
