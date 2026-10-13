<#
Research recorder for bench/extract (DATA.md).

Saves the MapleStory window's game area as lossless PNG every few seconds,
only while MapleStory is the window in front, into
Documents\MapleSyrup sessions\research <date time>\. By default it keeps the
bottom strip of each frame (the status bar, where HP, MP and EXP are printed),
which is all tasks A and B need; -Full keeps whole frames too.

Nothing leaves this PC: the frames stay here, and only the HP, MP and EXP crops
are copied from them later, so chat and names are never copied.

Run it by double-clicking record.cmd, or:
  powershell -NoProfile -ExecutionPolicy Bypass -File record.ps1 -Hud classic -World "Classic World"
Stop with Ctrl+C, or it stops by itself after -Minutes.
#>
param(
  [string]$Hud = '',
  [string]$World = '',
  [double]$Every = 3.0,
  [int]$Minutes = 15,
  [double]$Strip = 0.12,
  [switch]$Full,
  [string]$Process = 'MapleStory'
)
$ErrorActionPreference = 'Stop'

if ($Hud -eq '') {
  $answer = Read-Host 'HUD style: classic or modern [classic]'
  $Hud = if ($answer -eq '') { 'classic' } else { $answer }
}
$Hud = $Hud.Trim().ToLower()
if ($Hud -ne 'classic' -and $Hud -ne 'modern') { throw "HUD style must be classic or modern, not '$Hud'." }
if ($World -eq '') {
  $answer = Read-Host 'World [Classic World]'
  $World = if ($answer -eq '') { 'Classic World' } else { $answer.Trim() }
}

Add-Type -AssemblyName System.Drawing
Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class ResearchWin {
  [StructLayout(LayoutKind.Sequential)] public struct RECT { public int Left, Top, Right, Bottom; }
  [StructLayout(LayoutKind.Sequential)] public struct POINT { public int X, Y; }
  [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
  [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
  [DllImport("user32.dll")] public static extern bool GetClientRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern bool ClientToScreen(IntPtr h, ref POINT p);
  [DllImport("user32.dll")] public static extern bool IsIconic(IntPtr h);
}
"@
[ResearchWin]::SetProcessDPIAware() | Out-Null

function Find-Game {
  Get-Process -Name $Process -ErrorAction SilentlyContinue |
    Where-Object { $_.MainWindowHandle -ne [IntPtr]::Zero } | Select-Object -First 1
}

if (-not (Find-Game)) { throw "MapleStory is not running (no window for process '$Process'). Start the game first." }

$stamp = Get-Date -Format 'yyyy-MM-dd HH-mm-ss'
$root = Join-Path ([Environment]::GetFolderPath('MyDocuments')) 'MapleSyrup sessions'
$dir = Join-Path $root "research $stamp"
New-Item -ItemType Directory -Force -Path $dir | Out-Null

$frames = New-Object System.Collections.ArrayList
$started = Get-Date
$deadline = $started.AddMinutes($Minutes)
$n = 0
$warned = $false
Write-Host "Recording to: $dir"
Write-Host "HUD: $Hud, world: $World, a frame every $Every s, until $($deadline.ToShortTimeString()). Ctrl+C stops."
Write-Host "Frames are taken only while MapleStory is the window in front."

try {
  while ((Get-Date) -lt $deadline) {
    $game = Find-Game
    if ($game) {
      $hwnd = $game.MainWindowHandle
      if ([ResearchWin]::GetForegroundWindow() -eq $hwnd -and -not [ResearchWin]::IsIconic($hwnd)) {
        $r = New-Object 'ResearchWin+RECT'
        [ResearchWin]::GetClientRect($hwnd, [ref]$r) | Out-Null
        $p = New-Object 'ResearchWin+POINT'
        [ResearchWin]::ClientToScreen($hwnd, [ref]$p) | Out-Null
        $w = $r.Right - $r.Left
        $h = $r.Bottom - $r.Top
        if ($w -gt 0 -and $h -gt 0) {
          $bmp = New-Object System.Drawing.Bitmap $w, $h, ([System.Drawing.Imaging.PixelFormat]::Format24bppRgb)
          $g = [System.Drawing.Graphics]::FromImage($bmp)
          $g.CopyFromScreen($p.X, $p.Y, 0, 0, (New-Object System.Drawing.Size $w, $h))
          $g.Dispose()
          if (-not $warned) {
            $lit = 0
            foreach ($fx in 0.25, 0.5, 0.75) { foreach ($fy in 0.25, 0.5, 0.95) {
              $c = $bmp.GetPixel([int]($w * $fx), [int]($h * $fy)); $lit += $c.R + $c.G + $c.B } }
            if ($lit -eq 0) {
              Write-Warning 'The captured picture is black: the game may be in exclusive full screen. Switch it to windowed or borderless window and run again.'
            }
            $warned = $true
          }
          $rows = [Math]::Max(1, [int][Math]::Round($h * $Strip))
          $top = $h - $rows
          $strip = $bmp.Clone((New-Object System.Drawing.Rectangle 0, $top, $w, $rows), $bmp.PixelFormat)
          $name = 'strip-{0:D5}.png' -f $n
          $strip.Save((Join-Path $dir $name), [System.Drawing.Imaging.ImageFormat]::Png)
          $strip.Dispose()
          $entry = [ordered]@{ strip = $name; t = (Get-Date).ToString('o'); width = $w; height = $h; strip_top = $top }
          if ($Full) {
            $fullName = 'full-{0:D5}.png' -f $n
            $bmp.Save((Join-Path $dir $fullName), [System.Drawing.Imaging.ImageFormat]::Png)
            $entry.full = $fullName
          }
          $bmp.Dispose()
          [void]$frames.Add($entry)
          $n++
          if ($n % 20 -eq 0) { Write-Host "  $n frames" }
        }
      }
    }
    Start-Sleep -Milliseconds ([int]($Every * 1000))
  }
}
finally {
  $meta = [ordered]@{
    tool = 'bench/extract/capture/record.ps1'; started = $started.ToString('o'); stopped = (Get-Date).ToString('o')
    hud = $Hud; world = $World; every_s = $Every; strip_share = $Strip; full_frames = [bool]$Full
    process = $Process; frames = $frames
  }
  $meta | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 (Join-Path $dir 'meta.json')
  Write-Host "Saved $n frames in $dir"
}
