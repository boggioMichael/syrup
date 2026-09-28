# Shows a game's positions on the screen one after another, the way a
# Minesweeper window would show them: a borderless window on top, each
# picture for a second and a half, at its own size (no scaling). For the
# live test (windows-live.ps1): the coach, watching the screen, sees them.
param([Parameter(Mandatory = $true)][string]$Dir, [double]$Every = 1.5)

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$files = Get-ChildItem -Path $Dir -Filter *.png | Sort-Object Name
$form = New-Object System.Windows.Forms.Form
$form.FormBorderStyle = 'None'
$form.StartPosition = 'Manual'
$form.Location = New-Object System.Drawing.Point(120, 90)
$form.TopMost = $true
$form.ShowInTaskbar = $false
$box = New-Object System.Windows.Forms.PictureBox
$box.SizeMode = 'AutoSize'
$box.Location = New-Object System.Drawing.Point(0, 0)
$form.Controls.Add($box)
$form.Show()

function Wait-Showing([double]$seconds) {
    $end = (Get-Date).AddSeconds($seconds)
    while ((Get-Date) -lt $end) {
        [System.Windows.Forms.Application]::DoEvents()
        Start-Sleep -Milliseconds 20
    }
}

Wait-Showing 2
foreach ($f in $files) {
    $box.Image = [System.Drawing.Image]::FromFile($f.FullName)
    $form.ClientSize = $box.Image.Size
    Write-Host "showing $($f.Name)"
    Wait-Showing $Every
}
Wait-Showing 4
$form.Close()
