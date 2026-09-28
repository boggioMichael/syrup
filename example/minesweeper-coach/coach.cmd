@echo off
rem The Minesweeper coach: double-click to build it (the first time) and run it.
rem Close its window to stop it. Options go after the name, e.g.  coach.cmd --quiet
cd /d "%~dp0"
where cargo >nul 2>nul || (echo Building the coach needs Rust: https://rustup.rs & pause & exit /b 1)
cargo build --release || (pause & exit /b 1)
target\release\mines-coach.exe %*
pause
