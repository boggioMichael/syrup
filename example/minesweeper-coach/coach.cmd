@echo off
rem The Minesweeper coach: double-click to build it (the first time) and run it.
rem Close its window to stop it. Options go after the name, e.g.  coach.cmd --quiet
cd /d "%~dp0"
where cargo >nul 2>nul || (echo Building the coach needs Rust: https://rustup.rs & pause & exit /b 1)
cargo build --release
if not errorlevel 1 goto run
rem A windows-gnu toolchain next to llvm-mingw cannot link (llvm-mingw has no libgcc):
rem try again with the toolchain made for llvm-mingw, if it is installed.
rustup toolchain list | findstr /c:"-gnullvm" >nul || goto failed
echo.
echo Trying again with the gnullvm toolchain...
cargo +stable-x86_64-pc-windows-gnullvm build --release || goto failed
:run
target\release\mines-coach.exe %*
pause
exit /b
:failed
echo.
echo Could not build the coach. If linking failed with "-lgcc_eh", see README.md.
pause
exit /b 1
