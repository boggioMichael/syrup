#!/usr/bin/env bash
# Opens a window for the capture tests and names it in SYRUP_TEST_WINDOW.
# On Linux it first starts an X server.
set -euo pipefail

case "$RUNNER_OS" in
Linux)
    sudo apt-get update && sudo apt-get install -y xvfb python3-tk
    Xvfb :99 -screen 0 1280x800x24 &
    echo "DISPLAY=:99" >>"$GITHUB_ENV"
    title="Syrup capture test"
    sleep 1
    DISPLAY=:99 /usr/bin/python3 -c "import tkinter; w = tkinter.Tk(); w.title('$title'); w.geometry('320x200'); w.mainloop()" &
    ;;
macOS)
    file="$RUNNER_TEMP/syrup-capture-test.txt"
    echo "Syrup" >"$file"
    open -a TextEdit "$file"
    title="syrup-capture-test"
    ;;
Windows)
    notepad.exe &
    title="Notepad"
    ;;
esac

sleep 3
echo "SYRUP_TEST_WINDOW=$title" >>"$GITHUB_ENV"
echo "SYRUP_EXPECT_WINDOW=1" >>"$GITHUB_ENV"
