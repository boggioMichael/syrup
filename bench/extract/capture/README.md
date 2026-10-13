# Recording kit (DATA.md)

`record.ps1` records MapleStory sessions for the sealed test sets. It needs
nothing installed: it is Windows PowerShell and .NET's screen copy.

- It takes a frame every 3 seconds, only while MapleStory is the window in
  front, so nothing else on the screen is ever captured.
- It saves the bottom 12% of the game area (the status bar: HP, MP, EXP,
  level, quick slots) as lossless PNG. `-Full` saves whole frames too, for
  task C later.
- Frames go to `Documents\MapleSyrup sessions\research <date time>\`, with a
  `meta.json` (HUD style, world, frame sizes, times). They stay on the owner's
  PC. Only the HP, MP and EXP crops are copied from them, so chat and names
  are never copied.
- The game must run windowed or as a borderless window. The script warns when
  the picture comes back black (exclusive full screen).

Double-click `record.cmd`, answer two questions (HUD style, world), and play.
It stops after 15 minutes, or on Ctrl+C.
