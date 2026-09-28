"""When `import torch` fails on Windows with a DLL error, say which DLL and
why, as far as ctypes can tell: each torch DLL loaded on its own, the
Visual C++ runtime, the Python build, the DLLs already in the process.
Read by run.py's server.log; costs nothing when torch imports fine.
"""
from __future__ import annotations

import ctypes
import os
import platform
import sys


def _try(path: str) -> str:
    try:
        ctypes.WinDLL(path) if os.name == "nt" else ctypes.CDLL(path)
        return "ok"
    except OSError as e:
        return f"FAILED: {e}"


def explain_dll_failure(err: BaseException) -> None:
    print("=== torch could not be imported ===", flush=True)
    print(f"error: {err}")
    print(f"python: {sys.version.split()[0]} {platform.architecture()[0]} at {sys.executable}")
    print(f"windows: {platform.platform()} machine={platform.machine()}")
    if os.name != "nt":
        return
    for dll in ("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll", "msvcp140_1.dll", "concrt140.dll", "vcomp140.dll"):
        print(f"  system {dll}: {_try(dll)}")
    try:
        import importlib.util

        spec = importlib.util.find_spec("torch")
        lib = os.path.join(os.path.dirname(spec.origin), "lib")
        print(f"torch/lib: {lib}")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(lib)
        for name in ("libiomp5md.dll", "uv.dll", "c10.dll", "torch_cpu.dll", "torch_global_deps.dll", "torch_python.dll"):
            path = os.path.join(lib, name)
            if os.path.isfile(path):
                print(f"  {name} ({os.path.getsize(path)} bytes): {_try(path)}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  (could not probe torch/lib: {e})")
    try:
        import psutil  # noqa: F401
    except ImportError:
        pass
    loaded = []
    try:
        import ctypes.wintypes as wt

        psapi = ctypes.WinDLL("psapi")
        kernel32 = ctypes.WinDLL("kernel32")
        h = kernel32.GetCurrentProcess()
        mods = (wt.HMODULE * 1024)()
        needed = wt.DWORD()
        if psapi.EnumProcessModules(h, mods, ctypes.sizeof(mods), ctypes.byref(needed)):
            buf = ctypes.create_unicode_buffer(1024)
            for i in range(min(needed.value // ctypes.sizeof(wt.HMODULE), 1024)):
                psapi.GetModuleFileNameExW(h, mods[i], buf, 1024)
                loaded.append(buf.value)
    except Exception:  # noqa: BLE001
        pass
    interesting = [m for m in loaded if not m.lower().startswith("c:\\windows\\")]
    print("non-system DLLs already loaded when torch was imported:")
    for m in interesting:
        print(f"  {m}")
    print("=== end ===", flush=True)
