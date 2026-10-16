"""AeroSculpt Desktop: the whole app on this computer, simulating on its own graphics card.

    python desktop.py            (or AeroSculpt.exe from the desktop build, see desktop.spec)

Runs privately on 127.0.0.1 with no sign-in (one built-in user with every feature), keeps its
projects and results in your user folder (%LOCALAPPDATA%\\AeroSculpt on Windows), uses the NVIDIA
GPU when there is one (else the CPU), and opens the app in your browser. Nothing leaves the
computer. Close this window (or press Ctrl+C) to stop it.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import socket
import sys
import threading
import webbrowser
from pathlib import Path


def data_home() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    d = Path(base) / "AeroSculpt"
    d.mkdir(parents=True, exist_ok=True)
    return d


def free_port(preferred=8765):
    for port in (preferred, 0):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
            except OSError:
                continue
    raise RuntimeError("no free port")


def configure(port):
    """Environment for a private, single-user app; set before the app is imported (its settings
    are read at import)."""
    home = data_home()
    os.environ.setdefault("DATA_DIR", str(home / "data"))
    os.environ["HOST"] = "127.0.0.1"
    os.environ["PORT"] = str(port)
    os.environ.setdefault("PUBLIC_URL", f"http://127.0.0.1:{port}")
    os.environ.setdefault("OPEN_ACCESS", "1")          # no sign-in on this computer
    os.environ.setdefault("FUSION_BRIDGE", "1")        # Send to Fusion, when Fusion runs here
    os.environ.setdefault("NUMBA_CACHE_DIR", str(home / "compiled"))   # compiled kernels survive restarts
    return home


def dll_dirs():
    """Where a frozen app's CUDA DLLs are. Per process (DLL search folders are not inherited), so it runs
    before multiprocessing hands a child (the job worker) its task."""
    if not getattr(sys, "frozen", False):
        return
    # the bundled CUDA libraries (nvrtc, cudart, nvvm) live next to the app, as do the private DLLs of the
    # delvewheel-repaired CUDA wheels (<pkg>.libs), which numba-cuda would look for three folders above
    # itself, outside the bundle
    base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    for sub in list(base.glob("nvidia/*/bin")) + list(base.glob("nvidia/*/nvvm/bin")) + list(base.glob("*.libs")):
        os.add_dll_directory(str(sub))
        if str(sub) not in os.environ.get("PATH", ""):
            os.environ["PATH"] = str(sub) + os.pathsep + os.environ.get("PATH", "")


def check():
    """AeroSculpt.exe --check [--verbose]: what this computer can use (GPU, CAD), and why not."""
    configure(free_port())
    from aerosculpt.cfd import gpu
    from aerosculpt.geometry import cad
    ok = gpu.available()
    print("GPU:", f"yes ({gpu.device_name()})" if ok else f"no ({gpu._STATE.get('why')})")
    if not ok and "--verbose" in sys.argv and gpu._STATE.get("trace"):
        print(gpu._STATE["trace"])
    print("STEP/IGES (gmsh):", "yes" if cad.available() else "no")
    return 0


def main():
    dll_dirs()
    mp.freeze_support()
    if "--check" in sys.argv:
        return check()
    port = free_port()
    home = configure(port)
    url = f"http://127.0.0.1:{port}/app"
    print(f"AeroSculpt Desktop\n  open  {url}\n  data  {home}\n  Close this window to stop.\n", flush=True)
    if not os.environ.get("AEROSCULPT_NO_BROWSER"):
        threading.Timer(3.0, lambda: webbrowser.open(url)).start()
    import run
    return run.main(["serve", "--host", "127.0.0.1", "--port", str(port)])


if __name__ == "__main__":
    sys.exit(main())
