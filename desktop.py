"""AeroSculpt Desktop: the whole app on this computer, simulating on its own graphics card.

    python desktop.py            (or AeroSculpt.exe from the desktop build, see desktop.spec)
    AeroSculpt.exe --check       what this computer can use (GPU pack, STEP, licence), and why not
    AeroSculpt.exe --license KEY install a desktop licence key

Runs privately on 127.0.0.1 with no sign-in, keeps its projects and results in your user folder
(%LOCALAPPDATA%\\AeroSculpt on Windows), uses the NVIDIA GPU when the GPU pack is installed and
a card is present (else the CPU), and opens the app in your browser through a launch link that
carries a one-time token (only that browser may drive the app). Nothing leaves the computer
unless you ask (motor search, update check). Close this window (or press Ctrl+C) to stop it.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import secrets
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


def configure(port, token=None):
    """Environment for a private, single-user app; set before the app is imported (its settings
    are read at import)."""
    home = data_home()
    os.environ.setdefault("DATA_DIR", str(home / "data"))
    os.environ["HOST"] = "127.0.0.1"
    os.environ["PORT"] = str(port)
    os.environ.setdefault("PUBLIC_URL", f"http://127.0.0.1:{port}")
    os.environ.setdefault("OPEN_ACCESS", "1")          # no sign-in on this computer
    os.environ.setdefault("OPEN_ACCESS_PLAN", "license")   # features by the installed licence key (free tier without one)
    os.environ.setdefault("FUSION_BRIDGE", "0")        # Send to Fusion only once switched on in the app's settings
    os.environ.setdefault("NUMBA_CACHE_DIR", str(home / "compiled"))   # compiled kernels survive restarts
    if token:
        os.environ["LOCAL_TOKEN"] = token
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


def gpu_pack_installed():
    """Frozen app: whether the optional GPU pack (CUDA runtime + numba-cuda) is in place."""
    if not getattr(sys, "frozen", False):
        return True
    base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return (base / "nvidia").is_dir() and (base / "numba" / "cuda").is_dir()


def check():
    """AeroSculpt.exe --check [--verbose]: what this computer can use (GPU, CAD, licence), and why not."""
    configure(free_port())
    from aerosculpt import __version__, build_id
    from aerosculpt.cfd import gpu
    from aerosculpt.geometry import cad
    from aerosculpt import license as lic
    print(f"AeroSculpt {__version__} ({build_id()})")
    if not gpu_pack_installed():
        print("GPU: no (the GPU pack is not installed: unzip AeroSculpt-GPU-pack into the app folder)")
    else:
        ok = gpu.available()
        print("GPU:", f"yes ({gpu.device_name()})" if ok else f"no ({gpu._STATE.get('why')})")
        if not ok and "--verbose" in sys.argv and gpu._STATE.get("trace"):
            print(gpu._STATE["trace"])
    print("STEP/IGES (gmsh):", "yes" if cad.available() else "no (STL and OBJ only; gmsh is GPL and not bundled)")
    st = lic.installed()
    print("Licence:", f"{st['plan']} for {st.get('email')}" + (f" ({st['why']})" if st.get("why") else "") if st["installed"] else "none (free tier: draft fidelity, no exports)")
    return 0


def install_license(key):
    configure(free_port())
    from aerosculpt import license as lic
    try:
        st = lic.install(key)
    except Exception as e:  # noqa: BLE001
        print("Not installed:", e)
        return 1
    print(f"Licence installed: {st['plan']} for {st['email']}" + (f" ({st['why']})" if st.get("why") else ""))
    return 0


def main():
    dll_dirs()
    mp.freeze_support()
    if "--check" in sys.argv:
        return check()
    if "--license" in sys.argv:
        i = sys.argv.index("--license")
        return install_license(sys.argv[i + 1] if i + 1 < len(sys.argv) else "")
    port = free_port()
    token = secrets.token_urlsafe(18)
    home = configure(port, token)
    url = f"http://127.0.0.1:{port}/app?t={token}"
    print(f"AeroSculpt Desktop\n  open  {url}\n  data  {home}\n  Close this window to stop. --check lists what this computer can use.\n", flush=True)
    if not os.environ.get("AEROSCULPT_NO_BROWSER"):
        threading.Timer(3.0, lambda: webbrowser.open(url)).start()
    import run
    return run.main(["serve", "--host", "127.0.0.1", "--port", str(port)])


if __name__ == "__main__":
    sys.exit(main())
