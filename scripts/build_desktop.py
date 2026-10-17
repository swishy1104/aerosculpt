"""Build the desktop app for distribution.

    python scripts/build_desktop.py [--gpu-pack] [--installer] [--cpu-only] [--skip-tests]

1. stamps aerosculpt/BUILD with the git commit and writes THIRD-PARTY-NOTICES.txt
2. runs PyInstaller (desktop.spec); gmsh (GPL) is left out unless AEROSCULPT_BUNDLE_GMSH=1
3. --gpu-pack: moves the CUDA runtime and numba-cuda out of the app into a separate zip
   (AeroSculpt-GPU-pack-<version>.zip) so the base download stays small; users with an NVIDIA
   card unzip it over the app folder
4. signs AeroSculpt.exe when SIGN_CMD is set (e.g. signtool sign /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 /a "{file}")
5. --installer: compiles installer/AeroSculpt.iss with Inno Setup (ISCC on PATH or ISCC env) into AeroSculpt-Setup-<version>.exe
6. zips the app folder as AeroSculpt-<version>-windows.zip and runs AeroSculpt.exe --check

Outputs land in dist/.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
GPU_PARTS = ["nvidia", "cuda", "numba/cuda", "numba_cuda.libs", "cuda_bindings.libs", "cuda_core.libs"]
GPU_META_PREFIXES = ("numba_cuda-", "cuda_bindings-", "cuda_core-", "cuda_pathfinder-", "cuda_toolkit-", "nvidia_")


def run(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def version():
    sys.path.insert(0, str(ROOT))
    import aerosculpt
    return aerosculpt.__version__


def stamp():
    try:
        sha = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        sha = "dev"
    (ROOT / "aerosculpt" / "BUILD").write_text(sha + "\n", encoding="utf-8")
    run([sys.executable, str(ROOT / "scripts" / "third_party_notices.py")])
    return sha


def zip_dir(folder: Path, out: Path, prefix=""):
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in folder.rglob("*"):
            if f.is_file():
                z.write(f, prefix + str(f.relative_to(folder.parent)))
    print(f"wrote {out} ({out.stat().st_size // 1_000_000} MB)")


def split_gpu_pack(app: Path, ver: str):
    internal = app / "_internal"
    pack = DIST / "gpu-pack" / "AeroSculpt" / "_internal"
    if pack.parent.parent.exists():
        shutil.rmtree(pack.parent.parent)
    pack.mkdir(parents=True)
    moved = 0
    for part in GPU_PARTS:
        src = internal / part
        if src.exists():
            dst = pack / part
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            moved += 1
    for d in internal.iterdir():
        if d.is_dir() and d.name.startswith(GPU_META_PREFIXES) and d.name.endswith(".dist-info"):
            shutil.move(str(d), str(pack / d.name))
    out = DIST / f"AeroSculpt-GPU-pack-{ver}.zip"
    zip_dir(pack.parent, out)
    print(f"GPU pack: {moved} parts moved")
    return out


def sign(exe: Path):
    cmd = os.environ.get("SIGN_CMD")
    if not cmd:
        print("SIGN_CMD not set: the executable is not code-signed (Windows SmartScreen will warn users)")
        return
    run(cmd.replace("{file}", str(exe)), shell=True)


def installer(ver: str):
    iscc = os.environ.get("ISCC") or shutil.which("ISCC") or shutil.which("iscc")
    for guess in (r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe", r"C:\Program Files\Inno Setup 6\ISCC.exe"):
        if not iscc and Path(guess).exists():
            iscc = guess
    if not iscc:
        print("Inno Setup (ISCC) not found: skipping the installer (the zip is still built)")
        return None
    run([iscc, f"/DAppVersion={ver}", f"/DSourceDir={DIST / 'AeroSculpt'}", f"/DOutDir={DIST}", str(ROOT / "installer" / "AeroSculpt.iss")])
    out = DIST / f"AeroSculpt-Setup-{ver}.exe"
    if out.exists():
        sign(out)
    return out


def main(argv):
    ver = version()
    sha = stamp()
    print(f"AeroSculpt {ver} ({sha})")
    if "--skip-tests" not in argv:
        run([sys.executable, "-m", "pytest", "-q", "-x", "tests/test_rocket.py", "tests/test_api.py"], cwd=ROOT, env={**os.environ, "AEROSCULPT_GPU": "0"})
    env = dict(os.environ)
    if "--cpu-only" in argv:
        env["AEROSCULPT_CPU_ONLY"] = "1"
    run([sys.executable, "-m", "PyInstaller", "desktop.spec", "--noconfirm"], cwd=ROOT, env=env)
    app = DIST / "AeroSculpt"
    exe = app / "AeroSculpt.exe"
    shutil.copy(ROOT / "THIRD-PARTY-NOTICES.txt", app / "THIRD-PARTY-NOTICES.txt")
    sign(exe)
    if "--gpu-pack" in argv and "--cpu-only" not in argv:
        split_gpu_pack(app, ver)
    zip_dir(app, DIST / f"AeroSculpt-{ver}-windows.zip")
    if "--installer" in argv:
        installer(ver)
    print("self-check of the packaged app:")
    run([str(exe), "--check"], env={**os.environ, "LOCALAPPDATA": str(DIST / "check-home")})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
