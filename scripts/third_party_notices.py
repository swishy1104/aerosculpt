"""Write THIRD-PARTY-NOTICES.txt: every library the app ships or loads, its version and licence,
with the licence text where the package carries one. Run before a release build:

    python scripts/third_party_notices.py
"""
from __future__ import annotations

import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "THIRD-PARTY-NOTICES.txt"

# distributions the app imports (the runtime, the solvers, reports, security) and ships in the desktop build
DISTS = ["numpy", "scipy", "numba", "llvmlite", "starlette", "uvicorn", "anyio", "h11", "click", "bcrypt", "cryptography", "PyJWT",
         "matplotlib", "scikit-image", "requests", "pillow", "numba-cuda", "cuda-bindings", "cuda-core", "cuda-pathfinder",
         "nvidia-cuda-nvrtc-cu12", "nvidia-cuda-runtime-cu12", "nvidia-nvjitlink-cu12", "nvidia-cuda-nvcc-cu12", "nvidia-cuda-cccl-cu12",
         "pyinstaller"]
STATIC = [
    ("three.js", "0.170.0", "MIT", "Loaded in the browser from cdn.jsdelivr.net (3D view). https://github.com/mrdoob/three.js/blob/dev/LICENSE"),
    ("occt-import-js", "0.0.23", "MIT (OpenCASCADE: LGPL-2.1 with exception)", "Loaded in the browser from cdn.jsdelivr.net for STEP/IGES conversion. https://github.com/kovacsv/occt-import-js"),
    ("Inter, JetBrains Mono", "", "SIL Open Font License 1.1", "Loaded from fonts.googleapis.com."),
    ("NVIDIA CUDA runtime, NVRTC, NVVM, nvJitLink, CCCL", "12.x", "NVIDIA CUDA Toolkit EULA (redistributable runtime components)",
     "Shipped in the optional GPU pack as the nvidia-*-cu12 wheels. https://docs.nvidia.com/cuda/eula/"),
    ("gmsh", "4.x", "GPL-2.0-or-later", "NOT bundled in the desktop build. Install it yourself (pip install gmsh) for STEP/IGES import; the server offering it must honour the GPL."),
    ("NeuralFoil weights", "", "MIT (Peter Sharpe)", "Vendored NumPy port and weights in aerosculpt/aero/nf_weights. https://github.com/peterdsharpe/NeuralFoil"),
    ("OpenRocket technical documentation", "", "Reference", "The drag build-up and stability methods follow Niskanen's documentation and the cited literature; no OpenRocket code is used."),
]


def license_text(dist):
    try:
        files = dist.files or []
    except Exception:  # noqa: BLE001
        return ""
    for f in files:
        name = f.name.lower()
        if name.startswith(("license", "copying", "licence")) and "test" not in str(f).lower():
            try:
                return Path(dist.locate_file(f)).read_text(encoding="utf-8", errors="replace").strip()
            except OSError:
                continue
    return ""


def main():
    lines = ["AeroSculpt third-party notices", "=" * 32, "",
             "AeroSculpt ships or loads the following software. Each remains under its own licence, reproduced", "below where the package carries one.", ""]
    for name in DISTS:
        try:
            d = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        lic = d.metadata.get("License-Expression") or d.metadata.get("License") or ""
        if not lic or len(lic) > 80:
            lic = "; ".join(c.split("::")[-1].strip() for c in d.metadata.get_all("Classifier") or [] if c.startswith("License"))
        lines += [f"{d.metadata['Name']} {d.version} — {lic or 'see package'}", "-" * 72]
        txt = license_text(d)
        lines += ([txt, ""] if txt else [d.metadata.get("Home-page") or d.metadata.get("Project-URL", "") or "", ""])
    for name, ver, lic, note in STATIC:
        lines += [f"{name} {ver} — {lic}".strip(), "-" * 72, note, ""]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} kB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
