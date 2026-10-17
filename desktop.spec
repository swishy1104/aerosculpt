# PyInstaller build of AeroSculpt Desktop:   pyinstaller desktop.spec --noconfirm
# Output: dist/AeroSculpt/AeroSculpt.exe (one folder: zip it to share). See README "Desktop app".
from pathlib import Path

import os
import sys

import site

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_dynamic_libs, collect_submodules, copy_metadata

root = Path(SPECPATH)
# Anaconda keeps the Intel Fortran / OpenMP runtimes SciPy and NumPy load (libifcoremd, libmmd,
# svml_dispmd, libiomp5md) in Library/bin: put it on the search path so they are bundled
_libbin = Path(sys.prefix) / "Library" / "bin"
if _libbin.is_dir():
    os.environ["PATH"] = str(_libbin) + os.pathsep + os.environ.get("PATH", "")
datas = [(str(root / "aerosculpt" / "static"), "aerosculpt/static"),
         (str(root / "aerosculpt" / "aero" / "nf_weights"), "aerosculpt/aero/nf_weights"),
         (str(root / "aerosculpt" / "cfd" / "validation_results.json"), "aerosculpt/cfd")]
binaries = []
hidden = collect_submodules("aerosculpt") + ["run", "uvicorn.logging", "uvicorn.loops.auto", "uvicorn.protocols.http.auto",
                                             "uvicorn.protocols.websockets.auto", "uvicorn.lifespan.on",
                                             "skimage.measure", "skimage.measure._marching_cubes_lewiner", "jwt", "cryptography"]
for _stamp in (root / "aerosculpt" / "BUILD", root / "THIRD-PARTY-NOTICES.txt"):
    if _stamp.exists():
        datas.append((str(_stamp), "aerosculpt" if _stamp.name == "BUILD" else "."))
# gmsh (STEP/IGES import) is GPL: it is bundled only for in-house builds that honour that licence
BUNDLE_GMSH = bool(os.environ.get("AEROSCULPT_BUNDLE_GMSH"))
CPU_ONLY = bool(os.environ.get("AEROSCULPT_CPU_ONLY"))       # no CUDA pieces at all (the GPU pack ships separately)
if BUNDLE_GMSH:
    hidden.append("gmsh")
    binaries += collect_dynamic_libs("gmsh")
    try:
        import gmsh as _g
        _lib = Path(_g.__file__).parent
        # gmsh.py looks for its DLL beside itself (the bundle root) among other places; Anaconda keeps it in Lib/
        binaries += [(str(f), ".") for d in (_lib, _lib / "Lib", _lib.parent, _lib.parent.parent / "Library" / "bin") for f in d.glob("gmsh*.dll")]
    except Exception:
        pass
# the solver (numba, llvmlite), the GPU path (numba-cuda and NVIDIA's pip-packaged CUDA libraries),
# vortex surfaces (scikit-image), reports (matplotlib)
for pkg in ("numba", "llvmlite") + (() if CPU_ONLY else ("numba_cuda",)):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hidden += h
    except Exception:
        pass
# numba-cuda replaces numba's built-in numba.cuda through a site-packages .pth import hook, which a
# frozen app never runs: ship numba-cuda's package as numba/cuda on disk instead (numba's package
# path points there) and leave the old built-in one out (excludes below)
datas = [d for d in datas if not str(d[1]).replace("\\", "/").startswith("numba/cuda")]
binaries = [b for b in binaries if not str(b[1]).replace("\\", "/").startswith("numba/cuda")]
try:
    if not CPU_ONLY:
        import numba_cuda as _nc
        datas += [(str(Path(_nc.__file__).parent / "numba" / "cuda"), "numba/cuda")]
except Exception:
    pass
# NVIDIA's cuda.core picks its CUDA 12 or 13 build at import by swapping its own __path__ to cu12/ or cu13/,
# which a frozen importer cannot follow: ship the whole cuda namespace (core, bindings, pathfinder) as
# plain files that Python imports from disk, and keep it out of the archive (excludes below).
# Their delvewheel-repaired wheels load private DLLs (a hashed msvcp140, ...) from <pkg>.libs/ next to
# the package, found by a relative path, so those folders go in as they are.
for _sp in ([] if CPU_ONLY else site.getsitepackages()):
    if (Path(_sp) / "cuda" / "bindings").is_dir():
        datas += [(str(Path(_sp) / "cuda"), "cuda")]
        for _libs in Path(_sp).glob("*.libs"):
            if _libs.name.split(".")[0] in ("numba_cuda", "cuda_bindings", "cuda_core"):
                datas += [(str(_libs), _libs.name)]
        break
# cuda.pathfinder finds the CUDA DLLs through the installed distributions' file lists (dist-info RECORD),
# and numba-cuda locates nvvm and the runtime by distribution: keep that metadata
for dist in ([] if CPU_ONLY else ("cuda-bindings", "cuda-core", "cuda-pathfinder", "cuda-toolkit", "numba-cuda", "nvidia-cuda-nvrtc-cu12",
                                  "nvidia-cuda-runtime-cu12", "nvidia-nvjitlink-cu12", "nvidia-cuda-nvcc-cu12", "nvidia-cuda-cccl-cu12")):
    try:
        datas += copy_metadata(dist)
    except Exception:
        pass
# NVIDIA's pip-packaged CUDA pieces, in their own layout (numba-cuda finds nvvm/bin, nvvm/libdevice
# and the runtime DLLs by these relative paths); libdevice*.bc is a data file the compiler needs
for pkg in ([] if CPU_ONLY else ("nvidia.cuda_nvrtc", "nvidia.cuda_runtime", "nvidia.nvjitlink", "nvidia.cuda_nvcc", "nvidia.cuda_cccl")):
    try:
        binaries += collect_dynamic_libs(pkg)
        datas += collect_data_files(pkg, include_py_files=False)
    except Exception:
        pass

a = Analysis([str(root / "desktop.py")], pathex=[str(root)], binaries=binaries, datas=datas, hiddenimports=hidden,
             excludes=["tkinter", "IPython", "jupyter", "jupyter_client", "jupyter_core", "notebook", "nbformat", "nbconvert", "ipykernel",
                       "zmq", "tornado", "pytest", "PyQt5", "PyQt6", "PySide2", "PySide6", "wx", "gi", "torch", "tensorflow", "cv2",
                       "dask", "distributed", "h5py", "tables", "pandas", "bokeh", "panel", "holoviews", "hvplot", "xarray",
                       "pyarrow", "fsspec", "s3fs", "botocore", "boto3", "sqlalchemy", "sympy", "statsmodels", "sklearn",
                       "astropy", "lxml", "openpyxl", "xlrd", "sphinx", "babel", "docutils", "jedi", "parso", "numpydoc",
                       "Cython", "black", "mypy", "pylint", "imageio", "tifffile", "networkx", "numexpr", "bottleneck",
                       "pywt", "skimage.io", "skimage.data", "skimage.viewer", "matplotlib.tests", "scipy.tests", "numba.tests",
                       "numba.cuda", "cuda"] + ([] if BUNDLE_GMSH else ["gmsh"]),
             noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="AeroSculpt", console=True, icon=None)
coll = COLLECT(exe, a.binaries, a.datas, name="AeroSculpt")
