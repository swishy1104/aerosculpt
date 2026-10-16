# PyInstaller build of AeroSculpt Desktop:   pyinstaller desktop.spec --noconfirm
# Output: dist/AeroSculpt/AeroSculpt.exe (one folder: zip it to share). See README "Desktop app".
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs, collect_submodules

root = Path(SPECPATH)
datas = [(str(root / "aerosculpt" / "static"), "aerosculpt/static"),
         (str(root / "aerosculpt" / "aero" / "nf_weights"), "aerosculpt/aero/nf_weights"),
         (str(root / "aerosculpt" / "cfd" / "validation_results.json"), "aerosculpt/cfd")]
binaries = []
hidden = collect_submodules("aerosculpt") + ["run", "uvicorn.logging", "uvicorn.loops.auto", "uvicorn.protocols.http.auto",
                                             "uvicorn.protocols.websockets.auto", "uvicorn.lifespan.on",
                                             "skimage.measure", "skimage.measure._marching_cubes_lewiner", "jwt", "gmsh"]
binaries += collect_dynamic_libs("gmsh")
try:
    import gmsh as _g
    _lib = Path(_g.__file__).parent
    # gmsh.py looks for its DLL beside itself (the bundle root) among other places; Anaconda keeps it in Lib/
    binaries += [(str(f), ".") for d in (_lib, _lib / "Lib", _lib.parent, _lib.parent.parent / "Library" / "bin") for f in d.glob("gmsh*.dll")]
except Exception:
    pass
# the solver (numba, llvmlite), the GPU path (numba-cuda and NVIDIA's pip-packaged CUDA libraries),
# CAD exchange (gmsh with OpenCASCADE), vortex surfaces (scikit-image), reports (matplotlib)
for pkg in ("numba", "llvmlite", "numba_cuda", "cuda"):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hidden += h
    except Exception:
        pass
for pkg in ("nvidia.cuda_nvrtc", "nvidia.cuda_runtime", "nvidia.nvjitlink", "nvidia.cuda_nvcc"):
    try:
        binaries += [(src, f"nvidia/{pkg.split('.')[1]}/bin") for src, _ in collect_dynamic_libs(pkg)]
    except Exception:
        pass

a = Analysis([str(root / "desktop.py")], pathex=[str(root)], binaries=binaries, datas=datas, hiddenimports=hidden,
             excludes=["tkinter", "IPython", "jupyter", "jupyter_client", "jupyter_core", "notebook", "nbformat", "nbconvert", "ipykernel",
                       "zmq", "tornado", "pytest", "PyQt5", "PyQt6", "PySide2", "PySide6", "wx", "gi", "torch", "tensorflow", "cv2",
                       "dask", "distributed", "h5py", "tables", "pandas", "bokeh", "panel", "holoviews", "hvplot", "xarray",
                       "pyarrow", "fsspec", "s3fs", "botocore", "boto3", "sqlalchemy", "sympy", "statsmodels", "sklearn",
                       "astropy", "lxml", "openpyxl", "xlrd", "sphinx", "babel", "docutils", "jedi", "parso", "numpydoc",
                       "Cython", "black", "mypy", "pylint", "imageio", "tifffile", "networkx", "numexpr", "bottleneck",
                       "pywt", "skimage.io", "skimage.data", "skimage.viewer", "matplotlib.tests", "scipy.tests", "numba.tests"],
             noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="AeroSculpt", console=True, icon=None)
coll = COLLECT(exe, a.binaries, a.datas, name="AeroSculpt")
