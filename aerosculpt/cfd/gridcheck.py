"""Grid convergence: the same case on three grids, Richardson extrapolation and the Grid
Convergence Index (Roache 1994; procedure of Celik et al. 2008, J. Fluids Eng. 130, 078001).

The study refines by a constant ratio r = h_coarse / h_fine >= 1.3 (cells x 2.2 per step), runs the
two coarser grids below the one the user chose, and reports the chosen grid's value with an error
band (GCI_fine, 95 % confidence with safety factor 1.25) and the extrapolated grid-independent value.
"""
from __future__ import annotations

import math

CELLS_STEP = 2.2            # cell-count ratio between grids: r = 2.2^(1/3) = 1.30
FS = 1.25                   # safety factor for three-grid studies


def gci(f1, f2, f3, h1, h2, h3):
    """f1 fine, f2 medium, f3 coarse values on grids of spacing h1 < h2 < h3."""
    r21 = h2 / h1
    r32 = h3 / h2
    e21 = f2 - f1
    e32 = f3 - f2
    out = {"values": [f1, f2, f3], "h": [h1, h2, h3], "r21": r21, "r32": r32}
    if abs(e21) < 1e-12 * max(abs(f1), 1e-12):
        out.update({"order": None, "extrapolated": f1, "gci_fine": 0.0, "behaviour": "converged",
                    "note": "the two finest grids agree exactly"})
        return out
    ratio = e32 / e21
    s = 1.0 if ratio > 0 else -1.0
    p = 2.0
    ok = True
    for _ in range(100):
        try:
            q = math.log((r21 ** p - s) / (r32 ** p - s))
            p_new = abs(math.log(abs(ratio)) + q) / math.log(r21)
        except (ValueError, ZeroDivisionError):
            ok = False
            break
        if abs(p_new - p) < 1e-8:
            p = p_new
            break
        p = p_new
    behaviour = "monotonic" if s > 0 else "oscillatory"
    note = None
    if not ok or not math.isfinite(p) or p < 0.5 or p > 6.0:
        # outside the asymptotic range: no reliable extrapolation; GCI with a conservative order 1
        note = ("the grids are not yet in the asymptotic range (apparent order "
                f"{p:.2f}), so the band uses order 1" if ok and math.isfinite(p) else
                "no consistent order of convergence: the band uses order 1")
        p_used = 1.0
    else:
        p_used = p
    ea21 = abs((f1 - f2) / f1) if f1 else abs(f1 - f2)
    g = FS * ea21 / (r21 ** p_used - 1.0)
    ext = (r21 ** p_used * f1 - f2) / (r21 ** p_used - 1.0)
    out.update({"order": p if (ok and math.isfinite(p)) else None, "order_used": p_used, "extrapolated": ext,
                "gci_fine": g, "error_fine": ea21, "behaviour": behaviour, "note": note})
    return out
