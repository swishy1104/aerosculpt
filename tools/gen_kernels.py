"""Generate the unrolled D3Q19 lattice-Boltzmann kernels.

The stream+collide kernel is written out with all 19 populations as scalars so numba
keeps them in registers. Run this script whenever the lattice or collision model changes:

    python tools/gen_kernels.py

It overwrites aerosculpt/cfd/kernels.py. Do not edit that file by hand.
"""
from pathlib import Path

C = [(0, 0, 0),
     (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
     (1, 1, 0), (-1, -1, 0), (1, -1, 0), (-1, 1, 0),
     (1, 0, 1), (-1, 0, -1), (1, 0, -1), (-1, 0, 1),
     (0, 1, 1), (0, -1, -1), (0, 1, -1), (0, -1, 1)]
W = [1 / 3] + [1 / 18] * 6 + [1 / 36] * 12
OPP = [C.index((-a, -b, -c)) for (a, b, c) in C]
Q = len(C)

AX = ("x", "y", "z")


def src(i):
    """Index expressions of the cell a population with velocity c_i is pulled from (x - c_i)."""
    out = []
    for a, c in zip(AX, C[i]):
        out.append(a if c == 0 else (f"{a}m" if c == 1 else f"{a}p"))
    return ", ".join(out)


def away(i):
    """Index expressions of x + c_i (the fluid neighbour one link further from a wall)."""
    out = []
    for a, c in zip(AX, C[i]):
        out.append(a if c == 0 else (f"{a}p" if c == 1 else f"{a}m"))
    return ", ".join(out)


def lin(terms):
    """Build a +/- sum string from (coef, name) pairs, dropping zeros."""
    s = ""
    for coef, name in terms:
        if coef == 0:
            continue
        if coef == 1:
            s += f" + {name}"
        elif coef == -1:
            s += f" - {name}"
        else:
            s += f" + {coef}*{name}"
    s = s.strip()
    if s.startswith("+ "):
        s = s[2:]
    return s or "0.0"


def gen_gather_plain(ind):
    return [f"{ind}f{i} = f[{i}, {src(i)}]" for i in range(Q)]


def gen_gather_wall(ind):
    """Bouzidi linear interpolated bounce-back for every link that crosses the wall, plus the
    momentum-exchange force and moment of that link."""
    L = [f"{ind}f0 = f[0, x, y, z]"]
    for i in range(1, Q):
        j = OPP[i]
        cx, cy, cz = C[i]
        L += [
            f"{ind}q = qtab[b, {i}]",
            f"{ind}if q < 0.0:",
            f"{ind}    f{i} = f[{i}, {src(i)}]",
            f"{ind}else:",
            f"{ind}    fj = f[{j}, x, y, z]",
            f"{ind}    if q < 0.5:",
            f"{ind}        if flags[{away(i)}] != 1:",
            f"{ind}            f{i} = 2.0 * q * fj + (1.0 - 2.0 * q) * f[{j}, {away(i)}]",
            f"{ind}        else:",
            f"{ind}            f{i} = fj",
            f"{ind}    else:",
            f"{ind}        f{i} = fj / (2.0 * q) + (1.0 - 1.0 / (2.0 * q)) * f[{i}, x, y, z]",
            f"{ind}    s = fj + f{i}",
        ]
        # force on the body from this link: c_j (fj + fi) with c_j = -c_i, applied at x - q c_i
        dF = [-cx, -cy, -cz]
        L += [f"{ind}    dFx = {dF[0]}.0 * s",
              f"{ind}    dFy = {dF[1]}.0 * s",
              f"{ind}    dFz = {dF[2]}.0 * s",
              f"{ind}    Fx += dFx",
              f"{ind}    Fy += dFy",
              f"{ind}    Fz += dFz",
              f"{ind}    rx = x - q * {cx}.0 - refx",
              f"{ind}    ry = y - q * {cy}.0 - refy",
              f"{ind}    rz = z - q * {cz}.0 - refz",
              f"{ind}    Mx += ry * dFz - rz * dFy",
              f"{ind}    My += rz * dFx - rx * dFz",
              f"{ind}    Mz += rx * dFy - ry * dFx"]
    return L


def gen_moments(ind):
    L = [f"{ind}rho = " + " + ".join(f"f{i}" for i in range(Q))]
    for k, a in enumerate(AX):
        L.append(f"{ind}j{a} = " + lin([(C[i][k], f"f{i}") for i in range(Q)]))
    return L


def gen_collide(ind):
    L = gen_moments(ind)
    L += [
        f"{ind}irho = 1.0 / rho",
        f"{ind}ux = jx * irho",
        f"{ind}uy = jy * irho",
        f"{ind}uz = jz * irho",
    ]
    pairs = [("xx", 0, 0), ("yy", 1, 1), ("zz", 2, 2), ("xy", 0, 1), ("xz", 0, 2), ("yz", 1, 2)]
    for name, a, b in pairs:
        terms = [(C[i][a] * C[i][b], f"f{i}") for i in range(Q)]
        L.append(f"{ind}P{name} = " + lin(terms))
    L += [
        f"{ind}Nxx = Pxx - rho * (0.3333333333333333 + ux * ux)",
        f"{ind}Nyy = Pyy - rho * (0.3333333333333333 + uy * uy)",
        f"{ind}Nzz = Pzz - rho * (0.3333333333333333 + uz * uz)",
        f"{ind}Nxy = Pxy - rho * ux * uy",
        f"{ind}Nxz = Pxz - rho * ux * uz",
        f"{ind}Nyz = Pyz - rho * uy * uz",
        f"{ind}Qn = math.sqrt(Nxx * Nxx + Nyy * Nyy + Nzz * Nzz + 2.0 * (Nxy * Nxy + Nxz * Nxz + Nyz * Nyz))",
        f"{ind}tau = 0.5 * (tau0 + math.sqrt(tau0sq + smag * Qn * irho))",
        f"{ind}om1 = 1.0 - 1.0 / tau",
        f"{ind}usq = 1.5 * (ux * ux + uy * uy + uz * uz)",
        f"{ind}tr = Nxx + Nyy + Nzz",
        # bulk (trace) part relaxed separately with om1b: damps sound, leaves shear alone
        f"{ind}trb = tr * 0.3333333333333333 * (om1b - om1)",
    ]
    for i in range(Q):
        cx, cy, cz = C[i]
        w = W[i]
        cu = lin([(cx, "ux"), (cy, "uy"), (cz, "uz")])
        # H_i : N = sum c_a c_b N_ab - tr/3
        hterms = []
        if cx:
            hterms.append("Nxx")
        if cy:
            hterms.append("Nyy")
        if cz:
            hterms.append("Nzz")
        cross = []
        if cx and cy:
            cross.append((cx * cy * 2, "Nxy"))
        if cx and cz:
            cross.append((cx * cz * 2, "Nxz"))
        if cy and cz:
            cross.append((cy * cz * 2, "Nyz"))
        H = " + ".join(hterms) if hterms else "0.0"
        cr = lin(cross)
        Hs = f"({H}" + (f" + ({cr})" if cross else "") + " - tr * 0.3333333333333333)"
        L.append(f"{ind}cu = {cu}")
        c2m1 = cx * cx + cy * cy + cz * cz - 1
        bulk = f" + {4.5 * w * c2m1!r} * trb" if c2m1 else ""
        L.append(
            f"{ind}fn[{i}, x, y, z] = {w!r} * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - usq)"
            f" + om1 * {4.5 * w!r} * {Hs}{bulk}")
    return L


def gen_stream_collide():
    I3 = " " * 16
    L = [
        "@njit(parallel=True, fastmath=True, cache=True)",
        "def stream_collide(f, fn, flags, bidx, qtab, tau_x, tau_y, tau_z, feq_ff, smag, tau_bulk, ref, fpart):",
        '    """One fused pull-stream + regularized-BGK/Smagorinsky collision step.',
        "",
        "    f holds post-collision populations of the previous step; fn receives the new ones.",
        "    flags: 0 fluid, 1 solid, 2 far-field (Dirichlet equilibrium), 3 outlet.",
        "    Wall links use Bouzidi linear interpolated bounce-back with fractions qtab[b, i].",
        "    The stress trace relaxes with tau_bulk (Dellar bulk viscosity) to damp acoustics.",
        "    fpart[x] accumulates momentum-exchange force and moment (about ref) per x-slab.",
        '    """',
        "    nx, ny, nz = flags.shape",
        "    refx = ref[0]",
        "    refy = ref[1]",
        "    refz = ref[2]",
        "    om1b = 1.0 - 1.0 / tau_bulk",
        "    for x in prange(nx):",
        "        tau0x = tau_x[x]",
        "        Fx = 0.0",
        "        Fy = 0.0",
        "        Fz = 0.0",
        "        Mx = 0.0",
        "        My = 0.0",
        "        Mz = 0.0",
        "        xm = x - 1 if x > 0 else nx - 1",
        "        xp = x + 1 if x < nx - 1 else 0",
        "        for y in range(ny):",
        "            ym = y - 1 if y > 0 else ny - 1",
        "            yp = y + 1 if y < ny - 1 else 0",
        "            tau0y = max(tau0x, tau_y[y])",
        "            for z in range(nz):",
        "                zm = z - 1 if z > 0 else nz - 1",
        "                zp = z + 1 if z < nz - 1 else 0",
        "                tau0 = max(tau0y, tau_z[z])",
        "                tau0sq = tau0 * tau0",
        "                t = flags[x, y, z]",
        "                if t == 1:",
        "                    continue",
        "                if t == 2:",
    ]
    for i in range(Q):
        L.append(f"                    fn[{i}, x, y, z] = feq_ff[{i}]")
    L += [
        "                    continue",
        "                b = bidx[x, y, z]",
        "                if b < 0:",
    ]
    L += gen_gather_plain(I3 + "    ")
    L += ["                else:"]
    L += gen_gather_wall(I3 + "    ")
    L += gen_collide(I3)
    L += [
        "        fpart[x, 0] = Fx",
        "        fpart[x, 1] = Fy",
        "        fpart[x, 2] = Fz",
        "        fpart[x, 3] = Mx",
        "        fpart[x, 4] = My",
        "        fpart[x, 5] = Mz",
    ]
    return L


def gen_accumulate():
    I = " " * 16
    L = [
        "@njit(parallel=True, fastmath=True, cache=True)",
        "def accumulate(f, flags, acc):",
        '    """Add rho, ux, uy, uz of every non-solid cell into acc[4, nx, ny, nz]."""',
        "    nx, ny, nz = flags.shape",
        "    for x in prange(nx):",
        "        for y in range(ny):",
        "            for z in range(nz):",
        "                if flags[x, y, z] == 1:",
        "                    continue",
    ]
    for i in range(Q):
        L.append(f"{I}f{i} = f[{i}, x, y, z]")
    L += gen_moments(I)
    L += [
        f"{I}irho = 1.0 / rho",
        f"{I}acc[0, x, y, z] += rho",
        f"{I}acc[1, x, y, z] += jx * irho",
        f"{I}acc[2, x, y, z] += jy * irho",
        f"{I}acc[3, x, y, z] += jz * irho",
    ]
    return L


HEADER = '''"""GENERATED by tools/gen_kernels.py -- do not edit by hand.

D3Q19 lattice-Boltzmann kernels: regularized BGK collision with a Smagorinsky
sub-grid model, Bouzidi interpolated bounce-back walls, momentum-exchange forces.
"""
import math

import numpy as np
from numba import njit, prange

CX = np.array({cx}, dtype=np.int64)
CY = np.array({cy}, dtype=np.int64)
CZ = np.array({cz}, dtype=np.int64)
W = np.array({w}, dtype=np.float64)
OPP = np.array({opp}, dtype=np.int64)
Q = {q}


def feq_vector(rho, ux, uy, uz):
    """Equilibrium populations for one state, as a float32 array of length 19."""
    out = np.empty(Q, dtype=np.float32)
    usq = 1.5 * (ux * ux + uy * uy + uz * uz)
    for i in range(Q):
        cu = CX[i] * ux + CY[i] * uy + CZ[i] * uz
        out[i] = W[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - usq)
    return out


@njit(parallel=True, cache=True)
def fill_equilibrium(f, flags, feq, only_solid_to_fluid, prev_flags):
    """Set cells to the equilibrium feq. With only_solid_to_fluid, touch only cells that
    were solid in prev_flags and are now fluid (warm-start after a geometry change)."""
    nx, ny, nz = flags.shape
    for x in prange(nx):
        for y in range(ny):
            for z in range(nz):
                if only_solid_to_fluid:
                    if not (prev_flags[x, y, z] == 1 and flags[x, y, z] != 1):
                        continue
                for i in range(19):
                    f[i, x, y, z] = feq[i]


@njit(parallel=True, cache=True)
def outlet_copy(fn, flags):
    """Zero-gradient outlet: the last slab copies the populations of the one before it."""
    nx, ny, nz = flags.shape
    for y in prange(ny):
        for z in range(nz):
            if flags[nx - 1, y, z] == 3:
                for i in range(19):
                    fn[i, nx - 1, y, z] = fn[i, nx - 2, y, z]

'''


def main():
    out = HEADER.format(
        cx=[c[0] for c in C], cy=[c[1] for c in C], cz=[c[2] for c in C],
        w=W, opp=OPP, q=Q)
    out += "\n" + "\n".join(gen_stream_collide()) + "\n\n\n" + "\n".join(gen_accumulate()) + "\n"
    target = Path(__file__).resolve().parents[1] / "aerosculpt" / "cfd" / "kernels.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(out, encoding="utf-8")
    print(f"wrote {target} ({len(out.splitlines())} lines)")


if __name__ == "__main__":
    main()
