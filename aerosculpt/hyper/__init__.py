"""Supersonic and hypersonic tunnel: engineering aerothermodynamics on the surface mesh.

The lattice-Boltzmann tunnel is incompressible, so speeds above Mach 1.2 use the classic
local-inclination panel methods of the S/HABP family instead of a flow solver:
tangent-cone / tangent-wedge and modified Newtonian on windward faces, Prandtl-Meyer
expansion on leeward faces, an empirical base pressure, Eckert reference-temperature skin
friction and heating, Fay-Riddell stagnation heating and radiative-equilibrium wall
temperature checked against the material's service limit.
"""
