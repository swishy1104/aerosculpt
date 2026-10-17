"""Model and high-power rocketry: the component methods every rocket tool is built on.

  drag.py        drag build-up (skin friction with finish, nose and boat-tail pressure drag, base
                 drag, fin leading/trailing edges, launch lug) over Mach number, after Hoerner,
                 Barrowman and the OpenRocket technical documentation (Niskanen 2013)
  stability.py   Barrowman centre of pressure with Mach corrections, centre of gravity from the
                 parts and the motor as it burns, static margin in calibres
  flutter.py     fin flutter speed (NACA TN 4197 as applied in model rocketry)
  motors.py      thrust curves: RASP .eng and RockSim .rse files, a small built-in library,
                 ThrustCurve.org search (only when asked)
  flight.py      flight simulation on a launch rail with wind and recovery: apogee, rail-exit
                 speed, max speed and Mach, max-Q, optimal delay, descent and drift
  ork.py         OpenRocket .ork and RockSim .rkt import, Cd-vs-Mach export
  model.py       ties them together for the workspace, the optimizer and the report

These are fast (microseconds to milliseconds), so the optimizer can search thousands of designs
for apogee or drag while keeping the static margin, fin flutter and rail-exit speed within limits.
The CFD tunnels remain available for flow pictures and shape comparison.
"""
