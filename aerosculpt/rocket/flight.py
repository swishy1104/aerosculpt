"""Flight simulation: a point mass with the rocket's drag curve, on a launch rail, in a steady
wind, with recovery under a parachute.

  rail      the rocket slides along the rail (tilted by the launch angle) until it has travelled
            the rail length; it leaves only when thrust exceeds the weight component
  flight    thrust along the body axis; after the rail the rocket points into the relative wind
            (a stable rocket weathercocks), so the angle of attack is zero and a crosswind tilts
            the flight path upwind. Drag = 1/2 rho V_rel^2 Cd(Mach, Re) A_ref.
  motor     thrust from the curve; mass falls with the impulse delivered
  air       ISA at the site elevation plus altitude (optionally a temperature offset)
  recovery  the parachute opens at ejection (the motor delay after burnout, or at apogee when the
            delay is "optimal"); the rocket then falls at the parachute's terminal speed and
            drifts with the wind until the ground

Integration: RK4, 5 ms during the burn, 20 ms on coast, 50 ms under the parachute. Checked
against the closed-form Fehskens-Malewicki solution (constant thrust, quadratic drag) on the
validation page.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..cfd.analysis import isa

G0 = 9.80665


@dataclass
class Launch:
    rail_m: float = 1.2
    angle_deg: float = 0.0          # from vertical; positive tilts into the wind (upwind)
    wind_m_s: float = 0.0
    elevation_m: float = 0.0
    temperature_c: float = None     # None: ISA
    delay_s: float = None           # None: optimal (ejection at apogee)
    constant_air: bool = False      # sea-level air all the way up (the closed-form validation case)


@dataclass
class Parachute:
    diameter_m: float = 0.45
    cd: float = 0.75

    @property
    def cda(self):
        return self.cd * math.pi * self.diameter_m ** 2 / 4.0


def air_at(z, launch: Launch):
    a = isa(launch.elevation_m + (0.0 if launch.constant_air else max(z, 0.0)))
    if launch.temperature_c is not None:
        # same pressure, the chosen temperature at the site: density and sound speed follow
        dT = (launch.temperature_c + 273.15) - isa(launch.elevation_m)["T"]
        T = a["T"] + dT
        rho = a["p"] / (287.053 * T)
        mu = 1.458e-6 * T ** 1.5 / (T + 110.4)
        a = {"T": T, "p": a["p"], "rho": rho, "mu": mu, "nu": mu / rho, "a": math.sqrt(1.4 * 287.053 * T)}
    return a


def simulate(cd_fn, a_ref, length, dry_mass, motor, launch: Launch, chute: Parachute = None, max_time=600.0,
             cn_alpha=None):
    """cd_fn(mach, re) -> Cd on a_ref. Returns the summary and a decimated trajectory."""
    th = math.radians(launch.angle_deg)
    u = np.array([-math.sin(th), math.cos(th)])          # rail direction (x downwind, z up)
    w = np.array([launch.wind_m_s, 0.0])
    burn = motor.burn_time

    def mass(t):
        return dry_mass + motor.mass(t)

    def accel(t, pos, vel, on_rail, chute_open):
        m = mass(t)
        air = air_at(pos[1], launch)
        vrel = vel - w
        speed = float(np.linalg.norm(vrel))
        T = motor.thrust(t)
        if on_rail:
            axis = u
        elif speed > 1e-6:
            axis = vrel / speed
        else:
            axis = u
        f = T * axis + np.array([0.0, -m * G0])
        if speed > 1e-9:
            if chute_open:
                cda = chute.cda + 0.5 * a_ref
            else:
                mach = speed / air["a"]
                re = air["rho"] * speed * length / air["mu"]
                cda = cd_fn(mach, re) * a_ref
            f -= 0.5 * air["rho"] * speed * cda * vrel
        if on_rail:
            along = float(np.dot(f, u))
            if along < 0 and float(np.dot(vel, u)) <= 1e-9:
                along = 0.0
            f = along * u
        return f / m, air, speed, m

    t = 0.0
    pos = np.zeros(2)
    vel = np.zeros(2)
    on_rail, chute_open = True, False
    traj = []
    ev = {"rail_exit": None, "burnout": None, "apogee": None, "deploy": None, "landing": None}
    max_q, max_v, max_mach, max_acc, max_g = 0.0, 0.0, 0.0, 0.0, 0.0
    v_at_apogee = None
    deploy_v = None
    steps = 0
    while t < max_time:
        in_burn = t < burn
        dt = 0.005 if in_burn else (0.05 if chute_open else 0.02)
        # RK4
        a1, air, speed, m = accel(t, pos, vel, on_rail, chute_open)
        k1v, k1p = a1, vel
        a2, _, _, _ = accel(t + dt / 2, pos + k1p * dt / 2, vel + k1v * dt / 2, on_rail, chute_open)
        k2v, k2p = a2, vel + k1v * dt / 2
        a3, _, _, _ = accel(t + dt / 2, pos + k2p * dt / 2, vel + k2v * dt / 2, on_rail, chute_open)
        k3v, k3p = a3, vel + k2v * dt / 2
        a4, _, _, _ = accel(t + dt, pos + k3p * dt, vel + k3v * dt, on_rail, chute_open)
        k4v, k4p = a4, vel + k3v * dt
        new_vel = vel + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        new_pos = pos + dt / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
        if on_rail:
            s = float(np.dot(new_vel, u))
            if s < 0:
                new_vel[:] = 0
                new_pos = pos
            if float(np.dot(new_pos, u)) >= launch.rail_m:
                on_rail = False
                ev["rail_exit"] = {"t": t + dt, "v": float(np.linalg.norm(new_vel)), "z": float(new_pos[1])}
        mach = speed / air["a"]
        q = 0.5 * air["rho"] * speed * speed
        acc = float(np.linalg.norm(a1))
        if not chute_open:
            if q > max_q:
                max_q = q
            if speed > max_v:
                max_v, max_mach = speed, mach
                air_v = air
                z_v = float(pos[1])
            if acc > max_acc and t < burn:
                max_acc = acc
        if steps % (2 if in_burn else 1) == 0 or len(traj) == 0:
            traj.append((t, float(pos[0]), float(pos[1]), float(np.linalg.norm(vel)), mach, acc / G0 if t < burn else -acc / G0, m))
        if ev["burnout"] is None and t + dt >= burn > 0:
            ev["burnout"] = {"t": burn, "z": float(new_pos[1]), "v": float(np.linalg.norm(new_vel))}
        if ev["apogee"] is None and not on_rail and new_vel[1] < 0 and vel[1] >= 0:
            ev["apogee"] = {"t": t + dt, "z": float(new_pos[1]), "x": float(new_pos[0])}
            v_at_apogee = float(np.linalg.norm(new_vel - w))
        if chute is not None and not chute_open and ev["burnout"] is not None:
            t_deploy = (burn + launch.delay_s) if launch.delay_s is not None else None
            if (t_deploy is not None and t + dt >= t_deploy) or (t_deploy is None and ev["apogee"] is not None):
                chute_open = True
                deploy_v = float(np.linalg.norm(new_vel - w))
                ev["deploy"] = {"t": t + dt, "z": float(new_pos[1]), "v": deploy_v}
        pos, vel, t = new_pos, new_vel, t + dt
        steps += 1
        if pos[1] < 0 and not on_rail:
            ev["landing"] = {"t": t, "x": float(pos[0]), "v": float(np.linalg.norm(vel))}
            break
        if ev["apogee"] is not None and chute is None and pos[1] <= 0:
            break
    if ev["apogee"] is None and ev["burnout"] is not None:
        ev["apogee"] = {"t": t, "z": float(pos[1]), "x": float(pos[0])}
    apogee = ev["apogee"] or {"t": 0.0, "z": 0.0, "x": 0.0}
    optimal_delay = max(0.0, apogee["t"] - burn) if ev["burnout"] else None
    out = {
        "apogee_m": apogee["z"], "apogee_t": apogee["t"], "apogee_x": apogee["x"],
        "max_speed": max_v, "max_mach": max_mach, "max_acceleration_g": max_acc / G0, "max_q_pa": max_q,
        "altitude_at_max_speed": z_v if max_v > 0 else 0.0,
        "rail_exit_speed": ev["rail_exit"]["v"] if ev["rail_exit"] else 0.0,
        "rail_exit_t": ev["rail_exit"]["t"] if ev["rail_exit"] else None,
        "left_rail": ev["rail_exit"] is not None,
        "burnout": ev["burnout"], "optimal_delay": optimal_delay, "delay_used": launch.delay_s,
        "speed_at_apogee": v_at_apogee, "deploy": ev["deploy"], "deploy_speed": deploy_v,
        "landing": ev["landing"], "drift_m": ev["landing"]["x"] if ev["landing"] else None,
        "descent_time": (ev["landing"]["t"] - apogee["t"]) if ev["landing"] else None,
        "flight_time": ev["landing"]["t"] if ev["landing"] else t,
        "liftoff_mass": dry_mass + motor.total_mass_kg, "burnout_mass": dry_mass + motor.case_mass_kg,
        "thrust_to_weight": motor.avg_thrust / ((dry_mass + motor.total_mass_kg) * G0),
        "peak_thrust_to_weight": motor.peak_thrust / ((dry_mass + motor.total_mass_kg) * G0),
    }
    out["air_at_max_speed"] = air_v if max_v > 0 else air_at(0.0, launch)
    # decimate for charts: about 400 points
    step = max(1, len(traj) // 400)
    out["trajectory"] = [{"t": round(p[0], 3), "x": round(p[1], 2), "z": round(p[2], 2), "v": round(p[3], 2),
                          "mach": round(p[4], 4), "g": round(p[5], 2), "m": round(p[6], 4)} for p in traj[::step]]
    return out


def malewicki(T, m, cd, a_ref, rho, burn):
    """Closed-form boost and coast altitude for constant thrust, mass and drag coefficient
    (Fehskens-Malewicki, Estes TR-10): the validation case for the integrator."""
    k = 0.5 * rho * cd * a_ref
    q = math.sqrt((T - m * G0) / k)
    x = math.sqrt(k * (T - m * G0)) / m
    v_b = q * math.tanh(x * burn)
    z_b = (m / k) * math.log(math.cosh(x * burn))
    z_c = (m / (2 * k)) * math.log(1 + k * v_b * v_b / (m * G0))
    return {"v_burnout": v_b, "z_burnout": z_b, "apogee": z_b + z_c}
