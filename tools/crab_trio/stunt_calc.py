"""Ramp-jump stunt sizing for the crab cartwheel (docs/jumper_stunt_plan.md).

Rigid six-spoke star: mass 2.543 kg, inertia about the axle 0.0506 kg m^2 (in the
star pose, through the hub), spoke radius ~0.294 m. Prints:
 1. steady rolling speed down a slope (impact losses balance the drop per spoke),
 2. speed left at a kicker lip after climbing h,
 3. hub flight distance off a lip at angle th to a landing lip dh lower, with the
    landing speed and angle (vs the landing ramp's angle -> impact into the ramp).

  python tools/crab_trio/stunt_calc.py
"""
import math

G, M, I, R = 9.81, 2.543, 0.0506, 0.294
IC = I + M * R * R                 # about the contact point
K = IC / (M * R * R)               # rolling KE = K * (1/2) M v^2
C = 1 - M * R * R * (1 - math.cos(math.radians(60))) / IC   # spin kept per spoke change


def steady_speed(slope_deg):
    e = M * G * R * math.sin(math.radians(slope_deg)) / (1 - C * C)
    return math.sqrt(2 * e / IC) * R


def lip_speed(v, h):
    return math.sqrt(max(0.0, v * v - 2 * G * h / K))


def flight(v, th_deg, dh, pop=0.0):
    """Hub distance, flight time, landing speed and angle below horizontal; `pop` adds
    vertical speed from a leg push at the lip."""
    th = math.radians(th_deg)
    vx, vz = v * math.cos(th), v * math.sin(th) + pop
    t = (vz + math.sqrt(vz * vz + 2 * G * dh)) / G
    vzl = vz - G * t
    return vx * t, t, math.hypot(vx, vzl), math.degrees(math.atan2(-vzl, vx))


if __name__ == "__main__":
    print(f"rolling factor {K:.2f}; spin kept per spoke change {C:.3f} (energy {C * C:.2f})")
    print("1) rigid star steady speed (peak):", ", ".join(f"{s} deg {steady_speed(s):.2f} m/s" for s in (8, 12, 15, 20, 30)))
    print("2) lip speed after a kicker of rise h, from 2.0 m/s:", ", ".join(f"{h} m {lip_speed(2.0, h):.2f}" for h in (0.03, 0.05, 0.08, 0.1)))
    for v, th, dh, pop in ((1.4, 30, 0.2, 0), (2.0, 25, 0.1, 0), (1.5, 25, 0.1, 0.8), (2.0, 25, 0.1, 0.8)):
        x, t, vl, a = flight(v, th, dh, pop)
        print(f"3) lip {v} m/s at {th} deg, pop {pop} m/s, landing {dh} m lower: hub flies {x:.2f} m in {t:.2f} s, "
              f"lands {vl:.1f} m/s at {a:.0f} deg below horizontal")
