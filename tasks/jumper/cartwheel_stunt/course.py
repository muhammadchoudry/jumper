"""The stunt course: plateau, 15 deg hill, kicker, gap, landing ramp, run-out.

Sized in docs/jumper_stunt_plan.md (tools/crab_trio/stunt_calc.py). x runs along
the course in each env's own world (env_spacing 0: every env is its own MjData);
the reference's run-up and jump happen on the plateau (it lands in the star at
x ~ 1.75 m), and the hill starts at x = 2.4 m.

    plateau  z = 0            to x = 2.40
    hill     15 deg, 2.5 m    to (4.815, -0.647)
    flat     0.3 m            to (5.115, -0.647)
    kicker   12.5 then 25 deg, rise 6 cm, lip at (5.279, -0.587)
    gap      0.45 m (a pit 2.5 m down)
    landing  lip 10 cm lower at (5.729, -0.687), 40 deg down for 0.8 m
    run-out  flat at z = -1.201 to x = 30
"""

from __future__ import annotations

import math

import mujoco
import torch

PIT_Z = -2.5
HALF_WIDTH = 3.0
THICK = 0.3


def _pts():
    p = [(-6.0, 0.0), (2.4, 0.0)]

    def go(length, deg):
        x, z = p[-1]
        a = math.radians(deg)
        p.append((x + length * math.cos(a), z + length * math.sin(a)))

    go(2.5, -15.0)                               # hill
    go(0.3, 0.0)                                 # flat
    go(0.07, 12.5)                               # kicker foot
    rise = 0.06 - 0.07 * math.sin(math.radians(12.5))
    go(rise / math.sin(math.radians(25.0)), 25.0)  # kicker face
    return p


TAKEOFF = _pts()                                 # plateau ... kicker lip
LIP_X, LIP_Z = TAKEOFF[-1]
GAP = 0.45
LAND_X, LAND_Z = LIP_X + GAP, LIP_Z - 0.10
_a = math.radians(-40.0)
LANDING = [(LAND_X, LAND_Z), (LAND_X + 0.8 * math.cos(_a), LAND_Z + 0.8 * math.sin(_a))]
RUNOUT_X, RUNOUT_Z = LANDING[-1]
LANDING.append((30.0, RUNOUT_Z))
STAND_X = RUNOUT_X + 2.0                         # past here: tip down and stand


def _segment_box(body, x1, z1, x2, z2, name, **kw):
    dx, dz = x2 - x1, z2 - z1
    L = math.hypot(dx, dz)
    nx, nz = -dz / L, dx / L                     # surface normal (up)
    cx, cz = (x1 + x2) / 2 - nx * THICK / 2, (z1 + z2) / 2 - nz * THICK / 2
    phi = -math.atan2(dz, dx)
    body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=[L / 2, HALF_WIDTH, THICK / 2],
                  pos=[cx, 0.0, cz], quat=[math.cos(phi / 2), 0.0, math.sin(phi / 2), 0.0], **kw)


def add_course(spec: mujoco.MjSpec, body=None, material: str | None = None, **contact) -> None:
    """Add the course to `spec` (on `body`, default the world), and drop any ground
    plane to the bottom of the pit."""
    for g in spec.geoms:
        if g.type == mujoco.mjtGeom.mjGEOM_PLANE:
            g.pos = [g.pos[0], g.pos[1], PIT_Z]
            if not contact:
                contact = dict(contype=g.contype, conaffinity=g.conaffinity, condim=g.condim,
                               friction=list(g.friction), priority=g.priority)
                if material is None and g.material:
                    material = g.material
    body = body or spec.worldbody
    kw = dict(contact)
    if material:
        kw["material"] = material
    for i, ((x1, z1), (x2, z2)) in enumerate(zip(TAKEOFF[:-1], TAKEOFF[1:])):
        _segment_box(body, x1, z1, x2, z2, f"course_up_{i}", **kw)
    for i, ((x1, z1), (x2, z2)) in enumerate(zip(LANDING[:-1], LANDING[1:])):
        _segment_box(body, x1, z1, x2, z2, f"course_down_{i}", **kw)
    # pillars under the kicker and the landing ramp, down to the pit
    for name, (x0, x1), top in (("course_pillar_kick", (LIP_X - 0.45, LIP_X), LIP_Z - 0.12),
                                ("course_pillar_land", (LAND_X, LAND_X + 0.55), RUNOUT_Z - 0.05)):
        body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[(x1 - x0) / 2, HALF_WIDTH, (top - PIT_Z) / 2],
                      pos=[(x0 + x1) / 2, 0.0, (top + PIT_Z) / 2], **kw)


def spec_fn(spec: mujoco.MjSpec) -> None:
    """scene.spec_fn: the course on the terrain body (so contact sensors that match
    the terrain see it)."""
    terrain = None
    for b in spec.bodies:
        if b.name == "terrain":
            terrain = b
    add_course(spec, terrain)


_X = torch.tensor([p[0] for p in TAKEOFF] + [LAND_X - 1e-4] + [p[0] for p in LANDING])
_Z = torch.tensor([p[1] for p in TAKEOFF] + [PIT_Z] + [p[1] for p in LANDING])


def surface_z(x: torch.Tensor) -> torch.Tensor:
    """Course height under each x (the pit's floor over the gap)."""
    X, Z = _X.to(x.device, x.dtype), _Z.to(x.device, x.dtype)
    i = torch.searchsorted(X, x.contiguous()).clamp(1, len(X) - 1)
    x0, x1, z0, z1 = X[i - 1], X[i], Z[i - 1], Z[i]
    z = z0 + (z1 - z0) * ((x - x0) / (x1 - x0)).clamp(0, 1)
    gap = (x > LIP_X) & (x < LAND_X)
    return torch.where(gap, torch.full_like(z, PIT_Z), z)


if __name__ == "__main__":
    print("takeoff", [(round(a, 3), round(b, 3)) for a, b in TAKEOFF])
    print("landing", [(round(a, 3), round(b, 3)) for a, b in LANDING], "stand from x =", round(STAND_X, 2))
    print(surface_z(torch.tensor([0.0, 3.0, 5.2, 5.5, 6.0, 8.0])))
