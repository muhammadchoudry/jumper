"""The stunt courses: plateau, hill, kicker, gap, landing ramp, run-out.

Sized in docs/jumper_stunt_plan.md (tools/cartwheel/physics/stunt_calc.py). x runs along
the course in each env's own world (env_spacing 0: every env is its own MjData);
the reference's run-up and jump happen on the plateau (it lands in the star at
x ~ 1.75 m), and the hill starts at x = 2.4 m.

DEFAULT (jumper.cartwheel_stunt_pop / _fast):
    plateau  z = 0            to x = 2.40
    hill     15 deg, 2.5 m    to (4.815, -0.647)
    flat     0.3 m
    kicker   12.5 then 25 deg, rise 6 cm, lip at (5.279, -0.587)
    gap      0.45 m (a pit 2.5 m down)
    landing  lip 10 cm lower, 40 deg down for 0.8 m
    run-out  flat for 25 m

LONG (jumper.cartwheel_stunt_long): an 18 deg hill, a 0.75 m gap -- wider than the
crab is across -- with the landing lip 0.4 m below the kicker's, and a 45 deg
landing ramp easing to 20 deg before the run-out.

`COURSES[name]`; the module-level names (LIP_X, surface_z, spec_fn, ...) are
DEFAULT's, as before.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import torch

PIT_Z = -2.5
HALF_WIDTH = 3.0
THICK = 0.3


@dataclass
class Course:
    hill_deg: float = 15.0
    hill_len: float = 2.5
    gap: float = 0.45
    drop: float = 0.10                 # landing lip below the kicker lip
    land: tuple = ((40.0, 0.8),)       # landing ramp: (deg down, length) segments
    half_width: float = HALF_WIDTH
    TAKEOFF: list = field(init=False)
    LANDING: list = field(init=False)

    def __post_init__(self):
        p = [(-6.0, 0.0), (2.4, 0.0)]

        def go(pts, length, deg):
            x, z = pts[-1]
            a = math.radians(deg)
            pts.append((x + length * math.cos(a), z + length * math.sin(a)))

        go(p, self.hill_len, -self.hill_deg)       # hill
        go(p, 0.3, 0.0)                            # flat
        go(p, 0.07, 12.5)                          # kicker foot
        rise = 0.06 - 0.07 * math.sin(math.radians(12.5))
        go(p, rise / math.sin(math.radians(25.0)), 25.0)   # kicker face
        self.TAKEOFF = p
        self.LIP_X, self.LIP_Z = p[-1]
        self.LAND_X, self.LAND_Z = self.LIP_X + self.gap, self.LIP_Z - self.drop
        q = [(self.LAND_X, self.LAND_Z)]
        for deg, length in self.land:
            go(q, length, -deg)
        self.RUNOUT_X, self.RUNOUT_Z = q[-1]
        q.append((self.RUNOUT_X + 25.0, self.RUNOUT_Z))
        self.LANDING = q
        self.STAND_X = self.RUNOUT_X + 2.0         # past here: tip down and stand
        self._X = torch.tensor([a for a, _ in p] + [self.LAND_X - 1e-4] + [a for a, _ in q])
        self._Z = torch.tensor([b for _, b in p] + [PIT_Z] + [b for _, b in q])

    def _box(self, body, x1, z1, x2, z2, name, y0, hw, **kw):
        dx, dz = x2 - x1, z2 - z1
        L = math.hypot(dx, dz)
        nx, nz = -dz / L, dx / L                   # surface normal (up)
        cx, cz = (x1 + x2) / 2 - nx * THICK / 2, (z1 + z2) / 2 - nz * THICK / 2
        phi = -math.atan2(dz, dx)
        body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=[L / 2, hw, THICK / 2],
                      pos=[cx, y0, cz], quat=[math.cos(phi / 2), 0.0, math.sin(phi / 2), 0.0], **kw)

    def add_course(self, spec: mujoco.MjSpec, body=None, material: str | None = None, y0: float = 0.0,
                   half_width: float | None = None, **contact) -> None:
        """Add the course to `spec` (on `body`, default the world), centred on y = y0
        (renders only), and drop any ground plane to the bottom of the pit."""
        hw = half_width or self.half_width
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
        for i, ((x1, z1), (x2, z2)) in enumerate(zip(self.TAKEOFF[:-1], self.TAKEOFF[1:])):
            self._box(body, x1, z1, x2, z2, f"course_up_{i}", y0, hw, **kw)
        for i, ((x1, z1), (x2, z2)) in enumerate(zip(self.LANDING[:-1], self.LANDING[1:])):
            self._box(body, x1, z1, x2, z2, f"course_down_{i}", y0, hw, **kw)
        # pillars under the kicker and the landing ramp, down to the pit
        x_land = self.LANDING[1][0]
        for name, (x0, x1), top in (("course_pillar_kick", (self.LIP_X - 0.45, self.LIP_X), self.LIP_Z - 0.12),
                                    ("course_pillar_land", (self.LAND_X, self.LAND_X + 0.7 * (x_land - self.LAND_X)),
                                     self.RUNOUT_Z - 0.05)):
            body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=[(x1 - x0) / 2, hw, (top - PIT_Z) / 2],
                          pos=[(x0 + x1) / 2, y0, (top + PIT_Z) / 2], **kw)

    def spec_fn(self, spec: mujoco.MjSpec) -> None:
        """scene.spec_fn: the course on the terrain body (so contact sensors that match
        the terrain see it)."""
        terrain = None
        for b in spec.bodies:
            if b.name == "terrain":
                terrain = b
        self.add_course(spec, terrain)

    def surface_z(self, x: torch.Tensor) -> torch.Tensor:
        """Course height under each x (the pit's floor over the gap)."""
        X, Z = self._X.to(x.device, x.dtype), self._Z.to(x.device, x.dtype)
        i = torch.searchsorted(X, x.contiguous()).clamp(1, len(X) - 1)
        x0, x1, z0, z1 = X[i - 1], X[i], Z[i - 1], Z[i]
        z = z0 + (z1 - z0) * ((x - x0) / (x1 - x0)).clamp(0, 1)
        gap = (x > self.LIP_X) & (x < self.LAND_X)
        return torch.where(gap, torch.full_like(z, PIT_Z), z)


DEFAULT = Course()
LONG = Course(hill_deg=18.0, gap=0.75, drop=0.40, land=((45.0, 0.9), (20.0, 0.4)))
COURSES = {"default": DEFAULT, "long": LONG}
#: which course each stunt task runs on
TASK_COURSE = {"jumper.cartwheel_stunt_pop": "default", "jumper.cartwheel_stunt_fast": "default",
               "jumper.cartwheel_stunt_long": "long"}


def for_task(task: str) -> Course:
    return COURSES[TASK_COURSE.get(task, "default")]


# module-level names: DEFAULT's (the original course)
TAKEOFF, LANDING = DEFAULT.TAKEOFF, DEFAULT.LANDING
LIP_X, LIP_Z, LAND_X, LAND_Z = DEFAULT.LIP_X, DEFAULT.LIP_Z, DEFAULT.LAND_X, DEFAULT.LAND_Z
RUNOUT_X, RUNOUT_Z, STAND_X, GAP = DEFAULT.RUNOUT_X, DEFAULT.RUNOUT_Z, DEFAULT.STAND_X, DEFAULT.gap
add_course, spec_fn, surface_z = DEFAULT.add_course, DEFAULT.spec_fn, DEFAULT.surface_z


if __name__ == "__main__":
    for name, c in COURSES.items():
        print(name, "takeoff", [(round(a, 3), round(b, 3)) for a, b in c.TAKEOFF])
        print(name, "landing", [(round(a, 3), round(b, 3)) for a, b in c.LANDING], "stand from x =", round(c.STAND_X, 2))
