"""Plot a recorded rollout (full_record.py's npz) from above: the hub's path, and
the travel heading and uprightness over time (the cartwheel_eight_*_path.png figures).

  python -m tools.cartwheel.render.path_plot docs/media/cartwheel/traj/eight_try3.npz out.png [t_go=0.3]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

D = np.load(sys.argv[1])
out = sys.argv[2]
t_go = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
q, dt = D["qs"], float(D["dt"])
t = np.arange(len(q)) * dt - t_go
w, x, y, z = q[:, 3], q[:, 4], q[:, 5], q[:, 6]
axle = np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], 1)   # body z
heading = np.degrees(np.unwrap(np.arctan2(-axle[:, 0], axle[:, 1])))                          # up x axle
heading -= heading[0]
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
a.plot(q[:, 0] - q[0, 0], q[:, 1] - q[0, 1], color="crimson")
a.plot(0, 0, "go")
a.plot(q[-1, 0] - q[0, 0], q[-1, 1] - q[0, 1], "kx")
a.set_aspect("equal")
a.set_title("path from above (m)")
a.grid(alpha=0.3)
b.plot(t, heading, color="crimson", label="heading (deg)")
b.plot(t, 100 * axle[:, 2], color="0.75", label="upright x100")
b.set_xlabel("s after go")
b.legend()
b.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(out, dpi=100)
print("ok", out)
