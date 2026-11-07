# tools/cartwheel: the crab cartwheel and ramp-jump stunt tools

The training tasks themselves are in `tasks/jumper/cartwheel*`. These are the scripts around them.

```
tools/cartwheel/
  make_videos.py   render (or re-record and render) any of the published videos   <- start here
  refs/            kinematic reference motions the tasks imitate / spawn from
  physics/         scripted physics experiments and sizing (no learning)
  eval/            evaluate and record trained checkpoints
  render/          videos, films and plots from recorded rollouts
  NOTES_folded_leg_rolling.md   a parked alternative (folded-leg rolling)
scripts/cartwheel/ training: setup.sh, train.sh, runs.py (every run), train.command (Mac launcher)
tools/fire_skin.py, tools/neon_reef.py   display-only robot liveries
tools/crab_trio/   the dance trio and the robot helpers shared with it (stage.py, jump.py)
```

| Folder | Script | What it does |
|---|---|---|
| refs | `cartwheel_ref.py` … `cartwheel_ref5.py` | References 1–5 → `tasks/jumper/<task>/ref/*.npz` (5 = ten turns and the figure 8) |
| | `roll_loop_ref.py` | Spawn states for `cartwheel_roll` (rigid star rolling on edge) |
| | `record_runup.py` | Records the trained walk as the run-up (`runup_walk.npz`, used by refs 3–5) |
| physics | `limb_wheel.py` | Scripted stepping/pumping wheel gait; the "rigid star can't coast" measurement |
| | `hill_roll.py` | Rigid star down a slope (it falls over sideways: needs a policy to balance) |
| | `stunt_calc.py` | Ramp-jump sizing: steady speed, kicker loss, flight, landing |
| | `cartwheel*.py`, `wheel*.py`, `straight.py` | Earlier open-loop searches and the straight-limb spoke helpers (with their saved params) |
| eval | `full_eval.py` | Evaluate any `cartwheel_*` / stunt checkpoint from standing (`STOCH=1` for action noise) |
| | `full_record.py` | Record the best of N attempts to a rollout npz (`STOCH=1` likewise) |
| | `roll_eval.py`, `roll_record.py` | The same for `cartwheel_roll` |
| render | `traj_video.py` | Rollout → mp4 + contact sheet (`TOPVIEW=1`, `SLOPE_DEG=8`, `COURSE=1` or `long`) |
| | `film_stunt.py` | *The Inferno Gap*: the cinematic multi-camera cut with set, fire, replays and score |
| | `ref_video.py` | Reference npz → mp4 |
| | `path_plot.py` | Path from above + heading plot (figure-8 runs) |
| | `film_cartwheel.py`, `film_wheel.py` | Films of the early scripted searches |

Run every script as a module from the repo root, with the venv active:

```bash
python -m tools.cartwheel.refs.cartwheel_ref5            # rebuild references 5 (ten turns, figure 8)
python -m tools.cartwheel.physics.stunt_calc
python -m tools.cartwheel.eval.full_eval <ckpt.pt> 16 jumper.cartwheel_stunt_long
MUJOCO_GL=egl python -m tools.cartwheel.render.traj_video <rollout.npz> out.mp4
python -m tools.cartwheel.make_videos list
```

Scripts import one another by package path, e.g. `from tools.cartwheel.physics import wheel_s`
and `from tools.crab_trio import stage`. That works because `python -m` from the
repo root puts the root on the import path. Nothing rewrites `sys.path`;
`tests/test_layout.py` checks that.

More detail is in these docs:
- **training:** `docs/cartwheel_training.md`;
- **what each run taught us:** `docs/jumper_cartwheel.md` and `docs/jumper_stunt_plan.md`;
- **the videos:** `docs/media/cartwheel/README.md`.
