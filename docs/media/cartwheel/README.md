# Cartwheel and stunt videos: how each was made

Every video here is rendered from a **recorded rollout**: the robot's state at
every physics step from one simulated attempt by a trained policy. The rollouts
are saved in `traj/`, so each video can be re-rendered exactly without re-running
the simulation. To record a fresh rollout instead, use the `full_record.py`
command given for it. Rollouts can differ slightly between machines, which is why
the recordings are kept.

**The easy way:** `tools/cartwheel/make_videos.py` knows every video below.

```bash
python -m tools.cartwheel.make_videos list                  # what's here; * = the key videos
python -m tools.cartwheel.make_videos key                   # render the key videos from the saved rollouts
python -m tools.cartwheel.make_videos inferno_gap           # one video (exactly as published)
python -m tools.cartwheel.make_videos stunt_long --rerecord # record a fresh attempt first, then render
python -m tools.cartwheel.make_videos all --out /tmp/videos # everything, somewhere else
```

The table below is the same information, for running the steps by hand. Run
everything from the repo root with the venv active. Prefix render commands with
`MUJOCO_GL=egl` on a headless machine (`make_videos.py` sets that itself).

## Recorded rollouts → videos

| Video | Rollout (`traj/`) | Render | Re-record |
|---|---|---|---|
| `cartwheel_full_round8899.mp4` | `full_round8899.npz` | `traj_video.py` | `full_record.py tasks/jumper/cartwheel_full/checkpoints/full_v2_model_8899.pt out.npz` |
| `cartwheel_full_v3_round12898.mp4` | `full_v3_round12898.npz` | `traj_video.py` | `full_record.py tasks/jumper/cartwheel_full/checkpoints/full_v3_model_12898.pt out.npz` |
| `cartwheel_ten_turns_straight.mp4` | `ten_turns.npz` | `traj_video.py` | `full_record.py …/full_v3_model_12898.pt out.npz 4 0 jumper.cartwheel_ten` |
| `cartwheel_eight_attempt1_top.mp4`, `_path.png` | `eight_attempt1.npz` | `TOPVIEW=1 traj_video.py`; `path_plot.py` | `full_record.py <eight_v1 ckpt> out.npz N seed jumper.cartwheel_eight` |
| `cartwheel_eight_try2_top.mp4`, `_path.png` | `eight_try2.npz` | same | same, eight_v2 |
| `cartwheel_eight_try3_top.mp4`, `_path.png` | `eight_try3.npz` | same | same, eight_v3 |
| `cartwheel_ten_turns_straight_legs.mp4` + sheet | `ten_turns_straight_legs.npz` | `traj_video.py` | `full_record.py tasks/jumper/cartwheel_ten_straight/checkpoints/straight_v1_model_15897.pt out.npz 8 0 jumper.cartwheel_ten_straight` |
| `cartwheel_ten_turns_star_legs.mp4` + sheet | `ten_turns_star_legs.npz` | `traj_video.py` | `full_record.py tasks/jumper/cartwheel_ten_star/checkpoints/star_v1_model_18896.pt out.npz 6 0 jumper.cartwheel_ten_star` |
| `cartwheel_hill.mp4` + sheet | `hill.npz` | `SLOPE_DEG=8 traj_video.py` | `full_record.py tasks/jumper/cartwheel_hill/checkpoints/hill_v1_model_20600.pt out.npz 4 0 jumper.cartwheel_hill` |
| `cartwheel_stunt_pop.mp4` + sheet | `stunt_pop.npz` | `COURSE=1 traj_video.py` | `full_record.py tasks/jumper/cartwheel_stunt_pop/checkpoints/pop_v1_model_23599.pt out.npz 16 3 jumper.cartwheel_stunt_pop` |
| `cartwheel_stunt_fast.mp4` + sheet | `stunt_fast.npz` | `COURSE=1 traj_video.py` | `full_record.py tasks/jumper/cartwheel_stunt_fast/checkpoints/fast_v1_model_22300.pt out.npz 16 5 jumper.cartwheel_stunt_fast` |
| `cartwheel_stunt_long.mp4` + sheet | `stunt_long.npz` | `COURSE=long traj_video.py` | `full_record.py tasks/jumper/cartwheel_stunt_long/checkpoints/long_v1_model_25299.pt out.npz 16 7 jumper.cartwheel_stunt_long` |
| **`inferno_gap.mp4`** + sheet | `stunt_long.npz` | `film_stunt.py` (below) | as above |

Notes on the table:
- **Rendering:** `traj_video.py` is `python -m tools.cartwheel.render.traj_video <rollout.npz> <out.mp4>`. It also writes `<out>_sheet.png`.
- **Re-recording:** `full_record.py` is `python -m tools.cartwheel.eval.full_record <checkpoint> <out.npz> [N envs] [seed] [task]`. It runs N attempts and keeps the best one (on a stunt course, the one that gets furthest). `STOCH=1` samples actions with the policy's training noise; all the videos above use the default, noise-free actions.
- **Figure 8:** the eight videos used the checkpoints `tasks/jumper/cartwheel_eight/checkpoints/eight_v{1,2,3}_*.pt`. `path_plot.py` is `python -m tools.cartwheel.render.path_plot <rollout.npz> <out.png>`.
- **Roll task:** the two `cartwheel_roll_*` videos come from `roll_record.py` → `traj_video.py`. Only the last roll recording survives (`roll_last_recorded.npz`), so re-record from a checkpoint to get the other. `roll_v2_model_3999.pt` is in `tasks/jumper/cartwheel_roll/checkpoints/`; round 2300 of run 1 is only on the Mac mini (`~/Projects/jumper/logs/jumper/jumper.cartwheel_roll/2026-11-02_06-43-38/`).

## Reference motions (kinematic, no policy)

| Video | Command |
|---|---|
| `cartwheel_reference4c_lean_start.mp4` | `python -m tools.cartwheel.render.ref_video tasks/jumper/cartwheel_jump/ref/cartwheel_jump.npz out.mp4` |
| `cartwheel_runup_fixed.mp4` | `ref_video.py` on `tasks/jumper/cartwheel_runup/ref/cartwheel_runup.npz` (probably; it predates these notes) |

The reference npz files are written by `tools/cartwheel/refs/cartwheel_ref*.py`, `roll_loop_ref.py` and `record_runup.py`.

## Jump frame strips (`*_jump.png`)

8 frames grabbed from the rendered video around the gap, then tiled 4 × 2
(`make_videos.py` does this for the three stunt videos):

```bash
V=docs/media/cartwheel/cartwheel_stunt_long.mp4; T="4.1 4.3 4.45 4.55 4.65 4.8 5.1 9.2"
for t in $T; do ffmpeg -loglevel error -ss $t -i $V -frames:v 1 -vf scale=480:-1 /tmp/jf_$t.png; done
set -- $T; ffmpeg -y -i /tmp/jf_$1.png -i /tmp/jf_$2.png -i /tmp/jf_$3.png -i /tmp/jf_$4.png \
  -i /tmp/jf_$5.png -i /tmp/jf_$6.png -i /tmp/jf_$7.png -i /tmp/jf_$8.png \
  -filter_complex "[0][1][2][3]hstack=4[a];[4][5][6][7]hstack=4[b];[a][b]vstack" docs/media/cartwheel/cartwheel_stunt_long_jump.png
```

The times used: pop `5.0 5.7 5.9 6.1 6.3 6.5 6.8 9.2`; fast `3.8 4.1 4.25 4.4 4.55 4.7 5.0 9.0`; long as above.

## The Inferno Gap (`inferno_gap.mp4`)

```bash
python tools/fire_skin.py          # only if assets/jumper/jumper_fire.xml / assets/jumper/fire/ are missing
MUJOCO_GL=egl python -m tools.cartwheel.render.film_stunt docs/media/cartwheel/traj/stunt_long.npz docs/media/cartwheel/inferno_gap.mp4
```

- **Inputs:** the fire livery (`tools/fire_skin.py` → `assets/jumper/jumper_fire.xml` and the OBJ meshes and flame textures in `assets/jumper/fire/`) and the course geometry (`course.LONG`). The livery only changes how the robot looks; the physics model is unchanged.
- **What the script builds:** the set (desert, mesas, fire pit, torches, props, embers), the shots, slow motion and replays, the titles, a colour grade, and the synthesized score. Its textures and the score are generated each run.
- **Options:**
  - `PREVIEW` as a third argument renders at 960×408 without audio.
  - `STILLS=1` renders one frame per shot, to check framing.
- **Run time:** about 5 minutes on 2 CPU cores.
