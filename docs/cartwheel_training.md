# Cartwheel and stunt training: every run, and how to reproduce it

All training for the crab cartwheel and the ramp-jump stunt, Oct 31 – Nov 6, 2026,
on branch `cartwheel`. What each run taught us is in `docs/jumper_cartwheel.md`
(the cartwheel) and `docs/jumper_stunt_plan.md` (the stunt). This page covers how
each run was trained and how to run it again.

## Setup

Every run used:
- **Machine:** a Mac mini, CPU only, mjlab's native MuJoCo backend.
- **Scale:** 1024 parallel envs; 3000–4000 rounds (PPO iterations) per run.
- **Time:** about 3–5 hours per run.
- **Checkpoints:** every 50 rounds.
- **Training config:** PPO from rsl_rl. Each task's `rl_cfg.py` sets it, and every cartwheel task uses
  the same settings so checkpoints load across tasks: `jumper_ppo_baseline`,
  512-256-128 actor and critic, entropy 0.008, initial action std 0.5, seed 42.

```bash
bash scripts/cartwheel/setup.sh          # once: clone/update ~/Projects/jumper @ cartwheel, build .venv
cd ~/Projects/jumper && . .venv/bin/activate
python scripts/cartwheel/runs.py list    # every run below
python scripts/cartwheel/runs.py start hill
```

`runs.py start` calls `scripts/cartwheel/train.sh TASK CHECKPOINT ROUNDS ENVS`. That runs
`scripts/train.py` in the background under `caffeinate` (on a Mac) and logs to
`~/Projects/jumper-train-logs/train_<task>.log`. Checkpoints go to
`logs/jumper/<task>/<date>/model_<round>.pt`. On a Mac you can instead copy
`scripts/cartwheel/train.command` to `~/Projects`, set `RUN=` in it, and double-click it.

**Warm starts.** Almost every run continues an earlier policy (`--checkpoint`).
Round numbers carry on from it, so a 3000-round run from round 20600 ends at
23599. The checkpoints that later runs started from are committed under
`tasks/jumper/*/checkpoints/`. This is the "kept" checkpoint, chosen by evaluation, not always
the last round. Each run can therefore be repeated alone, without repeating the chain
before it.

**Exact code.** The tasks kept changing, and later edits were meant to leave earlier
tasks alone (new options default off). For a faithful repeat, run on the commit the run used:
`python scripts/cartwheel/runs.py start <run> --exact` prints a `git worktree`
command that does that.

**Randomness.** The seed is fixed (42), but multi-threaded CPU physics isn't
bit-for-bit reproducible. A repeat lands near the same result, not on the same weights.

## The runs

The original logs and every checkpoint are on the Mac mini under
`~/Projects/jumper/logs/jumper/<mac log>/`.

| # | Run | Date | Task | Starts from | Rounds | Code | Time | Kept checkpoint | Result |
|---|---|---|---|---|---|---|---|---|---|
| 1 | flat | Nov 1 | `cartwheel_flat` | scratch | 3000 | `1a32f12` | | – | Reference 2 (flat-crab star): 0 full cartwheels, falls 60–120° into the roll |
| 2 | runup_1 | Nov 1 | `cartwheel_runup` | scratch | stopped 2250 | `b9a0fd2` | | – | Loophole: lay ~70° off its edge |
| 3 | runup_2 | Nov 1 | `cartwheel_runup` | scratch | 3000 | `2bf722a` | | – | Full-rotation check added; 0 full. **A rigid star can't coast.** |
| 4 | roll_1 | Nov 2 | `cartwheel_roll` | scratch | stopped 2500 | `52a8a8a` | | – | Loophole: coin-spin with folded legs (parked idea, `tools/cartwheel/NOTES_folded_leg_rolling.md`) |
| 5 | **roll_2** | Nov 2 | `cartwheel_roll` | scratch | 4000 | `de983e5` | 4.5 h | `roll_v2_model_3999` | 15–16/16 roll 7–9 turns in 5 s |
| 6 | full_1 | Nov 2 | `cartwheel_full` | roll_v2 3999 | stopped 5500 | `14d4a40` | | – | Stuck in the trot (residual too big outside the roll) |
| 7 | full_2 | Nov 2 | `cartwheel_full` | full_1 round 4900 (Mac only) | 4000 | `3db68a2` | 4.6 h | `full_v2_model_8899` | 2/16 full cartwheels |
| 8 | **full_3** | Nov 2 | `cartwheel_full` | full_v2 8899 | 4000 | `31747c7` | 4.7 h | `full_v3_model_12898` | **12/16 full cartwheels**; 10+ turns on `cartwheel_ten` with no extra training |
| 9 | eight_1 | Nov 3 | `cartwheel_eight` | full_v3 12898 | 3000 | `5aebde8` | | `eight_v1_model_15897` | Steers left, never switches loops |
| 10 | eight_2 | Nov 3 | `cartwheel_eight` | eight_v1 15897 | 3000 | `e7f28fd` | | `eight_v2_model_18550` | 8–10 turns, left only; 0/16 figure 8s |
| 11 | eight_3 | Nov 4 | `cartwheel_eight` | eight_v2 18550 | 4000 | `a4f2c1d` | | `eight_v3_model_20550` | Tight circles both ways (loophole); figure 8 unsolved |
| 12 | ten_straight | Nov 4 | `cartwheel_ten_straight` | full_v3 12898 | 3000 | `356bf6d` | 3.2 h | `straight_v1_model_15897` | Legs ~3.5% short (from 6%), 5/6 full |
| 13 | ten_star | Nov 4 | `cartwheel_ten_star` | straight_v1 15897 | 3000 | `70fce20` | 3.3 h | `star_v1_model_18896` | Legs <2% short but bunched, 4/6 full |
| 14 | **hill** | Nov 5 | `cartwheel_hill` | star_v1 18896 | 3000 | `f9897ea` | 3.4 h | `hill_v1_model_20600` | **6/6 down an 8° hill**, star legs, ~1.3 m/s |
| 15 | stunt_pop | Nov 5 | `cartwheel_stunt_pop` | hill_v1 20600 | 3000 | `57c83b6` | 3.9 h | `pop_v1_model_23599` | 0.45 m gap ~1 in 16 |
| 16 | **stunt_fast** | Nov 5 | `cartwheel_stunt_fast` | hill_v1 20600 | 3000 | `57c83b6` | 4.0 h | `fast_v1_model_22300` | 0.45 m gap 9/16, 2.2 m/s at the lip |
| 17 | **stunt_long** | Nov 6 | `cartwheel_stunt_long` | fast_v1 22300 | 3000 | `26dd20d` | 3.5 h | `long_v1_model_25299` | **0.75 m gap** 5/16 (10/16 with action noise), 2.8 m/s |

Notes on the table:
- **Tasks:** every task id is prefixed `jumper.`.
- **Kept checkpoints:** all are in `tasks/jumper/<task>/checkpoints/`.
- **Times:** blank where the run's length wasn't recorded.
- **Earlier work:** before run 1, the original reference-tracking task `jumper.cartwheel` (reference 1, Oct 31) and the
  scripted physics searches (`tools/cartwheel/physics/`) showed that hand-made
  motions fail. Those were not RL runs on the Mac.

### The warm-start chain

```
roll_2 ─► full_1 ─► full_2 ─► full_3 ─┬─► eight_1 ─► eight_2 ─► eight_3
                                      └─► ten_straight ─► ten_star ─► hill ─┬─► stunt_pop
                                                                            └─► stunt_fast ─► stunt_long
```

## What changed in each step (and why)

1. **roll_2.** The roll is rewarded about the **world** axle and pays for forward travel, which closed the coin-spin loophole. The episode also ends if the hub drops below 0.15 m.
2. **full_1 → full_2.** Outside the roll the policy's correction is capped at ±1 × 0.4 rad. Raw roll-policy outputs had thrown the trot off.
3. **full_3.** Three changes:
   - **Terminations:** no rotation check after touch-down; the jump gets a looser 1.6 rad check.
   - **Practice:** 35% of practice starts at the jump.
   - **Spread legs:** star pose reward weight 1.5.
4. **eight_1–3.** These tried a heading target that followed the reference clock, then one that followed the distance rolled. Try 3's roll reward was gated by lean only, so circling paid; see the cartwheel doc.
5. **ten_straight.** Adds `legs_straight`, a reward on each limb's tip–hip length, kernel 0.02. That kernel was too strict and gave little gradient.
6. **ten_star.** Adds `spokes_even` (tips 60° apart in the wheel plane), `legs_straight` at kernel 0.05, and 25% standing starts so the trot keeps getting practice.
7. **hill.** Gravity tilted 8° toward +x (`slope_deg`), `spokes_even` weight 5 and kernel 0.25.
8. **stunt_pop / stunt_fast.** A real course of ramps built from boxes (`tasks/jumper/cartwheel_stunt/course.py`) and a stand-up phase triggered by position. Course terminations: too low above the surface, fell in the gap, off the side. A gap-clear bonus, and roll and travel caps raised to 10 rad/s and 3 m/s. *pop* keeps star legs except near the lip; *fast* frees the legs.
9. **stunt_long.** `course.LONG`: an 18° hill, a 0.75 m gap, the landing lip 0.4 m lower, and a 45°→20° landing. Adds a centre-line reward.

## Evaluating a checkpoint

```bash
python -m tools.cartwheel.eval.full_eval <checkpoint.pt> 16 <task>         # from standing; per-attempt report
STOCH=1 python -m tools.cartwheel.eval.full_eval ...                        # with the policy's action noise
python -m tools.cartwheel.eval.roll_eval <checkpoint.pt>                    # the roll task (rolling start)
```

`full_eval` prints, per attempt:
- how far through the move it got;
- turns about its own axle, path length, and heading change;
- leg spread and leg shortening, and how far the spokes are from evenly spaced;
- whether it ends upright, and why the attempt ended.

It adds course-specific lines on the stunt tasks (furthest point reached, gap cleared) and on the figure 8.

Several runs got *worse* near the end. Check a few checkpoints, not only the last: for example `full_eval` on rounds −1300, −600 and the final one.

## Videos

Each video is rendered from a recorded attempt by one of the kept checkpoints. See
`docs/media/cartwheel/README.md`, or run `python -m tools.cartwheel.make_videos list`.
