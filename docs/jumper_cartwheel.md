# jumper cartwheel: progress record

Goal: a real, physics-only cartwheel by the Jumper crab robot for the KingKong
Robotics crab robot challenge (Oct 30 – Nov 30, 2026): a run-up, a sideways jump
onto the legs, a roll on edge like a wheel, and a landing back on its feet, all
within the motors' limits (1.7464 N·m, kp 20, kd 0.5, 200 Hz control).

Branch `cartwheel` of github.com/muhammadchoudry/jumper. Tools are in
`tools/crab_trio/`, tasks in `tasks/jumper/cartwheel*`, videos in
`docs/media/cartwheel/`.

## Status (Nov 2, 2026, evening)

| piece | status |
|---|---|
| run-up (trot) | done: a physics recording of the trained `jumper.posture` walk (`record_runup.py`) |
| sideways jump onto one leg, leaning landing | trained inside `jumper.cartwheel_full`; the main point of failure (below) |
| **rolling on edge** | **done, in physics**: `jumper.cartwheel_roll`, 15–16 of 16 attempts roll 7–9 turns and 3.5–7.5 m in 5 s |
| tip back down and stand | trained inside `jumper.cartwheel_full`; works when the roll gets there |
| **whole move in one policy** | **first full cartwheels, in physics**: `jumper.cartwheel_full` round 8899, 2 of 16 from a standing start (6 of 16 in a second sample ran to the end); see below |

The rolling policy is kept in the repo at
`tasks/jumper/cartwheel_roll/checkpoints/roll_v2_model_3999.pt` (`*.pt` is
gitignored; this one is force-added). Video:
`docs/media/cartwheel/cartwheel_roll_v2_round2950.mp4`.

## What we learned

### 1. A rigid six-spoke star cannot coast

The first idea was the "flat crab": all six limbs straight and spread in the body
plane, a six-spoke star rolling on its spoke tips like a car wheel. Straight limbs
carry the load along the bone, so the pose fits within the motor limits, but:

- At every spoke change the next tip lands 60° from the last, and the impact
  keeps only ~59% of the spin (~35% of its energy) for the crab's mass (2.54 kg)
  and spoke length (~0.30 m), inertia 0.05 kg·m² about the axle.
- The hub must climb ~4 cm over each spoke (~1 J), which needs ≥ 2.7 rad/s just
  after each impact.
- Coasting a full turn would need ~1.9 J added per spoke (~11 J per turn). The
  run-up carries ~0.75 J.

Measured (`tools/crab_trio/limb_wheel.py hand`, physics, sideways balance held by
an invisible hand): the rigid star started at 2.4 rad/s **stops after ~25°**. This
is why every reference-tracking run on references 1–3 failed 60–120° into the
roll.

### 2. Hip motors cannot push the wheel round with one foot down

With one foot planted, a hip torque only trades the body spinning about its own
centre against the body travelling round the foot. Angular momentum about the
contact point changes only through gravity. "Push harder with the hips" made the
body sag and roll backward. What can add energy is straightening a bent leg under
load (pumping, as on a swing), push-off from the trailing leg, and short steps
that lose less at each landing. A hand-scripted stepping/pumping gait
(`limb_wheel.py`) reached about 1/3 turn before we handed the problem to RL.

### 3. Let RL find the gait, and watch for loopholes

`jumper.cartwheel_roll` starts episodes on edge already rolling (spawn states
`ref/roll_loop.npz` from `roll_loop_ref.py`, the rigid star at 2.7 rad/s). Nothing
prescribes the legs (residual scale 0.7 rad around the star). It is rewarded for
rolling forward, forward travel, staying on edge, rolling straight and a light
pull toward the star; shell contact −2.

- **Run 1 (reward on body-frame spin): loophole.** It folded its legs and spun
  like a coin with a wobbling axle: ~1800° of spin per episode while the wheel's
  orientation went back and forth, hub at 0.10–0.18 m, 0.1–0.7 m of travel. Video:
  `cartwheel_roll_round2300.mp4`. Kept as a parked alternative in
  `tools/crab_trio/NOTES_folded_leg_rolling.md`.
- **Run 2 (commit de983e5):** roll measured about the world axle (only rolling
  forward produces it), travel weight 3, episode ends if the hub drops below
  0.15 m. 4000 rounds on the Mac mini (CPU, 1024 envs, ~4.5 h); scores flattened
  around round 1750.

Final check (round 3999, 16 attempts from a rolling start, 5 s each,
`roll_eval.py`): all 16 stayed on edge for the full 5 s; 15 rolled
2600–3200° (7–9 turns), 3.5–7.5 m, worst lean ≤ 18° (most ≤ 6°); 1 mostly spun in
place (0.57 m). It rolls about 1 m/s with **legs part-bent into a rounder wheel**,
hub at 0.20–0.26 m (the straight star rides at ~0.29 m), each leg reaching forward
to land. That is the physics above: a rounder wheel and short steps lose much less
at each landing.

## References (kinematic, npz format of `tasks/jumper/jump/mdp/reference.py`)

| # | file | task | what |
|---|---|---|---|
| 1 | `cartwheel_ref.py` | `jumper.cartwheel` | crouch, sideways push, straight-spoke star, roll |
| 2 | `cartwheel_ref2.py` | `jumper.cartwheel_flat` | flat-crab star tips onto edge, rolls 360°, tips down |
| 3 | `cartwheel_ref3.py` | `jumper.cartwheel_runup` | recorded trot, tip onto edge while moving, roll |
| 4 | `cartwheel_ref4.py` | `jumper.cartwheel_jump` | trot, sideways jump, lands **leaning (55°)** on **one spoke planted ahead**, vaults over it and stands up on edge while rolling, then tips down and stands |
| – | `roll_loop_ref.py` | `jumper.cartwheel_roll` | spawn states only: the star on edge rolling at 2.7 rad/s |

Reference-tracking results: references 2 and 3 scored 0/32 full cartwheels from
standing. Run 3 first learned to lie ~70° off its edge, which the gravity-only
check could not see; fixed with a full-rotation termination (commit 2bf722a).

## Reproduce

Training on a Mac mini (CPU): `~/Projects/jumper-cartwheel-train.sh <task> <envs>
<iterations>` clones or pulls this branch into `~/Projects/jumper`, builds the
venv, and runs

```bash
python scripts/train.py --task jumper.cartwheel_roll --backend native --device cpu \
  --num_envs 1024 --max-iterations 4000 --headless --no-tensorboard
```

Evaluate and render a checkpoint (from the repo root):

```bash
python tools/crab_trio/roll_eval.py tasks/jumper/cartwheel_roll/checkpoints/roll_v2_model_3999.pt
python tools/crab_trio/roll_record.py tasks/jumper/cartwheel_roll/checkpoints/roll_v2_model_3999.pt x
MUJOCO_GL=egl python tools/crab_trio/traj_video.py /root/scratch/roll_traj.npz roll.mp4
```

Regenerate the references: `cd tools/crab_trio && python cartwheel_ref4.py`
(likewise `roll_loop_ref.py`, `record_runup.py`).

## In progress: `jumper.cartwheel_full`

One policy on reference 4's timeline (`tasks/jumper/cartwheel_full/`), each reward
and termination gated to its phase (`full_mdp.py`):

- **A** (stand, run-up, jump, landing, to 0.15 s into the roll): the jump task's
  imitation terms, full-rotation tracking, divergence terminations.
- **B** (roll, 2.55–4.51 s after go): `cartwheel_roll`'s rolling rewards; ends if
  it topples off its edge (after 0.4 s to stand up from the leaning landing) or the
  hub drops below 0.15 m.
- **C** (tip-down and stand, to the end of a 6.4 s episode): stand pose, height and
  gravity direction tracked, plus upright and settle.

Started from the rolling policy with `--checkpoint
tasks/jumper/cartwheel_roll/checkpoints/roll_v2_model_3999.pt`; the iteration
count continues from 3999. On the Mac mini, run with
`~/Projects/jumper-cartwheel-full.command`. Evaluate from a standing start with
`python tools/crab_trio/full_eval.py <model_N.pt>`: per env, how far through the
timeline it got, turns while rolling, and whether it ends upright.

### Results: `jumper.cartwheel_full`

**Run 1** (started from the rolling policy, one residual scale of 0.7 rad
everywhere): stuck. 0/16 got past the first second of the trot. The rolling
policy's raw outputs are several times the unit range, so they threw the joints
more than 6 rad² off the reference. With zero action, the reference tracks to
about 0.1 rad².

**Run 2** (commit 3db68a2, resumed from run 1 round 4900): outside the roll the
residual is the raw output capped at ±1 times 0.4 rad; the roll keeps 0.7 rad.
4000 rounds, about 4.6 h on the Mac mini.

Final check, round 8899 (`full_eval.py`, 16 attempts from a standing start):
- **2/16 complete the whole move**: rolled 2.9–3.0 turns, 2.2–2.8 m, ran to the
  end, upright and still.
- 13/16 got through the run-up and jump and into the roll, then ended with
  `rotation_diverged` just after landing. The full-rotation check still applies
  for the first 0.15 s of the roll (the end of phase A), and the leaning landing
  strays more than 1 rad from the reference's.
- 1/16 hub_too_low in the roll after about a turn.

A recorded run (`full_record.py`, best of 16; 6/16 ran to the end in that
sample): trot 1.7 m, jump at 1.9 s tipping 90° onto its edge with the hub up to
0.23 m, about 3.5 turns (1260°) over 3 m, tips down at 4.5 s, standing still by
4.9 s. Video: `docs/media/cartwheel/cartwheel_full_round8899.mp4`. Checkpoint:
`tasks/jumper/cartwheel_full/checkpoints/full_v2_model_8899.pt`.

Likely next fixes: end phase A at the landing itself (no full-rotation check once
it's down), or loosen that check across the jump and landing; start more practice
attempts at the jump; then train longer.

## Next
1. Make the landing reliable (13/16 fail there): end the full-rotation check at touch-down or loosen it over the jump, start more practice attempts at the jump, train longer.
2. Optional looks pass: reward straighter, more spread legs, closer to the star
   pictured originally.
3. Parked: folded-leg rolling (`NOTES_folded_leg_rolling.md`).
