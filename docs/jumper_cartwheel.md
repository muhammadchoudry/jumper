# jumper cartwheel: progress record

Goal: a real, physics-only cartwheel by the Jumper crab robot for the KingKong
Robotics crab robot challenge (Oct 30 – Nov 30, 2026): a run-up, a sideways jump
onto the legs, a roll on edge like a wheel, and a landing back on its feet, all
within the motors' limits (1.7464 N·m, kp 20, kd 0.5, 200 Hz control).

Branch `cartwheel` of github.com/muhammadchoudry/jumper. Tools are in
`tools/crab_trio/`, tasks in `tasks/jumper/cartwheel*`, videos in
`docs/media/cartwheel/`.

## Status (Nov 3, 2026, morning)

| piece | status |
|---|---|
| run-up (trot) | done: a physics recording of the trained `jumper.posture` walk (`record_runup.py`) |
| sideways jump onto one leg, leaning landing | trained inside `jumper.cartwheel_full`; the main point of failure (below) |
| **rolling on edge** | **done, in physics**: `jumper.cartwheel_roll`, 15–16 of 16 attempts roll 7–9 turns and 3.5–7.5 m in 5 s |
| tip back down and stand | trained inside `jumper.cartwheel_full`; works when the roll gets there |
| **whole move in one policy** | **12/16 full cartwheels from a standing start** (`jumper.cartwheel_full` run 3, round 12898; 15/16 run to the end, 3 of those land on their back) |

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

**Run 3** (commit 31747c7, resumed from run 2 round 8899, 4000 rounds, about 4.7 h):
- Terminations: the full-rotation and gravity checks (1 rad) during the run-up
  only; a looser full-rotation check (1.6 rad) during the jump; none after
  touch-down. The hub-height check waits 0.4 s after landing. On run 2's
  weights these rules alone gave **9/16** (from 2/16): the 1 rad check right
  after the leaning landing had been killing attempts that could roll on.
- 35% of practice starts around the jump and landing (`reset_focus`).
- Spread legs while rolling: `star_pose` weight 0.3 → 1.5, kernel 1.0 → 0.15 rad²
  (mean square), so about 0.15 rad of pumping stays cheap.

Results (`full_eval.py`, 16 from standing): round 9900 13/16; **final round
12898: 12/16 full cartwheels** — trot, jump, **3.5–3.9 turns** over 1.6–3.1 m,
upright and still at the end. 15/16 ran to the end; 3 of those finished on their
back, and 1 drifted off heading in the run-up. **Leg spread while rolling: 0.28 m
in every attempt** (run 2: 0.23–0.27 m; the straight star ~0.30 m), so the legs
are more spread but still a little bent. Rolling did not suffer: 3.5–3.9 turns vs
run 2's ~3. Video (a run that ends on its feet):
`docs/media/cartwheel/cartwheel_full_v3_round12898.mp4`. Checkpoint:
`tasks/jumper/cartwheel_full/checkpoints/full_v3_model_12898.pt`.

### Ten turns, and a figure 8 (reference 5)

`cartwheel_ref5.py` makes reference 4 with a ~5.5 s roll (about 10 turns at the
policy's pace) in two versions: **straight** (`jumper.cartwheel_ten`) and a
**figure 8** (`jumper.cartwheel_eight`: heading 0 → 2π over the first half of the
roll, a loop to the left, then back to 0, a loop to the right; loops ~1.1 m
across in the reference). The figure-8 task scores the roll against the
reference's heading at each moment, and its lean against what the turn needs:
tan(lean) = v · yaw rate / g, the way a rolling coin leans into a turn
(`full_mdp.py` `steer_*`).

- **Straight, 10+ turns: run 3's policy does it with no extra training.** On
  `jumper.cartwheel_ten`, 6/8 then 4/4 full cartwheels, **10.2–11 turns over
  9.2–10 m**, ending upright. Video:
  `docs/media/cartwheel/cartwheel_ten_turns_straight.mp4` (11 turns, 10.2 m).
- **Figure 8, try 1** (two tangent circles, 3000 rounds from run 3's policy,
  checkpoint `tasks/jumper/cartwheel_eight/checkpoints/eight_v1_model_15897.pt`):
  **it steers but never switches loops.** 10/16 rolled 4.7–5.7 turns while curving
  left 270–640°, then fell (hub too low) about where the path switches from the
  left loop to the right; 6/16 drifted off heading in the trot (a regression). The
  best attempt rolls one left loop, wider than the reference, then falls back
  inside it: `cartwheel_eight_attempt1_top.mp4`, `cartwheel_eight_attempt1_path.png`.
  Between tangent circles the lean has to flip from left to right in ~0.4 s.
- **Figure 8, try 2:** a gentler 8, two round loops joined by straight diagonals
  that cross in the middle (constant lean around each loop, upright on the
  diagonals; at most ~14° of lean at 1.1 m/s), a longer roll (5.5 reference turns,
  7.5 s; 11.8 s episodes), and 25% of practice starts between 30% and 70% of the
  roll (around the crossing), 20% at the jump. Continuing from try 1's policy.
  **Result (round 18550): still no figure 8, 0/16.** It rolls much longer — 9/16
  roll 8–10.5 turns over 7.6–10.8 m — but every one turns **left** only: net heading
  +115° to +560° where the 8 needs +240° then back to 0. The best attempt rolls the
  left loop, keeps turning left past where the right loop should start, then falls
  (`cartwheel_eight_try2_top.mp4`, `cartwheel_eight_try2_path.png`; checkpoint
  `eight_v2_model_18550.pt`). Across both tries it has never learned to turn right.
  Candidate causes: (1) the heading target runs on the reference's clock, not on how
  far the robot has rolled, so once it is ahead or behind, the target says "turn"
  at the wrong place; (2) a left/right asymmetry of the body rolling on its right
  edge (shell on one side, legs on the other). Next: index the heading target by
  distance rolled; check (2) with a mirrored 8 (right loop first).

- **Figure 8, try 3** (commit a4f2c1d: the 8 drawn by distance rolled from where
  each roll starts, left- or right-first at random; 4000 rounds from try 2):
  **no figure 8 — it learned to roll in small circles.** Best checkpoint, round
  20550 (`eight_v3_model_20550.pt`), 16 attempts from standing: 9/16 complete
  the cartwheel (trot, jump, ~9 turns, back on its feet), but each one rolls 3–4
  tight loops (~0.3–0.4 m radius, heading turning ~140°/s), covering only
  3.3–4.7 m: `cartwheel_eight_try3_top.mp4`, `cartwheel_eight_try3_path.png`. It
  **does** turn right as well as left (both directions show up), so the robot
  can turn either way. Why circles score: the roll reward was gated by lean only,
  and the 8's progress follows distance, so slow circling kept the lean target
  near zero and paid the roll reward without ever matching the heading.
  The final round (22549) collapsed in the trot: 15/16 drift off heading in the
  run-up (`rotation_diverged`) — fewer run-up practice starts, and longer
  episodes dominated by the roll.
  (`full_eval` / `full_record` had kept the mid-roll practice starts on until
  this check; fixed — the numbers above are from standing.)

### Straight legs (`jumper.cartwheel_ten_straight`, Nov 4)
Same reference as ten turns, plus a phase-B reward for leg length
(`legs_straight`: shortening = 1 − |tip − hip| / straight length per limb,
reward exp(−mean(e²)/0.02²), weight 3.0) and a weaker `star_pose` (0.3).
Warm-started from `full_v3_model_12898`, 3000 rounds on the Mac (3 h 12 min).

| policy | full from standing | turns | legs short avg / worst |
|---|---|---|---|
| v3 (start) | 4/4 | 10.0–10.7 | 5.6–7.2% / 40–55% |
| round 13850 | 0/8 (trot forgotten) | – | from the jump: 3.2–4.5% / 30–46% |
| **round 15897 (final)** | **5/6** | 8.0–10.2 | **2.2–5.1% / 35–58%** |

- Mid-run the trot fell apart (every attempt `rotation_diverged` in the run-up),
  then recovered by the end. Same forgetting as figure-8 try 3.
- Legs are straighter on average (about 3.5% vs 6%), but each roll still has
  moments where a leg bends hard (worst 35–58%). That is how it keeps the spin.
  It rolls a little slower: 8–10 turns in the same time instead of 10–11.
- The reward is too strict at 5% short (almost no credit), so progress is slow.
  Next try: sigma 0.05, more standing starts, keep run-up terminations.
- Checkpoint `tasks/jumper/cartwheel_ten_straight/checkpoints/straight_v1_model_15897.pt`;
  video `docs/media/cartwheel/cartwheel_ten_turns_straight_legs.mp4`.

### Straight and evenly spread (`jumper.cartwheel_ten_star`, Nov 4–5)
Adds `spokes_even` (each tip's angle round the hub in the wheel plane vs its star
spoke, 60 deg apart; kernel 0.35 rad, weight 3), `legs_straight` at sigma 0.05,
drops `star_pose` (out-of-plane tilt left free), and starts 25% of practice
standing (`start_fraction`). From `straight_v1_model_15897`, 3000 rounds.
`full_eval` now also reports spokes off even (rms, worst) and tilt.

| policy | full | turns | path | legs short avg / worst | spokes off even rms | tilt rms |
|---|---|---|---|---|---|---|
| v3 | 4/4 | 10.1–10.6 | 8.9–10.1 m | 5.6–6.1% / 40–61% | 25–27 deg | 23–24 deg |
| straight_v1 | 5/6 | 8–10 | 5–8 m | 2.7–3.6% / 49–57% | 34–36 deg | 23–24 deg |
| **star_v1 (18896)** | **4/6** | 8.0–8.5 | 4.4–6.0 m | **0.7–1.9% / 22–58%** | 27–29 deg | 20–23 deg |

- Straight legs: done (under 2% short on average).
- Even spread: not done. The video shows straight legs bunched, three or four
  together on the ground side, rather than a six-point star. Spacing barely
  moved (35 to 28 deg rms). Keeping several feet near the floor is how it keeps
  rolling with locked knees.
- It covers less ground per turn (about 0.7 m vs 0.95 m): more pivoting and
  scuffing, less clean rolling.
- The trot is erratic between checkpoints (round 18300: 2/6; round 17500: 5/6),
  and works better with the policy's training noise on than off.
- Checkpoint `tasks/jumper/cartwheel_ten_star/checkpoints/star_v1_model_18896.pt`;
  video `docs/media/cartwheel/cartwheel_ten_turns_star_legs.mp4`.

### Down a hill (`jumper.cartwheel_hill`, Nov 5)
The star task on an 8 deg downhill (gravity tilted toward +x, `slope_deg`;
`traj_video.py` with `SLOPE_DEG=8` draws the real hill), `spokes_even` weight 5,
kernel 0.25 rad. From `star_v1_model_18896`, 3000 rounds.

A scripted rigid star (`tools/crab_trio/hill_roll.py`) falls over sideways within a
second on any slope and spin: it balances on one tip at a time. At 4–6 rad/s
its legs also buckle on landing (kp 20). So the policy has to balance it.

| checkpoint | full | turns | legs short avg | spokes off even rms | hub speed |
|---|---|---|---|---|---|
| star_v1 on the hill | 1/4 | – | 3.5% | 32 deg | – |
| **20600 (kept)** | **6/6** | 7.0–7.5 | **0.8–1.3%** | **24 deg** | ~1.3 m/s |
| 21300 | 5/6 | 7.2–8.4 | 0.8–2.1% | 22–24 deg | ~1.4 m/s |
| 21895 (last) | 4/6 | 6.4–7.6 | 0.7–1.7% | 21–23 deg | ~1.3 m/s |

The star reads clearly on video, legs straight and spread, though not perfectly
even. Mean hub speed of ~1.3 m/s on 8 deg is above the rigid-star estimate of
1.0 m/s peak (stunt_calc). Checkpoint
`tasks/jumper/cartwheel_hill/checkpoints/hill_v1_model_20600.pt`; video
`docs/media/cartwheel/cartwheel_hill.mp4`.

`full_eval.py` now takes the task as a third argument and reports turns about the
robot's own axle, path length, and heading change (total and net), so it works
on a curving path.

## Next
1. The finish: 3/16 end on their back. Reward the tip-down direction (feet-first), or end attempts that land upside down.
2. Legs: 0.28 m vs the star's 0.30 m. Another run with `star_pose` raised (e.g. weight 3) if the look matters more than reliability.
3. Make the competition video: Neon Reef skin and stage, music.
4. Optional looks pass: reward straighter, more spread legs, closer to the star
   pictured originally.
5. Parked: folded-leg rolling (`NOTES_folded_leg_rolling.md`).
