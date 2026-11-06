# Ramp-jump stunt: sizing (Nov 5, 2026)

Goal: the crab cartwheels down a hill, up a kicker ramp, flies over a gap and lands
on a landing ramp on the other side (Evel Knievel style), with real physics.
Numbers from `tools/crab_trio/stunt_calc.py`; to be checked in simulation before
building the course.

## The crab as a wheel
- Mass 2.54 kg; inertia about the axle 0.051 kg m²; spoke radius 0.29 m (the star
  is 0.6 m across). Spoke tips on the ground are 0.29 m apart.
- Rolling energy is 1.23 × ½mv² (the spin adds 23%).
- A rigid star keeps 59% of its spin (35% of its energy) at each spoke change.
  That loss caps its speed, however long the hill.

## 1. How fast can it come down the hill?
| Hill | Rigid star, steady top speed |
|---|---|
| 8° | 1.0 m/s |
| 12° | 1.2 m/s |
| 15° | 1.4 m/s |
| 20° | 1.6 m/s |
| 30° | 1.9 m/s |

It reaches that speed within 2–3 spoke changes, so a hill of about 2–3 m is long
enough. A longer one adds nothing.

The current policy on the 8° test hill went **1.9 m/s**, twice the rigid figure,
but with its legs bunched (pivoting foot to foot, which loses less). The hill run
under way will show what speed straight, evenly spread legs give.

## 2. The kicker costs speed
Climbing h at the lip takes v² down by 2gh/1.23. From 2.0 m/s: rise 3 cm leaves
1.9 m/s, 5 cm leaves 1.8 m/s, 10 cm leaves 1.55 m/s. Each spoke change on the kicker
also costs about 65% of the energy. So the kicker must be short (one spoke step,
~0.15–0.3 m), shallow-rising (25°, rise 5–6 cm), and curved at its foot rather than
a sharp kink.

## 3. Flight
| At the lip | Landing lip | Hub flies | Lands at |
|---|---|---|---|
| 1.4 m/s, 30° | 0.2 m lower | 0.35 m | 2.4 m/s, 60° down |
| 2.0 m/s, 25° | 0.1 m lower | 0.46 m | 2.4 m/s, 42° down |
| 1.5 m/s, 25° + 0.8 m/s leg pop | 0.1 m lower | 0.48 m | 2.4 m/s, 56° down |
| 2.0 m/s, 25° + 0.8 m/s leg pop | 0.1 m lower | 0.70 m | 2.8 m/s, 50° down |

- **The gap has to be wider than a spoke step.** Anything under about 0.35 m the
  star simply steps across. 0.4–0.5 m reads as a real jump for a 0.6 m crab.
- **Landing:** a landing ramp sloped down 40–45° takes most of the impact. At
  2.4 m/s arriving 42–56° down, the speed into the ramp is only 0.1–0.4 m/s and
  the rest carries on as rolling. For comparison, the crab already survives about
  1.5 m/s straight down from its jump. The flat run-out after the ramp stops it
  within a few spoke changes.
- **In the air** it keeps spinning, about a third of a turn, and lands on whichever
  spoke comes round.

## Recommended course (pending the measured hill speed)
1. Start plateau, then a **15° hill, 2.5 m long** (0.65 m drop).
2. 0.3 m flat, then a **kicker: 25°, rise 6 cm**, curved foot.
3. **Gap 0.45 m.**
4. **Landing ramp:** lip 10 cm below the kicker lip, down at 40°, 0.8 m long, then
   3 m flat run-out.

This needs **about 2.0 m/s at the lip**. A rigid, evenly spread star gives only
1.3–1.4 m/s there, which clears about 0.3 m (a token gap). To reach 0.45–0.5 m it has to do one of these:
- **Roll faster on the approach** by moving its legs (as the flat-ground policies
  do), or
- **Pop off the lip:** a radial leg push as the last spoke leaves the kicker. The
  crab's standing jump reaches 1.55 m/s upward, and 0.8 m/s of pop adds about
  0.15–0.25 m of distance. This is the real-physics equivalent of a BMX rider
  pulling up off the lip.

## Next
1. Measure hub speed and spin from the hill run (`jumper.cartwheel_hill`).
2. Build the course from real tilted box geoms. The tilted-gravity trick only
   works for a single slope.
3. Scripted rigid-star check on the course, then RL on it: spawn rolling at the
   top, reward progress, clearing the gap, landing upright, and standing at the
   end.

## Results: push off the lip, run 1 (`jumper.cartwheel_stunt_pop`, Nov 5)
From `hill_v1_model_20600`, 3000 rounds (3 h 52 min on the Mac).
- **It can do the whole stunt:** trot, jump into the star, down the hill, up the
  kicker, over the 0.45 m gap, onto the landing ramp, and out onto the run-out.
  Video: `docs/media/cartwheel/cartwheel_stunt_pop.mp4`; the gap is around 5.7–6.5 s
  (`cartwheel_stunt_pop_jump.png`). The hub clears the kicker lip by ~14 cm.
- **Not reliable:** about 1 in 16 attempts clears the gap and finishes.
  - With deterministic actions, most attempts fall in the **trot** (rotation_diverged),
    as in the star runs. With the policy's own action noise (`STOCH=1` in
    full_eval / full_record) the trot mostly works.
  - The next failure is **drifting off the side** (up to 2.9 m sideways on a 6 m course).
  - Then **landing short**: many attempts reach the landing lip (x ≈ 5.72) and drop in.
- Checkpoint `tasks/jumper/cartwheel_stunt_pop/checkpoints/pop_v1_model_23599.pt`.

Fixes for a second run:
1. Hand the trot and jump to the hill policy (6/6 from standing) and switch to
   the stunt policy once it lands.
2. A heading / stay-on-the-centre-line reward on the hill.
3. Train more from spawns on the hill, so the kicker and gap get more practice.

Renders: `traj_video.py` with `COURSE=1` uses a side camera. The course is drawn
1.2 m wide and slid under the robot (the simulated course is 6 m wide and uniform
across).

## Results: roll faster, run 1 (`jumper.cartwheel_stunt_fast`, Nov 5)
From `hill_v1_model_20600`, 3000 rounds (4 h on the Mac). Best checkpoint: round
22300, not the last.
- **Much more reliable than the pop version.**
  - From standing, with deterministic actions, it trots and jumps every time.
  - It clears the gap in 6/8 and 9/16 attempts.
  - It clears the gap and finishes standing upright in about 3/16.
- **The physics matches the plan.** In the run on video it reaches the lip at
  **2.2 m/s** (the plan needed ~2.0). The hub is 21 cm above the kicker lip and
  23 cm above the landing lip as it crosses. It lands on the ramp, rolls out, and
  stands up on the run-out.
  Video: `docs/media/cartwheel/cartwheel_stunt_fast.mp4`; frames 3.8–5.0 s plus the
  end in `cartwheel_stunt_fast_jump.png`.
- **Where the rest fail:** drifting off the side (up to 2 m sideways in the
  good run), collapsing or tipping on the run-out, and a few short landings.
- Later checkpoints (23000, 23599) got worse: 0–3 clears out of 8.
- Checkpoint `tasks/jumper/cartwheel_stunt_fast/checkpoints/fast_v1_model_22300.pt`.
