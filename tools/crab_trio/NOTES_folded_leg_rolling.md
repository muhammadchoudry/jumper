# Alternative to revisit: folded-leg rolling

**Status:** parked (Nov 2, 2026). The main line is still the straight-legged "car
wheel" cartwheel: the flat six-spoke star rolling on its spoke tips.

## What it is

In the first `jumper.cartwheel_roll` run (legs free, rewarded for rolling), the
policy did not keep its legs out as spokes. It **folded its legs in and tumbled on
its edge**, a compact, rounder body turning over and over. The hub rode at
0.10–0.18 m instead of the star's 0.26–0.33 m.

As a cartwheel it scored as a loophole. The reward then measured spin about the
body's own axis, so a wobbling, coin-like spin counted. Real travel was only
0.1–0.7 m per 5 s, going back and forth. We changed the reward and stopped that
run.

## Why it might still be worth revisiting

- **It is what the robot found easy.** A folded crab is much rounder than the
  six-spoke star, so it loses far less energy at each contact. The rigid star loses
  ~2/3 of its spin energy at every spoke change and stops ~25 deg after a 2.4 rad/s
  start (`limb_wheel.py`). A rounder body avoids most of that.
- **It survives.** It stayed upright on edge for nearly the full 5 s episode in
  most attempts, which no straight-leg version has managed.
- **It could look good on video**: the crab curling into a ball-like tumble and
  springing back out into the star or onto its feet.

## What it would need

- Reward real forward rolling: rotation about the world axle (`roll_forward` as
  fixed in commit de983e5) plus net forward travel, but **without** the
  `hub_too_low` (< 0.15 m) termination, which now forbids the folded pose.
- A spawn or reference that starts folded and on edge, and the run-up and jump
  (reference 4) ending in a tuck instead of the star.
- Check that the shell isn't scraping. The run had little shell or upper-limb
  contact (undesired_contacts ~ -0.03/episode), but confirm on video.
- A finish: unfold to land on its feet.

## Where it is

- Task: `tasks/jumper/cartwheel_roll` at commit 52a8a8a (before the reward fix).
- Checkpoint: Mac mini,
  `~/Projects/jumper/logs/jumper/jumper.cartwheel_roll/2026-11-02_06-43-38/model_2300.pt`
  (and every 50 rounds before it). The behaviour was settled by round ~500.
- Eval: `/root/scratch/roll_eval.py` style metrics: time on edge, wheel turned by
  orientation vs integrated spin, forward travel, worst lean.
