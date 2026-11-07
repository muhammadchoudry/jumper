"""Every cartwheel / stunt training run, in order: what it trained, what it started
from, how long, the code it ran, and what came out. Full write-up:
docs/cartwheel_training.md.

  python scripts/cartwheel/runs.py list                 # the table
  python scripts/cartwheel/runs.py show hill            # one run, with its command
  python scripts/cartwheel/runs.py start hill           # start it (scripts/cartwheel/train.sh)
  python scripts/cartwheel/runs.py start hill --exact   # print how to run it on the exact code it used

Every run was 1024 envs on a Mac mini CPU (native backend), checkpoints every 50
rounds. "from" is the checkpoint it continued from. Round numbers continue from it,
so a run of 3000 rounds from round 20600 ends at 23599. The committed checkpoints
under tasks/jumper/*/checkpoints/ are the "kept" ones (not always the last round).
Runs marked mac_only started from a checkpoint that exists only in the Mac mini's
logs/ folder.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
CK = "tasks/jumper"

# name: dict(task, from, rounds, commit, date, hours, mac_log, kept, result)
RUNS = {
    # ---- reference tracking on references 1-3 (Oct 31 - Nov 1): never completed a cartwheel
    "flat": dict(task="jumper.cartwheel_flat", frm=None, rounds=3000, commit="1a32f12", date="2026-11-01",
                 mac_log="jumper.cartwheel_flat/2026-11-01_09-31-08", kept=None,
                 result="reference 2 (flat-crab star tips onto edge): 0 full cartwheels; fails 60-120 deg into the roll"),
    "runup_1": dict(task="jumper.cartwheel_runup", frm=None, rounds=3000, commit="b9a0fd2", date="2026-11-01",
                    mac_log="jumper.cartwheel_runup/2026-11-01_14-13-13", kept=None,
                    result="stopped at 2250: learned to lie ~70 deg off its edge (loophole)"),
    "runup_2": dict(task="jumper.cartwheel_runup", frm=None, rounds=3000, commit="2bf722a", date="2026-11-01",
                    mac_log="jumper.cartwheel_runup/2026-11-01_16-52-03", kept=None,
                    result="with a full-rotation check: 0 full cartwheels; a rigid star can't coast (see the doc)"),
    # ---- rolling on edge with free legs
    "roll_1": dict(task="jumper.cartwheel_roll", frm=None, rounds=5000, commit="52a8a8a", date="2026-11-02",
                   mac_log="jumper.cartwheel_roll/2026-11-02_06-43-38", kept=None,
                   result="stopped at 2500: coin-spin loophole (folded legs, spinning about a wobbling axle)"),
    "roll_2": dict(task="jumper.cartwheel_roll", frm=None, rounds=4000, commit="de983e5", date="2026-11-02",
                   hours=4.5, mac_log="jumper.cartwheel_roll/2026-11-02_09-33-25",
                   kept=f"{CK}/cartwheel_roll/checkpoints/roll_v2_model_3999.pt",
                   result="15-16/16 roll 7-9 turns, 3.5-7.5 m in 5 s"),
    # ---- the whole move in one policy
    "full_1": dict(task="jumper.cartwheel_full", frm=f"{CK}/cartwheel_roll/checkpoints/roll_v2_model_3999.pt",
                   rounds=4000, commit="14d4a40", date="2026-11-02", mac_log="jumper.cartwheel_full/2026-11-02_15-04-33",
                   kept=None, result="stopped at 5500: stuck in the trot (residual too large outside the roll)"),
    "full_2": dict(task="jumper.cartwheel_full", frm="logs/jumper/jumper.cartwheel_full/2026-11-02_15-04-33/model_4900.pt",
                   mac_only=True, rounds=4000, commit="3db68a2", date="2026-11-02", hours=4.6,
                   mac_log="jumper.cartwheel_full/2026-11-02_17-05-57",
                   kept=f"{CK}/cartwheel_full/checkpoints/full_v2_model_8899.pt",
                   result="2/16 full cartwheels; most killed by the rotation check right after landing"),
    "full_3": dict(task="jumper.cartwheel_full", frm=f"{CK}/cartwheel_full/checkpoints/full_v2_model_8899.pt",
                   rounds=4000, commit="31747c7", date="2026-11-02", hours=4.7,
                   mac_log="jumper.cartwheel_full/2026-11-02_22-52-44",
                   kept=f"{CK}/cartwheel_full/checkpoints/full_v3_model_12898.pt",
                   result="12/16 full cartwheels, 3.5-3.9 turns; also does 10+ turns on jumper.cartwheel_ten untrained"),
    # ---- figure 8 (not solved)
    "eight_1": dict(task="jumper.cartwheel_eight", frm=f"{CK}/cartwheel_full/checkpoints/full_v3_model_12898.pt",
                    rounds=3000, commit="5aebde8", date="2026-11-03", mac_log="jumper.cartwheel_eight/2026-11-03_09-08-14",
                    kept=f"{CK}/cartwheel_eight/checkpoints/eight_v1_model_15897.pt",
                    result="steers left, never switches loops"),
    "eight_2": dict(task="jumper.cartwheel_eight", frm=f"{CK}/cartwheel_eight/checkpoints/eight_v1_model_15897.pt",
                    rounds=3000, commit="e7f28fd", date="2026-11-03", mac_log="jumper.cartwheel_eight/2026-11-03_14-13-50",
                    kept=f"{CK}/cartwheel_eight/checkpoints/eight_v2_model_18550.pt",
                    result="rolls 8-10 turns but turns left only; 0/16 figure 8s"),
    "eight_3": dict(task="jumper.cartwheel_eight", frm=f"{CK}/cartwheel_eight/checkpoints/eight_v2_model_18550.pt",
                    rounds=4000, commit="a4f2c1d", date="2026-11-04", mac_log="jumper.cartwheel_eight/2026-11-04_07-38-25",
                    kept=f"{CK}/cartwheel_eight/checkpoints/eight_v3_model_20550.pt",
                    result="rolls tight circles both ways, not an 8 (loophole); last round lost the trot"),
    # ---- the look: straight legs, then straight and evenly spread
    "ten_straight": dict(task="jumper.cartwheel_ten_straight",
                         frm=f"{CK}/cartwheel_full/checkpoints/full_v3_model_12898.pt", rounds=3000, commit="356bf6d",
                         date="2026-11-04", hours=3.2, mac_log="jumper.cartwheel_ten_straight/2026-11-04_15-06-44",
                         kept=f"{CK}/cartwheel_ten_straight/checkpoints/straight_v1_model_15897.pt",
                         result="legs ~3.5% short (was 6%), 5/6 full, 8-10 turns"),
    "ten_star": dict(task="jumper.cartwheel_ten_star",
                     frm=f"{CK}/cartwheel_ten_straight/checkpoints/straight_v1_model_15897.pt", rounds=3000,
                     commit="70fce20", date="2026-11-04", hours=3.3, mac_log="jumper.cartwheel_ten_star/2026-11-04_21-17-43",
                     kept=f"{CK}/cartwheel_ten_star/checkpoints/star_v1_model_18896.pt",
                     result="legs <2% short but bunched (spokes ~28 deg rms off even), 4/6 full"),
    # ---- downhill and the ramp jump
    "hill": dict(task="jumper.cartwheel_hill", frm=f"{CK}/cartwheel_ten_star/checkpoints/star_v1_model_18896.pt",
                 rounds=3000, commit="f9897ea", date="2026-11-05", hours=3.4, mac_log="jumper.cartwheel_hill/2026-11-05_07-24-41",
                 kept=f"{CK}/cartwheel_hill/checkpoints/hill_v1_model_20600.pt",
                 result="6/6 full cartwheels down 8 deg, legs ~1% short, spokes 24 deg rms, ~1.3 m/s"),
    "stunt_pop": dict(task="jumper.cartwheel_stunt_pop", frm=f"{CK}/cartwheel_hill/checkpoints/hill_v1_model_20600.pt",
                      rounds=3000, commit="57c83b6", date="2026-11-05", hours=3.9,
                      mac_log="jumper.cartwheel_stunt_pop/2026-11-05_11-09-37",
                      kept=f"{CK}/cartwheel_stunt_pop/checkpoints/pop_v1_model_23599.pt",
                      result="clears the 0.45 m gap ~1 in 16; trot needs action noise"),
    "stunt_fast": dict(task="jumper.cartwheel_stunt_fast", frm=f"{CK}/cartwheel_hill/checkpoints/hill_v1_model_20600.pt",
                       rounds=3000, commit="57c83b6", date="2026-11-05", hours=4.0,
                       mac_log="jumper.cartwheel_stunt_fast/2026-11-05_15-01-49",
                       kept=f"{CK}/cartwheel_stunt_fast/checkpoints/fast_v1_model_22300.pt",
                       result="clears the 0.45 m gap 9/16 (2.2 m/s at the lip), ~3/16 end standing"),
    "stunt_long": dict(task="jumper.cartwheel_stunt_long",
                       frm=f"{CK}/cartwheel_stunt_fast/checkpoints/fast_v1_model_22300.pt", rounds=3000, commit="26dd20d",
                       date="2026-11-06", hours=3.5, mac_log="jumper.cartwheel_stunt_long/2026-11-06_15-00-32",
                       kept=f"{CK}/cartwheel_stunt_long/checkpoints/long_v1_model_25299.pt",
                       result="clears the 0.75 m gap 5/16 (10/16 with action noise), 2.8 m/s at the lip"),
}


def command(r) -> str:
    ck = f" {r['frm']}" if r["frm"] else ' ""'
    return f"bash scripts/cartwheel/train.sh {r['task']}{ck} {r['rounds']} 1024"


def exact(name, r) -> str:
    """The run on the exact code it used: a worktree at its commit, this repo's venv,
    and the starting checkpoint from this checkout (older commits don't have it)."""
    ck = f" --checkpoint {REPO / r['frm']}" if r["frm"] else ""
    return (f"git worktree add ../jumper-{name} {r['commit']} && cd ../jumper-{name} && \\\n"
            f"  PYTHONPATH=$PWD nohup {REPO}/.venv/bin/python scripts/train.py --task {r['task']} --backend native "
            f"--device cpu --num_envs 1024 --max-iterations {r['rounds']} --headless --no-tensorboard{ck} "
            f"> train_{name}.log 2>&1 &")


def main(argv):
    if not argv or argv[0] == "list":
        print(f"{'run':13s} {'date':10s} {'task':28s} {'rounds':>6s}  {'from':36s} kept / result")
        for n, r in RUNS.items():
            frm = pathlib.Path(r["frm"]).name if r["frm"] else "scratch"
            kept = pathlib.Path(r["kept"]).name if r["kept"] else "-"
            print(f"{n:13s} {r['date']:10s} {r['task']:28s} {r['rounds']:6d}  {frm:36s} {kept}: {r['result']}")
        return
    cmd, name = argv[0], argv[1]
    r = RUNS[name]
    if cmd == "show":
        for k, v in r.items():
            print(f"{k:8s} {v}")
        print("command ", command(r))
        print("exact   ", exact(name, r))
        return
    if cmd == "start":
        if "--exact" in argv:
            print(exact(name, r))
            return
        if r.get("mac_only") and not (REPO / r["frm"]).exists():
            raise SystemExit(f"{name} started from {r['frm']}, which is only in the Mac mini's logs/")
        args = [r["task"], r["frm"] or "", str(r["rounds"]), "1024"]
        subprocess.run(["bash", str(REPO / "scripts/cartwheel/train.sh"), *args], check=True, cwd=REPO)
        return
    raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
