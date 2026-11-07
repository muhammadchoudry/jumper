"""Make the cartwheel and stunt videos in docs/media/cartwheel/ from one place.

Each video is rendered from a recorded rollout in docs/media/cartwheel/traj/
(the state at every physics step of one attempt by a trained policy), so a render
reproduces the published video exactly. With --rerecord the rollout is first
recorded again from the policy's checkpoint (the same command that made it);
rollouts can differ slightly between machines.

  python -m tools.cartwheel.make_videos list
  python -m tools.cartwheel.make_videos inferno_gap stunt_long           # render these
  python -m tools.cartwheel.make_videos key                              # the key videos (KEY below)
  python -m tools.cartwheel.make_videos all
  python -m tools.cartwheel.make_videos stunt_fast --rerecord            # record again, then render
  python -m tools.cartwheel.make_videos inferno_gap --preview            # 960x408, no audio, quick
  python -m tools.cartwheel.make_videos hill --out /tmp/videos           # write somewhere else

Run from the repo root (it also works as `python tools/cartwheel/make_videos.py ...` from
anywhere); it uses the repo's venv python if the current one lacks mujoco.
Rendering needs an OpenGL context: MUJOCO_GL=egl is set for you (headless Linux);
on a Mac set MUJOCO_GL=glfw or cgl before running.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
MEDIA = REPO / "docs/media/cartwheel"
TRAJ = MEDIA / "traj"
CK = "tasks/jumper"

# name: how the video is made.
#   traj      rollout in docs/media/cartwheel/traj/
#   render    "traj" (render/traj_video.py), "film" (render/film_stunt.py), "ref" (render/ref_video.py)
#   env       extra environment for the renderer (camera / course / slope)
#   record    (checkpoint, task, num_envs, seed) for eval/full_record.py, or ("roll", ckpt, task) for roll_record.py
#   strip     times (s) for an 8-frame strip "<video>_jump.png" around the jump
#   path      also draw the path plot "<video stem minus _top>_path.png"
VIDEOS = {
    "inferno_gap": dict(
        out="inferno_gap.mp4", traj="stunt_long.npz", render="film",
        record=(f"{CK}/cartwheel_stunt_long/checkpoints/long_v1_model_25299.pt", "jumper.cartwheel_stunt_long", 16, 7),
        what="The Inferno Gap: fire livery, desert set and fire pit, multi-camera cut with slow-motion replays, "
             "titles and score (0.75 m gap)"),
    "stunt_long": dict(
        out="cartwheel_stunt_long.mp4", traj="stunt_long.npz", render="traj", env={"COURSE": "long"},
        record=(f"{CK}/cartwheel_stunt_long/checkpoints/long_v1_model_25299.pt", "jumper.cartwheel_stunt_long", 16, 7),
        strip="4.1 4.3 4.45 4.55 4.65 4.8 5.1 9.2", what="ramp jump over a 0.75 m gap (side camera)"),
    "stunt_fast": dict(
        out="cartwheel_stunt_fast.mp4", traj="stunt_fast.npz", render="traj", env={"COURSE": "1"},
        record=(f"{CK}/cartwheel_stunt_fast/checkpoints/fast_v1_model_22300.pt", "jumper.cartwheel_stunt_fast", 16, 5),
        strip="3.8 4.1 4.25 4.4 4.55 4.7 5.0 9.0", what="ramp jump over a 0.45 m gap, legs free to roll faster"),
    "stunt_pop": dict(
        out="cartwheel_stunt_pop.mp4", traj="stunt_pop.npz", render="traj", env={"COURSE": "1"},
        record=(f"{CK}/cartwheel_stunt_pop/checkpoints/pop_v1_model_23599.pt", "jumper.cartwheel_stunt_pop", 16, 3),
        strip="5.0 5.7 5.9 6.1 6.3 6.5 6.8 9.2", what="ramp jump over a 0.45 m gap, star legs, push off the lip"),
    "hill": dict(
        out="cartwheel_hill.mp4", traj="hill.npz", render="traj", env={"SLOPE_DEG": "8"},
        record=(f"{CK}/cartwheel_hill/checkpoints/hill_v1_model_20600.pt", "jumper.cartwheel_hill", 4, 0),
        what="star cartwheel down an 8 deg hill"),
    "ten_star_legs": dict(
        out="cartwheel_ten_turns_star_legs.mp4", traj="ten_turns_star_legs.npz", render="traj",
        record=(f"{CK}/cartwheel_ten_star/checkpoints/star_v1_model_18896.pt", "jumper.cartwheel_ten_star", 6, 0),
        what="~8 turns, legs straight and spread (flat ground)"),
    "ten_straight_legs": dict(
        out="cartwheel_ten_turns_straight_legs.mp4", traj="ten_turns_straight_legs.npz", render="traj",
        record=(f"{CK}/cartwheel_ten_straight/checkpoints/straight_v1_model_15897.pt", "jumper.cartwheel_ten_straight",
                8, 0),
        what="8-10 turns, legs kept straight"),
    "ten_turns": dict(
        out="cartwheel_ten_turns_straight.mp4", traj="ten_turns.npz", render="traj",
        record=(f"{CK}/cartwheel_full/checkpoints/full_v3_model_12898.pt", "jumper.cartwheel_ten", 4, 0),
        what="10+ turns in a straight line"),
    "full_v3": dict(
        out="cartwheel_full_v3_round12898.mp4", traj="full_v3_round12898.npz", render="traj",
        record=(f"{CK}/cartwheel_full/checkpoints/full_v3_model_12898.pt", "jumper.cartwheel_full", 16, 0),
        what="the whole cartwheel in one policy: trot, jump, ~3.5 turns, stand"),
    "full_v2": dict(
        out="cartwheel_full_round8899.mp4", traj="full_round8899.npz", render="traj",
        record=(f"{CK}/cartwheel_full/checkpoints/full_v2_model_8899.pt", "jumper.cartwheel_full", 16, 0),
        what="first full cartwheels (run 2)"),
    "eight_try3": dict(
        out="cartwheel_eight_try3_top.mp4", traj="eight_try3.npz", render="traj", env={"TOPVIEW": "1"}, path=True,
        record=(f"{CK}/cartwheel_eight/checkpoints/eight_v3_model_20550.pt", "jumper.cartwheel_eight", 16, 0),
        what="figure-8 try 3 (rolls in circles)"),
    "eight_try2": dict(
        out="cartwheel_eight_try2_top.mp4", traj="eight_try2.npz", render="traj", env={"TOPVIEW": "1"}, path=True,
        record=(f"{CK}/cartwheel_eight/checkpoints/eight_v2_model_18550.pt", "jumper.cartwheel_eight", 16, 0),
        what="figure-8 try 2 (turns left only)"),
    "eight_attempt1": dict(
        out="cartwheel_eight_attempt1_top.mp4", traj="eight_attempt1.npz", render="traj", env={"TOPVIEW": "1"},
        path=True,
        record=(f"{CK}/cartwheel_eight/checkpoints/eight_v1_model_15897.pt", "jumper.cartwheel_eight", 16, 0),
        what="figure-8 try 1 (one left loop)"),
    "roll_v2": dict(
        out="cartwheel_roll_v2_round2950.mp4", traj="roll_last_recorded.npz", render="traj",
        record=("roll", f"{CK}/cartwheel_roll/checkpoints/roll_v2_model_3999.pt", "jumper.cartwheel_roll"),
        what="rolling on edge from a rolling start (the roll task)"),
    "reference4c": dict(
        out="cartwheel_reference4c_lean_start.mp4", render="ref", ref=f"{CK}/cartwheel_jump/ref/cartwheel_jump.npz",
        what="reference 4 (kinematic): trot, jump onto one spoke leaning, roll, stand"),
}
KEY = ["inferno_gap", "stunt_long", "stunt_fast", "stunt_pop", "hill", "ten_star_legs", "ten_straight_legs",
       "ten_turns", "full_v3"]


def python() -> str:
    try:
        import mujoco  # noqa: F401
        return sys.executable
    except ImportError:
        venv = REPO / ".venv/bin/python"
        if venv.exists():
            return str(venv)
        raise SystemExit("needs mujoco: activate the repo's venv")


def run(cmd, env=None, cwd=REPO):
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    e = dict(os.environ)
    e.setdefault("MUJOCO_GL", "egl")
    e.update(env or {})
    subprocess.run([str(c) for c in cmd], check=True, cwd=cwd, env=e)


def strip(video: pathlib.Path, times: str, out: pathlib.Path) -> None:
    tmp = out.parent / f".strip_{out.stem}"
    tmp.mkdir(exist_ok=True)
    ts = times.split()
    for i, t in enumerate(ts):
        run(["ffmpeg", "-loglevel", "error", "-y", "-ss", t, "-i", video, "-frames:v", "1", "-vf", "scale=480:-1",
             tmp / f"{i}.png"])
    ins = sum((["-i", tmp / f"{i}.png"] for i in range(len(ts))), [])
    run(["ffmpeg", "-loglevel", "error", "-y", *ins, "-filter_complex",
         "[0][1][2][3]hstack=4[a];[4][5][6][7]hstack=4[b];[a][b]vstack", out])
    shutil.rmtree(tmp)


def make(name: str, out_dir: pathlib.Path, rerecord: bool, preview: bool) -> None:
    v = VIDEOS[name]
    py = python()
    out = out_dir / v["out"]
    print(f"[{name}] {v['what']} -> {out}", flush=True)
    if v["render"] == "ref":
        run([py, "-m", "tools.cartwheel.render.ref_video", REPO / v["ref"], out])
        return
    traj = TRAJ / v["traj"]
    if rerecord:                     # a fresh rollout goes next to the video (<out>/traj/), then renders from there
        traj = out_dir / "traj" / v["traj"]
        traj.parent.mkdir(parents=True, exist_ok=True)
        rec = v["record"]
        if rec[0] == "roll":
            run([py, "-m", "tools.cartwheel.eval.roll_record", rec[1], traj, rec[2]])
        else:
            ckpt, task, n, seed = rec
            run([py, "-m", "tools.cartwheel.eval.full_record", ckpt, traj, n, seed, task])
    if v["render"] == "film":
        run([py, "-m", "tools.cartwheel.render.film_stunt", traj, out] + (["PREVIEW"] if preview else []))
    else:
        run([py, "-m", "tools.cartwheel.render.traj_video", traj, out], env=v.get("env"))
    if v.get("strip"):
        strip(out, v["strip"], out.with_name(out.stem + "_jump.png"))
    if v.get("path"):
        run([py, "-m", "tools.cartwheel.render.path_plot", traj, out.with_name(out.stem.removesuffix("_top") + "_path.png")])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+", help="video names, 'key', 'all' or 'list'")
    ap.add_argument("--rerecord", action="store_true", help="record the rollout again from its checkpoint first")
    ap.add_argument("--preview", action="store_true", help="the Inferno Gap at 960x408 without audio")
    ap.add_argument("--out", type=pathlib.Path, default=MEDIA, help="output folder (default docs/media/cartwheel)")
    a = ap.parse_args()
    if a.names == ["list"]:
        for n, v in VIDEOS.items():
            print(f"{n:18s} {'*' if n in KEY else ' '} {v['out']:42s} {v['what']}")
        print("(* = in 'key')")
        return
    names = list(VIDEOS) if a.names == ["all"] else KEY if a.names == ["key"] else a.names
    bad = [n for n in names if n not in VIDEOS]
    if bad:
        raise SystemExit(f"unknown: {bad}; try 'list'")
    a.out.mkdir(parents=True, exist_ok=True)
    for n in names:
        make(n, a.out.resolve(), a.rerecord, a.preview)


if __name__ == "__main__":
    main()
