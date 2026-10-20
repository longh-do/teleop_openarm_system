#!/usr/bin/env python3
"""Draw both arms as the model sees them (front, side, top), to compare with photos of the real rig. Runs on the Mac.
  python3 tools/fk_plot.py out.png --q-left "deg x8" --q-right "deg x8" [--calib pi/calib.json] [--sign-right 1,1,-1,...]
"""
import argparse
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pi"))
import kinematics as kin  # noqa: E402


def chain_points(arm, c, q):
    p, R = kin.fk(arm, c, q)
    o = kin.elbow_wrist_points(arm, c, q)
    flange = p - R @ kin.TOOL
    pts = np.vstack([kin.BASE[arm], o, flange, p])
    # gripper direction marker: a short stub along the tool's x axis
    stub = p + R @ np.array([0.03, 0, 0])
    return pts, stub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--q-left", required=True)
    ap.add_argument("--q-right", required=True)
    ap.add_argument("--calib", default=kin.CALIB_PATH)
    ap.add_argument("--sign-left")
    ap.add_argument("--sign-right")
    ap.add_argument("--offset-left")
    ap.add_argument("--offset-right")
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    calib = kin.load_calib(a.calib)
    for arm in ("left", "right"):
        if getattr(a, f"sign_{arm}"):
            calib[arm]["sign"] = np.array([float(v) for v in getattr(a, f"sign_{arm}").split(",")])
        if getattr(a, f"offset_{arm}"):
            calib[arm]["offset"] = np.radians([float(v) for v in getattr(a, f"offset_{arm}").split(",")])
    q = {arm: np.radians([float(v) for v in getattr(a, f"q_{arm}").replace(",", " ").split()])[:7] for arm in ("left", "right")}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    views = [("FRONT (looking at the robot's face: robot-left on YOUR right)", 1, 2, False),
             ("SIDE (from the robot's right: forward = to the right)", 0, 2, False),
             ("TOP (forward = up)", 1, 0, True)]
    for ax, (title, i, j, flip) in zip(axes, views):
        for arm, col in (("left", "tab:blue"), ("right", "tab:red")):
            pts, stub = chain_points(arm, calib[arm], q[arm])
            xs, ys = pts[:, i], pts[:, j]
            if title.startswith("TOP"):
                xs, ys = -pts[:, 1], pts[:, 0]
                sx, sy = -stub[1], stub[0]
            else:
                sx, sy = stub[i], stub[j]
                if flip:
                    xs, sx = -xs, -sx
            ax.plot(xs, ys, "-o", color=col, lw=3, ms=5, label=f"{arm} arm")
            ax.plot([xs[-1], sx], [ys[-1], sy], color="k", lw=2)
            ax.annotate("shoulder", (xs[1], ys[1]), fontsize=7)
            ax.annotate("elbow (J4)", (xs[4], ys[4]), fontsize=7)
            ax.annotate("grip", (xs[-1], ys[-1]), fontsize=7)
        ax.plot([0.12, -0.12], [0, 0], color="gray", lw=6, alpha=0.4) if not title.startswith("TOP") else None
        ax.set_title(title, fontsize=9)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.7, 0.4)
    axes[0].legend(loc="lower left")
    fig.suptitle(a.title)
    fig.tight_layout()
    fig.savefig(a.out, dpi=80)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
