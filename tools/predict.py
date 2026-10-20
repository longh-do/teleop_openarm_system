#!/usr/bin/env python3
"""Say in words where the grasp point goes for each probe move, under the current calib.json (runs on the Mac).
  python3 tools/predict.py right --base "4:30,7:20" --probe "3:12,5:15"   (motor degrees, from the resting pose)
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pi"))
import kinematics as kin  # noqa: E402

REST = {"right": [-0.036, -0.9249, -0.0036, -1.0096, -0.2398, -0.0898, -0.2375],
        "left": [-1.2228, -0.083, -0.1898, -0.7002, -0.3466, 0.1047, -0.083]}


def words(d, arm):
    out = "outward (away from the pole)" if arm == "right" else "inward (toward the pole)"
    inn = "inward (toward the pole)" if arm == "right" else "outward (away from the pole)"
    names = [("forward", "backward"), (inn, out), ("up", "down")]  # +y is robot-left
    parts = [(abs(v), names[i][0 if v > 0 else 1], v) for i, v in enumerate(d)]
    parts.sort(reverse=True)
    main = [f"{n} {abs(v) * 100:.1f} cm" for m, n, v in parts if m > 0.25 * parts[0][0] and m > 0.003]
    return ", ".join(main) or "barely moves"


def parse(s):
    return [(int(k), np.radians(float(v))) for k, v in (p.split(":") for p in s.split(","))] if s else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("--base", default="")
    ap.add_argument("--probe", required=True)
    a = ap.parse_args()
    c = kin.load_calib()[a.arm]
    q0 = np.array(REST[a.arm])
    for j, d in parse(a.base):
        q0[j - 1] += d
    p0, R0 = kin.fk(a.arm, c, q0)
    print(f"{a.arm} arm base pose: grasp point {np.round(p0, 3)} m, model angles (deg) {np.round(np.degrees(kin.to_model(c, q0)), 1)}")
    for j, d in parse(a.probe):
        if j == 8:
            print(f"  J8 {np.degrees(d):+.0f}: gripper {'closes' if d < 0 else 'opens'} (calib says - = closed)")
            continue
        q = q0.copy()
        q[j - 1] += d
        p, R = kin.fk(a.arm, c, q)
        hand = R[:, 2] - R0[:, 2]  # tool z (along the fingers, pointing up the hand) change
        print(f"  J{j} {np.degrees(d):+.0f} deg: grasp point moves {words(p - p0, a.arm)}")


if __name__ == "__main__":
    main()
