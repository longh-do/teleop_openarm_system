#!/usr/bin/env python3
"""Turn an arm's twist joints back to the normal hanging pose with its own motors, slowly, one joint at a time.
  (motor_loop.py running with the arm ON and hanging, teleop_server.py NOT running: one brain at a time)
  .venv/bin/python tools/straighten.py right,left --then-off
Only the twist and wrist joints move (J7 wrist turn, J6 wrist bend, J5 forearm twist, J3 upper-arm twist), at 14 deg/s.
The shoulders and elbow stay where they are: hanging, these joints carry almost no weight. A joint that pushes 1.5 Nm
harder than at its start (touching something), or J6 going silent while J5 turns (its cable runs through the forearm
twist), stops that joint where it is. --then-off switches the arms off afterwards (limp, still hanging), so the next
start.sh starts from the straightened pose: a normal STOP would park back to the twisted pose the arm was switched on in.
"""
import argparse
import json
import math
import os
import socket
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pi"))
import kinematics as kin  # noqa: E402
from find_stops import SPEED, Abort, Link  # noqa: E402

NORMAL_MODEL_DEG = {"right": [0, 0, 0, 0, 90, 0, 0], "left": [0, 0, 0, 0, -90, 0, 0]}  # hanging straight down
ORDER = [7, 6, 5, 3]  # wrist first, upper-arm twist last
NAMES = {3: "J3 upper-arm twist", 5: "J5 forearm twist", 6: "J6 wrist bend", 7: "J7 wrist turn"}
PUSH = 1.5  # Nm more than at the start of the move = touching something
FRAMES = 3


def turn(link, c, j, want_motor):
    i = j - 1
    t = link.tick()
    tau0, miss6 = t["tau"][i], t["misses"][5]
    n_push = n_j6 = 0
    while abs(want_motor - link.q[i]) > 1e-4:
        link.q[i] += max(-SPEED * 0.01, min(SPEED * 0.01, want_motor - link.q[i]))
        t = link.tick()
        n_push = n_push + 1 if abs(t["tau"][i] - tau0) > PUSH else 0
        n_j6 = n_j6 + 1 if j == 5 and (t["stale"][5] or t["misses"][5] > miss6) else 0
        if n_push >= FRAMES or n_j6 >= FRAMES:
            link.q[i] = t["q"][i]  # stay where it really is
            link.hold(0.3)
            return "stopped early: " + ("pushing against something" if n_push >= FRAMES else "J6 stopped answering")
    link.hold(0.2)
    return "done"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms", nargs="?", default="right,left")
    ap.add_argument("--loop-port", type=int, default=47001)
    ap.add_argument("--tel-port", type=int, default=47002)
    ap.add_argument("--then-off", action="store_true", help="switch the arms off afterwards (they stay hanging)")
    a = ap.parse_args()
    calib = kin.load_calib()
    arms = [n for n in a.arms.split(",") if n]
    for arm in arms:
        c = calib[arm]
        link = Link(arm, a.loop_port, a.tel_port)
        want = kin.to_motor(c, np.radians(NORMAL_MODEL_DEG[arm]))
        lim = kin.motor_limits(arm, c, margin=math.radians(2))
        before = np.degrees(kin.to_model(c, link.q)) - NORMAL_MODEL_DEG[arm]
        print(f"[{arm}] degrees from normal before: " + "  ".join(f"J{j} {before[j - 1]:+.0f}" for j in range(1, 8)), flush=True)
        try:
            for j in ORDER:
                goal = float(np.clip(want[j - 1], *lim[j - 1]))
                if abs(goal - link.q[j - 1]) < math.radians(1):
                    continue
                print(f"[{arm}] {NAMES[j]}: {math.degrees(goal - link.q[j - 1]):+.0f} deg ... ", end="", flush=True)
                print(turn(link, c, j, goal), flush=True)
        except Abort as e:
            print(f"\n[{arm}] stopped: {e}", flush=True)
        after = np.degrees(kin.to_model(c, link.newest()["q"][:7])) - NORMAL_MODEL_DEG[arm]
        print(f"[{arm}] degrees from normal after:  " + "  ".join(f"J{j} {after[j - 1]:+.0f}" for j in range(1, 8)), flush=True)
        link.tel.close()
    if a.then_off:
        out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        for arm in arms:
            out.sendto(json.dumps({"t": "cmd", "cmd": "estop", "arm": arm}).encode(), ("127.0.0.1", a.loop_port))
        print("arms switched off (hanging); run pi/start.sh to switch them on from this pose", flush=True)


if __name__ == "__main__":
    main()
