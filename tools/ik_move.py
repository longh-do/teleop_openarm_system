#!/usr/bin/env python3
"""Calibration check without the headset: move one arm's grasp point along a straight line through the motor
loop, hold, and come back. If signs/zeros are right the gripper moves exactly the asked distance and direction.
  (motor_loop.py running in teleop mode, teleop_server.py NOT running)
  .venv/bin/python tools/ik_move.py left --dx 0.08          # 8 cm forward, hold 3 s, back
Rig frame: x forward, y left, z up.
"""
import argparse
import json
import os
import socket
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pi"))
import kinematics as kin  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("--dx", type=float, default=0.0)
    ap.add_argument("--dy", type=float, default=0.0)
    ap.add_argument("--dz", type=float, default=0.0)
    ap.add_argument("--move-s", type=float, default=3.0)
    ap.add_argument("--hold-s", type=float, default=3.0)
    ap.add_argument("--dry", action="store_true", help="plan only: print the joint changes, send nothing")
    a = ap.parse_args()
    c = kin.load_calib()[a.arm]
    tel = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    tel.bind(("127.0.0.1", 47002))  # fails if teleop_server is running: one brain at a time
    tel.settimeout(1.0)
    out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def latest():
        tel.setblocking(False)
        m = None
        try:
            while True:
                m = json.loads(tel.recv(65536))
        except BlockingIOError:
            pass
        tel.settimeout(1.0)
        if m is None:
            m = json.loads(tel.recv(65536))
        return m["arms"][a.arm]

    t = latest()
    if t["mode"] != "on":
        sys.exit(f"{a.arm} arm is {t['mode']} ({t['reason']}), not on")
    epoch, q0, g = t["epoch"], np.array(t["x"][:7]), t["x"][7]
    p0, R0 = kin.fk(a.arm, c, q0)
    d = np.array([a.dx, a.dy, a.dz])
    q_end, res, _ = q0, 0, 0
    for k in range(60):  # plan the whole path first
        q_end, res, _ = kin.ik(a.arm, c, p0 + d * (k + 1) / 60, R0, q_end)
    print(f"{a.arm}: grasp point now at {np.round(p0, 3)} m; move {np.round(d, 3)} m")
    print("  joint change (deg):", np.round(np.degrees(q_end - q0), 1), f" plan residual {res * 1000:.1f} mm")
    if a.dry:
        return
    if res > 0.01 or np.max(np.abs(q_end - q0)) > 0.6:
        sys.exit("refusing: target unreachable or joints would move > 0.6 rad")

    def run(frm, to, dur):
        q = frm[1]
        t0 = time.monotonic()
        while (s := (time.monotonic() - t0) / dur) < 1.0:
            k = (1 - np.cos(np.pi * s)) / 2
            q, _, _ = kin.ik(a.arm, c, frm[0] + (to - frm[0]) * k, R0, q)
            out.sendto(json.dumps({"t": "target", "arm": a.arm, "epoch": epoch, "q": [float(v) for v in q], "g": g}).encode(),
                       ("127.0.0.1", 47001))
            time.sleep(0.01)
        return q

    q = run((p0, q0), p0 + d, a.move_s)
    t_end = time.monotonic() + a.hold_s
    while time.monotonic() < t_end:
        out.sendto(json.dumps({"t": "target", "arm": a.arm, "epoch": epoch, "q": [float(v) for v in q], "g": g}).encode(),
                   ("127.0.0.1", 47001))
        time.sleep(0.02)
    t = latest()
    pm, _ = kin.fk(a.arm, c, np.array(t["q"][:7]))
    print(f"  at the far end: model says the grasp point moved {np.round((pm - p0) * 100, 1)} cm "
          f"(asked {np.round(d * 100, 1)} cm); loop mode {t['mode']} {t['reason']}")
    run((p0 + d, q), p0, a.move_s)
    time.sleep(0.5)
    print("  back at the start")


if __name__ == "__main__":
    main()
