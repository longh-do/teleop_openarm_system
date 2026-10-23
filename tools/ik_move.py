#!/usr/bin/env python3
"""Calibration check without the headset: move one arm's grasp point along a straight line through the motor
loop, hold, and come back. If signs/zeros are right the gripper moves exactly the asked distance and direction.
  (motor_loop.py running with the arm on; teleop_server.py stopped: one program sends targets)
  .venv/bin/python tools/ik_move.py left --dx 0.08          # 8 cm forward, hold 3 s, back
Rig frame: x forward, y left, z up.
"""
import argparse
import json
import os
import socket
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pi"))
import kinematics as kin  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arm", choices=("left", "right"))
    ap.add_argument("--dx", type=float, default=0.0)
    ap.add_argument("--dy", type=float, default=0.0)
    ap.add_argument("--dz", type=float, default=0.0)
    ap.add_argument("--move-s", type=float, default=3.0)
    ap.add_argument("--hold-s", type=float, default=3.0)
    ap.add_argument("--dry", action="store_true", help="plan only: print the joint changes, send nothing")
    ap.add_argument("--pre", default="", help="joint:motor-degrees offsets to reach a working pose first (joint space)")
    ap.add_argument("--moves", default="", help="several moves in one run, e.g. x:0.08,z:0.08,y:-0.08 (from the working pose)")
    a = ap.parse_args()
    steps = []  # --moves, parsed before anything moves
    if a.moves:
        axes = {"x": 0, "y": 1, "z": 2}
        for item in a.moves.split(","):
            k, v = item.split(":")
            d = np.zeros(3)
            d[axes[k]] = float(v)
            steps.append(d)
    c = kin.load_calib()[a.arm]
    tel = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    tel.bind(("127.0.0.1", 47002))  # busy while teleop_server.py runs (one program sends targets at a time)
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
            try:
                m = json.loads(tel.recv(65536))
            except TimeoutError:
                sys.exit("no telemetry from the motor loop: is it running?")
        if a.arm not in m["arms"]:
            sys.exit(f"the motor loop is not driving the {a.arm} arm")
        return m["arms"][a.arm]

    t = latest()
    if t["mode"] != "on":
        sys.exit(f"{a.arm} arm is {t['mode']} ({t['reason']}), not on")
    epoch, q0, g = t["epoch"], np.array(t["x"][:7]), t["x"][7]

    last = {"q": q0.copy()}

    def send(q):
        last["q"] = np.array(q, float)
        out.sendto(json.dumps({"t": "target", "arm": a.arm, "epoch": epoch, "q": [float(v) for v in q], "g": g}).encode(),
                   ("127.0.0.1", 47001))

    def keepalive():  # the loop brakes and re-syncs if targets stop for 100 ms: repeat the last one meanwhile
        while True:
            time.sleep(0.03)
            if not a.dry:
                send(last["q"])

    threading.Thread(target=keepalive, daemon=True).start()

    def joint_move(qa, qb, dur):
        t0 = time.monotonic()
        while (s := (time.monotonic() - t0) / dur) < 1.0:
            send(qa + (qb - qa) * (1 - np.cos(np.pi * s)) / 2)
            time.sleep(0.01)
        send(qb)

    q_rest = q0.copy()
    if a.pre:
        pre = np.zeros(7)
        for k, v in (p.split(":") for p in a.pre.split(",")):
            pre[int(k) - 1] = np.radians(float(v))
        if not a.dry:
            print(f"  going to the working pose (joint offsets {a.pre} deg)")
            joint_move(q0, q0 + pre, 8.0)
            time.sleep(1.0)
            if latest()["mode"] != "on":
                sys.exit(f"arm stopped while going to the working pose: {latest()['reason']}")
        q0 = q0 + pre
    if a.moves:
        for d in steps:
            moved = do_move(a, c, q0, d, send, latest)
            if moved is None:
                break
        if a.pre and not a.dry:
            print("  back to the resting pose")
            joint_move(q0, q_rest, 8.0)
            time.sleep(0.5)
        return
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
            send(q)  # through send(), so the keepalive repeats this pose and not an old one
            time.sleep(0.01)
        return q

    q = run((p0, q0), p0 + d, a.move_s)
    t_end = time.monotonic() + a.hold_s
    while time.monotonic() < t_end:
        send(q)
        time.sleep(0.02)
    t = latest()
    pm, _ = kin.fk(a.arm, c, np.array(t["q"][:7]))
    print(f"  at the far end: model says the grasp point moved {np.round((pm - p0) * 100, 1)} cm "
          f"(asked {np.round(d * 100, 1)} cm); loop mode {t['mode']} {t['reason']}")
    run((p0 + d, q), p0, a.move_s)
    time.sleep(0.5)
    print("  back at the start")


def do_move(a, c, q0, d, send, latest):
    """Straight-line move of the grasp point by d from the pose q0, hold, back. -> measured displacement or None."""
    p0, R0 = kin.fk(a.arm, c, q0)
    q_end = q0
    for k in range(60):
        q_end, res, _ = kin.ik(a.arm, c, p0 + d * (k + 1) / 60, R0, q_end)
    print(f"  move {np.round(d * 100, 1)} cm: joint change (deg) {np.round(np.degrees(q_end - q0), 1)}, plan residual {res * 1000:.1f} mm")
    if a.dry:
        return d
    if res > 0.01 or np.max(np.abs(q_end - q0)) > 0.8:
        print("  refusing this move: unreachable or joints would move > 0.8 rad")
        return None

    def line(frm, to, qs, dur):
        t0 = time.monotonic()
        q = qs
        while (s := (time.monotonic() - t0) / dur) < 1.0:
            q, _, _ = kin.ik(a.arm, c, frm + (to - frm) * (1 - np.cos(np.pi * s)) / 2, R0, q)
            send(q)
            time.sleep(0.01)
        q, _, _ = kin.ik(a.arm, c, to, R0, q)
        send(q)
        return q

    t_meas0 = latest()
    pm0, _ = kin.fk(a.arm, c, np.array(t_meas0["q"][:7]))
    q = line(p0, p0 + d, q0, a.move_s)
    t_end = time.monotonic() + a.hold_s
    while time.monotonic() < t_end:
        send(q)
        time.sleep(0.02)
    t = latest()
    pm, _ = kin.fk(a.arm, c, np.array(t["q"][:7]))
    print(f"    measured joints say the grasp point moved {np.round((pm - pm0) * 100, 1)} cm; loop {t['mode']} ({t['reason']})")
    if t["mode"] != "on":
        sys.exit("arm is not following any more: stopping the test")
    line(p0 + d, p0, q, a.move_s)
    t_end = time.monotonic() + 1.5
    while time.monotonic() < t_end:
        send(q0)
        time.sleep(0.02)
    return pm - pm0


if __name__ == "__main__":
    main()
