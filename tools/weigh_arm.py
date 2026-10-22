#!/usr/bin/env python3
"""Weigh one arm for gravity compensation: visit a set of still poses slowly, record the torque each motor needs to
hold the arm there, and fit how heavy each part is and where its centre of mass sits. Saves pi/gravity.json and says
how well it predicts poses it was not fitted on.
  (motor_loop.py running with the arm ON and hanging still, teleop_server.py NOT running)
  .venv/bin/python tools/weigh_arm.py right
Poses stay in front of and beside the arm, never inward toward the pole or the other arm; the elbow is bent for the
raised ones so the hand stays up. Moves at 0.25 rad/s; each pose is reached from both sides (friction averages out).
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pi"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import kinematics as kin  # noqa: E402
from find_stops import Abort, Link  # noqa: E402

SPEED = 0.25  # rad/s, fastest joint
HOLD, AVG = 2.0, 1.2  # s at each pose; the last AVG s are averaged
G = np.array([0.0, 0.0, 9.81])
KEEP_Y = 0.10  # m: elbow, wrist and hand stay at least this far from the body's centre line (the pole, the other arm)
# model degrees J1..J4 for the right arm (J5-J7 stay at their hanging values); the left arm mirrors J1/J2 signs
POSES = [(0, 0, 0, 30), (15, 0, 0, 0), (15, 0, 0, 60), (30, 0, 0, 90), (30, 15, 0, 45), (0, 20, 0, 0),
         (0, 20, 30, 60), (20, 25, -30, 90), (40, 10, 0, 100), (40, 0, 40, 70), (10, 30, -40, 30), (25, 5, 60, 110),
         (0, 10, -60, 90), (35, 20, 20, 20), (5, 5, 0, 120), (20, 0, -20, 80)]


def regressor(arm, x):
    """Holding torque per joint = sum over links k >= i of g . (a_i x (m_k (o_k - o_i) + R_k h_k)), linear in each
    link's mass m_k and first moment h_k (3, in its own frame); plus a constant offset per joint."""
    R, p = np.eye(3), kin.BASE[arm].copy()
    o, a, Rk = [], [], []
    for (xyz, ax), qi in zip(kin.CHAIN[arm], x):
        p = p + R @ np.array(xyz)
        o.append(p.copy())
        a.append(R @ np.array(ax))
        R = R @ kin.rot(ax, qi)
        Rk.append(R.copy())
    Y = np.zeros((7, 7 * 4 + 7))
    for i in range(7):
        for k in range(i, 7):
            Y[i, 4 * k] = G @ np.cross(a[i], o[k] - o[i])
            for c in range(3):
                Y[i, 4 * k + 1 + c] = G @ np.cross(a[i], Rk[k][:, c])
        Y[i, 28 + i] = 1.0
    return Y


def predict(arm, theta, x):
    return regressor(arm, x) @ theta


def move_to(link, q_to):
    q_from = link.q.copy()
    dur = max(1.0, float(np.max(np.abs(q_to - q_from))) / SPEED)
    t0 = time.monotonic()
    while (u := (time.monotonic() - t0) / dur) < 1.0:
        link.q = q_from + (q_to - q_from) * (1 - math.cos(math.pi * u)) / 2
        link.tick()
    link.q = q_to.copy()


def hold_and_average(link):
    t0, qs, taus = time.monotonic(), [], []
    while time.monotonic() - t0 < HOLD:
        t = link.tick()
        if link.fresh and time.monotonic() - t0 > HOLD - AVG:
            qs.append(t["q"][:7])
            taus.append(t["tau"][:7])
    return np.mean(qs, axis=0), np.mean(taus, axis=0), np.std(taus, axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arm")
    ap.add_argument("--loop-port", type=int, default=47001)
    ap.add_argument("--tel-port", type=int, default=47002)
    ap.add_argument("--save", action="store_true", help="write pi/gravity.json")
    a = ap.parse_args()
    c = kin.load_calib()[a.arm]
    link = Link(a.arm, a.loop_port, a.tel_port)
    q_home = link.q.copy()
    x = np.degrees(kin.to_model(c, link.q))
    if abs(x[0]) > 15 or abs(x[1]) > 15 or x[3] > 25:
        sys.exit(f"{a.arm}: the arm isn't hanging straight; STOP parks it")
    mirror = np.array([1, 1, 1, 1]) if a.arm == "right" else np.array([-1, -1, 1, 1])
    targets = []
    p_hang = kin.fk(a.arm, c, q_home)[0]
    for pose in POSES:
        xm = kin.to_model(c, q_home).copy()
        xm[:4] = np.radians(np.array(pose) * mirror)
        if a.arm == "left":  # J3 twist mirrors too, so the hand goes the same way relative to the body
            xm[2] = -xm[2]
        qm = kin.to_motor(c, xm)
        pts = np.vstack([kin.elbow_wrist_points(a.arm, c, qm)[3:], kin.fk(a.arm, c, qm)[0]])
        if np.min(np.abs(pts[:, 1])) < KEEP_Y or kin.fk(a.arm, c, qm)[0][2] < p_hang[2] + 0.02:
            print(f"  skipping pose {pose}: it would come within {KEEP_Y * 100:.0f} cm of the centre line or below the hanging hand")
            continue
        targets.append(qm)
    rows = []
    try:
        for n, tq in enumerate(targets + targets[::-1][1:]):  # out, then back: each pose from the other side too
            move_to(link, tq)
            q, tau, sd = hold_and_average(link)
            rows.append({"q": q.tolist(), "tau": tau.tolist(), "tau_sd": sd.tolist()})
            print(f"  pose {n + 1:2d}/{2 * len(targets) - 1}: model deg {np.round(np.degrees(kin.to_model(c, q))[:4]).astype(int).tolist()} "
                  f"holding torque J1-4 {np.round(tau[:4], 2).tolist()} Nm", flush=True)
        move_to(link, q_home)
        link.hold(0.5)
    except (Abort, KeyboardInterrupt) as e:
        print(f"stopped early: {e or 'Ctrl-C'}", flush=True)
        try:
            move_to(link, q_home)
        except Abort:
            pass
    if len(rows) < 12:
        sys.exit("too few poses to fit")
    X = [kin.to_model(c, np.array(r["q"])) for r in rows]
    T = [np.array(r["tau"]) * c["sign"] for r in rows]
    idx = np.arange(len(rows))
    test = idx % 4 == 3  # every 4th pose is held out to check the fit
    A = np.vstack([regressor(a.arm, X[i]) for i in idx[~test]])
    y = np.concatenate([T[i] for i in idx[~test]])
    theta = np.linalg.solve(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ y)
    err = np.array([predict(a.arm, theta, X[i]) - T[i] for i in idx[test]])
    print(f"fit check on {test.sum()} held-out poses: error RMS Nm J1-7 {np.round(np.sqrt(np.mean(err ** 2, axis=0)), 2).tolist()}", flush=True)
    theta = np.linalg.solve(np.vstack([regressor(a.arm, X[i]) for i in idx]).T @ np.vstack([regressor(a.arm, X[i]) for i in idx])
                            + 1e-3 * np.eye(A.shape[1]),
                            np.vstack([regressor(a.arm, X[i]) for i in idx]).T @ np.concatenate([T[i] for i in idx]))
    out = os.path.join(ROOT, "logs", time.strftime(f"weigh-{a.arm}-%Y%m%d-%H%M%S.json"))
    with open(out, "w") as f:
        json.dump({"arm": a.arm, "rows": rows, "theta": theta.tolist()}, f)
    print(f"data: {out}", flush=True)
    if a.save:
        path = os.path.join(ROOT, "pi", "gravity.json")
        g = json.load(open(path)) if os.path.exists(path) else {}
        g[a.arm] = {"theta": theta.tolist(), "date": time.strftime("%Y-%m-%d %H:%M"), "held_out_rms_nm": np.sqrt(np.mean(err ** 2, axis=0)).round(3).tolist()}
        with open(path + ".tmp", "w") as f:
            json.dump(g, f, indent=1)
        os.replace(path + ".tmp", path)
        print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
