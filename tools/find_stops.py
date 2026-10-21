#!/usr/bin/env python3
"""Find where each wrist joint really ends, gently, with its own motor; with --save, write those ranges to
pi/calib.json so teleop can use every joint all the way to its real end.
  (motor_loop.py running with the arm ON and hanging still, teleop_server.py NOT running: one brain at a time)
  .venv/bin/python tools/find_stops.py right,left --save
One joint at a time (J7 wrist turn, J6 wrist bend, J5 forearm twist), each way from where it is now: turn slowly
(14 deg/s) toward the robot's designed end. Stop when 3 telemetry frames in a row show it touching something (the
joint pushes 1.2 Nm harder than when it started, or falls 2 deg behind). Then back off 2 deg and return. J5 also
stops if J6 stops answering (J6's cable runs through the forearm twist). Saved range = the end found (or the
designed end) minus 2 deg, only if the whole run finished and every range makes sense.
"""
import argparse
import json
import math
import os
import socket
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pi"))
import kinematics as kin  # noqa: E402

SPEED = 0.25  # rad/s
PUSH = 1.2  # Nm more than at the start of the sweep = touching something
LAG = math.radians(2.0)  # the joint this far behind the loop's follower = blocked
FRAMES = 3  # consecutive telemetry frames that must agree before a sweep stops
MIN_TRAVEL = math.radians(5.0)  # trips this close to the start are ignored (the start's own settling)
BACKOFF = math.radians(2.0)
KEEP = 2.0  # deg kept from a real end in the saved range
# the robot's designed travel (model deg): J6/J7 from the model; J5 the model's +-90 about its MOTOR's zero (on this
# rig the hanging rest is 14-20 deg off it; 2026-10-21 both forearms met a real end at motor -91.5 deg)
DESIGN = {6: (-45.0, 45.0), 7: (-90.0, 90.0)}
NAMES = {5: "J5 forearm twist", 6: "J6 wrist bend", 7: "J7 wrist turn"}


def design(c, j):
    if j != 5:
        return DESIGN[j]
    a, b = (math.degrees(c["sign"][4] * math.radians(v) + c["offset"][4]) for v in (-90.0, 90.0))
    return min(a, b), max(a, b)


class Abort(Exception):
    pass


class Link:
    """Talks to the motor loop like the server does: newest telemetry in, joint targets out."""

    def __init__(self, arm, loop_port, tel_port):
        self.arm, self.loop_port = arm, loop_port
        self.tel = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.tel.bind(("127.0.0.1", tel_port))
        except OSError:
            sys.exit("telemetry port busy: stop teleop_server.py first (one brain at a time)")
        self.tel.settimeout(1.0)
        self.out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.t, self.fresh, self.rx_t = None, False, 0.0
        self.newest(wait=True)
        if self.t["mode"] != "on":
            sys.exit(f"{arm} arm is {self.t['mode']} ({self.t['reason']}), not on")
        self.epoch, self.g = self.t["epoch"], self.t["x"][7]
        self.q = np.array(self.t["x"][:7])  # motor rad, the target we send

    def newest(self, wait=False):
        m = None
        self.tel.setblocking(False)
        try:
            while True:
                m = json.loads(self.tel.recv(65536))
        except BlockingIOError:
            pass
        self.tel.settimeout(1.0)
        if m is None and wait:
            m = json.loads(self.tel.recv(65536))
        self.fresh = m is not None
        if m is not None:
            self.t, self.rx_t = m["arms"][self.arm], time.monotonic()
        return self.t

    def send(self):
        self.out.sendto(json.dumps({"t": "target", "arm": self.arm, "epoch": self.epoch,
                                    "q": [float(v) for v in self.q], "g": self.g}).encode(), ("127.0.0.1", self.loop_port))

    def tick(self):
        """Send the target, wait 10 ms, read telemetry; abort if the loop stopped following us or went quiet."""
        self.send()
        time.sleep(0.01)
        t = self.newest()
        if time.monotonic() - self.rx_t > 0.1:
            raise Abort("no telemetry from the motor loop for 0.1 s")
        if t["mode"] != "on" or t["epoch"] != self.epoch:
            raise Abort(f"the arm stopped following ({t['mode']}: {t['reason']})")
        return t

    def hold(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.tick()

    def ramp(self, i, to):
        """Move joint index i to motor angle `to` at SPEED (no contact checks: used for backing off and returning)."""
        while abs(to - self.q[i]) > 1e-4:
            self.q[i] += max(-SPEED * 0.01, min(SPEED * 0.01, to - self.q[i]))
            self.tick()


def sweep(link, c, j, end_model_deg):
    """Turn joint j toward end (model deg) until it touches something. -> (model deg reached, why, real end?)."""
    i = j - 1
    start = link.q[i]
    end = float(kin.to_motor(c, np.radians([end_model_deg] * 7))[i])
    d = 1.0 if end > start else -1.0
    link.hold(0.3)
    taus = []
    while len(taus) < 10:
        t = link.tick()
        if link.fresh:
            taus.append(t["tau"][i])
    tau0 = float(np.mean(taus))
    miss6 = link.t["misses"][5]
    why, real, at_end_t, n_push, n_lag, n_j6 = "designed end", False, None, 0, 0, 0
    while True:
        if (end - link.q[i]) * d > 0:
            link.q[i] += d * min(SPEED * 0.01, (end - link.q[i]) * d)
        elif at_end_t is None:
            at_end_t = time.monotonic()
        t = link.tick()
        q, x = t["q"][i], t["x"][i]
        if (link.q[i] - x) * d > math.radians(3):
            raise Abort(f"the motor loop's own limit stopped J{j} (restart it with the designed ranges in calib.json)")
        if link.fresh and (q - start) * d > MIN_TRAVEL:
            n_push = n_push + 1 if abs(t["tau"][i] - tau0) > PUSH else 0
            n_lag = n_lag + 1 if (x - q) * d > LAG else 0
            n_j6 = n_j6 + 1 if j == 5 and (t["stale"][5] or t["misses"][5] > miss6) else 0
            if n_push >= FRAMES:
                why, real = f"pushing {t['tau'][i] - tau0:+.1f} Nm", True
                break
            if n_lag >= FRAMES:
                why, real = f"stuck {math.degrees(x - q):+.1f} deg behind", True
                break
            if n_j6 >= FRAMES:
                why, real = "J6 stopped answering", True
                break
        if at_end_t is not None and abs(q - end) < math.radians(0.5):
            break
        if at_end_t is not None and time.monotonic() - at_end_t > 2.0:
            why, real = f"stopped {math.degrees(abs(end - q)):.1f} deg short", True
            break
    reached = float(link.t["q"][i])
    if real:
        link.q[i] = reached - d * BACKOFF  # stop pressing at once
        link.hold(0.3)
    link.ramp(i, start)
    link.hold(0.5)
    return math.degrees(c["sign"][i] * reached + c["offset"][i]), why, real


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms", help="right, left or right,left")
    ap.add_argument("--joints", default="7,6,5")
    ap.add_argument("--save", action="store_true", help="write the ranges found to pi/calib.json")
    ap.add_argument("--loop-port", type=int, default=47001)
    ap.add_argument("--tel-port", type=int, default=47002)
    a = ap.parse_args()
    found, ok = {}, True
    for arm in a.arms.split(","):
        c = kin.load_calib()[arm]
        link = Link(arm, a.loop_port, a.tel_port)
        q_start = link.q.copy()
        x = np.degrees(kin.to_model(c, link.q))
        if abs(x[0]) > 15 or abs(x[1]) > 15 or x[3] > 25:
            sys.exit(f"{arm}: the arm isn't hanging straight (shoulder {x[0]:.0f}/{x[1]:.0f}, elbow {x[3]:.0f} deg); "
                     "STOP parks it")
        print(f"{arm} arm: finding the wrist's real ends (each joint slowly each way, then back)", flush=True)
        found[arm] = {}
        try:
            for j in (int(v) for v in a.joints.split(",")):
                lo_d, hi_d = design(c, j)
                res = []
                for end in (hi_d, lo_d):
                    reached, why, real = sweep(link, c, j, end)
                    res.append((reached, real))
                    print(f"  {NAMES[j]} toward {end:+.0f}: stopped at {reached:+.1f} deg ({why})", flush=True)
                (hi_r, hi_real), (lo_r, lo_real) = res
                lo = lo_r + KEEP if lo_real else lo_d
                hi = hi_r - KEEP if hi_real else hi_d
                now_deg = x[j - 1]
                if not lo < now_deg < hi or hi - lo < 20:
                    print(f"  !! {NAMES[j]}: range {lo:+.1f} .. {hi:+.1f} doesn't make sense (start {now_deg:+.1f}): not saved")
                    ok = False
                    continue
                found[arm][j] = [round(lo * 2) / 2, round(hi * 2) / 2]
                old = np.degrees(c["limits"][j - 1])
                print(f"  -> {NAMES[j]} range {found[arm][j][0]:+.1f} .. {found[arm][j][1]:+.1f} deg "
                      f"(was {old[0]:+.1f} .. {old[1]:+.1f})", flush=True)
        except (Abort, KeyboardInterrupt) as e:
            print(f"  stopped early: {e or 'Ctrl-C'}", flush=True)
            ok = False
            if isinstance(e, KeyboardInterrupt):
                try:
                    for i in range(4, 7):
                        link.ramp(i, q_start[i])
                    link.hold(0.5)
                except Abort:
                    pass
        link.tel.close()
        if not ok:
            break
    if a.save and ok:
        with open(kin.CALIB_PATH) as f:
            raw = json.load(f)
        for arm, joints in found.items():
            lim = raw[arm].setdefault("limits_deg", {})
            for j, r in joints.items():
                lim[str(j)] = r
            raw[arm]["notes"] += f" {time.strftime('%Y-%m-%d %H:%M')} find_stops: " + ", ".join(
                f"J{j} {r[0]:+.1f}..{r[1]:+.1f}" for j, r in sorted(joints.items())) + "."
        tmp = kin.CALIB_PATH + ".tmp"
        with open(tmp, "w") as f:
            f.write(json.dumps(raw, indent=2) + "\n")
        os.replace(tmp, kin.CALIB_PATH)
        print(f"saved to {kin.CALIB_PATH}: restart the motor loop and the server to use them", flush=True)
    elif a.save:
        print("nothing saved (the run didn't finish cleanly)", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
