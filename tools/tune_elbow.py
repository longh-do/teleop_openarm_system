#!/usr/bin/env python3
"""Careful test of the two slow motors (J3 upper-arm twist, J4 elbow: DM4340) on one arm, one setting at a time.
For each setting: start the motor loop with it written to the motors' RAM (motors off; nothing saved, an arm power
cycle resets it), switch the arm on where it hangs, run a short motion pattern through the loop like the server does,
measure how far each joint trails and how much it chatters, and stop at once at the first sign of shaking (that
setting and every stronger one are then skipped). Ends by writing the factory values back.
  (teleop_server.py and the normal motor loop NOT running; run with sudo: it starts the motor loop)
  sudo .venv/bin/python tools/tune_elbow.py right
Pattern (~25 s per setting): lift the forearm 20 deg, two quick 10 deg elbow nods, a 1 Hz wiggle, the same for the
upper-arm twist, lower the forearm. The velocity lead is off for J3/J4 here, so 'lag' is the motor's own.
"""
import argparse
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pi"))
import kinematics as kin  # noqa: E402

FACTORY = (54.0, 2.0)  # KP_APR, ACC/DEC as shipped
SETTINGS = [FACTORY, (54.0, 30.0), (80.0, 30.0), (110.0, 30.0)]  # gentlest first; stop at the first that shakes
TUNED = (3, 4)
SHAKE_POS = 0.6  # deg peak-to-peak of a held joint over 0.4 s, after removing a smooth (quadratic) glide = shaking
SHAKE_TAU = 2.5  # Nm peak-to-peak of a held J3/J4 over 0.4 s, after removing a smooth glide = shaking
SHAKE_TAU_12 = 4.0  # the same for J1/J2 (their settings are not changed; this catches coupling)
FAR = 10.0  # deg behind while moving = give up


class Abort(Exception):
    pass


class Link:
    def __init__(self, arm, loop_port, tel_port):
        self.arm, self.loop_port = arm, loop_port
        self.tel = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.tel.bind(("127.0.0.1", tel_port))
        self.tel.setblocking(False)
        self.out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.t, self.rx_t, self.fresh = None, 0.0, False
        self.q = self.epoch = self.g = None

    def newest(self):
        m = None
        try:
            while True:
                m = json.loads(self.tel.recv(65536))
        except BlockingIOError:
            pass
        self.fresh = m is not None and self.arm in m.get("arms", {})
        if self.fresh:
            self.t, self.rx_t = m["arms"][self.arm], time.monotonic()
        return self.t

    def wait_on(self, timeout):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            t = self.newest()
            if t is not None and t["mode"] == "on":
                self.epoch, self.g, self.q = t["epoch"], t["x"][7], np.array(t["x"][:7])
                return t
            time.sleep(0.02)
        raise Abort(f"the arm didn't switch on ({self.t and self.t['mode']}: {self.t and self.t['reason']})")

    def tick(self):
        self.out.sendto(json.dumps({"t": "target", "arm": self.arm, "epoch": self.epoch,
                                    "q": [float(v) for v in self.q], "g": self.g}).encode(), ("127.0.0.1", self.loop_port))
        time.sleep(0.01)
        t = self.newest()
        if time.monotonic() - self.rx_t > 0.15:
            raise Abort("no telemetry from the motor loop")
        if t["mode"] != "on" or t["epoch"] != self.epoch:
            raise Abort(f"the loop stopped following ({t['mode']}: {t['reason']})")
        return t

    def cmd(self, c):
        self.out.sendto(json.dumps({"t": "cmd", "cmd": c, "arm": self.arm}).encode(), ("127.0.0.1", self.loop_port))


def pattern(q0, c):
    """-> list of (duration s, phase, joint index, function u->offset rad) relative to q0 (motor rad)."""
    up, nod, wig = math.radians(20), math.radians(10), math.radians(5)
    ease = lambda u: (1 - math.cos(math.pi * u)) / 2  # noqa: E731
    s4 = c["sign"][3]
    steps = [(3.0, "move", {3: lambda u: s4 * up * ease(u)}), (1.0, "hold", {3: lambda u: s4 * up})]
    for j in (3, 2):  # elbow, then upper-arm twist (index)
        base = {3: lambda u: s4 * up}
        for sgn in (1, -1, 1, -1):
            frm, to = (0.0, sgn * nod) if sgn > 0 else (nod, 0.0)
            if sgn < 0:
                frm, to = nod, 0.0
            steps.append((0.5, f"move J{j + 1}", {**base, j: (lambda f, t: lambda u: f + (t - f) * ease(u))(frm, to)}))
            steps.append((1.0, f"hold J{j + 1}", {**base, j: (lambda t: lambda u: t)(to)}))
        steps.append((3.0, f"sine J{j + 1}", {**base, j: lambda u: wig * math.sin(2 * math.pi * 3.0 * u)}))
        steps.append((0.8, f"hold J{j + 1}", base))
    steps.append((3.0, "move", {3: lambda u: s4 * up * (1 - ease(u))}))
    steps.append((0.5, "hold", {}))
    return steps


def run_setting(link, c, kp, acc, rec):
    """Drive the pattern, watching for shaking. -> None, or why it was stopped."""
    q0 = link.q.copy()
    x = np.degrees(kin.to_model(c, q0))
    if abs(x[0]) > 15 or abs(x[1]) > 15 or x[3] > 25:
        raise Abort(f"the arm isn't hanging straight (shoulder {x[0]:.0f}/{x[1]:.0f}, elbow {x[3]:.0f} deg)")
    win = []
    for dur, phase, f in pattern(q0, c):
        t0 = time.monotonic()
        while (u := (time.monotonic() - t0) / dur) < 1.0:
            q = q0.copy()
            for i, fn in f.items():
                q[i] = q0[i] + fn(u)
            link.q = q
            t = link.tick()
            if not link.fresh:
                continue
            now = time.monotonic()
            rec.append({"t": now, "phase": phase, "u": u, "x": t["x"][:7], "q": t["q"][:7], "tau": t["tau"][:7]})
            lag = [abs(math.degrees(t["q"][i] - t["x"][i])) for i in range(4)]
            if phase.startswith(("move", "sine")) and max(lag) > FAR:
                return f"J{int(np.argmax(lag)) + 1} {max(lag):.1f} deg behind"
            if phase.startswith("hold") and u * dur > 0.5:  # later part of a hold: is anything shaking back and forth?
                win.append(rec[-1])
                win = [r for r in win if now - r["t"] < 0.4 and r["phase"] == phase]
                if len(win) >= 15:
                    Q = np.degrees(np.array([r["q"] for r in win]))
                    T = np.array([r["tau"] for r in win])
                    k = np.arange(len(win))
                    for i in range(4):  # a joint still gliding into place is a smooth curve: take that out first
                        q_d = Q[:, i] - np.polyval(np.polyfit(k, Q[:, i], 2), k)
                        tau_d = T[:, i] - np.polyval(np.polyfit(k, T[:, i], 2), k)
                        lim = SHAKE_TAU if i + 1 in TUNED else SHAKE_TAU_12
                        if np.ptp(q_d) > SHAKE_POS:
                            return f"J{i + 1} shaking {np.ptp(q_d):.2f} deg"
                        if np.ptp(tau_d) > lim:
                            return f"J{i + 1} torque shaking {np.ptp(tau_d):.1f} Nm"
            else:
                win = []
    return None


def metrics(rec):
    """Per tuned joint: motor's own lag on the quick moves (ms), overshoot (deg), hold noise (deg), torque chatter
    in holds (Nm std), wiggle tracking error (deg RMS)."""
    out = {}
    for j in TUNED:
        i = j - 1
        lags, overs = [], []
        moves = [r for r in rec if r["phase"] == f"move J{j}"]
        # split into the four moves by gaps in time
        groups, cur = [], []
        for r in moves:
            if cur and r["t"] - cur[-1]["t"] > 0.2:
                groups.append(cur); cur = []
            cur.append(r)
        if cur:
            groups.append(cur)
        for g in groups:
            t = np.array([r["t"] for r in g]); xx = np.degrees([r["x"][i] for r in g]); qq = np.degrees([r["q"][i] for r in g])
            mid = (xx[0] + xx[-1]) / 2
            sgn = 1 if xx[-1] > xx[0] else -1
            tx = np.interp(0, sgn * (xx - mid), t) if np.ptp(xx) > 1 else None
            after = [r for r in rec if g[-1]["t"] < r["t"] < g[-1]["t"] + 1.0]
            tq_all = t.tolist() + [r["t"] for r in after]
            qq_all = qq.tolist() + np.degrees([r["q"][i] for r in after]).tolist()
            tq = np.interp(0, sgn * (np.array(qq_all) - mid), tq_all) if np.ptp(qq_all) > 1 else None
            if tx is not None and tq is not None:
                lags.append((tq - tx) * 1000)
            target = xx[-1]
            if after:
                overs.append(max(0.0, max(sgn * (np.degrees(r["q"][i]) - target) for r in after)))
        holds = [r for r in rec if r["phase"] == f"hold J{j}" and r["u"] > 0.5]
        hn = float(np.std(np.degrees([r["q"][i] - r["x"][i] for r in holds]))) if holds else None
        ch = float(np.std(np.diff([r["tau"][i] for r in holds]))) if len(holds) > 2 else None
        sine = [r for r in rec if r["phase"] == f"sine J{j}" and r["u"] > 0.2]
        se = float(np.sqrt(np.mean(np.degrees([r["q"][i] - r["x"][i] for r in sine]) ** 2))) if sine else None
        out[j] = {"lag_ms": round(float(np.median(lags)), 0) if lags else None,
                  "overshoot_deg": round(float(max(overs)), 2) if overs else None,
                  "hold_noise_deg": None if hn is None else round(hn, 3),
                  "chatter_nm": None if ch is None else round(ch, 3),
                  "wiggle_err_deg": None if se is None else round(se, 2)}
    return out


def gains_file(path, kp, acc):
    k = [None] * 7
    a = [None] * 7
    for j in TUNED:
        k[j - 1], a[j - 1] = kp, acc
    lead = [180.0, 180.0, 0.0, 0.0, 180.0, 180.0, 180.0]  # J3/J4 lead off: measure the motor itself
    with open(path, "w") as f:
        json.dump({"kp_apr": k, "acc": a, "lead_ms": lead}, f)


def start_loop(a, path, logf):
    cmd = [sys.executable, os.path.join(ROOT, "pi", "motor_loop.py"), "--arms", a.arm, "--gains-file", path,
           "--port", str(a.loop_port), "--tel-port", str(a.tel_port)] + a.loop_args
    return subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=ROOT)


def stop_loop(p, link=None):
    if p.poll() is None:
        p.send_signal(signal.SIGINT)  # parks slowly to where it started, then switches off
        try:
            p.wait(40)
        except subprocess.TimeoutExpired:
            p.send_signal(signal.SIGINT)  # second Ctrl-C: off now
            p.wait(10)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arm")
    ap.add_argument("--settings", default="", help="KP:ACC,KP:ACC,... (default: factory, then gentle steps)")
    ap.add_argument("--loop-port", type=int, default=47001)
    ap.add_argument("--tel-port", type=int, default=47002)
    ap.add_argument("--loop-args", default="", help="extra motor_loop.py arguments (simulator: --buses right=vcan2 ...)")
    ap.add_argument("--out", default=os.path.join(ROOT, "logs", time.strftime("tune-%Y%m%d-%H%M%S.json")))
    a = ap.parse_args()
    a.loop_args = a.loop_args.split()
    if os.geteuid() != 0:
        sys.exit("run with sudo (it starts the motor loop)")
    if subprocess.run(["pgrep", "-f", r"^\S*python\S* \S*pi/motor_loop.py"], capture_output=True).returncode == 0 and \
            not any(x.startswith("--buses") for x in a.loop_args):
        sys.exit("a motor loop is running: stop it first (and the server)")
    settings = [tuple(float(v) for v in s.split(":")) for s in a.settings.split(",")] if a.settings else SETTINGS
    c = kin.load_calib()[a.arm]
    link = Link(a.arm, a.loop_port, a.tel_port)
    results, path, failed = [], "/tmp/tune-gains.json", False
    logf = open("/tmp/tune-loop.log", "w")
    try:
        for kp, acc in settings:
            if failed:
                results.append({"kp_apr": kp, "acc": acc, "skipped": "a gentler setting already shook"})
                continue
            print(f"--- KP_APR {kp:.0f}, ACC {acc:.0f} on J3/J4 ({a.arm} arm)", flush=True)
            gains_file(path, kp, acc)
            p = start_loop(a, path, logf)
            rec, why = [], None
            try:
                link.wait_on(15)
                why = run_setting(link, c, kp, acc, rec)
            except Abort as e:
                why = str(e)
            if why:
                print(f"    STOPPED: {why}", flush=True)
                link.cmd("stop")
                failed = True
            stop_loop(p)
            r = {"kp_apr": kp, "acc": acc, "stopped": why, "metrics": metrics(rec) if rec else None}
            results.append(r)
            if r["metrics"]:
                for j, mm in r["metrics"].items():
                    print(f"    J{j}: lag {mm['lag_ms']} ms, overshoot {mm['overshoot_deg']} deg, hold noise {mm['hold_noise_deg']} deg, "
                          f"torque chatter {mm['chatter_nm']} Nm, wiggle error {mm['wiggle_err_deg']} deg", flush=True)
    except KeyboardInterrupt:
        print("Ctrl-C", flush=True)
    finally:
        print("--- writing the factory values back (KP_APR 54, ACC 2): the arm switches on and off where it hangs", flush=True)
        gains_file(path, *FACTORY)
        p = start_loop(a, path, logf)
        try:
            link.wait_on(15)
        except Abort as e:
            print(f"    restore: {e}", flush=True)
        stop_loop(p)
        with open(a.out, "w") as f:
            json.dump({"arm": a.arm, "results": results}, f, indent=1)
        print(f"results: {a.out}", flush=True)


if __name__ == "__main__":
    main()
