#!/usr/bin/env python3
"""Teleop brain: serves the Quest page + WebSocket over HTTPS (one port, one certificate), turns controller poses
into joint targets, and sends them to motor_loop.py over localhost UDP.

  .venv/bin/python pi/teleop_server.py --scale 0.5 --vmax 0.1      # S5: half scale, low caps
Open https://<pi-ip>:8443 on the Quest (Advanced -> Proceed once), press Enter AR.

Controls: squeeze = clutch (hold to drive that arm: the gripper follows the hand's position and angle, every joint
over its full range), trigger = gripper (after a squeeze it moves only once the trigger reaches the gripper's
current opening, so a held object isn't dropped), A/X = fine mode (moves and turns 0.3x),
B = align: stand beside the robot FACING THE SAME WAY IT FACES, point the right controller straight ahead, press B.
Y = STOP (park + power off). Hold a thumbstick 1 s while the arms are off = power on; the arms then hold where they are.
"""
import argparse
import asyncio
import json
import math
import os
import socket
import ssl
import sys
import time

import numpy as np
from aiohttp import WSMsgType, web

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kinematics as kin  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALIGN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "align.json")
# WebXR 'local' (x right, y up, z back) -> robot (x forward, y left, z up), operator facing robot-forward
M0 = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
SQ_ON, SQ_OFF = 0.7, 0.3
FINE = 0.3  # fine mode: turning 0.3x
FINE_POS = 0.25  # fine mode: moving 0.25x (4 cm of hand = 1 cm of gripper), for lining up a plug
FINE_LEAD = 0.3  # fine mode / insertion lock: the loop's velocity lead x0.3, so the gripper doesn't overshoot a stop
# thumbstick tilt while driving: forward/back = slide straight in/out along the gripper's pointing direction (the plug's
# axis); left/right = move to the operator's left/right, sideways to that axis (lining the plug up with a port)
SLIDE_V, SLIDE_V_FINE = 0.015, 0.005  # m/s at full tilt (normal / fine mode); response is squared for fine control
SLIDE_DZ = 0.15  # stick deadzone
# sliding in (or sideways) stops, with a buzz, once the gripper presses with --push-n newtons (estimated from the joint
# torques' change since the slide started), or as a backstop once the target is SLIDE_AHEAD past the measured grasp
# point (kept under the loop's 'fell behind' limit). 2026-10-22: the old 5 mm stop pressed with only a few newtons,
# too little to seat a plug; the real arm is stiff (16 Nm at 1.3 deg on J2), so 30 N comes within ~7 mm
SLIDE_AHEAD = 0.025
XR_TIMEOUT = 0.15  # s without controller data -> disengage
# No "ready pose": the arms never move on their own, driving starts from wherever they are (2026-10-21: raising both
# arms to point forward overloaded the left shoulder motors and the arm fell). From hanging straight down every
# direction but further down works; the IK bends the elbow for "up".
# joint speed the motor loop's follower allows (motor_loop VMAX); IK steps are kept under 90% of it, so the brain
# never asks for a jump the loop would refuse (19:36: from the straight-down pose the IK asked J4 for 29 deg at once)
LOOP_VMAX = np.array([1.0, 1.0, 0.6, 0.6, 2.0, 2.0, 2.0])
GRIP_RATE = 3.0  # rad/s
GRIP_CATCH = 0.05  # rad: after a squeeze the gripper waits until the trigger comes this close to its current opening
JOINT_NAMES = ["shoulder (fwd/back)", "shoulder (out/in)", "upper-arm twist", "elbow", "forearm twist", "wrist bend",
               "wrist turn"]


class OneEuro:
    def __init__(self, min_cutoff, beta, d_cutoff):
        self.mc, self.beta, self.dc = min_cutoff, beta, d_cutoff
        self.x = self.dx = None

    @staticmethod
    def _alpha(dt, cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self, x):
        self.x, self.dx = np.array(x, float), np.zeros(len(x))

    def __call__(self, x, dt):
        x = np.asarray(x, float)
        if self.x is None or dt <= 0:
            self.reset(x)
            return self.x.copy()
        dx = (x - self.x) / dt
        a_d = self._alpha(dt, self.dc)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.mc + self.beta * np.abs(self.dx)
        a = 1.0 / (1.0 + (1.0 / (2 * math.pi * cutoff)) / dt)
        self.x = a * x + (1 - a) * self.x
        return self.x.copy()


def quat_slerp(q0, q1, t):
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(d)
    return (math.sin((1 - t) * th) * q0 + math.sin(t * th) * q1) / math.sin(th)


def rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def cap_rotation(R_from, R_to, max_angle):
    w = kin.log_rot(R_to @ R_from.T)
    n = float(np.linalg.norm(w))
    if n <= max_angle:
        return R_to, False
    return kin.exp_rot(w * (max_angle / n)) @ R_from, True


class ArmBrain:
    def __init__(self, name, args, calib):
        self.name, self.args, self.calib = name, args, calib
        self.tel, self.tel_t = None, 0.0
        self.epoch = None
        self.q_cmd, self.g_cmd = None, None
        self.clutch = False  # Schmitt state of the squeeze
        self.engaged = False  # anchored and driving
        self.fine = False
        self.want_resume = False
        self.f_pos = OneEuro(args.min_cutoff, args.beta, args.d_cutoff)
        self.q_smooth = None
        self.anchor = None
        self.ee_p = self.ee_R = None
        self.last_buzz = None
        self.limit_buzz_t = 0.0
        self.ik_ms, self.ik_res = [], []
        self.resume_t = -10.0
        self.stick_t = None  # when this hand's thumbstick went down (hold 1 s with the arms off = power on)
        self.ee_v = np.zeros(3)  # commanded grasp-point velocity (smooth follower)
        self.lock = None  # insertion lock: {p0, axis, R} while the thumbstick is held or tilted during driving
        self.sliding, self.slide_v = False, 0.0  # thumbstick tilted: slide speed along the plug axis, m/s (+ = in)
        self.slide_side = 0.0  # and sideways, m/s (+ = the operator's right)
        self.grip_side = 0.0  # after a squeeze: +-1 while the trigger hasn't yet reached the gripper's opening
        self.blocked = []  # joints at their end that stop the gripper following the hand (for the headset panel)
        self.push_ref = None  # joint torques before the push (gravity), where and when they were taken
        self.push_f = np.zeros(3)  # estimated force the gripper presses with while sliding (N, robot frame, smoothed)
        self.push_now = 0.0  # ...along the direction it is sliding (for the headset panel)

    def on_tel(self, t, now):
        """New telemetry. If the loop reset its follower (epoch changed), re-sync and drop the clutch."""
        first = self.tel is None
        self.tel, self.tel_t = t, now
        if t["epoch"] != self.epoch:
            self.epoch = t["epoch"]
            self.q_cmd = np.array(t["x"][:7])
            self.g_cmd = t["x"][7]
            was = self.engaged
            self.engaged, self.anchor = False, None
            if was:
                return "resync"
        if first:
            self.last_buzz = t["buzz"]
        elif t["buzz"] != self.last_buzz:
            self.last_buzz = t["buzz"]
            return "buzz"
        return None

    @property
    def on(self):
        return self.tel is not None and self.tel["mode"] == "on" and time.monotonic() - self.tel_t < 0.3

    def engage(self, Cp, Cq, now):
        x = np.array(self.tel["x"][:7])
        self.q_cmd = x
        self.home = kin.to_model(self.calib, x)  # the IK's posture pull keeps this pose: squeezing alone moves nothing
        E0p, E0R = kin.fk(self.name, self.calib, x)
        self.f_pos.reset(Cp)
        self.q_smooth = np.array(Cq, float)
        self.anchor = {"C0p": np.array(Cp), "C0R": kin.quat_to_mat(Cq), "E0p": E0p, "E0R": E0R,
                       "s": self.args.scale * (FINE_POS if self.fine else 1.0),
                       "r": self.args.rot_scale * (FINE if self.fine else 1.0), "prev": np.eye(3), "acc": np.eye(3)}
        self.ee_p, self.ee_R = E0p.copy(), E0R.copy()
        self.ee_v, self.lock, self.sliding = np.zeros(3), None, False
        self.grip_side = None  # decided on the first trigger sample
        self.blocked = []
        self.engaged = True

    def set_lock(self, on):
        """Insertion lock: while on, the grasp point only slides along the gripper's pointing direction (the plug's
        axis) and the gripper keeps its angle; sideways hand motion and turning are ignored. Both switches re-anchor
        on the current (filtered) hand and commanded pose, so nothing jumps."""
        self.lock = {"p0": self.ee_p.copy(), "axis": -self.ee_R[:, 2].copy(), "R": self.ee_R.copy()} if on else None
        if self.lock and self.tel is not None:
            # torques before contact: the push force is their change. Back at the same spot soon after the last push,
            # keep the old ones: the plug may already be pressing, and that force must count
            now, q = time.monotonic(), np.array(self.tel["q"][:7])
            p_meas = kin.fk(self.name, self.calib, q)[0]
            r = self.push_ref
            if not (r and now - r["t"] < 20 and float(np.linalg.norm(p_meas - r["p"])) < 0.03):
                self.push_ref = {"tau0": np.array(self.tel["tau"][:7]), "p": p_meas}
            self.push_ref["t"] = now
            self.push_f, self.push_now = np.zeros(3), 0.0
        if self.lock:  # the operator's right (robot -y after B), minus any part along the plug axis; none if it is the axis
            side = np.array([0.0, -1.0, 0.0])
            side -= self.lock["axis"] * float(np.dot(side, self.lock["axis"]))
            n = float(np.linalg.norm(side))
            self.lock["side"] = side / n if n > 0.3 else None
        a = self.anchor
        a["C0p"], a["E0p"] = self.f_pos.x.copy(), self.ee_p.copy()
        a["C0R"], a["E0R"] = kin.quat_to_mat(self.q_smooth), self.ee_R.copy()
        if "acc" in a:  # fine-mode turn accumulator restarts from the new anchor
            a["prev"], a["acc"] = np.eye(3), np.eye(3)

    def drive(self, Cp, Cq, M, dt):
        """One pose sample while engaged -> new q_cmd. Cp/Cq in WebXR coordinates. Returns 'limit' on a cap hit."""
        a = self.anchor
        Cp_f = self.f_pos(np.asarray(Cp, float), dt)
        alpha = 1 - math.exp(-dt * 2 * math.pi * self.args.rot_cutoff)
        self.q_smooth = quat_slerp(self.q_smooth, Cq, alpha)
        CR = kin.quat_to_mat(self.q_smooth)
        target_p = a["E0p"] + a["s"] * (M @ (Cp_f - a["C0p"]))
        # the hand's turn since the squeeze, about the same room axes, applied to the gripper around its grasp point
        # (rot_scale 1 = as much as the hand, 0 = the gripper keeps its angle; fine mode 0.3x)
        turn = M @ (CR @ a["C0R"].T) @ M.T
        if a["r"] != 1.0:  # scaled: add up each frame's turn x r (scaling the total jumps when the hand passes 180 deg)
            a["acc"] = kin.exp_rot(kin.log_rot(turn @ a["prev"].T) * a["r"]) @ a["acc"]
            a["prev"] = turn
            turn = a["acc"]
        target_R = turn @ a["E0R"]
        slide_hit = False
        if self.lock and self.sliding:  # thumbstick: in/out along the plug's axis and sideways; the hand is ignored
            L = self.lock
            meas = None
            if self.tel is not None and self.push_ref is not None:
                q_meas = np.array(self.tel["q"][:7])
                meas = kin.fk(self.name, self.calib, q_meas)[0]
                f = kin.tool_force(self.name, self.calib, q_meas, np.array(self.tel["tau"][:7]) - self.push_ref["tau0"])
                self.push_f = 0.7 * self.push_f + 0.3 * f
                self.push_now = max(0.0, float(np.dot(self.push_f, L["axis"]))) if self.slide_v > 0 else 0.0
            # pushing in or sideways stops at --push-n newtons (or SLIDE_AHEAD past the real grasp point); pulling
            # out is not capped, so a plug can be unplugged
            for d, v, capped in ((L["axis"], self.slide_v, self.slide_v > 0), (L["side"], self.slide_side, True)):
                if d is None or v == 0.0:
                    continue
                if capped and meas is not None and (
                        float(np.dot(L["p0"] - meas, d)) * math.copysign(1.0, v) > SLIDE_AHEAD or
                        float(np.dot(self.push_f, d)) * math.copysign(1.0, v) > self.args.push_n):
                    slide_hit = True
                    continue
                L["p0"] = L["p0"] + d * v * dt
            target_p, target_R = L["p0"].copy(), L["R"]
        elif self.lock:  # insertion lock: only along the plug's axis, angle frozen
            L = self.lock
            target_p = L["p0"] + L["axis"] * float(np.dot(target_p - L["p0"], L["axis"]))
            target_R = L["R"]
        # smooth follower: speed AND acceleration limited, braking early enough to stop on the target (no hard-clamp
        # rubber band, no dead stop). A target moving at u trails by u^2 / (2 amax): 3 mm at 0.1 m/s.
        vmax, amax = (self.args.fine_vmax, self.args.fine_amax) if self.fine else (self.args.vmax, self.args.amax)
        e = target_p - self.ee_p
        d = float(np.linalg.norm(e))
        v_des = e * (min(vmax, math.sqrt(2 * amax * d)) / d) if d > 1e-9 else np.zeros(3)
        dv = v_des - self.ee_v
        if (n := float(np.linalg.norm(dv))) > amax * dt:
            dv *= amax * dt / n
        self.ee_v = self.ee_v + dv
        step = self.ee_v * dt
        if (n := float(np.linalg.norm(step))) > self.args.jump:
            step *= self.args.jump / n
        if float(np.linalg.norm(step)) >= d:  # would pass the target: land on it
            step = e
            self.ee_v = e / dt if dt > 0 else np.zeros(3)
        self.ee_p = self.ee_p + step
        self.ee_R, rcap = cap_rotation(self.ee_R, target_R, self.args.wmax * dt)
        self.trace = {"hand_cm": [round(float(v) * 100, 1) for v in M @ (np.asarray(Cp, float) - a["C0p"])],
                      "grip_cm": [round(float(v) * 100, 1) for v in self.ee_p - a["E0p"]]}
        t0 = time.perf_counter()
        q, res_p, res_r = kin.ik(self.name, self.calib, self.ee_p, self.ee_R, self.q_cmd, mode=self.args.ik,
                                 frozen=(self.tel or {}).get("skip", ()), home=self.home)
        self.ik_ms.append((time.perf_counter() - t0) * 1000)
        self.ik_res.append(res_p)
        dq = q - self.q_cmd
        over = float(np.max(np.abs(dq) / (0.9 * LOOP_VMAX * dt)))
        if over > 1.0:  # too fast for the joints: take the same direction, smaller step, and re-anchor the pose
            q = self.q_cmd + dq / over
            self.ee_p, self.ee_R = kin.fk(self.name, self.calib, q)
            self.ee_v = self.ee_v / over
            res_p = res_r = 0.0
        self.q_cmd = q
        x = kin.to_model(self.calib, q)
        self.trace["q_model_deg"] = [round(float(v), 1) for v in np.degrees(x)]
        self.trace["res_mm"] = round(res_p * 1000, 1)
        self.trace["res_deg"] = round(math.degrees(res_r), 1)
        self.trace["grip"] = None if self.g_cmd is None else round(float(self.g_cmd), 3)
        self.trace["lock"] = self.lock is not None
        self.trace["push_n"] = round(self.push_now, 1)
        lim, edge = self.calib["limits"], kin.MARGIN + math.radians(0.5)
        self.blocked = [j + 1 for j in range(7) if x[j] <= lim[j, 0] + edge or x[j] >= lim[j, 1] - edge] \
            if res_p > 0.005 or res_r > math.radians(3) else []
        if res_p > 0.01:  # unreachable: stop the commanded pose running away from the arm
            self.ee_p, self.ee_R = kin.fk(self.name, self.calib, q)
            self.ee_v = np.zeros(3)
            return "limit"
        # an angle the joints can't reach: buzz, and keep the target, so the gripper turns again as soon as the hand
        # comes back to where the gripper stopped
        return "limit" if res_r > math.radians(3) or slide_hit else None

    def grip(self, trigger, dt):
        g = self.calib.get("grip")
        if not g or self.g_cmd is None:
            return
        want = g["open"] + float(trigger) * (g["closed"] - g["open"])
        d = want - self.g_cmd
        if self.grip_side is None:  # just squeezed: don't jump to the trigger (a released trigger would drop the object)
            self.grip_side = 0.0 if abs(d) < GRIP_CATCH else math.copysign(1.0, d)
        if self.grip_side and d * self.grip_side > GRIP_CATCH:
            return  # wait until the trigger reaches the gripper's opening, then follow it
        self.grip_side = 0.0
        self.g_cmd += max(-GRIP_RATE * dt, min(GRIP_RATE * dt, d))


class Brain:
    def __init__(self, args):
        self.args = args
        calib = kin.load_calib()
        self.unverified = [a for a, c in calib.items() if not c["verified"]]
        self.arms = {a: ArmBrain(a, args, calib[a]) for a in ("left", "right")}
        self.yaw = 0.0
        if os.path.exists(ALIGN_PATH):
            with open(ALIGN_PATH) as f:
                self.yaw = json.load(f).get("yaw", 0.0)
        self.udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.ws = set()
        self.last_xr, self.last_send, self.last_ts = 0.0, 0.0, 0.0
        self.xr_times, self.page_skips, self.page_fps = [], 0, 0.0
        self.src_ts = []
        self.prev_buttons = {}
        # the page says when an AR session starts/ends: WebXR's 'local' space is re-centred each session, so the
        # alignment (B) is asked for again. B is also needed once after every server start: a saved yaw belongs to an
        # older AR session (2026-10-21: reusing one rotated every direction by ~90 deg).
        self.xr_active, self.aligned, self.xr_ws = False, False, None
        self.seen = {h: {"frames": 0, "ok": 0, "sq_max": 0.0, "tr_max": 0.0, "presses": [0, 0, 0]} for h in ("left", "right")}
        self.msg = ""
        os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
        self.log = open(os.path.join(ROOT, "logs", time.strftime("brain-%Y%m%d-%H%M%S.jsonl")), "a", buffering=1 << 16)
        self.event("start", args=vars(args), yaw=self.yaw, unverified=self.unverified)

    @property
    def M(self):
        return rotz(self.yaw) @ M0

    def event(self, kind, **kw):
        self.log.write(json.dumps({"t": round(time.time(), 4), "kind": kind, **kw}) + "\n")

    def say(self, text):
        self.msg = text
        print(time.strftime("%H:%M:%S"), text, flush=True)
        self.event("msg", text=text)

    def to_loop(self, msg):
        try:
            self.udp.sendto(json.dumps(msg).encode(), ("127.0.0.1", self.args.loop_port))
        except OSError:
            pass

    def send_targets(self):
        now = time.monotonic()
        for a in self.arms.values():
            if a.on and a.q_cmd is not None and a.epoch is not None:
                self.to_loop({"t": "target", "arm": a.name, "epoch": a.epoch,
                              "q": [round(float(v), 5) for v in a.q_cmd],
                              "g": None if a.g_cmd is None else round(float(a.g_cmd), 5),
                              "lead": FINE_LEAD if (a.fine or a.lock) else 1.0})
        self.last_send = time.monotonic()

    PATTERNS = {"fine": [[0.5, 40, 90], [0.5, 40, 0]], "align": [[0.8, 350, 0]],
                "stop": [[1.0, 150, 100], [1.0, 150, 100], [1.0, 150, 0]], "fault": [[1.0, 400, 0]],
                "resync": [[0.8, 120, 80], [0.8, 120, 0]], "lost": [[1.0, 80, 60], [1.0, 80, 0]], "limit": [[0.3, 25, 0]]}

    async def haptic(self, hand, kind):
        await self.broadcast({"t": "haptic", "hand": hand, "pattern": self.PATTERNS[kind]})

    async def broadcast(self, msg):
        data = json.dumps(msg)
        for ws in list(self.ws):
            if not ws.closed:
                try:
                    await ws.send_str(data)
                except ConnectionError:
                    pass

    async def on_xr(self, m):
        now = time.monotonic()
        ts = float(m.get("ts", 0.0))  # the headset's own clock: steadier than Wi-Fi arrival times
        dt = min(max((ts - self.last_ts) / 1000, 1e-3), 0.05) if self.last_ts else 1 / 72
        self.last_xr, self.last_ts = now, ts
        self.xr_times.append(now)
        self.src_ts.append(ts)  # headset clock: gaps here are the headset skipping frames, not the link
        self.page_fps = m.get("fps", 0)
        if now - getattr(self, "sample_t", 0.0) > 1.0:  # one raw sample a second, for debugging the headset side
            self.sample_t = now
            self.event("xr_sample", hands=m.get("hands"))
        for hand, a in self.arms.items():
            H = (m.get("hands") or {}).get(hand)
            if not H:
                if a.engaged:
                    a.engaged = False
                    self.say(f"{hand}: controller gone, holding")
                continue
            b = H.get("b", [0, 0, 0])
            prev = self.prev_buttons.get(hand, [0, 0, 0])
            self.prev_buttons[hand] = b
            edge = [bool(x) and not bool(y) for x, y in zip(b, prev)]
            sn = self.seen[hand]
            sn["frames"] += 1
            sn["ok"] += bool(H.get("ok"))
            sn["sq_max"] = max(sn["sq_max"], float(H.get("sq", 0)))
            sn["tr_max"] = max(sn["tr_max"], float(H.get("tr", 0)))
            for i, e in enumerate(edge):
                if e:
                    sn["presses"][i] += 1
                    self.event("button", hand=hand, button=["A/X", "B/Y", "stick"][i])
                    print(time.strftime("%H:%M:%S"), f"button {hand} {['A/X', 'B/Y', 'stick'][i]}", flush=True)
            if edge[0]:  # A / X: fine mode (re-anchor so the scale change can't jump)
                a.fine = not a.fine
                self.say(f"{hand}: fine mode {'on' if a.fine else 'off'}")
                if a.engaged and H.get("ok"):
                    a.engage(H["p"], H["q"], now)
                await self.haptic(hand, "fine")
            if edge[1] and hand == "right" and H.get("aim"):  # B: align
                ax, _, az = H["aim"]
                if math.hypot(ax, az) > 0.3:
                    self.yaw = -math.atan2(-ax, -az)
                    with open(ALIGN_PATH, "w") as f:
                        json.dump({"yaw": self.yaw}, f)
                    for o in self.arms.values():
                        o.engaged = False
                    self.aligned = True
                    self.say(f"aligned: robot forward = controller direction (yaw {math.degrees(self.yaw):+.0f} deg)")
                    self.event("align", yaw=self.yaw)
                    await self.haptic("right", "align")
            if edge[1] and hand == "left":  # Y: STOP
                await self.stop("Y button")
            if b[2] and a.stick_t is None:
                a.stick_t = now
            elif not b[2]:
                a.stick_t = None
            all_off = all(o.tel is not None and o.tel["mode"] == "off" for o in self.arms.values() if o.tel is not None) \
                and any(o.tel is not None for o in self.arms.values())
            if all_off and a.stick_t is not None and now - a.stick_t > 1.0:
                a.stick_t = now + 3600  # once per press
                self.to_loop({"t": "cmd", "cmd": "power_on", "arm": "all"})
                self.say("thumbstick held: power on (the arms hold where they are)")
                await self.haptic(hand, "align")
            sq = H.get("sq", 0.0)
            was = a.clutch
            a.clutch = sq > SQ_OFF if was else sq > SQ_ON
            ok = bool(H.get("ok"))
            if not a.clutch or not ok:
                if a.engaged:
                    a.engaged = False
                    self.event("release", arm=hand, why="tracking lost" if not ok else "squeeze released")
                    if not ok:
                        self.say(f"{hand}: tracking lost, holding")
                        await self.haptic(hand, "lost")
                a.want_resume = False
            elif not a.engaged and a.tel is not None:
                silent = [i + 1 for i, st in enumerate(a.tel.get("stale", [])) if st]
                if a.tel["mode"] == "hold" and not was and silent:
                    # resuming would only trip again: say why, once per squeeze
                    self.say(f"{hand}: can't drive, J{silent[0]} is not answering (check its cable). Y = STOP.")
                    await self.haptic(hand, "lost")
                elif a.tel["mode"] == "hold" and not a.want_resume and not was:
                    if now - a.resume_t < 3.0:
                        self.say(f"{hand}: still blocked ({a.tel['reason']}). Y = STOP.")
                        await self.haptic(hand, "lost")
                    else:
                        a.resume_t = now
                        self.to_loop({"t": "cmd", "cmd": "resume", "arm": hand})
                        a.want_resume = True
                elif a.on and not self.aligned:
                    if not was:
                        self.say(f"{hand}: press B to align first")
                        await self.haptic(hand, "lost")
                elif a.on and not was or (a.on and a.want_resume):
                    if self.unverified and hand in self.unverified and not self.args.allow_unverified:
                        if not was:
                            self.say(f"{hand}: calibration not verified (S1) - refusing to drive")
                    else:
                        a.engage(H["p"], H["q"], now)
                        a.want_resume = False
                        self.event("engage", arm=hand, E0=[round(v, 4) for v in a.anchor["E0p"]])
            if a.engaged:
                # thumbstick tilt: forward/back = in/out along the plug's axis, left/right = sideways; held down =
                # insertion lock (the hand drives, along the axis only)
                sx, sy = (float(v) for v in (H.get("stick") or [0, 0])[:2])
                vmax = SLIDE_V_FINE if a.fine else SLIDE_V
                tx, ty = (max(0.0, (abs(v) - SLIDE_DZ) / (1 - SLIDE_DZ)) for v in (sx, sy))
                sliding = tx > 0 or ty > 0
                a.slide_v = -math.copysign(ty * ty * vmax, sy)  # forward (y < 0) = in
                a.slide_side = math.copysign(tx * tx * vmax, sx)  # right (x > 0) = the operator's right
                want_lock = bool(b[2]) or sliding
                if want_lock != (a.lock is not None) or sliding != a.sliding:
                    a.sliding = sliding
                    a.set_lock(want_lock)  # re-anchors: switching never jumps
                    if sliding:
                        self.say(f"{hand}: thumbstick jog (in/out along the plug, left/right sideways)")
                    else:
                        self.say(f"{hand}: insertion lock {'on: straight in/out only' if a.lock else 'off'}")
                    await self.haptic(hand, "fine")
            if a.engaged:
                lim_hit = a.drive(H["p"], H["q"], self.M, dt) == "limit"
                if now - getattr(a, "trace_t", 0.0) > 0.1:  # 10 Hz trace: hand vs gripper (robot frame) + joints
                    a.trace_t = now
                    self.event("drive", arm=hand, **a.trace)
                if lim_hit and now - a.limit_buzz_t > 0.5:
                    a.limit_buzz_t = now
                    await self.haptic(hand, "limit")
            if a.engaged:  # trigger drives the gripper only while that arm is held; release freezes it
                a.grip(H.get("tr", 0.0), dt)
        self.send_targets()

    async def stop(self, why):
        for a in self.arms.values():
            a.engaged = False
        self.to_loop({"t": "cmd", "cmd": "stop", "arm": "all"})
        self.say(f"STOP ({why}): parking, then power off")
        await self.haptic("left", "stop")
        await self.haptic("right", "stop")

    async def on_cmd(self, m):
        cmd = m.get("cmd")
        if cmd == "stop":
            await self.stop("page")
        elif cmd == "power_on":
            self.to_loop({"t": "cmd", "cmd": "power_on", "arm": "all"})
            self.say("power on requested")
        elif cmd == "xr_start":
            self.xr_active, self.aligned = True, False
            for a in self.arms.values():
                a.engaged = False
            self.say("AR started: press B to align")
        elif cmd == "xr_end":
            self.xr_active = False
            for a in self.arms.values():
                a.engaged = False
            self.say("AR ended: arms hold")
        elif cmd == "ready":
            self.say("there is no ready pose any more: the arms hold where they are")

    def status(self):
        now = time.monotonic()
        recent = [t for t in self.xr_times if now - t < 1.0]
        arms = {}
        for n, a in self.arms.items():
            t = a.tel
            mode = "no motor loop" if t is None or now - a.tel_t > 0.5 else t["mode"]
            mode = {"starting_read": "starting", "preload": "starting", "enabling": "starting", "starting_hold": "starting"}.get(mode, mode)
            arms[n] = {"mode": mode, "clutch": a.engaged, "fine": a.fine, "msg": (t or {}).get("reason", "")}
        return {"t": "status", "arms": arms, "pose_hz": len(recent), "msg": self.msg, "hud": self.hud(arms)}

    @staticmethod
    def plain(reason):
        """The motor loop's reason, said simply for the headset panel."""
        r = reason or ""
        if "tracking error" in r:
            return "A joint fell behind (moved too fast, or bumped something)."
        if "torque" in r:
            return "A joint is pushing against something."
        if "silent" in r or "not answering" in r:
            return f"Motor {r.split()[0]} stopped answering."
        if "refused target" in r:
            return "That move was too sudden."
        if "hot" in r:
            return "A motor is hot: let it cool for a minute."
        if "bus not acknowledging" in r:
            return "The arm lost power or its cable."
        return r[:60]

    def hud(self, arms):
        """Headline, up to 3 detail lines, and a label per arm, for the guide panel inside the headset."""
        label = {}
        for n, st in arms.items():
            a, m = self.arms[n], st["mode"]
            if m == "no motor loop":
                label[n] = "NOT USED"
            elif m == "off":
                label[n] = "OFF"
            elif m == "starting":
                label[n] = "STARTING"
            elif m == "on":
                label[n] = ("DRIVING" if a.engaged else "READY") if not a.fine else "FINE"
            else:
                label[n] = {"hold": "PAUSED", "fault": "FAULT", "parking": "STOPPING"}.get(m, m.upper())
        used = [n for n in arms if label[n] != "NOT USED"]
        name = lambda ns: " and ".join(n.upper() for n in ns)  # noqa: E731
        def out(headline, detail, level):
            return {"headline": headline, "detail": detail, "level": level, "left": label["left"], "right": label["right"]}
        if not used:
            return out("Robot program not running", ["On the laptop, run: pi/start.sh"], "error")
        if faults := [n for n in used if arms[n]["mode"] == "fault"]:
            return out(f"{name(faults)} arm fault", [self.plain(arms[faults[0]]["msg"]), "Press Y to stop both arms."], "error")
        if [n for n in used if arms[n]["mode"] == "parking"]:
            return out("Stopping...", ["The arms lower and switch off.", "When they are OFF, hold the RIGHT thumbstick 1 s to start again."], "warn")
        if all(arms[n]["mode"] == "off" for n in used):
            return out("Arms are OFF", ["Stand clear of the arms.", "Hold the RIGHT thumbstick down for 1 second to switch them on."], "info")
        if [n for n in used if label[n] == "STARTING"]:
            return out("Arms starting...", ["They switch on and hold where they are. Stand clear."], "warn")
        if held := [n for n in used if arms[n]["mode"] == "hold"]:
            return out(f"{name(held)} arm paused", [self.plain(arms[held[0]]["msg"]),
                                                     "Squeeze again to continue, or press Y to stop."], "warn")
        if self.xr_active and not self.aligned:
            return out("Press B to align", ["Stand beside the robot, facing the way it faces.",
                                            "Point the RIGHT controller straight ahead, then press B."], "info")
        if driving := [n for n in used if self.arms[n].engaged]:
            ends = [f"{n.upper()} {JOINT_NAMES[self.arms[n].blocked[0] - 1]} is at its end" for n in driving if self.arms[n].blocked]
            pushes = [f"{n.upper()} pushing {self.arms[n].push_now:.0f} N (max {self.args.push_n:.0f})" for n in driving
                      if self.arms[n].sliding and self.arms[n].push_now > 3]
            return out(f"Driving {name(driving)}", pushes + ends + ["Release the grip to freeze the arm.", "Trigger closes the gripper. Y = STOP."], "ok")
        return out("Ready to drive", ["Hold a GRIP to move that arm. Trigger = gripper.",
                                      "A/X fine mode - Y = STOP"], "ok")

    async def tel_reader(self):
        loop = asyncio.get_running_loop()
        brain = self

        class Proto(asyncio.DatagramProtocol):
            def datagram_received(self, data, addr):
                try:
                    m = json.loads(data)
                except ValueError:
                    return
                now = time.monotonic()
                for name, t in m.get("arms", {}).items():
                    if name in brain.arms:
                        r = brain.arms[name].on_tel(t, now)
                        if r == "buzz":
                            brain.say(f"{name}: {t['reason']}")
                            asyncio.ensure_future(brain.haptic(name, "fault"))
                        elif r == "resync":
                            brain.say(f"{name}: arm reset ({t['reason']}), squeeze again to drive")
                            asyncio.ensure_future(brain.haptic(name, "resync"))

        await loop.create_datagram_endpoint(Proto, local_addr=("127.0.0.1", self.args.tel_port))

    async def ticker(self):
        """Keep-alive targets when the headset is quiet, status to pages, stats once a second."""
        n, stat_t = 0, time.monotonic()
        while True:
            await asyncio.sleep(0.01)
            now = time.monotonic()
            if now - self.last_xr > XR_TIMEOUT:
                for a in self.arms.values():
                    if a.engaged:
                        a.engaged = False
                        self.say(f"{a.name}: no headset data, holding")
            if now - self.last_send > 0.035:
                self.send_targets()
            n += 1
            if n % 10 == 0:  # ~10 Hz: drives the headset's guide panel
                await self.broadcast(self.status())
            if now - stat_t >= 1.0:
                self.stats(now - stat_t)
                stat_t = now

    def stats(self, span):
        ts = self.xr_times
        gaps = np.diff(ts) * 1000 if len(ts) > 1 else np.array([0.0])
        ik = [x for a in self.arms.values() for x in a.ik_ms]
        sg = np.diff(self.src_ts) if len(self.src_ts) > 1 else np.array([0.0])
        rec = {"pose_hz": round(len(ts) / span, 1), "p99_gap_ms": round(float(np.percentile(gaps, 99)), 1),
               "max_gap_ms": round(float(gaps.max()), 1), "page_fps": self.page_fps,
               "src_p99_gap_ms": round(float(np.percentile(sg, 99)), 1), "src_max_gap_ms": round(float(sg.max()), 1),
               "remote": getattr(self, "last_remote", None),
               "ik_p99_ms": round(float(np.percentile(ik, 99)), 2) if ik else None,
               "ik_res_max_mm": round(max((x for a in self.arms.values() for x in a.ik_res), default=0) * 1000, 1),
               "arms": {n: {"mode": (a.tel or {}).get("mode"), "engaged": a.engaged, "fine": a.fine} for n, a in self.arms.items()},
               "inputs": self.seen}
        self.event("stats", **rec)
        self.log.flush()
        if ts:
            print(f"pose {rec['pose_hz']:5.1f} Hz  p99 gap {rec['p99_gap_ms']:5.1f} ms  max {rec['max_gap_ms']:5.1f}  "
                  f"ik p99 {rec['ik_p99_ms']} ms  ik res max {rec['ik_res_max_mm']} mm  "
                  + "  ".join(f"{n}:{v['mode']}{'*' if v['engaged'] else ''}{'(fine)' if v['fine'] else ''}" for n, v in rec["arms"].items()),
                  flush=True)
            print("    inputs: " + "  ".join(f"{h}: {v['frames']} frames ({v['ok']} tracked) squeeze max {v['sq_max']:.2f} "
                                             f"trigger max {v['tr_max']:.2f} presses A/X,B/Y,stick {v['presses']}"
                                             for h, v in self.seen.items()) + f"  page fps {self.page_fps}", flush=True)
        self.seen = {h: {"frames": 0, "ok": 0, "sq_max": 0.0, "tr_max": 0.0, "presses": [0, 0, 0]} for h in ("left", "right")}
        self.xr_times = self.xr_times[-1:]
        self.src_ts = self.src_ts[-1:]
        for a in self.arms.values():
            a.ik_ms, a.ik_res = [], []


async def ws_handler(request):
    brain = request.app["brain"]
    ws = web.WebSocketResponse(heartbeat=5.0, max_msg_size=1 << 16)
    await ws.prepare(request)
    brain.ws.add(ws)
    brain.say(f"page connected from {request.remote}")
    try:
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            try:
                m = json.loads(msg.data)
            except ValueError:
                continue
            if m.get("t") == "xr":
                brain.last_remote = request.remote
                await brain.on_xr(m)
            elif m.get("t") == "cmd":
                if m.get("cmd") == "xr_start":
                    brain.xr_ws = ws
                await brain.on_cmd(m)
    finally:
        brain.ws.discard(ws)
        if brain.xr_ws is ws:  # the AR page went away without saying 'xr_end'
            brain.xr_ws, brain.xr_active = None, False
            for a in brain.arms.values():
                a.engaged = False
        brain.say(f"page disconnected ({request.remote})")
    return ws


async def index(request):
    return web.FileResponse(os.path.join(ROOT, "quest", "index.html"), headers={"Cache-Control": "no-store"})


async def status_json(request):
    return web.json_response(request.app["brain"].status())


async def cmd_http(request):
    """POST /cmd {"cmd": "stop"|"power_on", "arm": ...}: the same commands as the page, from a laptop."""
    m = await request.json()
    await request.app["brain"].on_cmd(m)
    return web.json_response(request.app["brain"].status())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8443)
    ap.add_argument("--http-port", type=int, default=0, help="also serve plain http on localhost (adb reverse)")
    ap.add_argument("--cert", default=os.path.join(ROOT, "certs", "cert.pem"))
    ap.add_argument("--key", default=os.path.join(ROOT, "certs", "key.pem"))
    ap.add_argument("--loop-port", type=int, default=47001)
    ap.add_argument("--tel-port", type=int, default=47002)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--vmax", type=float, default=0.15, help="end-effector speed cap m/s")
    ap.add_argument("--fine-vmax", type=float, default=0.03, help="fine mode grasp-point speed cap m/s")
    ap.add_argument("--amax", type=float, default=1.5, help="grasp-point acceleration limit m/s^2 (smooth follower)")
    ap.add_argument("--fine-amax", type=float, default=0.3, help="same, fine mode")
    ap.add_argument("--wmax", type=float, default=2.0, help="end-effector rotation cap rad/s (the wrist joints' own top speed)")
    ap.add_argument("--jump", type=float, default=0.05, help="max target step per sample, m")
    ap.add_argument("--push-n", type=float, default=30.0, help="N: the thumbstick push in stops at this force")
    ap.add_argument("--min-cutoff", type=float, default=2.0)
    # beta multiplies the hand speed in m/s: 0.04 (Enactic's value) made the filter a fixed ~2 Hz low-pass with
    # ~80 ms delay at every speed; 20 keeps the same still-hand steadiness with 11-22 ms delay (measured on the Pi)
    ap.add_argument("--beta", type=float, default=20.0)
    ap.add_argument("--d-cutoff", type=float, default=1.5)
    ap.add_argument("--rot-cutoff", type=float, default=4.0, help="Hz, orientation SLERP smoothing")
    ap.add_argument("--ik", choices=["position", "pose"], default="pose",
                    help="pose: the gripper follows the hand's position and angle (default); position: position only")
    ap.add_argument("--rot-scale", type=float, default=1.0, help="1 = the gripper turns as much as the hand (default)")
    ap.add_argument("--allow-unverified", action="store_true")
    args = ap.parse_args()

    brain = Brain(args)
    app = web.Application()
    app["brain"] = brain
    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/status", status_json)
    app.router.add_post("/cmd", cmd_http)

    async def start_bg(app):
        await brain.tel_reader()
        app["ticker"] = asyncio.ensure_future(brain.ticker())

    app.on_startup.append(start_bg)
    ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ctx.load_cert_chain(args.cert, args.key)

    async def run():
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, "0.0.0.0", args.port, ssl_context=ctx).start()
        if args.http_port:
            await web.TCPSite(runner, "127.0.0.1", args.http_port).start()
        brain.say(f"serving https://0.0.0.0:{args.port}  (scale {args.scale}, vmax {args.vmax} m/s)")
        if brain.unverified and not args.allow_unverified:
            brain.say(f"calibration not verified for {brain.unverified}: those arms will not follow the hand")
        await asyncio.Event().wait()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
