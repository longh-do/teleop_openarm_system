#!/usr/bin/env python3
"""Teleop brain: serves the Quest page + WebSocket over HTTPS (one port, one certificate), turns controller poses
into joint targets, and sends them to motor_loop.py over localhost UDP.

  .venv/bin/python pi/teleop_server.py --scale 0.5 --vmax 0.1      # S5: half scale, low caps
Open https://<pi-ip>:8443 on the Quest (Advanced -> Proceed once), press Enter AR.

Controls: squeeze = clutch (hold to drive that arm), trigger = gripper, A/X = fine mode,
B = align: stand beside the robot FACING THE SAME WAY IT FACES, point the right controller straight ahead, press B.
Y = STOP (park + power off).
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
FINE = 0.3
XR_TIMEOUT = 0.15  # s without controller data -> disengage
# a comfortable pose to start driving from (model degrees): forearm forward, elbow 90. Hanging straight down the arm
# is singular (it can't move straight up), so S5+ start here. Left J1 and J5 are mirrored.
READY_MODEL_DEG = kin.HOME_DEG  # same pose the position-mode IK is pulled toward
READY_SPEED = 0.3  # rad/s, fastest joint during the ready move
GRIP_RATE = 3.0  # rad/s


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
        self.traj = None  # (q_from, q_to, t0, duration): a joint-space move the brain plays (ready pose)
        self.resume_t = -10.0

    def start_ready(self, now):
        if not self.on or self.q_cmd is None:
            return False
        q_to = kin.to_motor(self.calib, np.radians(READY_MODEL_DEG[self.name]))
        dur = max(4.0, float(np.max(np.abs(q_to - self.q_cmd))) / READY_SPEED)
        self.engaged, self.traj = False, (self.q_cmd.copy(), q_to, now, dur)
        return True

    def play_traj(self, now):
        if self.traj is None:
            return
        qa, qb, t0, dur = self.traj
        u = min(1.0, (now - t0) / dur)
        self.q_cmd = qa + (qb - qa) * (1 - math.cos(math.pi * u)) / 2
        if u >= 1.0:
            self.traj = None

    def on_tel(self, t, now):
        """New telemetry. If the loop reset its follower (epoch changed), re-sync and drop the clutch."""
        first = self.tel is None
        self.tel, self.tel_t = t, now
        if t["epoch"] != self.epoch:
            self.epoch = t["epoch"]
            self.traj = None
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
        self.traj = None
        x = np.array(self.tel["x"][:7])
        self.q_cmd = x
        E0p, E0R = kin.fk(self.name, self.calib, x)
        self.f_pos.reset(Cp)
        self.q_smooth = np.array(Cq, float)
        self.anchor = {"C0p": np.array(Cp), "C0R": kin.quat_to_mat(Cq), "E0p": E0p, "E0R": E0R,
                       "s": self.args.scale * (FINE if self.fine else 1.0)}
        self.ee_p, self.ee_R = E0p.copy(), E0R.copy()
        self.engaged = True

    def drive(self, Cp, Cq, M, dt):
        """One pose sample while engaged -> new q_cmd. Cp/Cq in WebXR coordinates. Returns 'limit' on a cap hit."""
        a = self.anchor
        Cp_f = self.f_pos(np.asarray(Cp, float), dt)
        alpha = 1 - math.exp(-dt * 2 * math.pi * self.args.rot_cutoff)
        self.q_smooth = quat_slerp(self.q_smooth, Cq, alpha)
        CR = kin.quat_to_mat(self.q_smooth)
        target_p = a["E0p"] + a["s"] * (M @ (Cp_f - a["C0p"]))
        # rot_scale 0 keeps the gripper's angle fixed (position only); 1 follows the controller's rotation
        rel = kin.log_rot(M @ (CR @ a["C0R"].T) @ M.T) * self.args.rot_scale
        target_R = kin.exp_rot(rel) @ a["E0R"]
        vmax = self.args.vmax * (self.args.fine_vmax / self.args.vmax if self.fine else 1.0)
        step = target_p - self.ee_p
        n = float(np.linalg.norm(step))
        lim = min(vmax * dt, self.args.jump)
        capped = n > lim
        if capped:
            step *= lim / n
        self.ee_p = self.ee_p + step
        self.ee_R, rcap = cap_rotation(self.ee_R, target_R, self.args.wmax * dt)
        self.trace = {"hand_cm": [round(float(v) * 100, 1) for v in M @ (np.asarray(Cp, float) - a["C0p"])],
                      "grip_cm": [round(float(v) * 100, 1) for v in self.ee_p - a["E0p"]]}
        t0 = time.perf_counter()
        q, res_p, res_r = kin.ik(self.name, self.calib, self.ee_p, self.ee_R, self.q_cmd, mode=self.args.ik,
                                 frozen=(self.tel or {}).get("skip", ()))
        self.ik_ms.append((time.perf_counter() - t0) * 1000)
        self.ik_res.append(res_p)
        self.q_cmd = q
        self.trace["q_model_deg"] = [round(float(v), 1) for v in np.degrees(kin.to_model(self.calib, q))]
        self.trace["res_mm"] = round(res_p * 1000, 1)
        if res_p > 0.01:  # unreachable: stop the commanded pose running away from the arm
            self.ee_p, self.ee_R = kin.fk(self.name, self.calib, q)
            return "limit"
        return None

    def grip(self, trigger, dt):
        g = self.calib.get("grip")
        if not g or self.g_cmd is None:
            return
        want = g["open"] + float(trigger) * (g["closed"] - g["open"])
        self.g_cmd += max(-GRIP_RATE * dt, min(GRIP_RATE * dt, want - self.g_cmd))


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
        self.prev_buttons = {}
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
            if not a.engaged:
                a.play_traj(now)
            if a.on and a.q_cmd is not None and a.epoch is not None:
                self.to_loop({"t": "target", "arm": a.name, "epoch": a.epoch,
                              "q": [round(float(v), 5) for v in a.q_cmd],
                              "g": None if a.g_cmd is None else round(float(a.g_cmd), 5)})
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
                    self.say(f"aligned: robot forward = controller direction (yaw {math.degrees(self.yaw):+.0f} deg)")
                    self.event("align", yaw=self.yaw)
                    await self.haptic("right", "align")
            if edge[1] and hand == "left":  # Y: STOP
                await self.stop("Y button")
            if edge[2] and a.tel is not None and a.tel["mode"] == "hold":  # thumbstick in hold: resume, then ready pose
                self.to_loop({"t": "cmd", "cmd": "resume", "arm": hand})
                a.ready_after_resume = True
            if getattr(a, "ready_after_resume", False) and a.on:
                a.ready_after_resume = False
                if a.start_ready(now):
                    self.say(f"{hand}: moving to the ready pose")
            elif edge[2] and not a.engaged and a.start_ready(now):  # thumbstick click: go to the ready pose
                self.say(f"{hand}: moving to the ready pose")
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
                        self.say(f"{hand}: still blocked ({a.tel['reason']}). Thumbstick click = ready pose, Y = STOP.")
                        await self.haptic(hand, "lost")
                    else:
                        a.resume_t = now
                        self.to_loop({"t": "cmd", "cmd": "resume", "arm": hand})
                        a.want_resume = True
                elif a.on and not was or (a.on and a.want_resume):
                    if self.unverified and hand in self.unverified and not self.args.allow_unverified:
                        if not was:
                            self.say(f"{hand}: calibration not verified (S1) - refusing to drive")
                    else:
                        a.engage(H["p"], H["q"], now)
                        a.want_resume = False
                        self.event("engage", arm=hand, E0=[round(v, 4) for v in a.anchor["E0p"]])
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
        elif cmd == "ready":
            now = time.monotonic()
            names = [m.get("arm")] if m.get("arm") in self.arms else list(self.arms)
            moved = [n for n in names if self.arms[n].start_ready(now)]
            self.say(f"moving to the ready pose: {moved or 'no arm is on'}")

    def status(self):
        now = time.monotonic()
        recent = [t for t in self.xr_times if now - t < 1.0]
        arms = {}
        for n, a in self.arms.items():
            t = a.tel
            mode = "no motor loop" if t is None or now - a.tel_t > 0.5 else t["mode"]
            mode = {"starting_read": "starting", "preload": "starting", "enabling": "starting", "starting_hold": "starting"}.get(mode, mode)
            arms[n] = {"mode": mode, "clutch": a.engaged, "fine": a.fine, "msg": (t or {}).get("reason", "")}
        return {"t": "status", "arms": arms, "pose_hz": len(recent), "msg": self.msg}

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
            busy = any(a.traj is not None for a in self.arms.values())
            if now - self.last_send > (0.008 if busy else 0.035):
                self.send_targets()
            n += 1
            if n % 20 == 0:
                await self.broadcast(self.status())
            if now - stat_t >= 1.0:
                self.stats(now - stat_t)
                stat_t = now

    def stats(self, span):
        ts = self.xr_times
        gaps = np.diff(ts) * 1000 if len(ts) > 1 else np.array([0.0])
        ik = [x for a in self.arms.values() for x in a.ik_ms]
        rec = {"pose_hz": round(len(ts) / span, 1), "p99_gap_ms": round(float(np.percentile(gaps, 99)), 1),
               "max_gap_ms": round(float(gaps.max()), 1), "page_fps": self.page_fps,
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
                await brain.on_xr(m)
            elif m.get("t") == "cmd":
                await brain.on_cmd(m)
    finally:
        brain.ws.discard(ws)
        brain.say(f"page disconnected ({request.remote})")
    return ws


async def index(request):
    return web.FileResponse(os.path.join(ROOT, "quest", "index.html"), headers={"Cache-Control": "no-store"})


async def status_json(request):
    return web.json_response(request.app["brain"].status())


async def cmd_http(request):
    """POST /cmd {"cmd": "ready"|"stop"|"power_on", "arm": ...}: the same commands as the page, from a laptop."""
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
    ap.add_argument("--fine-vmax", type=float, default=0.08)
    ap.add_argument("--wmax", type=float, default=1.5, help="end-effector rotation cap rad/s")
    ap.add_argument("--jump", type=float, default=0.05, help="max target step per sample, m")
    ap.add_argument("--min-cutoff", type=float, default=2.0)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--d-cutoff", type=float, default=1.5)
    ap.add_argument("--rot-cutoff", type=float, default=4.0, help="Hz, orientation SLERP smoothing")
    ap.add_argument("--ik", choices=["position", "pose"], default="position",
                    help="position: follow the hand's position only, wrist stays relaxed (default); pose: also its angle")
    ap.add_argument("--rot-scale", type=float, default=0.0, help="0 = gripper keeps its angle (position only), 1 = follow rotation")
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
