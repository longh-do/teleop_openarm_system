#!/usr/bin/env python3
"""Real-time motor loop for the OpenArm arms (both buses in ONE process; see TRACKER D1).

Each cycle, per arm in turn: send 8 frames (J1-J7 position-velocity, J8 MIT), wait for that arm's 8 replies
(the two buses share one USB adapter, which drops replies when the bursts overlap), then the next arm.
Every joint follows its goal through a velocity/acceleration-limited smoother.

Targets come from the brain over localhost UDP (newest wins):
  {"t":"target","arm":"left","epoch":3,"q":[7 motor rad],"g":gripper motor rad or null}
  {"t":"cmd","cmd":"stop"|"power_on"|"resume"|"estop","arm":"left"|"right"|"all"}
Telemetry goes back at 50 Hz to --tel-port. The epoch is bumped whenever the follower is reset, and targets
stamped with an older epoch are ignored, so a stale target can never make the arm jump.

Usage on the Pi (root for SCHED_FIFO):
  sudo .venv/bin/python pi/motor_loop.py --arms left --read-only        # motors stay off, print angles
  sudo .venv/bin/python pi/motor_loop.py --arms left --test sine --duration 40
  sudo .venv/bin/python pi/motor_loop.py --arms right --test hold --duration 300
  sudo .venv/bin/python pi/motor_loop.py --arms left,right              # teleop: follow the brain
Ctrl-C once: park slowly to the start pose, then switch off. Twice: switch off at once (the arm drops).
"""
import argparse
import fcntl
import gc
import json
import math
import os
import signal
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import damiao as dm  # noqa: E402
import kinematics as kin  # noqa: E402

ARM_BUS = {"right": "can0", "left": "can1"}
JOINTS = range(1, 9)
VMAX = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 2.0, 6: 2.0, 7: 2.0, 8: 3.0}  # follower speed limit, rad/s
AMAX = {1: 4.0, 2: 4.0, 3: 4.0, 4: 4.0, 5: 4.0, 6: 4.0, 7: 4.0, 8: 12.0}  # follower acceleration limit, rad/s^2
PARK_V, PARK_A = 0.3, 1.0
BLOCK_TORQUE = {1: 15.0, 2: 15.0, 3: 10.0, 4: 10.0, 5: 4.0, 6: 4.0, 7: 4.0}  # Nm, sustained -> hold
TORQUE_CYCLES = 10  # consecutive cycles over BLOCK_TORQUE before tripping
GRIP_KP, GRIP_KD, GRIP_TORQUE = 8.0, 0.5, 1.0  # MIT gains; command clamped to +-GRIP_TORQUE/KP around the position
JUMP_GUARD = 0.5  # rad: a target this far from the follower is refused
GOAL_V_DECAY_S = 0.04  # forget the target's velocity estimate when targets stop for this long
ENABLE_HOLD_S, GRACE_CYCLES = 0.5, 10


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


class Arm:
    def __init__(self, name, args, calib):
        self.name, self.bus, self.args = name, ARM_BUS[name], args
        self.calib = calib
        self.lockf = open(f"/run/lock/openarm-{self.bus}.lock", "w")
        try:
            fcntl.flock(self.lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit(f"{self.bus} is locked by another program (one writer per bus)")
        self.sock = dm.open_bus(self.bus)
        self.vmax_reg, self.tmax_reg = dict(dm.VMAX), dict(dm.TMAX)
        n = 9
        self.pos, self.vel, self.tau = [0.0] * n, [0.0] * n, [0.0] * n
        self.state, self.temp = [-1] * n, [0] * n
        self.seen = [False] * n  # ever replied
        self.x, self.v, self.goal, self.goal_v = [0.0] * n, [0.0] * n, [0.0] * n, [0.0] * n
        self.gap_start = [None] * n
        self.misses, self.gaps = [0] * n, []  # gaps: (joint, ms)
        self.torque_count = [0] * n
        self.mode, self.reason, self.epoch, self.buzz = "off", "", 0, 0
        self.start_pose, self.park_pose, self.phase_t, self.park_t0 = None, None, 0.0, 0.0
        self.target_t, self.target_prev, self.brain_live = 0.0, None, False
        self.can_errors = 0
        self.err_max = [0.0] * n  # per joint, reset each stats second
        self.limits = None  # motor-space (lo, hi) per joint 1-7 when calibration is verified

    # ---- events -------------------------------------------------------------------------------------------
    def set_mode(self, mode, reason, log):
        if mode != self.mode or reason != self.reason:
            log.event("mode", arm=self.name, mode=mode, reason=reason)
            print(f"[{self.name}] {mode}: {reason}", flush=True)
        self.mode, self.reason = mode, reason

    def reset_follower(self, to_measured=True):
        for j in JOINTS:
            if to_measured:
                self.x[j] = self.pos[j]
            self.v[j], self.goal[j], self.goal_v[j] = 0.0, self.x[j], 0.0
        self.epoch += 1
        self.target_prev = None

    def hold(self, why, log, to_measured=False):
        """Stop where we are (or where the joints really are) and refuse brain targets until resume."""
        self.reset_follower(to_measured)
        self.buzz += 1
        self.set_mode("hold", why, log)

    # ---- bus ----------------------------------------------------------------------------------------------
    def read_registers(self, log):
        for j in JOINTS:
            for rid, table in ((dm.RID_VMAX, self.vmax_reg), (dm.RID_TMAX, self.tmax_reg)):
                v = dm.read_register(self.sock, j, rid)
                if v is not None and 0 < v < 1000:
                    table[j] = v
                else:
                    log.event("register_unread", arm=self.name, joint=j, rid=rid)
        log.event("registers", arm=self.name, vmax=self.vmax_reg, tmax=self.tmax_reg)

    def frame_for(self, j):
        if self.mode in ("off", "starting_read"):
            return j, dm.DISABLE
        if self.mode == "enabling":
            return j, dm.ENABLE
        if j == 8:
            lead = GRIP_TORQUE / GRIP_KP
            p = clamp(self.x[8], self.pos[8] - lead, self.pos[8] + lead) if self.seen[8] else self.x[8]
            return 8, dm.mit(p, 0.0, GRIP_KP, GRIP_KD, 0.0, self.vmax_reg[8], self.tmax_reg[8])
        speed = PARK_V if self.mode == "parking" else VMAX[j] * self.args.motor_speed_scale
        # the motors' own position loop trails its target like a ~0.14 s first-order lag (measured S2):
        # command ahead by speed x lag so the joint itself lands on the follower
        lead_ms = self.args.lead_ms_34 if j in (3, 4) else self.args.lead_ms
        lead = self.v[j] * lead_ms / 1000 if self.mode in ("on", "parking", "hold", "fault") else 0.0
        return 0x100 + j, dm.posvel(self.x[j] + lead, speed)

    def exchange(self, timeout):
        """Send one frame per joint, then collect this arm's replies until all 8 arrive or the timeout."""
        s, got = self.sock, 0
        s.setblocking(False)
        try:  # late replies from last cycle: still valid state
            while True:
                self._take(*dm.recv(s))
        except BlockingIOError:
            pass
        t0 = time.monotonic()
        for j in JOINTS:
            dm.send(s, *self.frame_for(j))
        deadline = t0 + timeout
        while got != 0x1FE:
            left = deadline - time.monotonic()
            if left <= 0:
                break
            s.settimeout(left)
            try:
                j = self._take(*dm.recv(s))
            except (TimeoutError, BlockingIOError):
                break
            if j:
                got |= 1 << j
        return got, t0

    def _take(self, can_id, d):
        if can_id & dm.CAN_ERR_FLAG:
            self.can_errors += 1
            return 0
        j = can_id - 0x10
        if not 1 <= j <= 8 or (d[0] & 0x0F) != j:
            return 0
        st, p, v, t, tm, tr = dm.decode(d, self.vmax_reg[j], self.tmax_reg[j])
        self.state[j], self.pos[j], self.vel[j], self.tau[j], self.temp[j] = st, p, v, t, max(tm, tr)
        self.seen[j] = True
        return j

    # ---- per-cycle logic ----------------------------------------------------------------------------------
    def after_exchange(self, got, t0, now, log):
        silence = self.args.silence_ms / 1000
        for j in JOINTS:
            if got >> j & 1:
                if self.gap_start[j] is not None:
                    ms = (now - self.gap_start[j]) * 1000
                    self.gaps.append((j, round(ms, 1)))
                    log.event("gap", arm=self.name, joint=j, ms=round(ms, 1), mode=self.mode)
                    self.gap_start[j] = None
            else:
                self.misses[j] += 1
                if self.gap_start[j] is None:
                    self.gap_start[j] = t0
                elif self.mode in ("on", "parking") and now - self.gap_start[j] > silence:
                    self.hold(f"J{j} silent for more than {self.args.silence_ms:.0f} ms", log)

    def start_sequence(self, now, log):
        """off -> starting_read (motors off, collect 8 fresh replies) -> preload -> enabling -> starting_hold -> on."""
        if self.mode == "starting_read":
            if not all(self.seen[j] and self.gap_start[j] is None for j in JOINTS):
                if now - self.phase_t > 1.0:
                    self.set_mode("off", "can't power on: not every joint is answering", log)
                return
            bad = [j for j in JOINTS if self.state[j] >= 8]
            if bad:
                self.set_mode("off", f"can't power on: J{bad[0]} reports {dm.status_name(self.state[bad[0]])}", log)
                return
            if self.limits is not None:
                out = [j for j in range(1, 8) if not self.limits[j - 1][0] <= self.pos[j] <= self.limits[j - 1][1]]
                if out:
                    j = out[0]
                    self.set_mode("off", f"can't power on: J{j} at {math.degrees(self.pos[j]):+.1f} deg is outside its limits", log)
                    return
            self.reset_follower(True)
            if self.start_pose is None:
                self.start_pose = list(self.pos)
                log.event("start_pose", arm=self.name, q=[round(p, 4) for p in self.pos[1:]])
            self.park_pose = list(self.start_pose)
            self.set_mode("preload", "targets = measured angles", log)
        elif self.mode == "preload":
            self.set_mode("enabling", "enable", log)
        elif self.mode == "enabling":
            self.phase_t = now
            self.set_mode("starting_hold", "holding 0.5 s", log)
        elif self.mode == "starting_hold" and now - self.phase_t > ENABLE_HOLD_S:
            self.target_t, self.brain_live = 0.0, False
            self.set_mode("on", "ready", log)

    def safety(self, now, log):
        if self.mode not in ("on", "parking", "starting_hold"):
            return
        grace = self.mode == "starting_hold" and now - self.phase_t < GRACE_CYCLES / self.args.hz
        for j in JOINTS:
            if self.gap_start[j] is not None:
                continue  # stale this cycle: ride through
            st = self.state[j]
            if st >= 8:
                return self.set_fault(f"J{j} reports {dm.status_name(st)}", log)
            if st != 1 and not grace:
                return self.set_fault(f"J{j} switched itself off", log)
            if j == 8:
                continue
            self.err_max[j] = max(self.err_max[j], abs(self.pos[j] - self.x[j]))
            if abs(self.pos[j] - self.x[j]) > self.args.max_err:
                return self.hold(f"J{j} tracking error {math.degrees(self.pos[j] - self.x[j]):+.1f} deg", log, True)
            if abs(self.tau[j]) > BLOCK_TORQUE[j]:
                self.torque_count[j] += 1
                if self.torque_count[j] >= TORQUE_CYCLES:
                    self.torque_count[j] = 0
                    return self.hold(f"J{j} torque {self.tau[j]:+.1f} Nm (blocked?)", log, True)
            else:
                self.torque_count[j] = 0

    def set_fault(self, why, log):
        self.reset_follower(False)
        self.buzz += 1
        self.set_mode("fault", why, log)

    def accept_target(self, msg, now, log):
        if self.mode != "on" or msg.get("epoch") != self.epoch:
            return
        q = msg["q"]
        if len(q) != 7 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in q):
            return
        worst = max(range(7), key=lambda i: abs(q[i] - self.x[i + 1]))
        if abs(q[worst] - self.x[worst + 1]) > JUMP_GUARD:
            return self.hold(f"refused target: J{worst + 1} jump {math.degrees(q[worst] - self.x[worst + 1]):+.0f} deg", log)
        if self.limits is not None:
            q = [clamp(v, *self.limits[i]) for i, v in enumerate(q)]
        g = msg.get("g")
        goal = list(q) + [g if isinstance(g, (int, float)) and math.isfinite(g) else self.goal[8]]
        if self.calib.get("grip"):
            lo, hi = sorted((self.calib["grip"]["open"], self.calib["grip"]["closed"]))
            goal[7] = clamp(goal[7], lo, hi)
        if self.target_prev is not None and 0.002 < now - self.target_t < 0.1:
            dt = now - self.target_t
            for j in JOINTS:
                self.goal_v[j] = clamp((goal[j - 1] - self.target_prev[j - 1]) / dt, -VMAX[j], VMAX[j])
        else:
            for j in JOINTS:
                self.goal_v[j] = 0.0
        for j in JOINTS:
            self.goal[j] = goal[j - 1]
        if not self.brain_live:
            self.reason = "following the brain"
        self.target_prev, self.target_t, self.brain_live = goal, now, True

    def follow(self, dt, now):
        if self.mode not in ("on", "hold", "parking", "fault", "starting_hold"):
            return
        if self.mode == "on" and self.brain_live and now - self.target_t > self.args.brain_timeout_ms / 1000:
            # brain went quiet: brake to a stop, and make the brain re-sync before it may move us again
            self.brain_live = False
            for j in JOINTS:
                a = AMAX[j]
                self.goal[j] = self.x[j] + self.v[j] * abs(self.v[j]) / (2 * a)
                self.goal_v[j] = 0.0
            self.epoch += 1
            self.target_prev = None
            self.reason = "brain quiet: braked to a stop"
        if now - self.target_t > GOAL_V_DECAY_S:
            for j in JOINTS:
                self.goal_v[j] = 0.0
        parking = self.mode == "parking"
        for j in JOINTS:
            vmax = PARK_V if parking else VMAX[j]
            amax = PARK_A if parking else AMAX[j]
            x, v, g, vg = self.x[j], self.v[j], self.goal[j], self.goal_v[j] if not parking else 0.0
            e = g - x
            if abs(e) < 1e-4 and abs(v - vg) <= amax * dt:
                self.x[j], self.v[j] = g, vg
                continue
            vd = clamp(vg + math.copysign(min(vmax, math.sqrt(2 * amax * abs(e))), e), -vmax, vmax)
            v += clamp(vd - v, -amax * dt, amax * dt)
            xn = x + v * dt
            if (g - xn) * e < 0:  # overshot: land on the goal
                xn, v = g, vg
            self.x[j], self.v[j] = xn, v

    def park_progress(self, now, log):
        if self.mode != "parking":
            return False
        done = all(abs(self.x[j] - self.goal[j]) < 1e-3 for j in JOINTS) and \
            all(abs(self.pos[j] - self.goal[j]) < 0.05 for j in range(1, 8))
        if done:
            if not self.phase_t:
                self.phase_t = now
            elif now - self.phase_t > 1.0:
                self.set_mode("off", "parked", log)
                return True
        elif now - self.park_t0 > self.args.park_timeout:
            self.hold("park did not finish in time", log, True)
        return False

    def command(self, cmd, now, log):
        if cmd == "power_on" and self.mode == "off":
            self.phase_t = now
            self.set_mode("starting_read", "reading joints (motors off)", log)
        elif cmd == "stop" and self.mode in ("on", "hold", "fault", "starting_hold"):
            self.reset_follower(self.mode != "on")
            for j in JOINTS:
                self.goal[j] = self.park_pose[j]
            self.phase_t, self.park_t0 = 0.0, now
            self.set_mode("parking", "STOP: parking slowly, then power off", log)
        elif cmd == "stop" and self.mode in ("starting_read", "preload", "enabling"):
            self.set_mode("off", "STOP during start", log)
        elif cmd == "resume" and self.mode == "hold":
            self.reset_follower(False)
            self.set_mode("on", "resumed", log)
        elif cmd == "estop" and self.mode != "off":
            self.set_mode("off", "E-STOP: motors off", log)

    def telemetry(self):
        r = lambda a: [round(a[j], 4) for j in JOINTS]  # noqa: E731
        return {"mode": self.mode, "reason": self.reason, "epoch": self.epoch, "buzz": self.buzz,
                "x": r(self.x), "q": r(self.pos), "tau": [round(self.tau[j], 2) for j in JOINTS],
                "st": self.state[1:], "temp": self.temp[1:], "stale": [self.gap_start[j] is not None for j in JOINTS],
                "misses": self.misses[1:], "can_errors": self.can_errors}

    def disable_now(self):
        for _ in range(3):
            for j in JOINTS:
                try:
                    dm.send(self.sock, j, dm.DISABLE)
                except OSError:
                    pass
            time.sleep(0.003)


class Log:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "a", buffering=1 << 16)
        self.path = path
        if os.environ.get("SUDO_UID"):  # run with sudo: keep logs owned by the normal user
            uid, gid = int(os.environ["SUDO_UID"]), int(os.environ.get("SUDO_GID", os.environ["SUDO_UID"]))
            for p in (os.path.dirname(path), path):
                os.chown(p, uid, gid)

    def event(self, kind, **kw):
        self.f.write(json.dumps({"t": round(time.time(), 4), "kind": kind, **kw}) + "\n")

    def flush(self):
        self.f.flush()


def find_other_writer():
    for pid in os.listdir("/proc"):
        if pid.isdigit() and int(pid) != os.getpid():
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    argv = f.read().decode(errors="replace").split("\0")
            except OSError:
                continue
            if not os.path.basename(argv[0]).startswith("python"):
                continue  # shells and sudo merely mention the name
            if any(a.endswith(("teleop_web.py", "motor_loop.py")) for a in argv[1:3]):
                return pid, " ".join(argv).strip()
    return None


def realtime(cpu, prio):
    notes = []
    try:
        os.sched_setaffinity(0, {cpu})
        notes.append(f"cpu {cpu}")
    except OSError as e:
        notes.append(f"no affinity ({e})")
    try:
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(prio))
        notes.append(f"SCHED_FIFO {prio}")
    except (PermissionError, OSError) as e:
        notes.append(f"no SCHED_FIFO ({e})")
    return ", ".join(notes)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="left,right")
    ap.add_argument("--buses", default="right=can0,left=can1", help="arm=bus map (vcan for the simulator)")
    ap.add_argument("--hz", type=float, default=200)
    ap.add_argument("--reply-timeout-ms", type=float, default=2.6)
    ap.add_argument("--silence-ms", type=float, default=150)
    ap.add_argument("--brain-timeout-ms", type=float, default=100)
    ap.add_argument("--max-err", type=float, default=0.15)
    ap.add_argument("--lead-ms", type=float, default=180, help="velocity lead to cancel the motors' position-loop lag")
    ap.add_argument("--lead-ms-34", type=float, default=350, help="same, for the DM4340 joints J3/J4")
    ap.add_argument("--motor-speed-scale", type=float, default=1.0, help="x the follower vmax, sent as the posvel speed limit")
    ap.add_argument("--park-timeout", type=float, default=30)
    ap.add_argument("--read-only", action="store_true", help="never enable; print angles once a second")
    ap.add_argument("--test", choices=["none", "hold", "sine"], default="none")
    ap.add_argument("--duration", type=float, default=40)
    ap.add_argument("--amp-deg", type=float, default=5)
    ap.add_argument("--freq", type=float, default=0.2)
    ap.add_argument("--test-joints", default="1,2,3,4,5,6,7")
    ap.add_argument("--port", type=int, default=47001)
    ap.add_argument("--tel-port", type=int, default=47002)
    ap.add_argument("--cpu", type=int, default=3)
    ap.add_argument("--prio", type=int, default=50)
    ap.add_argument("--log-dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs"))
    args = ap.parse_args()

    ARM_BUS.update(dict(kv.split("=") for kv in args.buses.split(",")))
    names = [a.strip() for a in args.arms.split(",") if a.strip()]
    assert all(n in ARM_BUS for n in names), f"--arms must be from {list(ARM_BUS)}"
    other = None if all(ARM_BUS[n].startswith("vcan") for n in names) else find_other_writer()
    if other:
        sys.exit(f"another program drives the arm buses (pid {other[0]}: {other[1][:80]}). Stop it first.")
    calib = kin.load_calib()
    log = Log(os.path.join(args.log_dir, time.strftime("motor-%Y%m%d-%H%M%S.jsonl")))
    arms = [Arm(n, args, {"grip": calib[n]["grip"]}) for n in names]
    for a in arms:
        if calib[a.name]["verified"]:
            lim = kin.motor_limits(a.name, calib[a.name])
            a.limits = [tuple(map(float, row)) for row in lim]
        a.read_registers(log)
    rt = realtime(args.cpu, args.prio)
    log.event("start", args=vars(args), realtime=rt, calib_verified={a.name: a.limits is not None for a in arms})
    print(f"motor loop: arms {names}, {args.hz:.0f} Hz, {rt}, log {log.path}", flush=True)
    if not any(a.limits is not None for a in arms):
        print("note: calibration not verified -> joint-limit check at start is skipped", flush=True)

    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.bind(("127.0.0.1", args.port))
    udp.setblocking(False)
    tel = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    stop_requests = {"n": 0}

    def on_signal(*_):
        stop_requests["n"] += 1

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    if not args.read_only:
        for a in arms:
            a.command("power_on", time.monotonic(), log)

    period = 1 / args.hz
    gc.collect()
    gc.freeze()
    gc.disable()
    t_next = time.monotonic()
    t_prev = t_next
    t_start = t_next
    test_t0, test_done = None, False
    test_joints = [int(j) for j in args.test_joints.split(",")]
    periods, works, late, overruns, cycles = [], [], [], 0, 0
    stat_t = t_next
    handled_stops = 0
    try:
        while True:
            now = time.monotonic()
            late.append(now - t_next)
            dt = clamp(now - t_prev, 0.5 * period, 2 * period)
            periods.append(now - t_prev)
            t_prev = now

            # 1. inputs: newest target per arm, commands in order
            newest = {}
            while True:
                try:
                    data = udp.recv(4096)
                except BlockingIOError:
                    break
                try:
                    msg = json.loads(data)
                except ValueError:
                    continue
                if msg.get("t") == "target" and msg.get("arm") in names:
                    newest[msg["arm"]] = msg
                elif msg.get("t") == "cmd":
                    for a in arms:
                        if msg.get("arm") in (a.name, "all"):
                            a.command(msg.get("cmd"), now, log)
            if stop_requests["n"] > handled_stops:
                handled_stops = stop_requests["n"]
                if handled_stops == 1:
                    print("Ctrl-C: parking, then power off (Ctrl-C again = off now)", flush=True)
                    for a in arms:
                        a.command("stop", now, log)
                else:
                    for a in arms:
                        a.command("estop", now, log)
            for a in arms:
                if a.name in newest and args.test == "none":
                    a.accept_target(newest[a.name], now, log)

            # test generators stand in for the brain (they start once every arm is on, then drive the arms still on)
            if args.test != "none" and test_t0 is None and all(a.mode == "on" for a in arms):
                test_t0 = now
                print(f"test {args.test}: {args.duration:.0f} s", flush=True)
            if test_t0 is not None and not test_done:
                tt = now - test_t0
                w, amp = 2 * math.pi * args.freq, math.radians(args.amp_deg)
                for a in arms:
                    if a.mode != "on":
                        continue
                    for j in JOINTS:
                        sine = args.test == "sine" and j in test_joints
                        a.goal[j] = a.start_pose[j] + (amp * math.sin(w * tt) if sine else 0.0)
                        a.goal_v[j] = amp * w * math.cos(w * tt) if sine else 0.0
                        a.target_t = now
                if tt > args.duration or handled_stops:
                    test_done = True
                    for a in arms:
                        a.command("stop", now, log)

            # 2. follower step, then exchange frames arm by arm
            for a in arms:
                a.follow(dt, now)
            for a in arms:
                got, t0 = a.exchange(args.reply_timeout_ms / 1000)
                a.after_exchange(got, t0, time.monotonic(), log)
            now2 = time.monotonic()
            for a in arms:
                if a.mode in ("starting_read", "preload", "enabling", "starting_hold"):
                    a.start_sequence(now2, log)
                a.safety(now2, log)
                a.park_progress(now2, log)

            # 3. telemetry at 50 Hz
            cycles += 1
            if cycles % max(1, int(args.hz / 50)) == 0:
                try:
                    tel.sendto(json.dumps({"t": "tel", "ts": now2, "hz": args.hz,
                                           "arms": {a.name: a.telemetry() for a in arms}}).encode(),
                               ("127.0.0.1", args.tel_port))
                except OSError:
                    pass

            # 4. stats once a second
            work = time.monotonic() - now
            works.append(work)
            if now2 - stat_t >= 1.0:
                stats(args, arms, periods, works, late, overruns, log, now2 - stat_t, t_start)
                periods, works, late, overruns = [], [], [], 0
                stat_t = now2
                gc.collect(0)

            # exit conditions
            if args.read_only and handled_stops:
                break
            if all(a.mode == "off" for a in arms) and (not args.read_only) and \
                    (handled_stops or test_t0 is not None or all(a.reason.startswith("can't") for a in arms)):
                break

            t_next += period
            delay = t_next - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                overruns += 1
                if delay < -period:
                    t_next = time.monotonic()  # fell far behind: don't try to catch up
    except BaseException as e:
        log.event("crash", error=repr(e))
        print(f"motor loop crashed ({e!r}): switching all motors off", flush=True)
        raise
    finally:
        for a in arms:
            a.disable_now()
        log.event("exit", gaps={a.name: a.gaps[-200:] for a in arms}, misses={a.name: a.misses[1:] for a in arms})
        log.flush()
        print("all motors off. gaps:", {a.name: len(a.gaps) for a in arms}, "log:", log.path, flush=True)


def pct(xs, p):
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(p / 100 * len(s)))]


def stats(args, arms, periods, works, late, overruns, log, span, t_start):
    n = len(periods)
    period = 1 / args.hz
    on_time = sum(1 for p in periods if p < 1.5 * period) / max(1, n)
    rec = {"hz": round(n / span, 1), "p99_period_ms": round(pct(periods, 99) * 1000, 2),
           "max_period_ms": round(max(periods, default=0) * 1000, 2), "on_time": round(on_time, 4),
           "p99_work_ms": round(pct(works, 99) * 1000, 2), "p99_wake_late_ms": round(pct(late, 99) * 1000, 3),
           "overruns": overruns,
           "arms": {a.name: {"mode": a.mode, "misses": a.misses[1:], "gaps": len(a.gaps), "can_errors": a.can_errors,
                             "max_err_deg": round(math.degrees(max(a.err_max[1:8])), 2),
                             "err_deg": [round(math.degrees(e), 2) for e in a.err_max[1:8]],
                             "max_tau": [round(a.tau[j], 1) for j in JOINTS],
                             "temp": a.temp[1:]} for a in arms}}
    log.event("stats", **rec)
    for a in arms:
        a.err_max = [0.0] * 9
    log.flush()
    parts = [f"{rec['hz']:6.1f} Hz", f"p99 {rec['p99_period_ms']:.2f} ms", f"max {rec['max_period_ms']:.2f}",
             f"work p99 {rec['p99_work_ms']:.2f}", f"on-time {on_time * 100:.2f}%", f"overruns {overruns}"]
    for a in arms:
        miss = sum(a.misses[1:])
        if args.read_only:
            parts.append(f"{a.name}: " + " ".join(f"{math.degrees(a.pos[j]):+6.1f}" for j in JOINTS))
        else:
            parts.append(f"{a.name} {a.mode} miss {miss} gaps {len(a.gaps)} err {rec['arms'][a.name]['max_err_deg']:.2f}deg")
    print(f"[{time.monotonic() - t_start:6.1f}s] " + " | ".join(parts), flush=True)


if __name__ == "__main__":
    main()
