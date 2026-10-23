#!/usr/bin/env python3
"""Find one gripper's two hard stops with a gentle push (MIT, KP 8, command kept within 0.1 rad of the
position, so under ~1 Nm). Only the gripper motor is switched on; the rest of the arm stays off.
  sudo .venv/bin/python tools/grip_stops.py left [lead]       (motor_loop.py stopped: it owns the bus)
Moves toward + until it stalls, then toward -, then back to where it started, then switches off. lead: rad the
command may run ahead of the position (default 0.1); the push is at most KP x lead Nm.
"""
import fcntl
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pi"))
import damiao as dm  # noqa: E402

ARM_BUS = {"right": "can0", "left": "can1"}
KP, KD, SPEED, HZ = 8.0, 0.5, 0.5, 100
LEAD = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1  # rad of push ahead of the position: KP * LEAD = max Nm
STALL_S, MAX_TRAVEL, MAX_S = 0.5, 4.0, 15.0


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ARM_BUS:
        sys.exit(__doc__)
    arm = sys.argv[1]
    bus = ARM_BUS[arm]
    lockf = open(f"/run/lock/openarm-{bus}.lock", "w")
    fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    s = dm.open_bus(bus, err_frames=False)
    vmax = dm.read_register(s, 8, dm.RID_VMAX) or 50.0
    tmax = dm.read_register(s, 8, dm.RID_TMAX) or 10.0

    def xfer(data):
        dm.send(s, 8, data)
        deadline = time.monotonic() + 0.02
        while (left := deadline - time.monotonic()) > 0:
            s.settimeout(left)
            try:
                can_id, d = dm.recv(s)
            except TimeoutError:
                break
            if can_id == 0x18:
                return dm.decode(d, vmax, tmax)
        return None

    r = xfer(dm.DISABLE)
    if r is None or r[0] != 0:
        sys.exit(f"gripper not answering or not off: {r}")
    p0 = r[1]
    print(f"{arm} gripper starts at {p0:+.3f} rad")
    stops = {}
    try:
        xfer(dm.mit(p0, 0, KP, KD, 0, vmax, tmax))
        r = xfer(dm.ENABLE)
        pos, t = r[1], r[1]
        for d in (+1, -1):
            best, best_t, t0 = pos, time.monotonic(), time.monotonic()
            while True:
                t = min(max(t + d * SPEED / HZ, pos - LEAD), pos + LEAD)
                r = xfer(dm.mit(t, 0, KP, KD, 0, vmax, tmax))
                if r is None:
                    continue
                st, pos, _, tau = r[:4]
                if st != 1:
                    raise RuntimeError(f"gripper state {dm.status_name(st)}")
                now = time.monotonic()
                if d * (pos - best) > 0.01:
                    best, best_t = pos, now
                if now - best_t > STALL_S:
                    stops[d] = round(pos, 4)
                    print(f"  stop moving {'+' if d > 0 else '-'}: {pos:+.3f} rad (torque {tau:+.2f} Nm)")
                    break
                if abs(pos - p0) > MAX_TRAVEL or now - t0 > MAX_S:
                    raise RuntimeError(f"no stop found moving {d:+d} (at {pos:+.3f})")
                time.sleep(1 / HZ)
            t = pos
        # back to the start, gently (friction may leave it a little short: give up after 4 s)
        t_end = time.monotonic() + 4.0
        while abs(pos - p0) > 0.03 and time.monotonic() < t_end:
            t = min(max(p0, pos - LEAD), pos + LEAD)
            r = xfer(dm.mit(t, 0, KP, KD, 0, vmax, tmax))
            if r:
                pos = r[1]
            time.sleep(1 / HZ)
    finally:
        for _ in range(3):
            dm.send(s, 8, dm.DISABLE)
            time.sleep(0.01)
        print("gripper off")
    print(f"RESULT {arm}: + stop {stops.get(1)}  - stop {stops.get(-1)}  travel {abs(stops[1] - stops[-1]):.3f} rad")


if __name__ == "__main__":
    main()
