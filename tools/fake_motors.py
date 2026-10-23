#!/usr/bin/env python3
"""Simulated DAMIAO arm on a vcan bus, for testing motor_loop.py without hardware.

  sudo ip link add dev vcan0 type vcan && sudo ip link set vcan0 up
  python3 tools/fake_motors.py vcan0 --drop-j1 3.0     # J1 also drops its replies for ~35 ms every 3 s
Answers enable/disable/clear, position-velocity (0x100+j) and MIT (j) frames and register reads (0x7FF, 0x33);
register writes (0x55) are stored and read back.
"""
import argparse
import math
import os
import random
import socket
import struct
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pi"))
import damiao as dm  # noqa: E402


def encode_reply(j, state, p, v, t, vmax, tmax):
    pu, vu, tu = dm.to_uint(p, -dm.PMAX, dm.PMAX, 16), dm.to_uint(v, -vmax, vmax, 12), dm.to_uint(t, -tmax, tmax, 12)
    return bytes([state << 4 | j, pu >> 8, pu & 0xFF, vu >> 4, (vu & 0xF) << 4 | tu >> 8, tu & 0xFF, 30, 31])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bus")
    ap.add_argument("--drop-j1", type=float, default=0, help="J1 silent ~35 ms every N s (0 = never)")
    ap.add_argument("--start", default="0,-0.5,0,0.8,0,0,0,0.3", help="J1..J8 start angles, motor rad")
    ap.add_argument("--dead", type=int, default=0, help="this joint never answers")
    ap.add_argument("--lag-ms", type=float, default=25.0, help="joint response time (the real J5-J7 lag ~180 ms)")
    ap.add_argument("--stop-k", type=float, default=30.0, help="Nm/rad an end stop pushes back (the real arm: ~700)")
    ap.add_argument("--stops", default="", help="hard end stops, joint:lo:hi in motor rad, e.g. 7:-0.9:1.1,6:-0.6:0.7")
    args = ap.parse_args()
    stops = {int(j): (float(lo), float(hi)) for j, lo, hi in (x.split(":") for x in args.stops.split(",") if x)}
    s = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    s.bind((args.bus,))
    pos = [0.0] + [float(x) for x in args.start.split(",")]
    vel, on, tgt, spd = [0.0] * 9, [False] * 9, list(pos), [0.0] * 9
    mit_kp, last = [0.0] * 9, time.monotonic()
    regs = {}  # (joint, register) -> value written (0x55); reads return it, as from a motor's RAM
    next_drop, drop_until = time.monotonic() + (args.drop_j1 or 1e9), 0.0
    while True:
        can_id, _, d = struct.unpack(dm.FRAME, s.recv(16))
        now = time.monotonic()
        dt, last = now - last, now
        for j in range(1, 9):  # integrate: first-order tracking capped by the speed limit
            if on[j]:
                if j == 8:
                    step = (tgt[j] - pos[j]) * min(1.0, dt * mit_kp[j] * 2)
                else:
                    step = max(-spd[j] * dt, min(spd[j] * dt, (tgt[j] - pos[j]) * min(1.0, dt * 1000 / args.lag_ms)))
                vel[j] = step / dt if dt > 0 else 0.0
                pos[j] += step
                if j in stops and not stops[j][0] <= pos[j] <= stops[j][1]:
                    pos[j], vel[j] = min(max(pos[j], stops[j][0]), stops[j][1]), 0.0
            else:
                vel[j] = 0.0
        if now > next_drop:
            drop_until, next_drop = now + random.uniform(0.03, 0.04), now + args.drop_j1
        if can_id == 0x7FF:
            j, rid = d[0], d[3]
            if d[2] == 0x55 and 1 <= j <= 8:
                regs[j, rid] = struct.unpack("<f", d[4:8])[0]
            elif d[2] == 0x33 and 1 <= j <= 8:
                val = {dm.RID_VMAX: dm.VMAX[j], dm.RID_TMAX: dm.TMAX[j], dm.RID_PMAX: dm.PMAX}.get(rid, 0.0)
                val = regs.get((j, rid), val)
                s.send(struct.pack(dm.FRAME, 0x10 + j, 8, bytes([j, 0, 0x33, rid]) + struct.pack("<f", val)))
            continue
        if 0x101 <= can_id <= 0x108:
            j = can_id - 0x100
            tgt[j], spd[j] = struct.unpack("<ff", d)
        elif 1 <= can_id <= 8:
            j = can_id
            if d == dm.ENABLE:
                on[j], tgt[j] = True, pos[j] if j == 8 else tgt[j]
            elif d == dm.DISABLE:
                on[j] = False
            elif d != dm.CLEAR:
                tgt[j] = dm.to_float(d[0] << 8 | d[1], -dm.PMAX, dm.PMAX, 16)
                mit_kp[j] = dm.to_float(d[3] << 8 & 0xF00 | d[4], 0, 500, 12)
        else:
            continue
        if j == 1 and now < drop_until or j == args.dead:
            continue
        tau = 0.5 * math.sin(now) if on[j] else 0.0
        if on[j] and j in stops:  # pressing an end stop (only once the joint is at it): torque grows with how far the
            lo, hi = stops[j]      # command is past it
            if pos[j] >= hi - 1e-4:
                tau += args.stop_k * max(tgt[j] - hi, 0.0)
            elif pos[j] <= lo + 1e-4:
                tau += args.stop_k * min(tgt[j] - lo, 0.0)
        s.send(struct.pack(dm.FRAME, 0x10 + j, 8, encode_reply(j, int(on[j]), pos[j], vel[j], tau, dm.VMAX[j], dm.TMAX[j])))


if __name__ == "__main__":
    main()
