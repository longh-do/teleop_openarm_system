#!/usr/bin/env python3
"""Live 'how far from normal' readout while you turn an arm's joints BY HAND (motors stay off, nothing moves).

  on the Pi, in the repository:  python3 tools/untwist.py left [seconds, default 180]
Each number is degrees away from the normal hanging pose; turn that joint until it is near 0. If a number grows,
you are turning the wrong way. '!' marks a joint outside its allowed range; one far outside it makes the motor loop
refuse to power on. Only while the motor loop is NOT running (it owns the bus).
"""
import os
import socket
import struct
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pi"))
import numpy as np  # noqa: E402

import kinematics as kin  # noqa: E402

BUS = {"right": "can0", "left": "can1"}
NORMAL_MODEL_DEG = {"right": [0, 0, 0, 0, 90, 0, 0], "left": [0, 0, 0, 0, -90, 0, 0]}  # hanging straight down


def read_arm(s):
    q = {}
    for j in range(1, 8):
        s.send(struct.pack("<IB3x8s", j, 8, bytes.fromhex("FFFFFFFFFFFFFFFD")))  # 'motor off': it replies with its angle
        end = time.time() + 0.01
        while time.time() < end:
            try:
                cid, _, d = struct.unpack("<IB3x8s", s.recv(16))
            except OSError:
                continue
            if (cid & 0x7FF) == 0x10 + j:
                q[j] = (d[1] << 8 | d[2]) / 65535 * 25 - 12.5
                break
    return q


def main():
    arm = sys.argv[1] if len(sys.argv) > 1 else "left"
    if arm not in BUS:
        sys.exit(__doc__)
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 180
    # pi/motor_loop.py under any python, by relative or absolute path (the gain-tuning tool starts it by absolute path)
    if subprocess.run(["pgrep", "-f", "^[^ ]*python[^ ]* [^ ]*pi/motor_loop.py"], capture_output=True).returncode == 0:
        sys.exit("the motor loop is running: stop it first (Y on the headset, or Ctrl-C it)")
    c = kin.load_calib()[arm]
    lim = np.degrees(c["limits"])
    s = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    s.bind((BUS[arm],))
    s.settimeout(0.005)
    print(f"{arm} arm, motors OFF. Turn each joint by hand until its number is near 0 (degrees from normal hanging).")
    end = time.time() + seconds
    while time.time() < end:
        q = read_arm(s)
        if len(q) < 7:
            print(f"\r  only {len(q)}/7 joints answer (power or cable?)          ", end="", flush=True)
            time.sleep(0.3)
            continue
        x = np.degrees(kin.to_model(c, [q[j] for j in range(1, 8)]))
        off = x - np.array(NORMAL_MODEL_DEG[arm])
        parts = []
        for i in range(7):
            out = not (lim[i, 0] - 1 <= x[i] <= lim[i, 1] + 1)
            parts.append(f"J{i + 1} {off[i]:+5.0f}{'!' if out else ' '}")
        print("\r  " + "  ".join(parts) + "   ('!' = outside its allowed range)", end="", flush=True)
        time.sleep(0.25)
    print()


if __name__ == "__main__":
    main()
