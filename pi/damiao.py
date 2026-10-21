"""DAMIAO motor protocol over raw SocketCAN (classic CAN, 1 Mbit/s). Stdlib only.

J1-J7 run position-velocity mode (id 0x100+joint, <ff pos rad, speed limit rad/s>);
J8 (gripper) runs MIT mode (id = joint, packed p/v/kp/kd/t). Replies come back on 0x010+joint.
"""
import socket
import struct

FRAME = "=IB3x8s"
ENABLE, DISABLE, CLEAR = bytes.fromhex("FFFFFFFFFFFFFFFC"), bytes.fromhex("FFFFFFFFFFFFFFFD"), bytes.fromhex("FFFFFFFFFFFFFFFB")
PMAX = 12.5
# DM8009 (J1-2), DM4340 (J3-4), DM4310 (J5-8). Defaults; motor_loop reads the real values from registers 0x16/0x17.
VMAX = {1: 45.0, 2: 45.0, 3: 8.0, 4: 8.0, 5: 30.0, 6: 30.0, 7: 30.0, 8: 50.0}
TMAX = {1: 54.0, 2: 54.0, 3: 28.0, 4: 28.0, 5: 10.0, 6: 10.0, 7: 10.0, 8: 10.0}
RID_PMAX, RID_VMAX, RID_TMAX, RID_CTRL_MODE = 0x15, 0x16, 0x17, 0x0A
RID_ACC, RID_DEC, RID_KP_APR, RID_KI_APR = 0x04, 0x05, 0x1B, 0x1C
STATUS = {0: "off", 1: "on", 8: "over-voltage", 9: "under-voltage", 0xA: "over-current",
          0xB: "MOSFET overheat", 0xC: "coil overheat", 0xD: "comms lost", 0xE: "overload"}
CAN_ERR_FLAG = 0x20000000


def status_name(code):
    return STATUS.get(code, f"code {code:#x}")


def to_uint(x, lo, hi, bits):
    x = min(max(x, lo), hi)
    return int((x - lo) * ((1 << bits) - 1) / (hi - lo))


def to_float(u, lo, hi, bits):
    return u * (hi - lo) / ((1 << bits) - 1) + lo


def posvel(p, v):
    return struct.pack("<ff", p, v)


def mit(p, v, kp, kd, t, vmax=50.0, tmax=10.0):
    p, v = to_uint(p, -PMAX, PMAX, 16), to_uint(v, -vmax, vmax, 12)
    kp, kd, t = to_uint(kp, 0, 500, 12), to_uint(kd, 0, 5, 12), to_uint(t, -tmax, tmax, 12)
    return bytes([p >> 8, p & 0xFF, v >> 4, (v & 0xF) << 4 | kp >> 8, kp & 0xFF,
                  kd >> 4, (kd & 0xF) << 4 | t >> 8, t & 0xFF])


def decode(d, vmax, tmax):
    """-> (state, pos rad, vel rad/s, torque Nm, mosfet degC, rotor degC). state 0 off, 1 on, >= 8 fault."""
    return (d[0] >> 4, to_float(d[1] << 8 | d[2], -PMAX, PMAX, 16),
            to_float(d[3] << 4 | d[4] >> 4, -vmax, vmax, 12),
            to_float((d[4] & 0xF) << 8 | d[5], -tmax, tmax, 12), d[6], d[7])


def open_bus(channel, err_frames=True):
    """Raw socket that sees replies 0x010-0x01F (and register replies, which also use 0x01x) plus error frames."""
    s = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    s.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER, struct.pack("=II", 0x010, 0x7F0))
    if err_frames:
        s.setsockopt(socket.SOL_CAN_RAW, getattr(socket, "CAN_RAW_ERR_FILTER", 2), struct.pack("=I", 0x1FFFFFFF))
    s.bind((channel,))
    return s


def send(s, can_id, data):
    s.send(struct.pack(FRAME, can_id, 8, data))


def recv(s):
    """-> (can_id, data). Raises BlockingIOError / TimeoutError per the socket's timeout."""
    can_id, _, d = struct.unpack(FRAME, s.recv(16))
    return can_id, d


def read_register(s, motor, rid, timeout=0.05, fmt="<f"):
    """Query a motor register (only while nothing else is talking on this bus). -> value or None."""
    import time
    send(s, 0x7FF, bytes([motor, 0, 0x33, rid, 0, 0, 0, 0]))
    deadline = time.monotonic() + timeout
    while (left := deadline - time.monotonic()) > 0:
        s.settimeout(left)
        try:
            can_id, d = recv(s)
        except (TimeoutError, BlockingIOError):
            break
        if not can_id & CAN_ERR_FLAG and d[2] in (0x33, 0x55) and d[0] == motor and d[3] == rid:
            return struct.unpack(fmt, d[4:8])[0]
    return None


def write_register(s, motor, rid, value, timeout=0.05, fmt="<f"):
    """Write a motor register in RAM only (0x55; lost at power-off, nothing saved to flash: that would need 0xAA,
    which this module never sends). Only while nothing else is talking on this bus. -> value read back, or None."""
    send(s, 0x7FF, bytes([motor, 0, 0x55, rid]) + struct.pack(fmt, value))
    import time
    time.sleep(0.002)
    return read_register(s, motor, rid, timeout, fmt)


def _selftest():
    assert mit(0, 0, 0, 0, 0) == bytes.fromhex("7FFF7FF0000007FF")
    assert abs(decode(bytes.fromhex("0882417FF7FA1E1C"), 50, 10)[1] - 0.2203) < 1e-3
    assert decode(bytes.fromhex("117FBF7FF8311D19"), 45, 54)[0] == 1


_selftest()
