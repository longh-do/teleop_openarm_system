"""Forward/inverse kinematics for both OpenArm arms (numpy only).

Geometry, axes, limits, TOOL and the posture idea come from Monty's OpenArm retarget,
itself transcribed from openarm_mujoco v2 openarm_bimanual.xml. Rig frame: x forward, y left, z up, arms
hanging at model q = 0.

Motor angles and model angles are related per joint by the calibration (calib.json, measured in S1):
    q_model = sign * q_motor + offset
Everything in this module works in MOTOR radians at its interface; the model frame stays inside.
"""
import json
import math
import os

import numpy as np

#         body pos (m)          joint axis
RIGHT = [((0.0, -0.0625, 0.0), (0.0, -1.0, 0.0)),
         ((0.0, -0.06, 0.0), (-1.0, 0.0, 0.0)),
         ((0.0, 0.0, -0.06625), (0.0, 0.0, -1.0)),
         ((0.0, 0.0, -0.15375), (0.0, -1.0, 0.0)),
         ((0.0, 0.0, -0.0955), (0.0, 0.0, -1.0)),
         ((0.0, 0.0, -0.1205), (0.0, 1.0, 0.0)),
         ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))]
LEFT = [((0.0, 0.0625, 0.0), (0.0, 1.0, 0.0)),  # mirror: y offsets flip, and with them the sense of J1 and J6
        ((0.0, 0.06, 0.0), (-1.0, 0.0, 0.0)),
        ((0.0, 0.0, -0.06625), (0.0, 0.0, -1.0)),
        ((0.0, 0.0, -0.15375), (0.0, -1.0, 0.0)),
        ((0.0, 0.0, -0.0955), (0.0, 0.0, -1.0)),
        ((0.0, 0.0, -0.1205), (0.0, -1.0, 0.0)),
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))]
CHAIN = {"right": RIGHT, "left": LEFT}
BASE = {"right": np.array([0.0, -0.031, 0.0]), "left": np.array([0.0, 0.031, 0.0])}
_RL = [(-1.3963, 3.4907), (-0.1745, 3.3161), (-1.5708, 1.5708), (0.0, 2.4435), (-1.5708, 1.5708), (-0.7854, 0.7854), (-1.5708, 1.5708)]
MODEL_LIMITS = {"right": np.array(_RL), "left": np.array([(-3.4907, 1.3963), (-3.3161, 0.1745)] + _RL[2:])}
TOOL = np.array([0.0, 0.0, -0.10])  # grasp point in the flange frame
MARGIN = math.radians(5)
POSTURE = 0.08
DAMP = 0.05

CALIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib.json")


def load_calib(path=CALIB_PATH):
    """-> {arm: {"sign": np(7), "offset": np(7) model rad, "limits": np(7, 2) model rad, "grip": {"open", "closed"}
    motor rad | None, "verified": bool}}. calib.json gives offset_deg (model = sign * motor + offset) and optional
    limits_deg {joint: [lo, hi]} overriding the model's limits (in model degrees)."""
    out = {a: {"sign": np.ones(7), "offset": np.zeros(7), "limits": MODEL_LIMITS[a].copy(), "grip": None,
               "verified": False} for a in CHAIN}
    if os.path.exists(path):
        with open(path) as f:
            raw = json.load(f)
        for a, c in raw.items():
            if a not in out:
                continue
            o = out[a]
            o["sign"] = np.array(c["sign"], float)
            o["offset"] = np.radians(c["offset_deg"]) if "offset_deg" in c else np.array(c.get("offset", np.zeros(7)), float)
            for j, (lo, hi) in c.get("limits_deg", {}).items():
                o["limits"][int(j) - 1] = np.radians([lo, hi])
            o["grip"], o["verified"] = c.get("grip"), bool(c.get("verified", False))
    return out


def to_model(c, q):
    return c["sign"] * np.asarray(q, float)[:7] + c["offset"]


def to_motor(c, x):
    return (np.asarray(x, float) - c["offset"]) * c["sign"]


def motor_limits(arm, c, margin=0.0):
    """Joint limits in MOTOR radians (lo, hi) per joint, shrunk by margin."""
    lim = c["limits"]
    a, b = to_motor(c, lim[:, 0] + margin), to_motor(c, lim[:, 1] - margin)
    return np.stack([np.minimum(a, b), np.maximum(a, b)], axis=1)


def rot(axis, t):
    a = np.asarray(axis, float)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K


def quat_to_mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def log_rot(R):
    """rotation matrix -> axis * angle"""
    th = math.acos(min(1.0, max(-1.0, (np.trace(R) - 1) / 2)))
    if th < 1e-8:
        return np.zeros(3)
    if th > math.pi - 1e-4:  # near 180 deg: use the diagonal
        ax = np.sqrt(np.maximum((np.diag(R) + 1) / 2, 0))
        i = int(np.argmax(ax))
        ax[(i + 1) % 3] = math.copysign(ax[(i + 1) % 3], R[i, (i + 1) % 3])
        ax[(i + 2) % 3] = math.copysign(ax[(i + 2) % 3], R[i, (i + 2) % 3])
        return ax / np.linalg.norm(ax) * th
    return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) * th / (2 * math.sin(th))


def exp_rot(w):
    th = float(np.linalg.norm(w))
    return np.eye(3) if th < 1e-12 else rot(w / th, th)


def _fk_model_slow(arm, x):
    """Reference implementation (numpy, readable); _fk_model is the same in plain floats, ~5x faster on the Pi."""
    R, p, origins, axes = np.eye(3), BASE[arm].copy(), [], []
    for (xyz, ax), qi in zip(CHAIN[arm], x):
        p = p + R @ xyz
        origins.append(p)
        axes.append(R @ ax)
        R = R @ rot(ax, qi)
    return p + R @ TOOL, R, np.array(origins), np.array(axes)


# every joint axis is +-x, +-y or +-z: (index, sign) per joint
_AXIS = {a: [(int(np.argmax(np.abs(ax))), float(np.sign(ax[int(np.argmax(np.abs(ax)))]))) for _, ax in CHAIN[a]] for a in CHAIN}
_COLS = {0: (1, 2, 1.0), 1: (2, 0, 1.0), 2: (0, 1, 1.0)}  # R @ rot_k(t): columns (i, j) rotate as c*i + s*j, -s*i + c*j


def _fk_model(arm, x):
    """tool position + rotation, and each joint's origin and axis, in the rig frame (model angles)."""
    # columns of R as three [x, y, z] lists
    C = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    px, py, pz = (float(v) for v in BASE[arm])
    origins, axes = [], []
    for (xyz, _), (k, sg), qi in zip(CHAIN[arm], _AXIS[arm], x):
        a, b, c = xyz
        px += C[0][0] * a + C[1][0] * b + C[2][0] * c
        py += C[0][1] * a + C[1][1] * b + C[2][1] * c
        pz += C[0][2] * a + C[1][2] * b + C[2][2] * c
        origins.append((px, py, pz))
        ck = C[k]
        axes.append((sg * ck[0], sg * ck[1], sg * ck[2]))
        i, j, _ = _COLS[k]
        cs, sn = math.cos(qi), sg * math.sin(float(qi))
        ci, cj = C[i], C[j]
        C[i] = [cs * ci[0] + sn * cj[0], cs * ci[1] + sn * cj[1], cs * ci[2] + sn * cj[2]]
        C[j] = [-sn * ci[0] + cs * cj[0], -sn * ci[1] + cs * cj[1], -sn * ci[2] + cs * cj[2]]
    R = np.array(C).T
    p = np.array((px, py, pz)) + R @ TOOL
    return p, R, np.array(origins), np.array(axes)


def fk(arm, c, q_motor):
    """-> (position (3,), rotation (3,3)) of the grasp point in the rig frame."""
    p, R, _, _ = _fk_model(arm, to_model(c, q_motor))
    return p, R


def elbow_wrist_points(arm, c, q_motor):
    """Joint origins (for calibration plots / checks)."""
    _, _, o, _ = _fk_model(arm, to_model(c, q_motor))
    return o


def _jac(arm, x):
    p, R, o, a = _fk_model(arm, x)
    return p, R, np.vstack([np.cross(a, p - o).T, a.T])


# position mode: the cost of moving each joint (J3 upper-arm twist and the wrist are expensive, so small hand moves
# use the shoulder and elbow like a person does), and the per-step pull toward the home posture
JOINT_COST = np.array([1.0, 1.0, 3.0, 1.0, 4.0, 4.0, 4.0])
HOME_PULL = np.array([0.01, 0.01, 0.03, 0.01, 0.3, 0.3, 0.3])
# home posture (model degrees): the ready pose, forearm forward, elbow 90, wrist at rest. Not the middle of each
# joint's range: on this rig that is an arm stuck out sideways, and pulling toward it made the arm drift on its own
HOME_DEG = {"right": [20, 0, 0, 90, 90, 0, 0], "left": [-20, 0, 0, 90, -90, 0, 0]}


def ik(arm, c, want_p, want_R, seed_motor, iters=6, tol=2e-4, rot_weight=0.5, mode="pose", frozen=()):
    """Damped least squares from the seed (warm start), limits with a 5 deg margin (widened to admit the seed),
    then one posture step in the null space. -> (q_motor (7,), residual position error m, residual rotation rad).
    mode "position": only the grasp point's position is a task; the hand's angle follows the arm, the wrist joints are
    weighted expensive and pulled to the middle of their range (S5: holding a fixed hand angle while moving sideways
    twisted J5 far enough to pull the J6 cable)."""
    if mode == "position":
        return _ik_position(arm, c, want_p, seed_motor, iters, tol, frozen)
    x = to_model(c, seed_motor)
    lim = c["limits"]
    lo, hi = np.minimum(lim[:, 0] + MARGIN, x), np.maximum(lim[:, 1] - MARGIN, x)
    W = np.diag([1, 1, 1, rot_weight, rot_weight, rot_weight])
    for _ in range(iters):
        p, R, J = _jac(arm, x)
        e = np.concatenate([want_p - p, log_rot(want_R @ R.T)])
        if np.linalg.norm(e[:3]) < tol and np.linalg.norm(e[3:]) < 5 * tol:
            break
        Jw, ew = W @ J, W @ e
        dx = Jw.T @ np.linalg.solve(Jw @ Jw.T + DAMP ** 2 * np.eye(6), ew)
        if (n := np.linalg.norm(dx)) > 0.15:
            dx *= 0.15 / n
        x = np.clip(x + dx, lo, hi)
    span = lim[:, 1] - lim[:, 0]
    pull = POSTURE * (lim.mean(1) - x) / span
    pull[(x < lim[:, 0] + MARGIN) | (x > lim[:, 1] - MARGIN)] = 0.0
    _, _, J = _jac(arm, x)
    pinv = J.T @ np.linalg.inv(J @ J.T + DAMP ** 2 * np.eye(6))
    x = np.clip(x + (np.eye(7) - pinv @ J) @ pull, lo, hi)
    p, R, _, _ = _fk_model(arm, x)
    return to_motor(c, x), float(np.linalg.norm(want_p - p)), float(np.linalg.norm(log_rot(want_R @ R.T)))


def _ik_position(arm, c, want_p, seed_motor, iters, tol, frozen=()):
    x = to_model(c, seed_motor)
    lim = c["limits"]
    lo, hi = np.minimum(lim[:, 0] + MARGIN, x), np.maximum(lim[:, 1] - MARGIN, x)
    w = 1 / JOINT_COST
    for j in frozen:  # a dead motor: the solver may not use it
        w = w.copy()
        w[j - 1] = 0.0
    Winv = np.diag(w)
    for _ in range(iters):
        p, R, J = _jac(arm, x)
        e = want_p - p
        if np.linalg.norm(e) < tol:
            break
        Jp = J[:3]
        dx = Winv @ Jp.T @ np.linalg.solve(Jp @ Winv @ Jp.T + DAMP ** 2 * np.eye(3), e)
        if (n := np.linalg.norm(dx)) > 0.15:
            dx *= 0.15 / n
        x = np.clip(x + dx, lo, hi)
    home = np.clip(np.radians(HOME_DEG[arm]), lim[:, 0] + MARGIN, lim[:, 1] - MARGIN)
    pull = HOME_PULL * (home - x)
    for j in frozen:
        pull[j - 1] = 0.0
    _, _, J = _jac(arm, x)
    Jp = J[:3]
    pinv = Winv @ Jp.T @ np.linalg.inv(Jp @ Winv @ Jp.T + DAMP ** 2 * np.eye(3))
    x = np.clip(x + (np.eye(7) - pinv @ Jp) @ pull, lo, hi)
    p, R, _, _ = _fk_model(arm, x)
    return to_motor(c, x), float(np.linalg.norm(want_p - p)), 0.0
