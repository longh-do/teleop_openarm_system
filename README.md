# teleop_openarm_system

Teleoperate both arms of an [OpenArm](https://openarm.dev) robot with a **Meta Quest 3S**. You stand next to the robot, see it through the headset's passthrough, and the grippers follow your controllers. Everything runs on a **Raspberry Pi 5** next to the arms; the headset only needs a browser.

```
Quest 3S browser (WebXR, 90 Hz)
   │  WebSocket over HTTPS (Wi-Fi)
   ▼
pi/teleop_server.py   clutch, filtering, speed caps, inverse kinematics
   │  UDP on localhost (joint targets ↓, telemetry ↑)
   ▼
pi/motor_loop.py      200 Hz real-time loop, both CAN buses, smoothing, safety
   │  SocketCAN, classic CAN 1 Mbit/s (one DM-USB2FDCAN adapter, two buses)
   ▼
2 × OpenArm: 8 DAMIAO motors each (J1–J7 position-velocity mode, J8 gripper MIT mode)
```

## Files

| Path | What it is |
|---|---|
| `pi/motor_loop.py` | The only program that talks to the motors. One process drives both buses: it sends one arm's frames, waits for that arm's replies, then does the other arm (the shared adapter drops replies when both bursts overlap). Velocity/acceleration-limited follower, ride-through of short reply gaps, tracking-error / torque / temperature guards, slow park on STOP. Also has test modes (`--read-only`, `--test hold/sine/probe`). |
| `pi/teleop_server.py` | Serves the headset page and its WebSocket on one HTTPS port. Squeeze = clutch, trigger = gripper, One Euro filter, speed caps, IK (position-only by default), sends joint targets to the loop. `POST /cmd` accepts `ready`, `stop`, `power_on`. |
| `pi/kinematics.py` | Forward/inverse kinematics for both arms (numpy). Geometry from the `openarm_mujoco` v2 bimanual model. |
| `pi/damiao.py` | DAMIAO CAN protocol: frame packing and reply decoding. |
| `pi/calib.json` | Per-joint sign, zero offset and extra limits that map this rig's motor angles to the model, plus each gripper's open/closed stops. |
| `pi/setup.sh` | Creates the Python venv and a self-signed HTTPS certificate for the Pi's IP. |
| `quest/index.html` | The headset page (plain HTML/JS, no build step). |
| `tools/` | Simulated motors on `vcan` (`fake_motors.py`), a fake headset (`fake_quest.py`), calibration helpers (`grip_stops.py`, `ik_move.py`, `predict.py`, `fk_plot.py`), and `quest_cdp.py` for debugging the page in the headset over USB. |

## Run it

On the Pi (Python 3.13, numpy, `aiohttp`; the arms on `can0` = right and `can1` = left at 1 Mbit/s):

```bash
sh pi/setup.sh                                               # once: venv + HTTPS certificate
sudo .venv/bin/python pi/motor_loop.py --arms left,right     # arms switch on and hold where they are
.venv/bin/python pi/teleop_server.py                         # 1:1 scale, 0.15 m/s gripper speed cap
```

On the Quest: open `https://<pi-ip>:8443`, accept the self-signed certificate once (Advanced → Proceed), tap **Enter AR**. The Quest's guardian boundary must include where you stand, or it pauses the AR session.

**Stop:** press **Y** in the headset, the STOP button on the page, or Ctrl-C on the motor loop. The arms park slowly to where they started, then switch off.

## Controls

| Control | Does |
|---|---|
| **B** (right) | Align, once per session: stand beside the robot facing the way it faces, point the right controller straight ahead, press B |
| **Grip**, hold | Drive that arm: its gripper follows your hand. Release to freeze. |
| **Trigger** (while holding grip) | Close that gripper; release to open |
| **A / X** | Fine mode: slower and smaller, for precise moves |
| **Thumbstick click** | Move that arm to the ready pose (forearm forward) |
| **Y** (left) | STOP: park both arms and switch off |

## Safety behaviour

- Targets carry an "epoch": whenever the loop resets an arm (start, freeze, resume), older targets are ignored, so a stale target can't make the arm jump.
- A joint that falls behind its target, pushes too hard, or gets hot freezes the arm (it backs off its obstacle a little); squeezing again resumes.
- Short silences from a motor (a few tens of ms) are ridden through; more than 150 ms freezes the arm instead of switching it off.
- If the headset or the server goes quiet, the arm brakes to a stop and holds.
- One program per bus (a lock file, plus a check for other drivers).

## Try it without the robot

```bash
sudo ip link add dev vcan0 type vcan && sudo ip link set vcan0 up
python3 tools/fake_motors.py vcan0 &
sudo .venv/bin/python pi/motor_loop.py --buses right=vcan0 --arms right --cpu 2
.venv/bin/python pi/teleop_server.py --allow-unverified &
.venv/bin/python tools/fake_quest.py --hand right            # squeezes and moves a fake hand
```

## Known limitations

- The elbow motors (DM4340, J3/J4) follow their position command a few degrees behind while moving. A velocity lead hides most of it; MIT mode with gravity feed-forward would fix it properly.
- The gripper only follows your hand's position; the hand's twist is ignored by default (`--ik pose` to try it).
- Calibration in `pi/calib.json` is specific to this rig: zeros were taken with both arms hanging straight down.
- One left-wrist motor (J6) lost communication twice during testing; `--skip left:6` runs that arm without it.
