# teleop_openarm_system

Teleoperate both arms of an [OpenArm](https://openarm.dev) robot with a **Meta Quest 3 / 3S**. You stand next to the robot and see it through the headset's passthrough. While you hold a controller's grip button, that arm's gripper follows the controller's position **and** angle. Everything runs on a **Raspberry Pi 5** beside the arms; the headset only needs its browser.

- **Full 6-DoF following:** relative to where you squeezed (clutch), with hand filtering and a smooth, speed- and acceleration-limited follower.
- **IK for the 7-DoF arm:** position first, then the gripper's angle with what is left. Every joint uses its full measured range, and a joint that reaches its end drops out so the others make up for it.
- **Precision tools:** gripper on the trigger, fine mode, and an insertion lock with a force-limited thumbstick push (for example, pushing a connector into a port).
- **200 Hz motor loop** on both CAN buses: per-joint gains in motor RAM, latency compensation, and safety holds (tracking error, torque, temperature, silent motors, stale targets).
- **A guide panel inside the headset** that tells the operator what to do next, plus haptic cues.
- **A simulator and calibration tools.**

## How it works

```
Quest browser (WebXR, passthrough, ~90 Hz controller poses)
   │  WebSocket: HTTPS over Wi-Fi, or HTTP over the USB cable (adb reverse)
   ▼
pi/teleop_server.py   clutch, filtering, smooth follower, inverse kinematics, gripper, guide panel
   │  UDP on localhost (joint targets ↓, telemetry ↑)
   ▼
pi/motor_loop.py      200 Hz real-time loop for both CAN buses, follower + latency lead, safety holds
   │  SocketCAN, classic CAN 1 Mbit/s (one DM-USB2FDCAN adapter, two buses)
   ▼
2 × OpenArm: 8 DAMIAO motors each (J1–J7 position-velocity mode, J8 gripper MIT mode)
```

## Hardware

- Two OpenArm arms. Each has DM8009 shoulders (J1, J2), DM4340 upper-arm twist and elbow (J3, J4), and DM4310 wrist and gripper motors (J5–J8).
- One DAMIAO DM-USB2FDCAN adapter: `can0` = right arm, `can1` = left arm, both at 1 Mbit/s.
- A Raspberry Pi 5 running Raspberry Pi OS / Debian with Python 3.11 or newer.
- A Meta Quest 3 or 3S. A USB-C cable to the Pi is optional; it gives a steadier link than Wi-Fi and needs the headset in developer mode with USB debugging allowed.
- A power supply that covers both arms' peak current (see [Known limitations](#known-limitations)).

## Setup

On the Pi, in the repository:

```bash
sudo apt install python3-numpy python3-venv adb   # adb only for the USB link
sh pi/setup.sh                                    # venv with aiohttp + a self-signed HTTPS certificate for the Pi's address
```

`pi/calib.json` describes one particular robot. Calibrate yours before driving it (see [Calibration](#calibration)).

## Run

```bash
pi/start.sh          # both arms; pi/start.sh right for one
```

`start.sh` does the following:
1. Stops any running copy; the arms park first.
2. Resets both CAN buses.
3. Starts the server and the motor loop. **The arms switch on and hold where they are.**
4. If the headset is on the USB cable, links it and opens the teleop page. If the headset isn't ready yet, it keeps retrying in the background for a minute.

Logs go to `/tmp/brain.log` and `/tmp/loop.log`, and as JSON lines to `logs/`.

To start the two programs by hand instead:

```bash
sudo .venv/bin/python pi/motor_loop.py --arms left,right      # root for real-time scheduling
.venv/bin/python pi/teleop_server.py --http-port 8080          # 8080: plain HTTP for the USB link
```

On the headset, open `https://<pi-address>:8443` over Wi-Fi (accept the self-signed certificate once), or `http://localhost:8080` over the USB cable. Tap **START**. The headset's boundary must include where you stand, or it pauses the AR session.

**Stop:** press **Y**, the STOP button on the page, or Ctrl-C on the motor loop. The arms park slowly to where they started, then switch off. **Start again:** with the arms off, hold the **right thumbstick** down for 1 s.

## Controls

| Control | Does |
|---|---|
| **B** (right) | Align, once per AR session: stand beside the robot facing the way it faces, point the right controller straight ahead, press B. |
| **Grip**, hold | Drive that arm. Its gripper follows where your hand is and how it is turned, so hold the controller the way the gripper points. Release: the arm holds where it is. |
| **Trigger** (while holding grip) | Close the gripper. It moves at 1.2 rad/s (`teleop_server.py --grip-rate`) and squeezes with up to 3 Nm (`motor_loop.py --grip-torque`). After a new squeeze it waits until the trigger reaches the gripper's current opening, so a held object stays held. |
| **Thumbstick tilt** (while holding grip) | Forward/back: slide the gripper straight in or out along the way it points. Left/right: slide sideways. Your hand is ignored meanwhile. Speed is up to 1.5 cm/s (0.8 cm/s in fine mode). Sliding in or sideways stops at 30 N (`--push-n`, estimated from the joint torques); pulling out is not limited. |
| **Thumbstick press**, hold (while holding grip) | Insertion lock: your hand moves the gripper only straight in or out, with its angle frozen. |
| **A / X** | Fine mode: half the motion and turning, at up to 6 cm/s. |
| **Y** (left) | STOP: park both arms and switch them off. |
| **Right thumbstick**, hold 1 s (arms off) | Switch the arms on; they hold where they are. |

The defaults are a 1:1 motion scale and a 0.35 m/s gripper speed cap. `teleop_server.py --help` lists every setting.

## Calibration

`pi/calib.json` maps each motor's angle to the kinematic model (geometry from the [openarm_mujoco](https://github.com/enactic/openarm_mujoco) v2 bimanual model):

- `sign`, `offset_deg`: the conversion `model = sign × motor + offset`. Model zero is the arm hanging straight down.
- `limits_deg`: real end stops that are closer than the model's limits. The IK keeps 2° inside them.
- `grip`: the gripper motor's open and closed stops.
- `verified`: the server refuses to drive an arm that isn't verified (unless `--allow-unverified`), and the motor loop doesn't enforce its joint limits.

These tools help you calibrate:

| Tool | Use |
|---|---|
| `tools/untwist.py` | Motors off: live readout of how far each joint is from the hanging pose while you turn the joints by hand. |
| `tools/fk_plot.py`, `tools/predict.py` | Draw the arms as the model sees them, or say where the grasp point goes for a joint move, to compare with the real robot. They run on any computer. |
| `tools/ik_move.py` | Move one grasp point along a straight line and back through the motor loop. Wrong signs or zeros show up as the wrong distance or direction. |
| `tools/find_stops.py` | Find each wrist joint's real end stops gently with its own motor; `--save` writes them to `calib.json`. |
| `tools/grip_stops.py` | Find a gripper's open and closed stops (run it with the motor loop stopped). |
| `tools/straighten.py` | Turn twisted joints back to the hanging pose with their motors. |

## Motor tuning

`pi/gains.json` is written to the motors' RAM at every start and power-on, never to their flash. It sets each joint's position-loop gain (KP_APR) and acceleration, plus `lead_ms`, how far ahead the motor loop aims to cancel each joint's lag.

The shipped values:
- **Elbow motors (J3/J4):** KP_APR 250 instead of the factory 54, which cuts their lag from about 450 ms to 140 ms.
- **Shoulders (J1/J2):** factory gain with an 80 ms lead. A lead matching their full lag makes them resonate near 3 Hz.

`tools/tune_gains.py` tests KP_APR settings one step at a time. It measures lag and chatter, stops at the first sign of shaking, and writes the factory values back at the end. Check new settings in a short teleop session with all joints together: settings that are calm for one joint group at a time can still resonate together.

## Safety behaviour

- **Stale targets:** targets carry an epoch. Whenever the loop resets an arm (start, hold, resume), older targets are ignored, so a stale target can't make the arm jump. A target too far from the current command is refused.
- **Holds:** a joint that falls behind, pushes too hard, or gets hot pauses the arm and backs off its obstacle a little. Squeeze again to continue.
- **Silent motors:** short gaps are ridden through; more than 150 ms pauses the arm. A motor that switched itself off faults the arm; press Y, then switch on again.
- **Letting go:** when driving stops (grip released, tracking or the headset link lost, re-aligning, leaving AR), a joint that is pressing against something holds slightly back from it instead of pushing ever harder; a joint still moving finishes its last command.
- **Server silence:** if the server goes quiet, the motor loop brakes to a stop and holds.
- **Start-up:** the arms never move on their own, and only one program may drive the buses (a lock file plus a check for other drivers).
- **Network:** there is **no authentication**. Anyone who can reach the Pi's port can drive the arms, so use a trusted network.

## Try it without the robot

On Linux (the motor loop needs SocketCAN):

```bash
sudo ip link add dev vcan0 type vcan && sudo ip link set vcan0 up
python3 tools/fake_motors.py vcan0 &                                      # a simulated arm
sudo .venv/bin/python pi/motor_loop.py --buses right=vcan0 --arms right &
.venv/bin/python pi/teleop_server.py &
.venv/bin/python tools/fake_quest.py --hand right                         # aligns, squeezes, moves a fake hand
```

`tools/hud_flow_test.py` walks the whole operator flow (AR start, align, drive, release, STOP, power on) and prints what the headset panel would say. Run it against a server started with `--port 8444`. `tools/quest_cdp.py` shows the page's console from inside the headset over USB.

## Repository layout

| Path | What it is |
|---|---|
| `pi/motor_loop.py` | The only program that talks to the motors. Per cycle it sends one arm's frames and waits for that arm's replies, then does the other arm (the shared adapter drops replies when both bursts overlap). |
| `pi/teleop_server.py` | Serves the headset page and its WebSocket; clutch, filtering, IK, gripper, guide panel. `POST /cmd` accepts `stop` and `power_on`. |
| `pi/kinematics.py` | Forward and inverse kinematics for both arms (numpy). |
| `pi/damiao.py` | DAMIAO CAN protocol: frame packing, reply decoding, register access. |
| `pi/calib.json`, `pi/gains.json` | Calibration and motor gains (see above). |
| `pi/start.sh`, `pi/setup.sh`, `pi/headset_link.sh` | Start everything, one-time setup, USB link to the headset. |
| `quest/index.html` | The headset page (plain HTML/JS, no build step). |
| `tools/` | Simulator, calibration, tuning and debugging tools (above). |

## Known limitations

- **Wrist range:** the wrist bends much less than a human wrist (J6 about ±30–45°). When the gripper can't turn as far as your hand (a wrist joint at its end, or the arm straight), it stops there; the controller buzzes and the headset names the joint at its end. Turning further doesn't build up, so when you turn back the gripper follows at once; its angle relative to your hand has then shifted by the excess. Squeeze again to re-anchor.
- **Straight arm:** with the arm straight, the hand can't move quickly along the arm's own direction (a kinematic singularity).
- **Hard pushes:** pushing very hard (shoulder torques of about 17–30 Nm) can dip the motors' supply until a shoulder motor reboots. The arm then faults: press Y, then switch on again. A stronger power supply helps, and so does not driving the arms into the robot's own frame.
- **No collision avoidance:** there is none, either between the arms or with the surroundings.
- **Wi-Fi delays:** Wi-Fi adds occasional delays; the USB link is steadier.
