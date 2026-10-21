#!/bin/sh
# One command to start teleop on the Pi:  ~/teleop_openarm_system/pi/start.sh          (both arms)
#                                          ~/teleop_openarm_system/pi/start.sh right    (one arm)
# 1. stops old copies (arms park first), 2. resets both CAN buses (needed after any arm power unplug),
# 3. starts the motor loop (both arms switch on, then the server raises them to the ready pose),
# 4. starts the server (https :8443 for Wi-Fi, http :8080 for the USB cable), 5. sets up the USB link
# (adb reverse) and opens the teleop page in the headset if it is plugged in.
cd "$(dirname "$0")/.." || exit 1
ARMS="${1:-left,right}"

echo "stopping old copies (arms park first if on)..."
sudo pkill -INT -f "^.venv/bin/python pi/motor_loop.py"
pkill -TERM -f "^.venv/bin/python pi/teleop_server.py"
i=0
while pgrep -f "^.venv/bin/python pi/motor_loop.py" >/dev/null && [ $i -lt 30 ]; do sleep 1; i=$((i + 1)); done

echo "resetting CAN buses..."
for b in can0 can1; do sudo ip link set "$b" down; sudo ip link set "$b" up type can bitrate 1000000; done

echo "starting the server and the motor loop ($ARMS): THE ARMS SWITCH ON, then rise to point forward. Stand clear."
setsid nohup .venv/bin/python pi/teleop_server.py --http-port 8080 > /tmp/brain.log 2>&1 &
sleep 2
sudo setsid nohup .venv/bin/python pi/motor_loop.py --arms "$ARMS" > /tmp/loop.log 2>&1 &

echo "USB link to the headset..."
adb kill-server >/dev/null 2>&1
sudo adb start-server >/dev/null 2>&1
timeout 5 adb wait-for-device 2>/dev/null  # the server needs a moment to find the headset
if adb devices | grep -q "device$"; then
  adb reverse tcp:8080 tcp:8080 && echo "  headset on USB: open http://localhost:8080"
  adb shell am start -a android.intent.action.VIEW -d http://localhost:8080 com.oculus.browser >/dev/null 2>&1
else
  echo "  headset not on USB (or not allowed): use https://$(ip -4 -br addr show eth0 | awk '{print $3}' | cut -d/ -f1):8443"
fi

sleep 6
echo "--- arms:"
grep -vE "^\[ *[0-9.]+s\]" /tmp/loop.log | tail -4
