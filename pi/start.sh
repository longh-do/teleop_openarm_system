#!/bin/sh
# Start teleop on the Pi:  pi/start.sh            (both arms)
#                          pi/start.sh right      (one arm)
# 1. stops running copies (the arms park first), 2. resets both CAN buses (needed after an arm's power was off),
# 3. starts the server (https :8443 for Wi-Fi, http :8080 for the USB cable) and the motor loop (the arms switch on
# and hold where they are), 4. links the headset over USB (pi/headset_link.sh; retried in the background for 60 s if
# the headset isn't ready yet).
cd "$(dirname "$0")/.." || exit 1
ARMS="${1:-left,right}"
IP=$(hostname -I | awk '{print $1}')

echo "stopping running copies (arms park first if on)..."
sudo pkill -INT -f "^.venv/bin/python pi/motor_loop.py"
pkill -TERM -f "^.venv/bin/python pi/teleop_server.py"
i=0
while pgrep -f "^.venv/bin/python pi/motor_loop.py" >/dev/null && [ $i -lt 30 ]; do sleep 1; i=$((i + 1)); done

echo "resetting CAN buses..."
for b in can0 can1; do sudo ip link set "$b" down; sudo ip link set "$b" up type can bitrate 1000000; done

echo "starting the server and the motor loop ($ARMS): THE ARMS SWITCH ON and hold where they are. Stand clear."
setsid nohup .venv/bin/python pi/teleop_server.py --http-port 8080 > /tmp/brain.log 2>&1 &
sleep 2
sudo setsid nohup .venv/bin/python pi/motor_loop.py --arms "$ARMS" > /tmp/loop.log 2>&1 &

echo "USB link to the headset..."
for p in $(pgrep -f "pi/headset_link.sh"); do kill "$p"; done  # a retry left from an earlier start
adb kill-server >/dev/null 2>&1
sudo adb start-server >/dev/null 2>&1
timeout 5 adb wait-for-device 2>/dev/null  # the adb server needs a moment to find the headset
if adb devices | grep -q "device$"; then
  sh pi/headset_link.sh
else
  # just plugged in, waking up, or waiting for "Allow USB debugging": keep trying in the background
  echo "  headset not ready yet: retrying for 60 s in the background (log: /tmp/usb_link.log)"
  echo "  (over Wi-Fi instead: https://$IP:8443)"
  setsid nohup sh pi/headset_link.sh 60 > /tmp/usb_link.log 2>&1 < /dev/null &
fi

sleep 6
echo "--- arms:"
grep -vE "^\[ *[0-9.]+s\]" /tmp/loop.log | tail -4
