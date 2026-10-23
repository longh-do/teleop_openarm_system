#!/bin/sh
# Link the headset's teleop page to the Pi over the USB cable (adb reverse), then open the page in the headset only
# if no open page reconnects by itself.   pi/headset_link.sh [seconds to wait for the headset, default 0]
# start.sh runs it, in the background with 60 s of retries if the headset isn't ready yet.
cd "$(dirname "$0")/.." || exit 1
WAIT="${1:-0}"
LOG="${BRAIN_LOG:-/tmp/brain.log}"
end=$(($(date +%s) + WAIT))
until adb devices | grep -q "device$"; do
  if [ "$(date +%s)" -ge "$end" ]; then
    IP=$(ip route get 1.1.1.1 2>/dev/null | awk '{for (i = 1; i < NF; i++) if ($i == "src") {print $(i + 1); exit}}')
    [ -n "$IP" ] || IP=$(hostname -I | awk '{print $1}')
    echo "  headset not on USB (or USB debugging not allowed): use https://$IP:8443"
    exit 1
  fi
  sleep 2
done
adb reverse tcp:8080 tcp:8080 >/dev/null && echo "  headset on USB: open http://localhost:8080"
# an open teleop page reconnects by itself within a second or two: launching the browser again while the headset
# switches in or out of AR can hang it, so the page is opened only if none reconnects
i=0
while ! grep -q "page connected" "$LOG" && [ $i -lt 6 ]; do sleep 1; i=$((i + 1)); done
if grep -q "page connected" "$LOG"; then
  echo "  the teleop page in the headset reconnected by itself"
else
  adb shell am start -a android.intent.action.VIEW -d http://localhost:8080 com.oculus.browser >/dev/null 2>&1
  echo "  opened the teleop page in the headset"
fi
