#!/bin/sh
# One-time setup on the Pi: venv (system numpy + aiohttp) and a self-signed cert for the Quest (HTTPS/WebXR).
set -e
cd "$(dirname "$0")/.."
[ -d .venv ] || python3 -m venv --system-site-packages .venv
.venv/bin/pip install -q aiohttp
# the address the Pi reaches the network with (the first address if there is no route out)
IP=$(ip route get 1.1.1.1 2>/dev/null | awk '{for (i = 1; i < NF; i++) if ($i == "src") {print $(i + 1); exit}}')
[ -n "$IP" ] || IP=$(hostname -I | awk '{print $1}')
SAN="IP:127.0.0.1,DNS:localhost,DNS:$(hostname).local"
for a in $(hostname -I); do case $a in *.*) SAN="$SAN,IP:$a" ;; esac; done  # every IPv4 address of the Pi
mkdir -p certs logs
if [ ! -f certs/cert.pem ] || ! openssl x509 -in certs/cert.pem -noout -text | grep -q "IP Address:$IP"; then
  openssl req -x509 -newkey rsa:2048 -nodes -days 3650 -keyout certs/key.pem -out certs/cert.pem \
    -subj "/CN=$(hostname)" -addext "subjectAltName=$SAN"
fi
echo "ready: https://$IP:8443"
