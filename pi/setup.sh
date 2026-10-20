#!/bin/sh
# One-time setup on the Pi: venv (system numpy + aiohttp) and a self-signed cert for the Quest (HTTPS/WebXR).
set -e
cd "$(dirname "$0")/.."
[ -d .venv ] || python3 -m venv --system-site-packages .venv
.venv/bin/pip install -q aiohttp
IP=$(ip -4 -br addr show eth0 | awk '{print $3}' | cut -d/ -f1)
mkdir -p certs logs
if [ ! -f certs/cert.pem ] || ! openssl x509 -in certs/cert.pem -noout -text | grep -q "IP Address:$IP"; then
  openssl req -x509 -newkey rsa:2048 -nodes -days 3650 -keyout certs/key.pem -out certs/cert.pem \
    -subj "/CN=$(hostname)" -addext "subjectAltName=IP:$IP,IP:127.0.0.1,DNS:$(hostname).local,DNS:localhost"
fi
echo "ready: https://$IP:8443"
