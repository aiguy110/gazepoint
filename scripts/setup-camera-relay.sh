#!/usr/bin/env bash
# One-time, on the target, with sudo: let gazepoint and a video call (Teams,
# Meet, ...) use the webcam at the same time. v4l2loopback lets only one app
# stream from a loopback device, so this replaces v4l2-relayd@libcamera with
# scripts/camera-relay.py, which feeds the same libcamera source into both
# "libcamera Camera" (pick it in the call app) and "gazepoint Camera" (the
# camera.index default). The camera runs only while either has a reader.
# Re-run after changing camera-relay.py. Undo:
#   systemctl disable --now gazepoint-camera-relay; systemctl unmask v4l2-relayd@libcamera
#   systemctl enable --now v4l2-relayd@libcamera
set -euo pipefail
[ "$(id -u)" = 0 ] || exec sudo "$0" "$@"
conf=/etc/v4l2-relayd.d/libcamera.conf
[ -f "$conf" ] || { echo "$conf not found (the libcamera v4l2-relayd source)" >&2; exit 1; }
install -m755 "$(dirname "$0")/camera-relay.py" /usr/local/sbin/gazepoint-camera-relay
cat >/etc/systemd/system/gazepoint-camera-relay.service <<EOF
[Unit]
Description=Webcam relay for gazepoint and video calls (libcamera -> 2 loopbacks)
After=modprobe@v4l2loopback.service systemd-logind.service
Wants=modprobe@v4l2loopback.service

[Service]
EnvironmentFile=$conf
ExecStart=/usr/local/sbin/gazepoint-camera-relay "libcamera Camera" "gazepoint Camera"
Restart=always
RestartSec=2

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl disable --now v4l2-relayd@libcamera.service 2>/dev/null || true
systemctl mask v4l2-relayd@libcamera.service  # it would grab the sensor
systemctl enable gazepoint-camera-relay.service
systemctl restart gazepoint-camera-relay.service
sleep 2
systemctl --no-pager status gazepoint-camera-relay.service | head -12
