#!/usr/bin/env bash
# One-time, on the target, with sudo: let the user read keyboards (hotkey) and
# create a virtual pointer (/dev/uinput, needed on Wayland). Log out/in afterwards.
set -euo pipefail
[ "$(id -u)" = 0 ] || exec sudo "$0" "$@"
user=${SUDO_USER:?run via sudo as the desktop user}
usermod -aG input "$user"
cat > /etc/udev/rules.d/60-gazepoint-uinput.rules <<'RULE'
KERNEL=="uinput", GROUP="input", MODE="0660", OPTIONS+="static_node=uinput"
RULE
echo uinput > /etc/modules-load.d/gazepoint-uinput.conf
modprobe uinput
udevadm control --reload-rules
udevadm trigger --name-match=uinput || true
echo "added $user to 'input' and granted /dev/uinput; log out and back in for the group to apply"
