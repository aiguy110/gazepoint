#!/usr/bin/env bash
# One-time, on the target (Meteor Lake or newer, Ubuntu 24.04), with sudo: install
# Intel's userspace NPU driver (Level Zero backend + compiler, from
# intel/linux-npu-driver; not in Ubuntu's archive) and the Level Zero loader it
# needs, and add the user to 'render' for /dev/accel/accel0. The kernel driver
# (intel_vpu) and firmware come with the kernel. Log out and back in afterwards
# (or reboot); until then gaze.device = auto falls back to the CPU.
set -euo pipefail
[ "$(id -u)" = 0 ] || exec sudo "$0" "$@"
user=${SUDO_USER:?run via sudo as the desktop user}
ver=1.38.0
build=20260910-34487311128
ze=https://snapshot.ppa.launchpadcontent.net/kobuk-team/intel-graphics/ubuntu/20260830T100000Z/pool/main/l/level-zero-loader/libze1_1.32.0-1~24.04~ppa1_amd64.deb
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cd "$tmp"
curl -fsSLO "https://github.com/intel/linux-npu-driver/releases/download/v$ver/linux-npu-driver-v$ver.$build-ubuntu2404.tar.gz"
tar -xf linux-npu-driver-*.tar.gz
curl -fsSLO "$ze"
apt-get install -y libtbb12
# intel-fw-npu is skipped: the kernel's firmware works and replacing it needs a reboot.
dpkg -i libze1_*.deb intel-level-zero-npu_*.deb intel-driver-compiler-npu_*.deb
usermod -aG render "$user"
echo "NPU driver $ver installed and $user added to 'render'; log out and back in, then: gazepoint doctor"
