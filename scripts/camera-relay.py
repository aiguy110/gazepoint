#!/usr/bin/python3
"""Fan one camera out to several v4l2loopback devices, so gazepoint and a video
call can both use the webcam.

v4l2loopback (0.15) lets only one reader stream from a device at a time, and the
sensor can only be opened by one pipeline, so v4l2-relayd (one source, one
loopback) can't serve two apps. This replaces it for one source: the camera
pipeline is tee'd into a loopback per card label (created if missing), and it
runs only while at least one of them has a reader. Otherwise black frames are
written slowly, which keeps the devices advertised as capture devices (Chrome
needs that with exclusive caps).

Takes the card labels as arguments; the source comes from the environment, in
v4l2-relayd's conf format (VIDEOSRC, FORMAT, WIDTH, HEIGHT; e.g.
/etc/v4l2-relayd.d/libcamera.conf). Runs as root (system python3 + GStreamer).
Installed by scripts/setup-camera-relay.sh.
"""

import fcntl
import glob
import os
import select
import struct
import sys
import time

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst  # noqa: E402

# v4l2loopback.h: V4L2_EVENT_PRIVATE_START + 0x08E00000 + 1; payload is the
# number of readers. Any fd on the device can subscribe.
EVENT_CLIENT_USAGE = 0x08000000 + 0x08E00000 + 1
VIDIOC_SUBSCRIBE_EVENT = 0x4020565A
VIDIOC_DQEVENT = 0x80885659  # sizeof(struct v4l2_event) == 136 on 64-bit
V4L2_EVENT_SUB_FL_SEND_INITIAL = 1
CTL_ADD = 0x40487E01  # _IOW('~', 1, struct v4l2_loopback_config)

IDLE_PERIOD_MS = 500
RETRY_S = 2.0


def log(msg):
    print(msg, flush=True)


def ensure_device(label: str) -> str:
    """/dev/videoN of the loopback with this card label, created if missing."""
    for name in glob.glob("/sys/devices/virtual/video4linux/*/name"):
        with open(name) as f:
            if f.read().strip() == label:
                return "/dev/" + name.split("/")[-2]
    # output_nr, capture_nr, card_label[32], min/max width/height (0 = module
    # default), max_buffers, max_openers, debug, announce_all_caps (0 =
    # exclusive caps, needed by Chrome/WebRTC)
    conf = bytearray(struct.pack("ii32sIIIIiiii", -1, -1, label.encode()[:31], 0, 0, 0, 0, 0, 0, 0, 0))
    fd = os.open("/dev/v4l2loopback", os.O_RDWR)
    try:
        nr = fcntl.ioctl(fd, CTL_ADD, conf)
    finally:
        os.close(fd)
    log(f"created /dev/video{nr} {label!r}")
    return f"/dev/video{nr}"


class Relay:
    def __init__(self, labels: list[str]):
        self.fmt = os.environ.get("FORMAT", "NV12")
        self.w = int(os.environ.get("WIDTH", 1280))
        self.h = int(os.environ.get("HEIGHT", 720))
        self.videosrc = os.environ["VIDEOSRC"]
        caps = f"video/x-raw,format={self.fmt},width={self.w},height={self.h}"
        self.devices = [ensure_device(label) for label in labels]

        branches = " ".join(
            f"t. ! queue leaky=downstream max-size-buffers=2 ! videoconvert ! v4l2sink device={d} sync=false"
            for d in self.devices
        )
        self.out = Gst.parse_launch(
            f"appsrc name=src is-live=true format=time do-timestamp=true caps={caps},framerate=30/1 "
            f"! tee name=t {branches}"
        )
        self.src = self.out.get_by_name("src")
        self.out.get_bus().add_signal_watch()
        self.out.get_bus().connect("message::error", self._out_error)
        self.out.set_state(Gst.State.PLAYING)

        self.caps = caps
        self.inp = None
        self.last_frame = 0.0
        self.retry_at = 0.0
        self.black = self._black_frame()

        self.readers = {}
        for d in self.devices:
            fd = os.open(d, os.O_RDWR | os.O_NONBLOCK)
            sub = struct.pack("III5I", EVENT_CLIENT_USAGE, 0, V4L2_EVENT_SUB_FL_SEND_INITIAL, *[0] * 5)
            fcntl.ioctl(fd, VIDIOC_SUBSCRIBE_EVENT, sub)
            self.readers[fd] = 0
            GLib.unix_fd_add_full(GLib.PRIORITY_DEFAULT, fd, GLib.IOCondition.PRI, self._on_event, d)
            self._drain(fd, d)
        GLib.timeout_add(IDLE_PERIOD_MS, self._tick)
        log(f"relaying {self.videosrc!r} to {', '.join(self.devices)}")

    def _black_frame(self) -> Gst.Buffer:
        if self.fmt != "NV12":
            raise SystemExit(f"FORMAT={self.fmt}: only NV12 idle frames are implemented")
        y = self.w * self.h
        return Gst.Buffer.new_wrapped(b"\x10" * y + b"\x80" * (y // 2))

    def _drain(self, fd, dev):
        buf = bytearray(136)
        while select.select([], [], [fd], 0)[2]:
            try:
                fcntl.ioctl(fd, VIDIOC_DQEVENT, buf)
            except OSError:
                break
            kind, count = struct.unpack_from("I4xI", buf)
            if kind == EVENT_CLIENT_USAGE and count != self.readers[fd]:
                self.readers[fd] = count
                log(f"{dev}: {count} reader(s)")
        self._update()

    def _on_event(self, fd, _cond, dev):
        self._drain(fd, dev)
        return True

    def _update(self):
        want = any(self.readers.values())
        if want and self.inp is None and time.monotonic() >= self.retry_at:
            self._start()
        elif not want and self.inp is not None:
            self._stop("no readers")

    def _start(self):
        log("starting camera")
        self.inp = Gst.parse_launch(
            f"{self.videosrc} ! videoconvert ! videoscale ! {self.caps} "
            "! appsink name=sink emit-signals=true max-buffers=2 drop=true sync=false"
        )
        self.inp.get_by_name("sink").connect("new-sample", self._on_sample)
        bus = self.inp.get_bus()
        bus.add_signal_watch()
        bus.connect("message::error", self._in_error)
        bus.connect("message::eos", lambda *_: self._fail("end of stream"))
        if self.inp.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            self._fail("failed to start")

    def _stop(self, why):
        log(f"stopping camera ({why})")
        inp, self.inp = self.inp, None
        inp.get_bus().remove_signal_watch()
        inp.set_state(Gst.State.NULL)

    def _fail(self, why):
        if self.inp is not None:
            self._stop(why)
            self.retry_at = time.monotonic() + RETRY_S

    def _in_error(self, _bus, msg):
        err, _ = msg.parse_error()
        GLib.idle_add(lambda: self._fail(f"error: {err.message}") and False)

    def _out_error(self, _bus, msg):
        err, dbg = msg.parse_error()
        log(f"output error: {err.message} ({dbg}); exiting")
        self.loop.quit()

    def _on_sample(self, sink):
        sample = sink.emit("pull-sample")
        buf = sample.get_buffer().copy()  # shares memory; new timestamps on push
        buf.pts = buf.dts = buf.duration = Gst.CLOCK_TIME_NONE
        self.last_frame = time.monotonic()
        self.src.emit("push-buffer", buf)
        return Gst.FlowReturn.OK

    def _tick(self):
        self._update()  # retries a failed start
        if time.monotonic() - self.last_frame > IDLE_PERIOD_MS / 1000:
            self.src.emit("push-buffer", self.black.copy())
        return True

    def run(self):
        self.loop = GLib.MainLoop()
        for sig in (15, 2):  # SIGTERM, SIGINT
            GLib.unix_signal_add(GLib.PRIORITY_HIGH, sig, self.loop.quit)
        self.loop.run()
        if self.inp is not None:
            self._stop("exiting")
        self.out.set_state(Gst.State.NULL)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(f"usage: {sys.argv[0]} LABEL [LABEL...]")
    Gst.init(None)
    Relay(sys.argv[1:]).run()
