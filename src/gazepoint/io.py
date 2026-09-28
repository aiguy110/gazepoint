"""Host I/O: pointer warping, global hotkey / click listening (evdev), smoothing."""

from __future__ import annotations

import math
import os
import select
import threading
import time
from collections.abc import Callable

from .monitors import Monitor, desktop_bbox, session_type

# ---------------------------------------------------------------- smoothing


class OneEuro:
    """One Euro filter (Casiez et al. 2012): low jitter when slow, low lag when fast."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.05, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(dt: float, cutoff: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        self.x = self.dx = self.t = None

    def __call__(self, x: float, t: float) -> float:
        if self.t is None:
            self.x, self.dx, self.t = x, 0.0, t
            return x
        dt = max(t - self.t, 1e-4)
        dx = (x - self.x) / dt
        self.dx += self._alpha(dt, self.d_cutoff) * (dx - self.dx)
        cutoff = self.min_cutoff + self.beta * abs(self.dx)
        self.x += self._alpha(dt, cutoff) * (x - self.x)
        self.t = t
        return self.x


# ---------------------------------------------------------------- pointer


class X11Pointer:
    def __init__(self):
        from Xlib import display

        self.d = display.Display()
        self.root = self.d.screen().root

    def move(self, x: float, y: float):
        self.root.warp_pointer(int(round(x)), int(round(y)))
        self.d.sync()

    def position(self) -> tuple[int, int] | None:
        p = self.root.query_pointer()
        return p.root_x, p.root_y

    def close(self):
        self.d.close()


class UInputPointer:
    """Virtual absolute pointer (like QEMU's usb-tablet). Compositors map its
    range onto the whole desktop, which works on Wayland where warping doesn't.
    Needs write access to /dev/uinput (scripts/setup-input-perms.sh)."""

    RANGE = 32767

    def __init__(self, mons: list[Monitor]):
        from evdev import AbsInfo, UInput
        from evdev import ecodes as e

        self.bbox = desktop_bbox(mons)
        absinfo = AbsInfo(value=0, min=0, max=self.RANGE, fuzz=0, flat=0, resolution=0)
        self.e = e
        self.ui = UInput(
            {
                e.EV_KEY: [e.BTN_LEFT, e.BTN_RIGHT, e.BTN_MIDDLE],
                e.EV_ABS: [(e.ABS_X, absinfo), (e.ABS_Y, absinfo)],
                e.EV_REL: [e.REL_WHEEL],
            },
            name="gazepoint-pointer",
        )
        time.sleep(0.3)  # let the compositor pick the device up

    def move(self, x: float, y: float):
        bx, by, bw, bh = self.bbox
        ax = int(round((x - bx) / max(bw - 1, 1) * self.RANGE))
        ay = int(round((y - by) / max(bh - 1, 1) * self.RANGE))
        self.ui.write(self.e.EV_ABS, self.e.ABS_X, min(max(ax, 0), self.RANGE))
        self.ui.write(self.e.EV_ABS, self.e.ABS_Y, min(max(ay, 0), self.RANGE))
        self.ui.syn()

    def position(self):
        return None  # not observable on Wayland

    def close(self):
        self.ui.close()


def make_pointer(backend: str, mons: list[Monitor]):
    if backend == "auto":
        backend = "x11" if session_type() == "x11" else "uinput"
    return X11Pointer() if backend == "x11" else UInputPointer(mons)


# ---------------------------------------------------------------- evdev input


def input_devices(key_code: int):
    """(keyboards having key_code, mice having BTN_LEFT) that we can read."""
    from evdev import InputDevice, list_devices
    from evdev import ecodes as e

    kbds, mice = [], []
    for path in list_devices():
        try:
            dev = InputDevice(path)
        except OSError:
            continue
        keys = dev.capabilities().get(e.EV_KEY, [])
        if dev.name == "gazepoint-pointer":
            dev.close()
        elif key_code in keys:
            kbds.append(dev)
        elif e.BTN_LEFT in keys:
            mice.append(dev)
        else:
            dev.close()
    return kbds, mice


def readable_input() -> bool:
    return any(os.access(f"/dev/input/{n}", os.R_OK) for n in os.listdir("/dev/input") if n.startswith("event"))


class InputListener(threading.Thread):
    """Calls on_key(down: bool, t) for the hotkey and on_click(t) for left clicks.

    Does not grab devices: the hotkey still reaches applications, so pick a key
    whose lone press/release is harmless (Right Ctrl by default).
    """

    def __init__(self, key_name: str, on_key: Callable, on_click: Callable | None = None):
        super().__init__(daemon=True, name="input")
        from evdev import ecodes as e

        self.e = e
        self.key_name = key_name
        self.code = e.ecodes[key_name]
        # The same key on the other side (KEY_RIGHTCTRL <-> KEY_LEFTCTRL): pressing it is
        # the usual reason the hotkey "does nothing", so say so once.
        other = key_name.replace("LEFT", "\0").replace("RIGHT", "LEFT").replace("\0", "RIGHT")
        self.sibling = (other, e.ecodes[other]) if other != key_name and other in e.ecodes else None
        self._hinted = False
        self.on_key, self.on_click = on_key, on_click
        self.kbds, self.mice = input_devices(self.code)
        if not self.kbds:
            raise RuntimeError(
                f"no readable keyboard with {key_name}; is your user in the 'input' group? "
                "(scripts/setup-input-perms.sh)"
            )
        self._stop = False

    def _rescan(self, devs: dict):
        """Pick up keyboards that (re)connected since startup, e.g. Bluetooth after sleep."""
        known = {d.path for d in devs.values()}
        kbds, mice = input_devices(self.code)
        for d in kbds + mice:
            if d.path in known or (d in mice and not self.on_click):
                d.close()
                continue
            devs[d.fd] = d
            print(f"input: listening on {d.name}")

    def run(self):
        e = self.e
        devs = {d.fd: d for d in self.kbds + (self.mice if self.on_click else [])}
        last_scan = time.monotonic()
        while not self._stop:
            if time.monotonic() - last_scan > 2:
                self._rescan(devs)
                last_scan = time.monotonic()
            r, _, _ = select.select(list(devs), [], [], 0.2)
            for fd in r:
                try:
                    events = list(devs[fd].read())
                except OSError:
                    print(f"input: lost {devs[fd].name}")
                    devs.pop(fd).close()
                    continue
                for ev in events:
                    if ev.type != e.EV_KEY:
                        continue
                    t = time.monotonic()
                    if ev.code == self.code and ev.value in (0, 1):
                        self.on_key(ev.value == 1, t)
                    elif ev.code == e.BTN_LEFT and ev.value == 1 and self.on_click:
                        self.on_click(t)
                    elif self.sibling and ev.code == self.sibling[1] and ev.value == 1 and not self._hinted:
                        self._hinted = True
                        print(f"input: saw {self.sibling[0]}, but the hotkey is {self.key_name} "
                              "(change [run] hotkey in ~/.config/gazepoint/config.toml)")

    def stop(self):
        self._stop = True
