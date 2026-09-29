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

    def move_rel(self, dx: float, dy: float):
        x, y = self.position()
        self.move(x + dx, y + dy)

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
        # A second, relative device for moving the pointer from wherever it is
        # (its position can't be read on Wayland). The compositor applies its
        # pointer acceleration to this one.
        self.rel = UInput(
            {e.EV_KEY: [e.BTN_LEFT], e.EV_REL: [e.REL_X, e.REL_Y]}, name="gazepoint-pointer-rel"
        )
        self.frac = [0.0, 0.0]
        time.sleep(0.3)  # let the compositor pick the devices up

    def move(self, x: float, y: float):
        bx, by, bw, bh = self.bbox
        ax = int(round((x - bx) / max(bw - 1, 1) * self.RANGE))
        ay = int(round((y - by) / max(bh - 1, 1) * self.RANGE))
        self.ui.write(self.e.EV_ABS, self.e.ABS_X, min(max(ax, 0), self.RANGE))
        self.ui.write(self.e.EV_ABS, self.e.ABS_Y, min(max(ay, 0), self.RANGE))
        self.ui.syn()

    def position(self):
        return None  # not observable on Wayland

    def move_rel(self, dx: float, dy: float):
        """Relative motion in (roughly, after acceleration) px; carries fractions over."""
        self.frac[0] += dx
        self.frac[1] += dy
        ix, iy = int(self.frac[0]), int(self.frac[1])
        if ix or iy:
            self.frac[0] -= ix
            self.frac[1] -= iy
            self.rel.write(self.e.EV_REL, self.e.REL_X, ix)
            self.rel.write(self.e.EV_REL, self.e.REL_Y, iy)
            self.rel.syn()

    def close(self):
        self.ui.close()
        self.rel.close()


def make_pointer(backend: str, mons: list[Monitor]):
    if backend == "auto":
        backend = "x11" if session_type() == "x11" else "uinput"
    return X11Pointer() if backend == "x11" else UInputPointer(mons)


# ---------------------------------------------------------------- evdev input


def input_devices(key_codes: set[int]):
    """(keyboards having any of key_codes, mice having BTN_LEFT) that we can read."""
    from evdev import InputDevice, list_devices
    from evdev import ecodes as e

    kbds, mice = [], []
    for path in list_devices():
        try:
            dev = InputDevice(path)
        except OSError:
            continue
        keys = dev.capabilities().get(e.EV_KEY, [])
        if dev.name.startswith("gazepoint-pointer"):
            dev.close()
        elif key_codes & set(keys):
            kbds.append(dev)
        elif e.BTN_LEFT in keys:
            mice.append(dev)
        else:
            dev.close()
    return kbds, mice


def readable_input() -> bool:
    return any(os.access(f"/dev/input/{n}", os.R_OK) for n in os.listdir("/dev/input") if n.startswith("event"))


class PointerDecoder:
    """Turns raw evdev events from pointing devices into log records: button
    down/up, wheel steps, touch begin/end (touchpads), and relative motion
    coalesced to at most one record per `motion_s` per device. Touchpad
    tap-to-click is synthesised by libinput, so it never shows up here."""

    BUTTONS = {0x110: "left", 0x111: "right", 0x112: "middle", 0x113: "side", 0x114: "extra"}

    def __init__(self, emit: Callable[[dict], None], motion_s: float = 0.05):
        from evdev import ecodes as e

        self.e = e
        self.emit = emit
        self.motion_s = motion_s
        self._motion: dict[str, list] = {}  # dev -> [dx, dy, t_first]

    def feed(self, dev: str, type_: int, code: int, value: int, t: float):
        e = self.e
        if type_ == e.EV_KEY and value in (0, 1):
            if code in self.BUTTONS:
                self.flush(dev)
                self.emit({"type": "button", "t": t, "dev": dev, "button": self.BUTTONS[code], "down": value == 1})
            elif code == e.BTN_TOUCH:
                self.emit({"type": "touch", "t": t, "dev": dev, "down": value == 1})
        elif type_ == e.EV_REL:
            if code in (e.REL_X, e.REL_Y):
                m = self._motion.setdefault(dev, [0, 0, t])
                m[0 if code == e.REL_X else 1] += value
            elif code in (e.REL_WHEEL, e.REL_HWHEEL):
                self.emit({"type": "wheel", "t": t, "dev": dev,
                           "dx" if code == e.REL_HWHEEL else "dy": value})
        elif type_ == e.EV_SYN:
            m = self._motion.get(dev)
            if m and t - m[2] >= self.motion_s:
                self.flush(dev, t)

    def flush(self, dev: str | None = None, t: float | None = None):
        """Emit pending motion for dev (or every device)."""
        for d in [dev] if dev else list(self._motion):
            m = self._motion.pop(d, None)
            if m and (m[0] or m[1]):
                self.emit({"type": "motion", "t": t if t is not None else m[2], "dev": d, "dx": m[0], "dy": m[1]})


class InputListener(threading.Thread):
    """Calls on_key(down: bool, t) for the hotkey, on_reaim(down, t) for the
    re-aim key and on_slow(down, t) for the slow-gain key (if any), on_click(t) for left clicks, and on_pointer(record)
    for PointerDecoder records from every pointing device.

    Does not grab devices: the keys still reach applications, so pick ones
    whose lone press/release is harmless (Right Ctrl by default).
    """

    def __init__(self, key_name: str, on_key: Callable, on_click: Callable | None = None,
                 reaim_name: str = "", on_reaim: Callable | None = None,
                 on_pointer: Callable[[dict], None] | None = None,
                 slow_name: str = "", on_slow: Callable | None = None):
        super().__init__(daemon=True, name="input")
        from evdev import ecodes as e

        self.e = e
        self.key_name = key_name
        self.handlers = {e.ecodes[key_name]: on_key}
        if reaim_name and on_reaim:
            self.handlers[e.ecodes[reaim_name]] = on_reaim
        if slow_name and on_slow:
            self.handlers[e.ecodes[slow_name]] = on_slow
        # The same key on the other side (KEY_RIGHTCTRL <-> KEY_LEFTCTRL): pressing it is
        # the usual reason the hotkey "does nothing", so say so once.
        other = key_name.replace("LEFT", "\0").replace("RIGHT", "LEFT").replace("\0", "RIGHT")
        self.sibling = (other, e.ecodes[other]) if other != key_name and other in e.ecodes else None
        self._hinted = False
        self.on_click = on_click
        self.pointer = PointerDecoder(on_pointer) if on_pointer else None
        self.read_mice = bool(on_click or on_pointer)
        self.kbds, self.mice = input_devices(set(self.handlers))
        if not self.kbds:
            raise RuntimeError(
                f"no readable keyboard with {key_name}; is your user in the 'input' group? "
                "(scripts/setup-input-perms.sh)"
            )
        self._stop = False

    def _rescan(self, devs: dict):
        """Pick up keyboards that (re)connected since startup, e.g. Bluetooth after sleep."""
        known = {d.path for d in devs.values()}
        kbds, mice = input_devices(set(self.handlers))
        for d in kbds + mice:
            if d.path in known or (d in mice and not self.read_mice):
                d.close()
                continue
            devs[d.fd] = d
            print(f"input: listening on {d.name}")

    def run(self):
        e = self.e
        devs = {d.fd: d for d in self.kbds + (self.mice if self.read_mice else [])}
        last_scan = time.monotonic()
        while not self._stop:
            if time.monotonic() - last_scan > 2:
                self._rescan(devs)
                last_scan = time.monotonic()
            r, _, _ = select.select(list(devs), [], [], 0.2)
            if self.pointer and not r:
                self.pointer.flush()
            for fd in r:
                try:
                    events = list(devs[fd].read())
                except OSError:
                    print(f"input: lost {devs[fd].name}")
                    devs.pop(fd).close()
                    continue
                for ev in events:
                    t = time.monotonic()
                    if self.pointer:
                        self.pointer.feed(devs[fd].name, ev.type, ev.code, ev.value, t)
                    if ev.type != e.EV_KEY:
                        continue
                    handler = self.handlers.get(ev.code)
                    if handler and ev.value in (0, 1):
                        handler(ev.value == 1, t)
                    elif ev.code == e.BTN_LEFT and ev.value == 1 and self.on_click:
                        self.on_click(t)
                    elif self.sibling and ev.code == self.sibling[1] and ev.value == 1 and not self._hinted:
                        self._hinted = True
                        print(f"input: saw {self.sibling[0]}, but the hotkey is {self.key_name} "
                              "(change [run] hotkey in ~/.config/gazepoint/config.toml)")

    def stop(self):
        self._stop = True
