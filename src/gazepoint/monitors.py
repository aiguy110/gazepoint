"""Monitor discovery: identities (EDID), physical sizes, and desktop layout.

Identity is derived from EDID so the same physical monitor gets the same id
regardless of connector, dock, or session type. Layout (logical px) comes from
xrandr on X11 or Mutter's DisplayConfig D-Bus API on GNOME/Wayland.
"""

from __future__ import annotations

import glob
import hashlib
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

BUILTIN_PREFIXES = ("eDP", "LVDS", "DSI")


@dataclass
class Monitor:
    id: str
    name: str
    connector: str
    x: int
    y: int
    w: int  # logical px
    h: int
    width_mm: float
    height_mm: float
    builtin: bool
    primary: bool = False

    def to_json(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- EDID


def parse_edid(b: bytes) -> dict | None:
    if len(b) < 128 or b[:8] != b"\x00\xff\xff\xff\xff\xff\xff\x00":
        return None
    m = (b[8] << 8) | b[9]
    vendor = "".join(chr(((m >> s) & 0x1F) + 64) for s in (10, 5, 0))
    product = b[10] | (b[11] << 8)
    serial_num = int.from_bytes(b[12:16], "little")
    name = serial_str = None
    w_mm = h_mm = None
    for off in (54, 72, 90, 108):
        d = b[off : off + 18]
        if d[0] or d[1]:  # detailed timing descriptor
            if w_mm is None:
                w_mm = d[12] | ((d[14] & 0xF0) << 4)
                h_mm = d[13] | ((d[14] & 0x0F) << 8)
            continue
        text = d[5:18].split(b"\x0a")[0].decode("ascii", "replace").strip()
        if d[3] == 0xFC:
            name = text
        elif d[3] == 0xFF:
            serial_str = text
    if not w_mm or not h_mm:
        w_mm, h_mm = b[21] * 10, b[22] * 10
    return {
        "vendor": vendor,
        "product": product,
        "serial_num": serial_num,
        "serial_str": serial_str,
        "name": name,
        "width_mm": float(w_mm),
        "height_mm": float(h_mm),
    }


def edid_id(e: dict) -> str:
    prod = e["name"] or f"0x{e['product']:04x}"
    serial = e["serial_str"] or (f"0x{e['serial_num']:08x}" if e["serial_num"] else "")
    return f"{e['vendor']}:{prod}:{serial}"


def sysfs_edids() -> dict[str, dict]:
    """connector name (as in sysfs, e.g. 'HDMI-A-1') -> parsed EDID, connected only."""
    out = {}
    for d in glob.glob("/sys/class/drm/card*-*"):
        p = Path(d)
        try:
            if (p / "status").read_text().strip() != "connected":
                continue
            e = parse_edid((p / "edid").read_bytes())
        except OSError:
            continue
        if e:
            out[p.name.split("-", 1)[1]] = e
    return out


def _norm_connector(name: str) -> str:
    # DRM 'HDMI-A-1' vs Xorg/Mutter 'HDMI-1'; nvidia/amdgpu differ in numbering,
    # so this is only a last-resort match.
    return name.replace("HDMI-A-", "HDMI-").replace("DVI-D-", "DVI-")


# ---------------------------------------------------------------- session


def session_type() -> str:
    """'x11', 'wayland', or 'unknown'. Falls back to loginctl so it works over ssh."""
    st = os.environ.get("XDG_SESSION_TYPE", "")
    if st in ("x11", "wayland"):
        return st
    try:
        uid = str(os.getuid())
        sessions = subprocess.run(
            ["loginctl", "list-sessions", "--no-legend"], capture_output=True, text=True, timeout=5
        ).stdout.split("\n")
        for line in sessions:
            parts = line.split()
            if len(parts) >= 2 and parts[1] == uid:
                t = subprocess.run(
                    ["loginctl", "show-session", parts[0], "-p", "Type", "--value"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                ).stdout.strip()
                if t in ("x11", "wayland"):
                    return t
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def desktop() -> str:
    return os.environ.get("XDG_CURRENT_DESKTOP", "")


# ---------------------------------------------------------------- backends

_XRANDR_RE = re.compile(
    r"^(\S+) connected( primary)? (\d+)x(\d+)\+(\d+)\+(\d+)\s*(normal|left|right|inverted)?"
)


def from_xrandr() -> list[Monitor]:
    out = subprocess.run(["xrandr", "--prop"], capture_output=True, text=True, check=True).stdout
    lines = out.split("\n")
    sys_edids = sysfs_edids()
    mons = []
    i = 0
    while i < len(lines):
        m = _XRANDR_RE.match(lines[i])
        i += 1
        if not m:
            continue
        conn, primary, w, h, x, y, rot = m.groups()
        mm = re.search(r"(\d+)mm x (\d+)mm", lines[i - 1])
        edid_hex = ""
        while i < len(lines) and (lines[i].startswith("\t") or lines[i].startswith(" ")):
            if lines[i].strip() == "EDID:":
                i += 1
                while i < len(lines) and re.fullmatch(r"\s+[0-9a-f]+", lines[i]):
                    edid_hex += lines[i].strip()
                    i += 1
                continue
            i += 1
        e = parse_edid(bytes.fromhex(edid_hex)) if edid_hex else None
        if e is None:
            e = sys_edids.get(conn) or sys_edids.get(conn.replace("HDMI-", "HDMI-A-"))
        wmm, hmm = (e["width_mm"], e["height_mm"]) if e else (0.0, 0.0)
        if (not wmm or not hmm) and mm:
            wmm, hmm = float(mm.group(1)), float(mm.group(2))
        if rot in ("left", "right") and e:
            wmm, hmm = hmm, wmm
        mons.append(
            Monitor(
                id=edid_id(e) if e else f"conn:{conn}",
                name=(e and e["name"]) or conn,
                connector=conn,
                x=int(x),
                y=int(y),
                w=int(w),
                h=int(h),
                width_mm=wmm,
                height_mm=hmm,
                builtin=conn.startswith(BUILTIN_PREFIXES),
                primary=bool(primary),
            )
        )
    return mons


def _v(variant):
    """jeepney returns variants as (signature, value)."""
    return variant[1] if isinstance(variant, tuple) and len(variant) == 2 else variant


def from_mutter() -> list[Monitor]:
    from jeepney import DBusAddress, new_method_call
    from jeepney.io.blocking import open_dbus_connection

    addr = DBusAddress(
        "/org/gnome/Mutter/DisplayConfig",
        bus_name="org.gnome.Mutter.DisplayConfig",
        interface="org.gnome.Mutter.DisplayConfig",
    )
    with open_dbus_connection(bus="SESSION") as conn:
        reply = conn.send_and_get_reply(new_method_call(addr, "GetCurrentState"))
    _, monitors, logical_monitors, props = reply.body
    layout_mode = _v(props.get("layout-mode", ("u", 1)))  # 1 = logical, 2 = physical

    phys = {}  # connector -> (spec, current mode (w, h), props)
    for spec, modes, mprops in monitors:
        cur = next((m for m in modes if _v(m[6].get("is-current", ("b", False)))), None)
        if cur:
            phys[spec[0]] = (spec, (cur[1], cur[2]), mprops)

    sys_edids = sysfs_edids()
    mons = []
    for x, y, scale, transform, primary, specs, _lprops in logical_monitors:
        for spec in specs:
            if spec[0] not in phys:
                continue
            (conn, vendor, product, serial), (mw, mh), mprops = phys[spec[0]]
            e = None
            for c, cand in sys_edids.items():
                if cand["vendor"] == vendor and (
                    (serial and serial in (cand["serial_str"], f"0x{cand['serial_num']:08x}"))
                    or (product and product == cand["name"])
                ):
                    e = cand
                    break
            if e is None:
                e = sys_edids.get(conn) or next(
                    (v for c, v in sys_edids.items() if _norm_connector(c) == _norm_connector(conn)),
                    None,
                )
            if transform % 2 == 1:  # 90 / 270 degrees
                mw, mh = mh, mw
            w, h = (round(mw / scale), round(mh / scale)) if layout_mode == 1 else (mw, mh)
            wmm, hmm = (e["width_mm"], e["height_mm"]) if e else (0.0, 0.0)
            if transform % 2 == 1:
                wmm, hmm = hmm, wmm
            builtin = bool(_v(mprops.get("is-builtin", ("b", False)))) or conn.startswith(
                BUILTIN_PREFIXES
            )
            mons.append(
                Monitor(
                    id=edid_id(e) if e else f"{vendor}:{product}:{serial}",
                    name=_v(mprops.get("display-name", ("s", product))) or product,
                    connector=conn,
                    x=int(x),
                    y=int(y),
                    w=int(w),
                    h=int(h),
                    width_mm=wmm,
                    height_mm=hmm,
                    builtin=builtin,
                    primary=bool(primary),
                )
            )
    return mons


def detect() -> list[Monitor]:
    st = session_type()
    if st == "wayland" and "GNOME" in desktop().upper():
        mons = from_mutter()
    elif st == "wayland":
        # Try Mutter anyway (desktop env var is often missing over ssh), then XWayland.
        try:
            mons = from_mutter()
        except Exception:
            mons = from_xrandr()
    else:
        mons = from_xrandr()
    # Disambiguate identical monitors that report no serial.
    seen: dict[str, int] = {}
    for m in mons:
        n = seen.get(m.id, 0)
        seen[m.id] = n + 1
        if n:
            m.id = f"{m.id}#{m.connector}"
    missing = [m.connector for m in mons if not (m.width_mm and m.height_mm)]
    if missing:
        raise RuntimeError(f"no physical size for {missing}; set it via config (TODO) or check EDID")
    return mons


def profile_key(mons: list[Monitor]) -> str:
    """Stable key for the set of *external* monitors (the built-in panel is always there)."""
    ext = sorted(m.id for m in mons if not m.builtin)
    return "+".join(ext) if ext else "laptop-only"


def profile_hash(key: str) -> str:
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def desktop_bbox(mons: list[Monitor]) -> tuple[int, int, int, int]:
    x0 = min(m.x for m in mons)
    y0 = min(m.y for m in mons)
    x1 = max(m.x + m.w for m in mons)
    y1 = max(m.y + m.h for m in mons)
    return x0, y0, x1 - x0, y1 - y0
