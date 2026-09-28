from gazepoint.monitors import Monitor, edid_id, parse_edid, profile_key


def make_edid(vendor="DEL", product=0xA0B1, serial=1234, name="DELL U2720Q", serial_str="ABC123", w_mm=597, h_mm=336):
    b = bytearray(128)
    b[0:8] = b"\x00\xff\xff\xff\xff\xff\xff\x00"
    m = sum((ord(c) - 64) << s for c, s in zip(vendor, (10, 5, 0)))
    b[8], b[9] = m >> 8, m & 0xFF
    b[10], b[11] = product & 0xFF, product >> 8
    b[12:16] = serial.to_bytes(4, "little")
    b[21], b[22] = w_mm // 10, h_mm // 10
    dtd = bytearray(18)
    dtd[0] = 1  # nonzero pixel clock => timing descriptor
    dtd[12], dtd[13], dtd[14] = w_mm & 0xFF, h_mm & 0xFF, ((w_mm >> 8) << 4) | (h_mm >> 8)
    b[54:72] = dtd
    for off, tag, text in ((72, 0xFC, name), (90, 0xFF, serial_str)):
        d = bytearray(18)
        d[3] = tag
        d[5:18] = (text.encode() + b"\x0a").ljust(13, b" ")[:13]
        b[off : off + 18] = d
    return bytes(b)


def test_parse_edid():
    e = parse_edid(make_edid())
    assert e["vendor"] == "DEL" and e["name"] == "DELL U2720Q" and e["serial_str"] == "ABC123"
    assert (e["width_mm"], e["height_mm"]) == (597, 336)
    assert edid_id(e) == "DEL:DELL U2720Q:ABC123"


def test_profile_key_ignores_builtin_and_order():
    a = Monitor("A", "a", "DP-1", 0, 0, 1, 1, 1, 1, False)
    b = Monitor("B", "b", "DP-2", 0, 0, 1, 1, 1, 1, False)
    lap = Monitor("L", "l", "eDP-1", 0, 0, 1, 1, 1, 1, True)
    assert profile_key([b, lap, a]) == profile_key([a, b]) == "A+B"
    assert profile_key([lap]) == "laptop-only"
