"""Per-unit compute (MACs) and output tensor sizes for split-computing.

Port of the architecture model used by the live DNN Split Advisor at
https://menti.ameer.itbabylon.org/split.html (224x224x3 input).
"""
import math


def _conv(cin, cout, k, s, h):
    ho = math.ceil(h / s)
    return ho * ho * cout * cin * k * k, ho


def resnet18():
    units = []
    macs, h = _conv(3, 64, 7, 2, 224)
    h = math.ceil(h / 2)  # max-pool 3x3/2
    c = 64
    units.append(("conv1+pool", macs, h * h * c))
    for li, cout in enumerate([64, 128, 256, 512]):
        for b in range(2):
            s = 2 if (b == 0 and li > 0) else 1
            m1, h1 = _conv(c, cout, 3, s, h)
            m2, _ = _conv(cout, cout, 3, 1, h1)
            m = m1 + m2
            if s == 2 or c != cout:
                m += _conv(c, cout, 1, s, h)[0]
            h, c = h1, cout
            units.append((f"layer{li + 1}.{b}", m, h * h * c))
    units.append(("avgpool", h * h * c, c))
    units.append(("fc", c * 1000, 1000))
    return units


def mobilenetv2():
    units = []
    h, c = 112, 32
    units.append(("stem", 112 * 112 * 32 * 3 * 9, h * h * c))
    cfg = [(1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2), (6, 64, 4, 2),
           (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]
    idx = 1
    for t, cout, n, s0 in cfg:
        for i in range(n):
            s = s0 if i == 0 else 1
            hid, ho = c * t, math.ceil(h / s)
            m = 0
            if t != 1:
                m += h * h * c * hid      # expand 1x1
            m += ho * ho * hid * 9        # depthwise 3x3
            m += ho * ho * hid * cout     # project 1x1
            h, c = ho, cout
            units.append((f"block{idx}", m, h * h * c))
            idx += 1
    units.append(("conv1x1", h * h * c * 1280, h * h * 1280))
    c = 1280
    units.append(("avgpool", h * h * c, c))
    units.append(("fc", c * 1000, 1000))
    return units


MODELS = {"resnet18": resnet18, "mobilenetv2": mobilenetv2}
JPEG_BYTES = 20 * 1024


def split_points(model, bytes_per_elem=4):
    """List of (label, cumulative MACs on device, bytes to upload). Last point = all on device."""
    units = MODELS[model]()
    pts = [("input (JPEG)", 0, JPEG_BYTES)]
    cum = 0
    for i, (name, macs, out) in enumerate(units):
        cum += macs
        pts.append((name, cum, 0 if i == len(units) - 1 else out * bytes_per_elem))
    return pts, cum
