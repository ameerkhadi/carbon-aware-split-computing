"""Latency, energy and carbon of every device/edge/cloud split, driven by measured RTTs.

Latency follows the live DNN Split Advisor. Energy and carbon are an explicit,
configurable model: every default below is an ASSUMPTION, not a measurement,
and is listed in the README so it can be replaced with measured values.
"""
from dataclasses import dataclass, field
import json
import numpy as np

from .models import split_points

OVERHEAD_MS = 3.0


@dataclass
class Device:
    name: str = "Raspberry Pi 4 (CPU)"
    gmacs: float = 5.0            # GMAC/s (same default as the live advisor)
    p_active_w: float = 6.0       # ASSUMPTION: board power while computing
    p_idle_w: float = 2.7         # ASSUMPTION: board power while waiting for the reply
    p_radio_w: float = 1.0        # ASSUMPTION: extra power while transmitting
    uplink_mbps: float = 10.0
    grid_g_per_kwh: float = 690.0  # ASSUMPTION: Iraq grid carbon intensity


@dataclass
class Tier:
    id: str
    name: str
    gmacs: float                  # server throughput, GMAC/s
    j_per_gmac: float             # ASSUMPTION: server energy per GMAC incl. PUE
    grid_g_per_kwh: float         # ASSUMPTION: grid carbon intensity of the region
    rtts_ms: np.ndarray = field(default_factory=lambda: np.array([]))
    lost: int = 0


NET_J_PER_BYTE = 0.006 * 3.6e6 / 1e9  # ASSUMPTION: 0.006 kWh/GB network transport -> J/byte


def load_trace(path):
    d = json.load(open(path, encoding="utf-8"))
    return np.array(d["rtts"], dtype=float), int(d.get("lost", 0)), d.get("name", "")


def evaluate(model, device, tier, deadline_ms, bytes_per_elem=4):
    pts, total = split_points(model, bytes_per_elem)
    rows = []
    n = len(tier.rtts_ms) + tier.lost
    for k, (label, cum, nbytes) in enumerate(pts):
        t_local = cum / (device.gmacs * 1e9) * 1e3
        last = k == len(pts) - 1
        t_tx = 0.0 if last else nbytes * 8 / (device.uplink_mbps * 1e6) * 1e3
        t_remote = 0.0 if last else (total - cum) / (tier.gmacs * 1e9) * 1e3 + OVERHEAD_MS
        if last:
            lat = np.array([t_local])
            hit = float(t_local <= deadline_ms)
        else:
            lat = t_local + t_tx + t_remote + tier.rtts_ms
            hit = float((lat <= deadline_ms).sum()) / n if n else 0.0
        wait_ms = 0.0 if last else float(np.median(tier.rtts_ms)) + t_remote
        e_dev = (device.p_active_w * t_local + device.p_radio_w * t_tx
                 + device.p_idle_w * wait_ms) / 1e3                       # J
        e_srv = 0.0 if last else (total - cum) / 1e9 * tier.j_per_gmac     # J
        e_net = 0.0 if last else nbytes * NET_J_PER_BYTE                   # J
        g_co2 = (e_dev * device.grid_g_per_kwh
                 + (e_srv + e_net) * tier.grid_g_per_kwh) / 3.6e6          # g
        rows.append(dict(k=k, split=label, bytes=nbytes,
                         p50=float(np.percentile(lat, 50)), p95=float(np.percentile(lat, 95)),
                         hit=hit, e_dev_mj=e_dev * 1e3, e_total_mj=(e_dev + e_srv + e_net) * 1e3,
                         mg_co2=g_co2 * 1e3))
    return rows


def best(rows, objective, min_hit):
    ok = [r for r in rows if r["hit"] >= min_hit]
    return min(ok, key=lambda r: r[objective]) if ok else None
