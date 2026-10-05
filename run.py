"""Carbon-aware split computing from Babylon, Iraq.

Usage:
    python run.py                 # use the trace snapshot in data/
    python run.py --fetch 72      # download the last 72 h from the live platform first
"""
import argparse, json, urllib.request
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from carbonsplit.evaluate import Device, Tier, evaluate, best, load_trace

API = "https://menti.ameer.itbabylon.org/api/rtt-samples?id={id}&hours={h}&mode=all"
TRACES = {"uae": "AWS me-central-1 (UAE)", "fra": "DigitalOcean FRA1 (Frankfurt)"}
COLORS = {"edge": "#2a9d8f", "uae": "#e9a23b", "fra": "#5b6dc8", "device": "#888888"}


def fetch(hours):
    for tid in TRACES:
        with urllib.request.urlopen(API.format(id=tid, h=hours), timeout=60) as r:
            open(f"data/rtt_{tid}.json", "wb").write(r.read())


def tiers():
    out = [Tier("edge", "Local edge in Babylon (what-if)", 500, 0.36, 690.0,
                rtts_ms=np.full(1000, 5.0))]
    for tid, name in TRACES.items():
        rtts, lost, _ = load_trace(f"data/rtt_{tid}.json")
        grid = 420.0 if tid == "uae" else 380.0   # ASSUMPTION: UAE / Germany grid g/kWh
        out.append(Tier(tid, name, 1000, 0.36, grid, rtts_ms=rtts, lost=lost))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", type=int, default=0, help="hours of live trace to download first")
    ap.add_argument("--model", default="resnet18", choices=["resnet18", "mobilenetv2"])
    ap.add_argument("--deadline", type=float, default=150.0)
    ap.add_argument("--min-hit", type=float, default=0.95)
    a = ap.parse_args()
    if a.fetch:
        fetch(a.fetch)

    dev, ts = Device(), tiers()
    results = {t.id: evaluate(a.model, dev, t, a.deadline) for t in ts}
    summary = {"model": a.model, "deadline_ms": a.deadline, "min_hit": a.min_hit, "tiers": {}}

    # Figure 1: measured RTT distributions
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for t in ts[1:]:
        r = np.sort(t.rtts_ms)
        ax.plot(r, np.linspace(0, 1, len(r)), color=COLORS[t.id], lw=2,
                label=f"{t.name}  (n={len(r):,}, loss={t.lost / (len(r) + t.lost):.1%})")
    ax.set_xscale("log"); ax.set_xlabel("Round-trip time from Babylon (ms, log)")
    ax.set_ylabel("CDF"); ax.grid(alpha=.3); ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Measured RTT from the University of Babylon edge node")
    fig.tight_layout(); fig.savefig("figures/rtt_cdf.png", dpi=160); plt.close(fig)

    # Figure 2: latency vs carbon for every split point and tier
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for t in ts:
        rows = results[t.id][:-1]
        ax.plot([r["p95"] for r in rows], [r["mg_co2"] for r in rows], "o-", ms=3,
                color=COLORS[t.id], label=t.name, alpha=.85)
        b = best(results[t.id], "mg_co2", a.min_hit)
        if b:
            ax.plot(b["p95"], b["mg_co2"], "*", ms=16, color=COLORS[t.id], mec="k")
        summary["tiers"][t.id] = {"name": t.name, "carbon_best": b,
                                  "latency_best": best(results[t.id], "p95", a.min_hit)}
    loc = results["uae"][-1]
    ax.plot(loc["p95"], loc["mg_co2"], "s", ms=9, color=COLORS["device"], label="All on device")
    summary["all_on_device"] = loc
    ax.axvline(a.deadline, color="crimson", ls="--", lw=1)
    ax.text(a.deadline, ax.get_ylim()[1] * .95, f" deadline {a.deadline:.0f} ms", color="crimson", fontsize=8, va="top")
    ax.set_xscale("log"); ax.set_xlabel("p95 end-to-end latency (ms, log)")
    ax.set_ylabel("mg CO$_2$e per inference"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title(f"{a.model}: latency vs carbon per split point "
                 "(star = greenest plan meeting the deadline)", fontsize=9)
    fig.tight_layout(); fig.savefig("figures/latency_vs_carbon.png", dpi=160); plt.close(fig)

    # Figure 3: greenest feasible plan as the deadline tightens
    dls = np.arange(60, 420, 10)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for t in ts:
        ys = []
        for d in dls:
            b = best(evaluate(a.model, dev, t, d), "mg_co2", a.min_hit)
            ys.append(b["mg_co2"] if b else np.nan)
        ax.plot(dls, ys, color=COLORS[t.id], lw=2, label=t.name)
    ax.axhline(loc["mg_co2"], color=COLORS["device"], ls=":", label="All on device")
    ax.set_xlabel("Deadline (ms)"); ax.set_ylabel("Lowest mg CO$_2$e meeting deadline")
    ax.grid(alpha=.3); ax.legend(fontsize=8); ax.set_title("Greenest feasible plan per tier vs deadline (missing = infeasible)", fontsize=10)
    fig.tight_layout(); fig.savefig("figures/deadline_vs_carbon.png", dpi=160); plt.close(fig)


    # Figure 4: how the greenest plan changes with uplink bandwidth (UAE tier)
    import dataclasses
    bws = [0.25, 0.5, 1, 2, 5, 10, 20]
    xs, ys, labels = [], [], []
    for bw in bws:
        d2 = dataclasses.replace(dev, uplink_mbps=bw)
        rows = evaluate(a.model, d2, ts[1], 400.0)
        b = best(rows, "mg_co2", a.min_hit)
        xs.append(bw); ys.append(b["mg_co2"] if b else np.nan); labels.append(("all on device" if b["bytes"] == 0 else b["split"]) if b else "none")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(xs, ys, "o-", color=COLORS["uae"], lw=2)
    for x, y, l in zip(xs, ys, labels):
        ax.annotate(l, (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7)
    ax.set_xscale("log"); ax.set_xlabel("Device uplink (Mbit/s, log)"); ax.set_ylabel("mg CO$_2$e per inference")
    ax.grid(alpha=.3); ax.set_title("Greenest split vs uplink bandwidth (UAE tier, 400 ms deadline)", fontsize=10)
    fig.tight_layout(); fig.savefig("figures/bandwidth_vs_split.png", dpi=160); plt.close(fig)
    summary["bandwidth_sweep"] = dict(zip(map(str, bws), labels))

    json.dump(summary, open("figures/summary.json", "w"), indent=2)
    print(json.dumps({k: (v["carbon_best"] and {x: v["carbon_best"][x] for x in ("split", "p95", "hit", "mg_co2")})
                      for k, v in summary["tiers"].items()}, indent=1))
    print("all on device:", {x: round(loc[x], 2) for x in ("p95", "mg_co2")})


if __name__ == "__main__":
    main()
