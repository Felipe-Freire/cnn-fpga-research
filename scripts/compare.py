"""Compare the cnn-optimization analytical VU13P estimator with Vitis HLS and Vivado results.

Usage:
    python scripts/compare.py [--configs u16 u8 kkt kkt_pf1]

Writes results/validation_summary.{csv,md} and results/validation_resources.png
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from model.configs import CLOCK_PERIOD_NS, CONFIGS, LAYERS  # noqa: E402
from model.hw_estimator_ref import estimate_for_config  # noqa: E402
from scripts.reports import parse_vivado_util_hier  # noqa: E402

RES = REPO / "results"
BRAM18_BITS = 18 * 1024


def _load(p: Path):
    return json.loads(p.read_text()) if p.exists() else {}


def collect(names):
    rows = []
    for name in names:
        cfg = CONFIGS[name]
        est = estimate_for_config(cfg)["total"]
        cs = _load(RES / name / "csynth.json")
        viv = _load(RES / name / "vivado" / "vivado.json")
        emu = _load(RES / name / "emulation.json")
        met = _load(REPO / "model" / "artifacts" / cfg.trained_model / "metrics.json")
        csr = cs.get("resources", {})
        vs = viv.get("post_synth", {})
        vr = viv.get("post_route", {})
        rows.append({
            "config": name,
            "w_bits": "/".join(str(cfg.w_bits[l]) for l in LAYERS),
            "a_bits(in/conv/dense)": f"{cfg.input_bits}/{cfg.a_bits['conv1d_1']}/{cfg.a_bits['dense_1']}",
            "conv_PF": cfg.conv_pf,
            "est_DSP": est["DSP"], "est_LUT": est["LUT"], "est_FF": est["FF"], "est_BRAM_bits": est["BRAM_bits"],
            "hls_DSP": csr.get("DSP"), "hls_LUT": csr.get("LUT"), "hls_FF": csr.get("FF"), "hls_BRAM18": csr.get("BRAM_18K"),
            "syn_DSP": vs.get("DSP"), "syn_LUT": vs.get("LUT"), "syn_FF": vs.get("FF"), "syn_BRAM_tile": vs.get("BRAM_tile"),
            "route_DSP": vr.get("DSP"), "route_LUT": vr.get("LUT"), "route_LUTRAM": vr.get("LUT_memory"),
            "route_FF": vr.get("FF"), "route_BRAM_tile": vr.get("BRAM_tile"),
            "latency_cycles": cs.get("latency_worst_cycles"),
            "latency_ns": (cs.get("latency_worst_cycles") or 0) * CLOCK_PERIOD_NS if cs else None,
            "II_cycles": cs.get("interval_max_cycles"),
            "route_WNS_ns": viv.get("post_route_wns_ns"),
            "SP_holdout_qkeras": (met.get("holdout_at_val_threshold") or {}).get("sp"),
            "SP_holdout_hls_knee": emu.get("sp_hls_holdout"),
            "SP_holdout_qkeras_knee": emu.get("sp_qkeras_holdout"),
        })
    return pd.DataFrame(rows)


def _fmt(v, nd=0):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "–"
    return f"{v:,.{nd}f}"


def _ratio(real, est):
    if real is None or est is None or pd.isna(real) or pd.isna(est):
        return "–"
    if est == 0:
        return "∞" if real > 0 else "1.0×"
    return f"{real / est:.2f}×"


_KINDS = [("conv1d_1", "conv_1d"), ("dense_1", "config9"), ("output_dense", "config12"),
          ("relu", "relu"), ("pool", "pooling"), ("sigmoid", "sigmoid")]


def layer_breakdown(name: str):
    """Post-synthesis LUT/FF/DSP per layer from the hierarchical report (depth-1 instances).
    Whatever Vivado inlined into the top (e.g. a conv with {0,1} weights) lands in 'top/glue'."""
    p = RES / name / "vivado" / "post_synth_util_hier.rpt"
    if not p.exists():
        return {}
    rows = parse_vivado_util_hier(p)
    top, agg = rows[0], {}
    for r in rows:
        if r["depth"] != 1 or r["instance"].startswith("("):
            continue
        kind = next((k for k, pat in _KINDS if pat in r["instance"]), "top/glue")
        a = agg.setdefault(kind, [0, 0, 0])
        a[0] += r.get("Total LUTs", 0)
        a[1] += r.get("FFs", 0)
        a[2] += r.get("DSP Blocks", 0)
    s = [sum(v[i] for v in agg.values()) for i in range(3)]
    g = agg.setdefault("top/glue", [0, 0, 0])
    g[0] += top["Total LUTs"] - s[0]
    g[1] += top["FFs"] - s[1]
    g[2] += top["DSP Blocks"] - s[2]
    return agg


def write_markdown(df: pd.DataFrame, path: Path):
    lines = [
        "# Validation: analytical VU13P estimator vs. hls4ml synthesis",
        "",
        f"Device xcvu13p-flga2577-2-e, clock {1e3 / CLOCK_PERIOD_NS:.0f} MHz, hls4ml io_parallel / Latency / RF=1, "
        "Vitis HLS + Vivado 2026.1.1 (out-of-context). `est_*` = ORIGINAL cnn-optimization estimator "
        "(`estimate_hardware_resources`, commit f05e91a; replaced since, see cost_model_check.md), "
        "`hls_*` = Vitis HLS csynth estimate, `route_*` = Vivado post-place&route.",
        "",
        "## Resources",
        "",
        "| config | w (conv/dense/out) | a (in/conv/dense) | conv PF | DSP est → route | LUT est → HLS → route | "
        "FF est → HLS → route | BRAM est [bits] → route [tiles] |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r.config} | {r.w_bits} | {r['a_bits(in/conv/dense)']} | {r.conv_PF} "
            f"| {_fmt(r.est_DSP)} → **{_fmt(r.route_DSP)}** "
            f"| {_fmt(r.est_LUT)} → {_fmt(r.hls_LUT)} → **{_fmt(r.route_LUT)}** "
            f"| {_fmt(r.est_FF)} → {_fmt(r.hls_FF)} → **{_fmt(r.route_FF)}** "
            f"| {_fmt(r.est_BRAM_bits)} → **{_fmt(r.route_BRAM_tile, 1)}** |"
        )
    lines += [
        "",
        "## Ratio real (post-route) / estimated",
        "",
        "| config | DSP | LUT | FF |",
        "|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        lines.append(f"| {r.config} | {_ratio(r.route_DSP, r.est_DSP)} | {_ratio(r.route_LUT, r.est_LUT)} "
                     f"| {_ratio(r.route_FF, r.est_FF)} |")
    lines += [
        "",
        "## Per layer (post-synthesis LUT / FF / DSP vs. estimator LUT)",
        "",
        "`top/glue` holds logic Vivado inlined into the top: ReLU/pool/sigmoid and, in `kkt`, the whole",
        "Conv1D (its 2-bit weights take only the values {0, 1}, so it reduces to wires and adders).",
        "",
        "| config | layer | est. LUT | real LUT | real FF | real DSP |",
        "|---|---|---|---|---|---|",
    ]
    for name in df["config"]:
        est = {b["layer"]: b["LUT"] for b in estimate_for_config(CONFIGS[name])["breakdown"]}
        for layer, (lut, ff, dsp) in sorted(layer_breakdown(name).items()):
            lines.append(f"| {name} | {layer} | {_fmt(est.get(layer))} | {lut:,} | {ff:,} | {dsp:,} |")
    lines += [
        "",
        "## Timing and physics performance",
        "",
        "| config | latency [cycles] | latency [ns] | II [cycles] | post-route WNS [ns] | SP holdout QKeras | SP holdout hls4ml C++ |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r.config} | {_fmt(r.latency_cycles)} | {_fmt(r.latency_ns, 1)} | {_fmt(r.II_cycles)} "
            f"| {_fmt(r.route_WNS_ns, 3)} | {_fmt(100 * r.SP_holdout_qkeras_knee, 2)} % "
            f"| {_fmt(100 * r.SP_holdout_hls_knee, 2)} % |"
        )
    path.write_text("\n".join(lines) + "\n")


def plot(df: pd.DataFrame, path: Path):
    # Light-mode reference palette, categorical slots 1-3 (validated all-pairs for dot plots)
    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    series = [("est", "Analytical estimator", "#2a78d6", "o"),
              ("hls", "Vitis HLS (csynth)", "#eb6834", "s"),
              ("route", "Vivado post-route", "#1baf7a", "D")]
    panels = [("LUT", "LUTs"), ("FF", "Flip-flops"), ("DSP", "DSP48E2  (zero drawn on the bottom edge)")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), facecolor=surface)
    x = range(len(df))
    for ax, (res, title) in zip(axes, panels):
        ax.set_facecolor(surface)
        all_vals = []
        for i, (prefix, label, color, marker) in enumerate(series):
            col = f"{prefix}_{res}"
            if col not in df:
                continue
            vals = df[col].astype(float).fillna(0)
            all_vals += list(vals)
            ax.scatter([xi + (i - 1) * 0.18 for xi in x], vals.clip(lower=0.8), s=64, color=color, marker=marker,
                       label=label, edgecolors=surface, linewidths=2, zorder=3)
        ax.set_yscale("log")
        positive = [v for v in all_vals if v > 0]
        lo = 0.6 if any(v == 0 for v in all_vals) else min(positive) / 2
        ax.set_ylim(lo, max(positive) * 2)
        ax.set_xticks(list(x), df["config"], color=ink)
        ax.set_title(title, color=ink, fontsize=11, loc="left")
        ax.grid(axis="y", color=grid, linewidth=0.8, zorder=0)
        ax.tick_params(colors=ink2, labelsize=9)
        for s in ax.spines.values():
            s.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.text(0.01, 1.07, "VU13P resources: cnn-optimization estimate vs. synthesis (log scale)",
             color=ink, fontsize=13, ha="left")
    fig.legend(handles, labels, loc="upper left", ncol=3, frameon=False, labelcolor=ink, bbox_to_anchor=(0.005, 1.04))
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=surface)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    args = ap.parse_args()
    df = collect(args.configs)
    df.to_csv(RES / "validation_summary.csv", index=False)
    write_markdown(df, RES / "validation_summary.md")
    plot(df, RES / "validation_resources.png")
    print((RES / "validation_summary.md").read_text())


if __name__ == "__main__":
    main()
