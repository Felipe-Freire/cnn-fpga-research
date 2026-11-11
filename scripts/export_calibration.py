"""Export the synthesized design points as a calibration dataset for cnn-optimization's cost model.

For every synthesized configuration it stores: bit widths, Conv1D ParallelizationFactor, the
integer weights/biases that were actually synthesized (from the QAT models), post-synthesis
per-layer and post-route total utilization, latency and II. cnn-optimization recomputes its cost
features from the integer weights and fits its coefficients on these points, so no FPGA tool is
needed on that side.

Usage:
    python scripts/export_calibration.py --out ../cnn-optimization/src/calibration/vu13p_hls4ml_synthesis.json
"""

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from model.configs import CLOCK_PERIOD_NS, CONFIGS, FPGA_PART, LAYERS  # noqa: E402
from model.loading import load_config_model  # noqa: E402
from scripts.compare import layer_breakdown  # noqa: E402

RES = REPO / "results"


def integer_params(qmodel, name, bits):
    layer = qmodel.get_layer(name)
    integer = min(2, max(0, bits - 1))
    step = 2.0 ** (integer + 1 - bits)  # LSB of quantized_bits(bits, integer, alpha=1)
    k = np.asarray(layer.kernel_quantizer_internal(layer.kernel))
    b = np.asarray(layer.bias_quantizer_internal(layer.bias))
    k_int = np.rint(k / step).astype(int).reshape(-1, k.shape[-1])  # (fan_in, n_out)
    b_int = np.rint(b / step).astype(int)
    assert np.allclose(k_int * step, k.reshape(-1, k.shape[-1])) and np.allclose(b_int * step, b)
    return k_int.tolist(), b_int.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    args = ap.parse_args()

    points = []
    for name in args.configs:
        cfg = CONFIGS[name]
        qm = load_config_model(cfg)
        viv = json.loads((RES / name / "vivado" / "vivado.json").read_text())
        cs = json.loads((RES / name / "csynth.json").read_text())
        in_bits = {"conv1d_1": cfg.input_bits, "dense_1": cfg.a_bits["conv1d_1"], "output_dense": cfg.a_bits["dense_1"]}
        layers = {}
        for lname in LAYERS:
            k_int, b_int = integer_params(qm, lname, cfg.w_bits[lname])
            layers[lname] = {
                "w_bits": cfg.w_bits[lname],
                "a_in_bits": in_bits[lname],
                "positions_parallel": 25 if (lname == "conv1d_1" and cfg.conv_pf == 25) else 1,
                "relu_after": lname in ("conv1d_1", "dense_1"),
                "kernel_int": k_int,
                "bias_int": b_int,
            }
        per_layer = {k: {"LUT": v[0], "FF": v[1], "DSP": v[2]} for k, v in layer_breakdown(name).items()}
        points.append({
            "name": name,
            "conv_parallelization_factor": cfg.conv_pf,
            "w_bits": cfg.w_bits,
            "a_bits": cfg.a_bits,
            "input_bits": cfg.input_bits,
            "layers": layers,
            "post_route": viv["post_route"],
            "post_synth_per_layer": per_layer,
            "latency_cycles": cs["latency_worst_cycles"],
            "ii_cycles": cs["interval_max_cycles"],
            "post_route_wns_ns": viv["post_route_wns_ns"],
        })

    out = {
        "description": "hls4ml synthesis points of the compact QCNN1D (v2, 25 rings) on the VU13P",
        "provenance": {
            "repository": "cnn-fpga-research",
            "date": str(date.today()),
            "flow": "QKeras 0.9.0 -> hls4ml 1.3.0 (Vitis backend, io_parallel, Latency, RF=1, accum auto) "
                    "-> Vitis HLS 2026.1.1 -> Vivado 2026.1.1 out-of-context synth + place & route",
            "part": FPGA_PART,
            "clock_period_ns": CLOCK_PERIOD_NS,
            "notes": "Conv1D multiplier_limit patched to scale with ParallelizationFactor (hls4ml 1.3 workaround). "
                     "Per-layer numbers are post-synthesis (hierarchical report); totals are post-route. "
                     "'top/glue' = logic Vivado inlined into the top (ReLU, pooling, sigmoid, input, "
                     "and the whole Conv1D in 'kkt').",
        },
        "points": points,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {len(points)} points to {args.out}")


if __name__ == "__main__":
    main()
