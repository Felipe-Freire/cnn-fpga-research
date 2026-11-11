"""Convert the trained QKeras QCNN1D to hls4ml (Vitis backend), check bit-accuracy, run C synthesis
and out-of-context Vivado synthesis on the AMD Virtex UltraScale+ VU13P.

Usage (Vitis/Vivado must be on PATH, e.g. `source scripts/env.sh`):
    python hls4ml/build_hls.py --configs u16 u8 kkt kkt_pf1 [--no-csynth] [--no-vsynth] [--impl]

Outputs:
    hls4ml/projects/<cfg>/            hls4ml/Vitis HLS project (git-ignored, regenerable)
    results/<cfg>/hls4ml_config.yaml  layer precisions actually synthesized
    results/<cfg>/emulation.json      QKeras vs hls4ml C++ agreement on the holdout set
    results/<cfg>/csynth.json         Vitis HLS estimates (latency, II, resources)
    results/<cfg>/vivado/*.rpt        post-synthesis utilization / timing (OOC)
"""

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import yaml  # noqa: E402
import hls4ml  # noqa: E402

from model.configs import CLOCK_PERIOD_NS, CONFIGS, FPGA_PART  # noqa: E402
from model.loading import load_config_model  # noqa: E402
from model.ringer_data import compute_sp_with_threshold  # noqa: E402
from scripts.reports import parse_csynth_xml, parse_vivado_util, parse_wns  # noqa: E402

PROJECT = "qcnn1d"
ART = REPO / "model" / "artifacts"
PRJ_ROOT = REPO / "hls4ml" / "projects"
RES_ROOT = REPO / "results"


@contextlib.contextmanager
def keras_v2_dispatch():
    """hls4ml >= 1.0 routes models to its Keras-3 frontend whenever the `keras` package is 3.x
    (TF 2.20 installs it). QKeras models are tf_keras (Keras 2) objects, so temporarily report the
    tf_keras version to make hls4ml use its Keras-2/QKeras frontend."""
    import keras as keras3
    import tf_keras

    orig = keras3.__version__
    keras3.__version__ = tf_keras.__version__
    try:
        yield
    finally:
        keras3.__version__ = orig


def make_hls_config(qmodel, cfg):
    with keras_v2_dispatch():
        hc = hls4ml.utils.config_from_keras_model(
            qmodel, granularity="name", backend="Vitis", default_precision="fixed<16,6>"
        )
    hc["Model"]["Strategy"] = "Latency"
    hc["Model"]["ReuseFactor"] = 1
    layers = hc["LayerName"]
    # Fixed-point rings at the FPGA input (unsigned, all fractional bits; matches input_quant).
    # hls4ml parses QKeras quantized_bits(b, 0, keep_negative=False) as the *signed* fixed<b,1>,
    # which drops one LSB w.r.t. QKeras; force the exact unsigned type.
    layers["input_rings"]["Precision"] = {"result": f"ufixed<{cfg.input_bits},0>"}
    if "input_quant" in layers:
        layers["input_quant"]["Precision"] = {"result": f"ufixed<{cfg.input_bits},0,RND_CONV,SAT>"}
    # Exact (overflow-free) accumulator / pre-activation widths inferred by hls4ml
    for name in ("conv1d_1", "dense_1", "output_dense"):
        prec = layers[name].setdefault("Precision", {})
        prec["accum"] = "auto"
        prec["result"] = "auto"
    layers["conv1d_1"]["ParallelizationFactor"] = cfg.conv_pf
    return hc


def patch_conv_multiplier_limit(prj_dir: Path, pf: int) -> None:
    """hls4ml 1.3 workaround. conv_1d_latency_cl carries
        #pragma HLS ALLOCATION operation instances=mul limit=mult_config::multiplier_limit
    with multiplier_limit = K*C_in*C_out - n_zeros (24 - n_zeros here), i.e. NOT scaled by the
    ParallelizationFactor. With PF = 25, Vitis HLS must then share 24 multipliers among 600
    products and the II degrades (II = 9 for u8, II = 20 for u16 in our runs), which silently
    defeats the PF. Scale the limit by PF so that PF = 25 really means 600 parallel products."""
    params = prj_dir / "firmware" / "parameters.h"
    text = params.read_text()
    old = "multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;"
    blocks = text.split("struct ")
    n_patched = 0
    for i, blk in enumerate(blocks):
        if blk.startswith("config") and "_mult : nnet::dense_config" in blk.split("\n", 1)[0] and old in blk:
            blocks[i] = blk.replace(
                old, f"multiplier_limit = (DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor) * {pf};"
            )
            n_patched += 1
    if n_patched != 1:
        raise RuntimeError(f"expected exactly one Conv1D mult_config in {params}, patched {n_patched}")
    params.write_text("struct ".join(blocks))


def run_vivado_ooc(hdl_dir: Path, out_dir: Path, impl: bool):
    tcl = REPO / "scripts" / "vivado_synth_ooc.tcl"
    cmd = [
        "vivado", "-mode", "batch", "-nojournal", "-nolog", "-source", str(tcl),
        "-tclargs", str(hdl_dir), PROJECT, FPGA_PART, str(CLOCK_PERIOD_NS), str(out_dir), str(int(impl)),
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "vivado_stdout.txt", "w") as log:
        subprocess.run(cmd, cwd=out_dir, stdout=log, stderr=subprocess.STDOUT, check=True)
    res = {"post_synth": parse_vivado_util(out_dir / "post_synth_util.rpt"),
           "post_synth_wns_ns": parse_wns(out_dir / "post_synth_timing.rpt")}
    if impl:
        res["post_route"] = parse_vivado_util(out_dir / "post_route_util.rpt")
        res["post_route_wns_ns"] = parse_wns(out_dir / "post_route_timing.rpt")
    (out_dir / "vivado.json").write_text(json.dumps(res, indent=2))
    return res


def build_one(cfg, X_ho, y_ho, do_csynth: bool, do_vsynth: bool, impl: bool):
    res_dir = RES_ROOT / cfg.name
    res_dir.mkdir(parents=True, exist_ok=True)
    prj_dir = PRJ_ROOT / cfg.name
    if prj_dir.exists():
        shutil.rmtree(prj_dir)

    qmodel = load_config_model(cfg)
    hc = make_hls_config(qmodel, cfg)
    with keras_v2_dispatch():
        hm = hls4ml.converters.convert_from_keras_model(
            qmodel, hls_config=hc, output_dir=str(prj_dir), project_name=PROJECT, backend="Vitis",
            part=FPGA_PART, clock_period=CLOCK_PERIOD_NS, io_type="io_parallel",
        )
    (res_dir / "hls4ml_config.yaml").write_text(yaml.safe_dump(hc, sort_keys=False))

    # --- functional check: QKeras vs hls4ml C++ (compiled with g++) on the whole holdout set.
    # Rings are rounded to the input fixed-point grid upstream (QKeras quantizer, round-half-even);
    # the HLS float->ap_ufixed conversion in predict() truncates, so feed both the rounded values.
    hm.compile()
    X_ho = qmodel.get_layer("input_quant")(X_ho).numpy()
    y_q = qmodel.predict(X_ho, batch_size=4096, verbose=0).ravel()
    y_h = np.asarray(hm.predict(np.ascontiguousarray(X_ho, dtype=np.float32))).ravel()
    sp_q, _, _, thr_q = compute_sp_with_threshold(y_ho, y_q)
    sp_h = compute_sp_with_threshold(y_ho, y_h)[0]
    agree = float(np.mean((y_q >= thr_q) == (y_h >= thr_q)))
    emu = {
        "max_abs_diff": float(np.max(np.abs(y_q - y_h))),
        "mean_abs_diff": float(np.mean(np.abs(y_q - y_h))),
        "decision_agreement_at_qkeras_knee": agree,
        "sp_qkeras_holdout": sp_q,
        "sp_hls_holdout": sp_h,
    }
    (res_dir / "emulation.json").write_text(json.dumps(emu, indent=2))
    print(f"[{cfg.name}] emulation: {emu}")

    summary = {"config": cfg.name, "emulation": emu}
    if cfg.conv_pf > 1:
        patch_conv_multiplier_limit(prj_dir, cfg.conv_pf)  # build() reuses the written project
    if do_csynth:
        hm.build(csim=False, synth=True, export=False, vsynth=False, log_to_stdout=False)
        rpt = prj_dir / f"{PROJECT}_prj" / "solution1" / "syn" / "report"
        cs = parse_csynth_xml(rpt / "csynth.xml")
        (res_dir / "csynth.json").write_text(json.dumps(cs, indent=2))
        shutil.copy(rpt / "csynth.rpt", res_dir / "csynth.rpt")
        summary["csynth"] = cs
        print(f"[{cfg.name}] csynth: {cs['resources']} latency={cs['latency_worst_cycles']} II={cs['interval_max_cycles']}")
        if do_vsynth:
            hdl = prj_dir / f"{PROJECT}_prj" / "solution1" / "syn" / "verilog"
            viv = run_vivado_ooc(hdl, res_dir / "vivado", impl)
            summary["vivado"] = viv
            print(f"[{cfg.name}] vivado: {viv}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--no-csynth", action="store_true")
    ap.add_argument("--no-vsynth", action="store_true")
    ap.add_argument("--impl", action="store_true", help="also place & route (OOC)")
    args = ap.parse_args()

    if not args.no_csynth and shutil.which("vitis-run") is None:
        sys.exit("vitis-run not found on PATH: source scripts/env.sh first")
    ho = np.load(ART / "holdout.npz")
    X_ho, y_ho = ho["X"].astype(np.float32), ho["y"]
    for name in args.configs:
        build_one(CONFIGS[name], X_ho, y_ho, not args.no_csynth, not args.no_vsynth, args.impl)


if __name__ == "__main__":
    main()
