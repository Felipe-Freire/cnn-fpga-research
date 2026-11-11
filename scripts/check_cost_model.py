"""Out-of-sample check of cnn-optimization's calibrated VU13P cost model against synthesis.

    python scripts/check_cost_model.py predict --tag round2   # BEFORE synthesizing: freeze predictions
    python scripts/check_cost_model.py compare --tag round2   # after build_hls.py --impl: predicted vs post-route

`predict` loads every configuration's QKeras model, builds the cost model from the calibration file
currently in cnn-optimization (src/calibration/vu13p_hls4ml_synthesis.json) and writes
results/cost_model_predictions_<tag>.json (refuses to overwrite) with a timestamp and the list of points used in the fit.
`compare` reads that frozen file and the Vivado reports and writes results/cost_model_check_<tag>.md.
Configurations that were part of the fit are reported as in-sample, the rest as out-of-sample.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from model.configs import CNN_OPT_DIR, CONFIGS  # noqa: E402

sys.path.insert(1, str(CNN_OPT_DIR))
from src.hardware import CostModel, CALIBRATION_PATH, layer_specs_from_model  # noqa: E402  (cnn-optimization)

from model.loading import load_config_model  # noqa: E402

RES = REPO / "results"
RESOURCES = ("DSP", "LUT", "FF")


def predict(names, tag):
    cm = CostModel.from_calibration(CALIBRATION_PATH)
    fitted = [c["name"] for c in cm.calibration if c["in_domain"]]
    out = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "calibration_file": str(CALIBRATION_PATH),
        "fitted_on": fitted,
        "coefficients": {"beta_lut": cm.beta_lut, "gamma_lut": cm.gamma_lut, "beta_ff": cm.beta_ff, "delta_ff": cm.delta_ff},
        "predictions": {},
    }
    for name in names:
        cfg = CONFIGS[name]
        qm = load_config_model(cfg)
        specs = layer_specs_from_model(qm, cfg.w_bits, cfg.a_bits, input_bits=cfg.input_bits)
        est = cm.estimate(specs)
        out["predictions"][name] = {
            "in_domain": cfg.conv_pf > 1,
            "in_sample": name in fitted,
            **{r: float(est[r]) for r in RESOURCES},
            "per_layer": {s.name: {k: float(v) for k, v in cm.layer_resources(s).items()} for s in specs},
        }
        print(f"[{name}] predicted {', '.join(f'{r}={est[r]:,.0f}' for r in RESOURCES)}"
              f"{'' if cfg.conv_pf > 1 else '  (outside calibrated domain: PF=1)'}")
    pred_file = RES / f"cost_model_predictions_{tag}.json"
    if pred_file.exists():
        sys.exit(f"{pred_file} already exists: predictions are frozen once per round (use a new --tag)")
    pred_file.write_text(json.dumps(out, indent=2))
    print(f"frozen predictions -> {pred_file}")


def compare(tag):
    pred = json.loads((RES / f"cost_model_predictions_{tag}.json").read_text())
    lines = [
        "# Calibrated cost model (cnn-optimization) vs. synthesis",
        "",
        f"Predictions frozen at {pred['created']} (before the new syntheses), cost model fitted on "
        f"{', '.join(pred['fitted_on'])}. Measured = Vivado post-route (out-of-context, VU13P, 240 MHz).",
        "",
        "| config | sample | DSP pred → real | LUT pred → real (err) | FF pred → real (err) |",
        "|---|---|---|---|---|",
    ]
    for name, p in pred["predictions"].items():
        viv = RES / name / "vivado" / "vivado.json"
        if not viv.exists():
            continue
        m = json.loads(viv.read_text())["post_route"]
        sample = "in-sample" if p["in_sample"] else ("out-of-sample" if p["in_domain"] else "outside domain (PF=1)")

        def cell(r):
            err = (p[r] - m[r]) / m[r] * 100 if m[r] else 0.0
            return f"{p[r]:,.0f} → **{m[r]:,.0f}** ({err:+.1f} %)"

        lines.append(f"| {name} | {sample} | {p['DSP']:,.0f} → **{m['DSP']:,.0f}** | {cell('LUT')} | {cell('FF')} |")
    out = RES / f"cost_model_check_{tag}.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["predict", "compare"])
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--tag", required=True, help="calibration round, e.g. round2")
    args = ap.parse_args()
    predict(args.configs, args.tag) if args.stage == "predict" else compare(args.tag)


if __name__ == "__main__":
    main()
