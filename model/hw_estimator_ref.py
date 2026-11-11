"""ORIGINAL analytical VU13P cost model of cnn-optimization, reproduced verbatim for comparison.

Historical: it was replaced in cnn-optimization (branch fix/activation-mapping) by a structural model
calibrated on these syntheses; that model is checked by scripts/check_cost_model.py.

Source: cnn-optimization, commit f05e91a, src/hardware.py
    - estimate_hardware_resources()      (formulas copied unchanged)
    - compute_model_hardware_costs()     (per-layer activation-bit lookup reproduced unchanged:
                                          b_a = act_bits[layer.name], fallback 8)

Nothing here is "corrected" on purpose: the goal is to compare what the optimization loop
actually used against post-synthesis numbers.
"""

import math
from typing import Dict, List

from model.configs import LAYERS, V2, QConfig
from model.qcnn1d import layer_macs_params


def estimate_hardware_resources(model_config: List[Dict]) -> Dict[str, float]:
    total_dsp = total_lut = total_ff = total_bits = 0.0
    for layer in model_config:
        w_bits = float(layer["w_bits"])
        a_bits = float(layer["a_bits"])
        params = float(layer["params"])
        macc_ops = float(layer["macc"])
        if w_bits <= 8 and a_bits <= 8:
            total_lut += macc_ops * (w_bits * a_bits) / 18.0
        else:
            total_dsp += math.ceil(macc_ops / 64.0)
            total_lut += macc_ops * (w_bits + a_bits) / 36.0
        total_ff += params * w_bits + (macc_ops * a_bits) / 16.0
        total_bits += params * w_bits
    return {"DSP": total_dsp, "LUT": total_lut, "FF": total_ff, "BRAM_bits": total_bits}


def estimate_for_config(cfg: QConfig, default_act_bits: int = 8) -> Dict[str, object]:
    """Same per-layer bookkeeping as compute_model_hardware_costs (incl. its act-bit lookup by layer name)."""
    mp = layer_macs_params(V2["input_dim"], V2["filters"], V2["kernel"], V2["dense_units"])
    rows, breakdown = [], []
    for name in LAYERS:
        b_w = int(cfg.w_bits[name])
        b_a = int(cfg.a_bits.get(name, default_act_bits))  # NB: output activation of *this* layer, fallback 8
        row = {"w_bits": b_w, "a_bits": b_a, "params": mp[name]["params"], "macc": mp[name]["macs"]}
        rows.append(row)
        per_layer = estimate_hardware_resources([row])
        breakdown.append({"layer": name, **row, "bops": row["macc"] * b_w * b_a, **per_layer})
    total = estimate_hardware_resources(rows)
    total["BOPs"] = float(sum(b["bops"] for b in breakdown))
    return {"total": total, "breakdown": breakdown}
