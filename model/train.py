"""Train the FP32 reference and the QKeras (QAT, warm-start) models for each precision config.

Recipe mirrors cnn-optimization notebook 01 (FP32: Adam 1.5e-3, 18 epochs, batch 512, SP early
stopping; QAT: warm-start from FP32, Adam 1e-3, 18 epochs, ReduceLROnPlateau). A single
train/validation split is used instead of 5-fold CV: here the networks only need realistic
trained weights for synthesis, not a CV estimate of SP.

Usage:
    python model/train.py --data /path/to/mc21_13p6TeV.Zee.JF17.2sigma.5M.et5_eta0.h5 [--configs u16 u8 kkt]
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import tf_keras as keras  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from model.configs import CONFIGS, V2  # noqa: E402
from model.qcnn1d import (  # noqa: E402
    SPCallback,
    build_compact_cnn1d_fp32,
    build_compact_qcnn1d,
    quantized_weight_stats,
    transfer_weights_to_qmodel,
)
from model.ringer_data import compute_sp_with_threshold, load_ringer_v2, sp_at_threshold  # noqa: E402

ART = REPO / "model" / "artifacts"
SEED = 42


def evaluate(model, X_val, y_val, X_ho, y_ho):
    p_val = model.predict(X_val, batch_size=4096, verbose=0).ravel()
    sp_v, pd_v, fa_v, thr = compute_sp_with_threshold(y_val, p_val)
    p_ho = model.predict(X_ho, batch_size=4096, verbose=0).ravel()
    sp_h, pd_h, fa_h = sp_at_threshold(y_ho, p_ho, thr)
    sp_knee_h = compute_sp_with_threshold(y_ho, p_ho)[0]
    return {
        "val": {"sp": sp_v, "pd": pd_v, "fa": fa_v, "threshold": thr},
        "holdout_at_val_threshold": {"sp": sp_h, "pd": pd_h, "fa": fa_h},
        "holdout_sp_knee": sp_knee_h,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Ringer HDF5 file (et5_eta0 in the reference study)")
    ap.add_argument("--configs", nargs="+", default=["u16", "u8", "kkt"])
    ap.add_argument("--epochs", type=int, default=18)
    args = ap.parse_args()

    keras.utils.set_random_seed(SEED)
    ART.mkdir(parents=True, exist_ok=True)

    X_dev, X_ho, y_dev, y_ho = load_ringer_v2(args.data, seed=SEED)
    X_tr, X_val, y_tr, y_val = train_test_split(X_dev, y_dev, test_size=0.20, stratify=y_dev, random_state=SEED)
    np.savez_compressed(ART / "holdout.npz", X=X_ho, y=y_ho)
    print(f"train {X_tr.shape}  val {X_val.shape}  holdout {X_ho.shape}")

    fp32_path = ART / "fp32.h5"
    fp32 = build_compact_cnn1d_fp32(V2["input_dim"], V2["filters"], V2["kernel"], V2["dense_units"], learning_rate=1.5e-3)
    if fp32_path.exists():
        fp32.load_weights(str(fp32_path))
    else:
        cb = SPCallback((X_val, y_val), patience=8)
        lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)
        fp32.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=args.epochs, batch_size=512, callbacks=[cb, lr], verbose=2)
        cb.restore_best_weights()
        fp32.save_weights(str(fp32_path))
    fp32_metrics = evaluate(fp32, X_val, y_val, X_ho, y_ho)
    (ART / "fp32_metrics.json").write_text(json.dumps(fp32_metrics, indent=2))
    print("FP32:", json.dumps(fp32_metrics))

    for key in dict.fromkeys(CONFIGS[c].trained_model for c in args.configs if CONFIGS[c].frozen_weights is None):
        cfg = CONFIGS[key]
        out = ART / key
        out.mkdir(exist_ok=True)
        keras.utils.set_random_seed(SEED)
        qm = build_compact_qcnn1d(
            V2["input_dim"], V2["filters"], V2["kernel"], V2["dense_units"],
            weights_bits=cfg.w_bits, act_bits=cfg.a_bits, input_bits=cfg.input_bits, learning_rate=1e-3,
        )
        transfer_weights_to_qmodel(fp32, qm)
        cb = SPCallback((X_val, y_val), patience=8)
        lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)
        qm.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=args.epochs, batch_size=512, callbacks=[cb, lr], verbose=2)
        cb.restore_best_weights()
        qm.save(str(out / "qmodel.h5"))
        m = evaluate(qm, X_val, y_val, X_ho, y_ho)
        m["weight_stats"] = quantized_weight_stats(qm)
        m["config"] = {"w_bits": cfg.w_bits, "a_bits": cfg.a_bits, "input_bits": cfg.input_bits}
        (out / "metrics.json").write_text(json.dumps(m, indent=2))
        print(f"[{key}]", json.dumps({k: v for k, v in m.items() if k != "weight_stats"}))
        print(f"[{key}] weight stats:", json.dumps(m["weight_stats"]))


if __name__ == "__main__":
    main()
