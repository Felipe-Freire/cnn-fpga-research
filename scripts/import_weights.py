"""Freeze externally trained weights inside this repository, with provenance.

cnn-optimization overwrites its exported models on every run, so a configuration must never point at them
directly. This script copies (or reconstructs) the weights into results/<config>/source_weights.weights.h5
and writes results/<config>/source.json (origin, sha256, date). model/loading.py loads that frozen copy.

    # from a model exported by cnn-optimization (.keras archive or .weights.h5)
    python scripts/import_weights.py --config mp_final --from ../cnn-optimization/results/.../best_model.keras
    # from the integer weight codes stored in a calibration file (e.g. a point already synthesized)
    python scripts/import_weights.py --config mp_v1 --from-calibration <json> --point mp_final
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from model.configs import CONFIGS, LAYERS, V2  # noqa: E402
from model.qcnn1d import build_compact_qcnn1d  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from", dest="source")
    src.add_argument("--from-calibration", dest="calibration")
    ap.add_argument("--point", help="point name inside the calibration file")
    args = ap.parse_args()

    cfg = CONFIGS[args.config]
    if cfg.frozen_weights is None:
        sys.exit(f"config {cfg.name} does not use frozen external weights")
    model = build_compact_qcnn1d(V2["input_dim"], V2["filters"], V2["kernel"], V2["dense_units"],
                                 weights_bits=cfg.w_bits, act_bits=cfg.a_bits, input_bits=cfg.input_bits)
    if args.source:
        source = Path(args.source).resolve()
        model.load_weights(str(source))  # works for .keras archives whose config QKeras cannot deserialize
        provenance = {"origin": str(source), "origin_sha256": _sha256(source)}
    else:
        points = {p["name"]: p for p in json.loads(Path(args.calibration).read_text())["points"]}
        point = points[args.point]
        for name in LAYERS:
            lay = point["layers"][name]
            bits = lay["w_bits"]
            assert bits == cfg.w_bits[name], f"{name}: {bits} bits in calibration vs {cfg.w_bits[name]} in config"
            lsb = 2.0 ** (min(2, max(0, bits - 1)) + 1 - bits)
            layer = model.get_layer(name)
            k = (np.asarray(lay["kernel_int"], dtype=np.float32) * lsb).reshape(layer.kernel.shape)
            b = np.asarray(lay["bias_int"], dtype=np.float32) * lsb
            layer.set_weights([k, b])
        provenance = {"origin": f"integer weight codes of point '{args.point}' in {Path(args.calibration).resolve()}",
                      "note": "exact: the codes are the quantized weights that were synthesized"}
    out = cfg.frozen_weights_path
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(out))
    provenance.update({"config": cfg.name, "w_bits": cfg.w_bits, "a_bits": cfg.a_bits, "input_bits": cfg.input_bits,
                       "frozen_file": str(out.relative_to(REPO)), "frozen_sha256": _sha256(out),
                       "imported": datetime.now().isoformat(timespec="seconds")})
    (out.parent / "source.json").write_text(json.dumps(provenance, indent=2))
    print(f"[{cfg.name}] weights frozen in {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
