"""Load the QKeras model of a configuration (trained here, or weights trained in cnn-optimization)."""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

from pathlib import Path

from qkeras.utils import load_qmodel

from model.configs import V2, QConfig
from model.qcnn1d import build_compact_qcnn1d

ART = Path(__file__).resolve().parent / "artifacts"


def load_config_model(cfg: QConfig, artifacts_dir: Path = ART):
    """QKeras model with the configuration's quantizers and trained weights.

    Weights coming from cnn-optimization are frozen in this repository (scripts/import_weights.py) and loaded
    into a freshly built architecture: its exporter saves QKeras models in the ``.keras`` format, whose
    QActivation layers cannot be deserialized by ``tf_keras``/QKeras (TypeError in ``from_config``), but the
    weights inside the archive load fine.
    """
    if cfg.frozen_weights is not None:
        path = cfg.frozen_weights_path
        if not path.exists():
            raise FileNotFoundError(f"{path}: freeze the weights first with scripts/import_weights.py")
        model = build_compact_qcnn1d(V2["input_dim"], V2["filters"], V2["kernel"], V2["dense_units"],
                                     weights_bits=cfg.w_bits, act_bits=cfg.a_bits, input_bits=cfg.input_bits)
        model.load_weights(str(path))
        return model
    return load_qmodel(str(artifacts_dir / cfg.trained_model / "qmodel.h5"), compile=False)
