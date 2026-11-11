"""Compact CNN1D (FP32 and QKeras) mirrored from cnn-optimization src/models.py (commit f05e91a).

Architecture (v2, 25 rings):
    Input(25) -> Reshape(25,1) -> Conv1D(8, k=3, same) -> ReLU -> MaxPool1D(2) -> Flatten(96)
              -> Dense(8) -> ReLU -> Dense(1, sigmoid)
    params = 32 + 776 + 9 = 817 ; MACs = 600 + 768 + 8 = 1376

The quantizer conventions are kept identical to the original (integer bits = min(2, b - 1),
alpha = 1, bias quantized with the weight quantizer). The only addition is an optional explicit
input quantizer (`input_quant`), required to fix the precision of the rings entering the FPGA.
"""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

from typing import Dict, Optional

import numpy as np
import tf_keras as keras
from qkeras import QActivation, QConv1D, QDense, quantized_bits, quantized_relu
from tf_keras.callbacks import Callback

from model.ringer_data import compute_sp_with_threshold


def build_compact_cnn1d_fp32(input_dim=25, num_filters=8, kernel_size=3, dense_units=8, learning_rate=1e-3):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,), name="input_rings"),
        keras.layers.Reshape((input_dim, 1), name="reshape_1d"),
        keras.layers.Conv1D(num_filters, kernel_size, padding="same", activation="relu", name="conv1d_1"),
        keras.layers.MaxPooling1D(pool_size=2, name="maxpool_1"),
        keras.layers.Flatten(name="flatten_1"),
        keras.layers.Dense(dense_units, activation="relu", name="dense_1"),
        keras.layers.Dense(1, activation="sigmoid", name="output_dense"),
    ], name="Compact_CNN1D_FP32")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def _int_bits(b: int) -> int:
    return min(2, max(0, b - 1))


def build_compact_qcnn1d(
    input_dim: int = 25,
    num_filters: int = 8,
    kernel_size: int = 3,
    dense_units: int = 8,
    weights_bits: Optional[Dict[str, int]] = None,
    act_bits: Optional[Dict[str, int]] = None,
    input_bits: Optional[int] = None,
    learning_rate: float = 1e-3,
):
    weights_bits = weights_bits or {}
    act_bits = act_bits or {}
    b_conv = int(weights_bits.get("conv1d_1", 8))
    b_dense = int(weights_bits.get("dense_1", 8))
    b_out = int(weights_bits.get("output_dense", 8))
    a_conv = int(act_bits.get("conv1d_1", 8))
    a_dense = int(act_bits.get("dense_1", 8))

    layers = [keras.layers.Input(shape=(input_dim,), name="input_rings")]
    if input_bits is not None:
        # Normalized rings live in [0, 0.74] for this dataset -> unsigned, 0 integer bits.
        layers.append(QActivation(quantized_bits(input_bits, 0, keep_negative=False, alpha=1), name="input_quant"))
    layers += [
        keras.layers.Reshape((input_dim, 1), name="reshape_1d"),
        QConv1D(
            filters=num_filters,
            kernel_size=kernel_size,
            padding="same",
            kernel_quantizer=quantized_bits(b_conv, _int_bits(b_conv), alpha=1),
            bias_quantizer=quantized_bits(b_conv, _int_bits(b_conv), alpha=1),
            name="conv1d_1",
        ),
        QActivation(quantized_relu(a_conv, _int_bits(a_conv)), name="relu_conv1"),
        keras.layers.MaxPooling1D(pool_size=2, name="maxpool_1"),
        keras.layers.Flatten(name="flatten_1"),
        QDense(
            dense_units,
            kernel_quantizer=quantized_bits(b_dense, _int_bits(b_dense), alpha=1),
            bias_quantizer=quantized_bits(b_dense, _int_bits(b_dense), alpha=1),
            name="dense_1",
        ),
        QActivation(quantized_relu(a_dense, _int_bits(a_dense)), name="relu_dense1"),
        QDense(
            1,
            kernel_quantizer=quantized_bits(b_out, _int_bits(b_out), alpha=1),
            bias_quantizer=quantized_bits(b_out, _int_bits(b_out), alpha=1),
            activation="sigmoid",
            name="output_dense",
        ),
    ]
    model = keras.Sequential(layers, name="Compact_QCNN1D")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def transfer_weights_to_qmodel(source_model, target_qmodel) -> None:
    """Warm-start: copy FP32 weights into same-named QKeras layers (as in cnn-optimization)."""
    src = {l.name: l for l in source_model.layers}
    for t in target_qmodel.layers:
        if t.name in src:
            sw, tw = src[t.name].get_weights(), t.get_weights()
            if len(sw) == len(tw) and all(a.shape == b.shape for a, b in zip(sw, tw)):
                t.set_weights(sw)


class SPCallback(Callback):
    """Tracks validation SP each epoch, keeps best weights, early-stops on SP (as in cnn-optimization)."""

    def __init__(self, val_data, patience=15):
        super().__init__()
        self.val_x, self.val_y = val_data
        self.patience = patience
        self.best_sp, self.best_epoch, self.best_weights, self.best_threshold, self.wait = -1.0, 0, None, 0.5, 0

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.val_x, batch_size=2048, verbose=0).ravel()
        sp, _, _, thr = compute_sp_with_threshold(self.val_y, y_pred)
        if logs is not None:
            logs["val_sp"] = sp
        if sp > self.best_sp:
            self.best_sp, self.best_epoch, self.best_threshold, self.wait = sp, epoch, thr, 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True

    def restore_best_weights(self):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)


def layer_macs_params(input_dim=25, num_filters=8, kernel_size=3, dense_units=8):
    """Analytic MACs / params per layer for the compact CNN1D ('same' padding, pool 2)."""
    l_out = input_dim
    flat = (l_out // 2) * num_filters
    return {
        "conv1d_1": {"macs": l_out * kernel_size * 1 * num_filters, "params": kernel_size * num_filters + num_filters},
        "dense_1": {"macs": flat * dense_units, "params": flat * dense_units + dense_units},
        "output_dense": {"macs": dense_units, "params": dense_units + 1},
    }


def quantized_weight_stats(qmodel) -> Dict[str, Dict[str, float]]:
    """Fraction of zero / power-of-two quantized weights per layer (they cost no multiplier in HLS)."""
    out = {}
    for layer in qmodel.layers:
        if hasattr(layer, "kernel_quantizer_internal") and layer.kernel_quantizer_internal is not None:
            wq = np.asarray(layer.kernel_quantizer_internal(layer.kernel)).ravel()
            nz = wq[wq != 0]
            pow2 = np.isclose(np.log2(np.abs(nz)), np.round(np.log2(np.abs(nz)))) if nz.size else np.array([])
            out[layer.name] = {
                "n": int(wq.size),
                "zero_frac": float(np.mean(wq == 0)),
                "pow2_frac_of_nonzero": float(np.mean(pow2)) if nz.size else 0.0,
                "distinct_values": int(np.unique(wq).size),
            }
    return out
