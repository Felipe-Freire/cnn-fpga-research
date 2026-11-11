"""Precision / parallelism configurations synthesized to validate the cnn-optimization cost model.

The three precision points are exactly the ones visited by the primal-dual loop of
cnn-optimization (commit f05e91a, notebooks/01_quantized_cnn1d_optimization.ipynb):

    iteration 1   -> u16 : w = [16, 16, 16], a = [16, 16]   (starting point)
    iteration 2   -> u8  : w = [ 8,  8,  8], a = [ 8,  8]   ("8-bit uniform" reference)
    iteration 3+  -> kkt : w = [ 2,  5,  8], a = [ 8,  8]   (reported "KKT optimum")

`input_bits` is the fixed-point width of the L1-normalized rings entering the FPGA. The QKeras
model in cnn-optimization has no input quantizer (float input); here it is made explicit so that
QKeras and the HLS C++ are bit-exact. It is set equal to the activation width the analytical
estimator assumes for the first layer (8 b for u8/kkt, 16 b for u16), so every multiplication in
hardware is exactly w_l x a_l bits, as the estimator assumes.

`conv_pf` is the hls4ml ParallelizationFactor of the Conv1D layer in io_parallel. With PF = 25 all
25 output positions are computed in parallel (600 physical multipliers, II = 1), which is what the
estimator implicitly assumes when it counts the 600 conv MACs as hardware. hls4ml's default is
PF = 1 (24 multipliers reused 25 times). `kkt_pf1` shows the impact of that choice.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

LAYERS = ("conv1d_1", "dense_1", "output_dense")

# v2 topology (cnn-optimization src/metrics.py CONFIG_VERSOES[2])
V2 = {"input_dim": 25, "kernel": 3, "filters": 8, "dense_units": 8}


@dataclass(frozen=True)
class QConfig:
    name: str
    w_bits: Dict[str, int]
    a_bits: Dict[str, int]
    input_bits: int
    conv_pf: int = 25
    model_key: Optional[str] = None  # trained model to reuse (defaults to `name`)
    description: str = ""
    frozen_weights: Optional[str] = None  # weights trained elsewhere, frozen here by scripts/import_weights.py

    @property
    def trained_model(self) -> str:
        return self.model_key or self.name

    @property
    def frozen_weights_path(self) -> Optional[Path]:
        return None if self.frozen_weights is None else REPO_DIR / self.frozen_weights


REPO_DIR = Path(__file__).resolve().parents[1]
# Sibling repository with the optimization study (override with the CNN_OPT_DIR environment variable)
CNN_OPT_DIR = Path(os.environ.get("CNN_OPT_DIR", REPO_DIR.parent / "cnn-optimization"))


CONFIGS: Dict[str, QConfig] = {
    "u16": QConfig(
        name="u16",
        w_bits={"conv1d_1": 16, "dense_1": 16, "output_dense": 16},
        a_bits={"conv1d_1": 16, "dense_1": 16},
        input_bits=16,
        description="Primal-dual iteration 1 (16-bit uniform)",
    ),
    "u8": QConfig(
        name="u8",
        w_bits={"conv1d_1": 8, "dense_1": 8, "output_dense": 8},
        a_bits={"conv1d_1": 8, "dense_1": 8},
        input_bits=8,
        description="Primal-dual iteration 2 (8-bit uniform reference)",
    ),
    "kkt": QConfig(
        name="kkt",
        w_bits={"conv1d_1": 2, "dense_1": 5, "output_dense": 8},
        a_bits={"conv1d_1": 8, "dense_1": 8},
        input_bits=8,
        description="Reported KKT optimum (w=[2,5,8], a=[8,8])",
    ),
    "kkt_pf1": QConfig(
        name="kkt_pf1",
        w_bits={"conv1d_1": 2, "dense_1": 5, "output_dense": 8},
        a_bits={"conv1d_1": 8, "dense_1": 8},
        input_bits=8,
        conv_pf=1,
        model_key="kkt",
        description="Same network as kkt, hls4ml default Conv1D ParallelizationFactor=1",
    ),
    # --- cnn-optimization, branch fix/activation-mapping (calibrated cost model + Lagrange/KKT + QAT loop).
    # Weights come from cnn-optimization and are frozen in results/<name>/ (provenance in source.json).
    "mp_v1": QConfig(
        name="mp_v1",
        w_bits={"conv1d_1": 3, "dense_1": 4, "output_dense": 16},
        a_bits={"conv1d_1": 8, "dense_1": 12},
        input_bits=8,
        frozen_weights="results/mp_v1/source_weights.weights.h5",
        description="1st final of fix/activation-mapping (cost model fitted on 3 points); calibration point",
    ),
    "mp_b10k": QConfig(
        name="mp_b10k",
        w_bits={"conv1d_1": 4, "dense_1": 6, "output_dense": 16},
        a_bits={"conv1d_1": 8, "dense_1": 12},
        input_bits=8,
        description="Exact optimum at B_LUT = 10,368 with the 3-point model, trained here; calibration point",
    ),
    "mp_final": QConfig(
        name="mp_final",
        w_bits={"conv1d_1": 4, "dense_1": 4, "output_dense": 16},
        a_bits={"conv1d_1": 8, "dense_1": 16},
        input_bits=8,
        frozen_weights="results/mp_final/source_weights.weights.h5",
        description="Final of fix/activation-mapping after recalibration (5 points) and CV selection; held out",
    ),
}

# Target device / clock
FPGA_PART = "xcvu13p-flga2577-2-e"   # AMD Virtex UltraScale+ VU13P (ATLAS GCM-class device)
CLOCK_PERIOD_NS = 4.1667             # 240 MHz = 6 x 40 MHz LHC bunch clock
