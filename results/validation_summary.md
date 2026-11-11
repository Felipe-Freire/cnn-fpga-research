# Validation: analytical VU13P estimator vs. hls4ml synthesis

Device xcvu13p-flga2577-2-e, clock 240 MHz, hls4ml io_parallel / Latency / RF=1, Vitis HLS + Vivado 2026.1.1 (out-of-context). `est_*` = ORIGINAL cnn-optimization estimator (`estimate_hardware_resources`, commit f05e91a; replaced since, see cost_model_check.md), `hls_*` = Vitis HLS csynth estimate, `route_*` = Vivado post-place&route.

## Resources

| config | w (conv/dense/out) | a (in/conv/dense) | conv PF | DSP est → route | LUT est → HLS → route | FF est → HLS → route | BRAM est [bits] → route [tiles] |
|---|---|---|---|---|---|---|---|
| u16 | 16/16/16 | 16/16/16 | 25 | 23 → **1,241** | 1,221 → 92,810 → **29,571** | 14,444 → 31,992 → **17,629** | 13,072 → **0.5** |
| u8 | 8/8/8 | 8/8/8 | 25 | 0 → **0** | 4,892 → 70,715 → **23,309** | 7,224 → 10,561 → **9,002** | 6,536 → **0.5** |
| kkt | 2/5/8 | 8/8/8 | 25 | 0 → **0** | 2,268 → 25,946 → **6,923** | 4,704 → 3,032 → **2,984** | 4,016 → **0.5** |
| kkt_pf1 | 2/5/8 | 8/8/8 | 1 | 0 → **0** | 2,268 → 32,779 → **14,589** | 4,704 → 18,542 → **15,792** | 4,016 → **0.5** |
| mp_v1 | 3/4/16 | 8/8/12 | 25 | 13 → **7** | 1,147 → 28,316 → **7,785** | 4,224 → 3,214 → **2,962** | 3,344 → **0.5** |
| mp_b10k | 4/6/16 | 8/8/12 | 25 | 13 → **10** | 1,456 → 43,167 → **12,011** | 5,808 → 5,591 → **3,870** | 4,928 → **0.5** |
| mp_final | 4/4/16 | 8/8/16 | 25 | 13 → **7** | 1,499 → 26,425 → **6,783** | 4,448 → 3,693 → **2,585** | 3,376 → **0.5** |

## Ratio real (post-route) / estimated

| config | DSP | LUT | FF |
|---|---|---|---|
| u16 | 53.96× | 24.21× | 1.22× |
| u8 | 1.0× | 4.76× | 1.25× |
| kkt | 1.0× | 3.05× | 0.63× |
| kkt_pf1 | 1.0× | 6.43× | 3.36× |
| mp_v1 | 0.54× | 6.79× | 0.70× |
| mp_b10k | 0.77× | 8.25× | 0.67× |
| mp_final | 0.54× | 4.53× | 0.58× |

## Per layer (post-synthesis LUT / FF / DSP vs. estimator LUT)

`top/glue` holds logic Vivado inlined into the top: ReLU/pool/sigmoid and, in `kkt`, the whole
Conv1D (its 2-bit weights take only the values {0, 1}, so it reduces to wires and adders).

| config | layer | est. LUT | real LUT | real FF | real DSP |
|---|---|---|---|---|---|
| u16 | conv1d_1 | 533 | 8,354 | 1,097 | 472 |
| u16 | dense_1 | 683 | 16,071 | 6,493 | 758 |
| u16 | output_dense | 5 | 126 | 0 | 11 |
| u16 | relu | – | 200 | 0 | 0 |
| u16 | top/glue | – | 5,757 | 9,972 | 0 |
| u8 | conv1d_1 | 2,133 | 7,212 | 777 | 0 |
| u8 | dense_1 | 2,731 | 12,135 | 2,802 | 0 |
| u8 | output_dense | 28 | 185 | 45 | 0 |
| u8 | relu | – | 165 | 0 | 0 |
| u8 | top/glue | – | 4,536 | 5,377 | 0 |
| kkt | dense_1 | 1,707 | 3,827 | 941 | 0 |
| kkt | output_dense | 28 | 216 | 48 | 0 |
| kkt | pool | – | 288 | 0 | 0 |
| kkt | top/glue | – | 2,910 | 1,994 | 0 |
| kkt_pf1 | conv1d_1 | 533 | 936 | 2,176 | 0 |
| kkt_pf1 | dense_1 | 1,707 | 3,935 | 2,284 | 0 |
| kkt_pf1 | output_dense | 28 | 294 | 126 | 0 |
| kkt_pf1 | pool | – | 63 | 565 | 0 |
| kkt_pf1 | relu | – | 24 | 1,194 | 0 |
| kkt_pf1 | top/glue | – | 9,896 | 9,446 | 0 |
| mp_v1 | dense_1 | 341 | 3,276 | 632 | 0 |
| mp_v1 | output_dense | 5 | 138 | 55 | 7 |
| mp_v1 | relu | – | 120 | 0 | 0 |
| mp_v1 | top/glue | – | 4,559 | 2,276 | 0 |
| mp_b10k | dense_1 | 384 | 6,975 | 1,465 | 0 |
| mp_b10k | output_dense | 5 | 85 | 28 | 10 |
| mp_b10k | top/glue | – | 5,432 | 2,378 | 0 |
| mp_final | dense_1 | 427 | 2,145 | 356 | 0 |
| mp_final | output_dense | 5 | 113 | 31 | 7 |
| mp_final | pool | – | 343 | 0 | 0 |
| mp_final | relu | – | 99 | 0 | 0 |
| mp_final | top/glue | – | 4,326 | 2,197 | 0 |

## Timing and physics performance

| config | latency [cycles] | latency [ns] | II [cycles] | post-route WNS [ns] | SP holdout QKeras | SP holdout hls4ml C++ |
|---|---|---|---|---|---|---|
| u16 | 12 | 50.0 | 1 | 0.231 | 97.92 % | 97.91 % |
| u8 | 12 | 50.0 | 1 | 0.124 | 97.88 % | 97.87 % |
| kkt | 9 | 37.5 | 1 | 0.647 | 97.75 % | 97.75 % |
| kkt_pf1 | 41 | 170.8 | 25 | 0.526 | 97.75 % | 97.75 % |
| mp_v1 | 9 | 37.5 | 1 | 0.432 | 97.65 % | 97.65 % |
| mp_b10k | 9 | 37.5 | 1 | 0.648 | 97.87 % | 97.86 % |
| mp_final | 9 | 37.5 | 1 | 0.949 | 97.64 % | 97.64 % |
