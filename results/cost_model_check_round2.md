# Calibrated cost model (cnn-optimization) vs. synthesis

Predictions frozen at 2026-11-10T11:46:43 (before the new syntheses), cost model fitted on u16, u8, kkt, mp_v1, mp_b10k. Measured = Vivado post-route (out-of-context, VU13P, 240 MHz).

| config | sample | DSP pred → real | LUT pred → real (err) | FF pred → real (err) |
|---|---|---|---|---|
| u16 | in-sample | 1,245 → **1,241** | 29,457 → **29,571** (-0.4 %) | 17,629 → **17,629** (+0.0 %) |
| u8 | in-sample | 0 → **0** | 23,402 → **23,309** (+0.4 %) | 8,903 → **9,002** (-1.1 %) |
| kkt | in-sample | 0 → **0** | 8,363 → **6,923** (+20.8 %) | 3,104 → **2,984** (+4.0 %) |
| kkt_pf1 | outside domain (PF=1) | 0 → **0** | 8,363 → **14,589** (-42.7 %) | 3,104 → **15,792** (-80.3 %) |
| mp_v1 | in-sample | 8 → **7** | 7,320 → **7,785** (-6.0 %) | 2,595 → **2,962** (-12.4 %) |
| mp_b10k | in-sample | 8 → **10** | 11,355 → **12,011** (-5.5 %) | 4,215 → **3,870** (+8.9 %) |
| mp_final | out-of-sample | 8 → **7** | 7,893 → **6,783** (+16.4 %) | 2,728 → **2,585** (+5.5 %) |
