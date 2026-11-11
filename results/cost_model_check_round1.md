# Calibrated cost model (cnn-optimization) vs. synthesis

Predictions frozen at 2026-11-10T11:25:00 (before the new syntheses), cost model fitted on u16, u8, kkt. Measured = Vivado post-route (out-of-context, VU13P, 240 MHz).

| config | sample | DSP pred → real | LUT pred → real (err) | FF pred → real (err) |
|---|---|---|---|---|
| u16 | in-sample | 1,245 → **1,241** | 27,668 → **29,571** (-6.4 %) | 17,629 → **17,629** (+0.0 %) |
| u8 | in-sample | 0 → **0** | 25,398 → **23,309** (+9.0 %) | 9,141 → **9,002** (+1.5 %) |
| kkt | in-sample | 0 → **0** | 6,865 → **6,923** (-0.8 %) | 2,471 → **2,984** (-17.2 %) |
| kkt_pf1 | outside domain (PF=1) | 0 → **0** | 6,865 → **14,589** (-52.9 %) | 2,471 → **15,792** (-84.4 %) |
| mp_v1 | out-of-sample | 8 → **7** | 5,072 → **7,785** (-34.8 %) | 1,875 → **2,962** (-36.7 %) |
| mp_b10k | out-of-sample | 8 → **10** | 10,251 → **12,011** (-14.7 %) | 3,739 → **3,870** (-3.4 %) |
