# cnn-fpga-research

Validação em hardware (hls4ml → Vitis HLS → Vivado, AMD Virtex UltraScale+ **VU13P**) da CNN1D
compacta do estudo de otimização [`cnn-optimization`](../cnn-optimization) (ATLAS L0 Ringer, `v2` = 25 anéis).

Perguntas que este repositório responde, sempre contra síntese + place & route reais:

1. o estimador **original** de `cnn-optimization` (commit `f05e91a`) correspondia à realidade? (não: 3–54×);
2. o modelo de custo **calibrado** que o substituiu (branch `fix/activation-mapping`) acerta configurações
   que não viu? (`scripts/check_cost_model.py`, previsões congeladas antes de cada síntese);
3. as configurações escolhidas pela otimização cabem e fecham timing na VU13P?

Os pontos sintetizados alimentam a calibração do `cnn-optimization` (`scripts/export_calibration.py`).

## Estrutura

```
cnn-fpga-research/
├── model/                 # Rede base (espelho de cnn-optimization@f05e91a), dados Ringer, treino QAT
│   ├── configs.py         # Configurações sintetizadas, part e clock
│   ├── loading.py         # Carrega o modelo de cada configuração (treinado aqui ou pesos congelados)
│   ├── qcnn1d.py          # CNN1D FP32/QKeras idêntica à original (+ quantizador explícito de entrada)
│   ├── ringer_data.py     # Fatiamento v2, norm1, índice SP (cópia de src/metrics.py)
│   ├── hw_estimator_ref.py# Estimador analítico original, copiado sem correções (para comparação)
│   └── train.py           # FP32 + QAT warm-start (mesma receita do notebook 01)
├── hls4ml/
│   └── build_hls.py       # QKeras → hls4ml (Vitis), emulação C++ bit-a-bit, csynth, Vivado OOC
├── scripts/
│   ├── env.sh             # Vitis/Vivado no PATH + venv
│   ├── vivado_synth_ooc.tcl  # Síntese + place&route out-of-context com restrição de clock
│   ├── reports.py         # Parsers de csynth.xml e relatórios do Vivado
│   ├── compare.py         # Estimador original × HLS × Vivado → results/validation_summary.{md,csv,png}
│   ├── check_cost_model.py   # Modelo calibrado (cnn-optimization) × Vivado, previsões congeladas por rodada
│   ├── import_weights.py  # Congela pesos treinados no cnn-optimization em results/<cfg>/ (com procedência)
│   └── export_calibration.py # Pontos sintetizados → cnn-optimization/src/calibration/*.json
├── results/               # Relatórios versionados (csynth, utilização, timing, emulação, resumo)
├── hls/  rtl/  verif/     # (reservado: HLS manual, RTL direto, testbenches)
```

## Rede base

```
Input(25) → [input_quant: ufixed<b_in,0>] → Reshape(25,1) → QConv1D(8, k=3, same) → QReLU
→ MaxPool1D(2) → Flatten(96) → QDense(8) → QReLU → QDense(1) → sigmoid
817 parâmetros, 1376 MACs (600 conv + 768 dense + 8 saída)
```

Quantizadores idênticos aos de `cnn-optimization/src/models.py` (`integer = min(2, b−1)`, `alpha=1`).
A única adição é `input_quant`: a rede original recebe anéis em float, mas no FPGA a entrada
precisa de um formato de ponto fixo. Usamos `b_in` igual à largura de ativação que o estimador
assume na 1ª camada (8 b para u8/kkt, 16 b para u16), de modo que cada multiplicação no hardware
é exatamente `w_l × a_l` bits — a comparação mais favorável possível ao estimador.

## Configurações

| nome | w (conv/dense/out) | a (in/conv/dense) | Conv1D PF | origem |
|---|---|---|---|---|
| `u16` | 16/16/16 | 16/16/16 | 25 | iteração 1 do laço primal-dual |
| `u8` | 8/8/8 | 8/8/8 | 25 | iteração 2 ("8 bits uniforme") |
| `kkt` | 2/5/8 | 8/8/8 | 25 | "ótimo KKT" reportado |
| `kkt_pf1` | 2/5/8 | 8/8/8 | 1 | mesma rede, PF padrão do hls4ml |
| `mp_v1` | 3/4/16 | 8/8/12 | 25 | 1ª final de `fix/activation-mapping` (pesos congelados) |
| `mp_b10k` | 4/6/16 | 8/8/12 | 25 | ótimo exato em B_LUT = 10.368 com o modelo de 3 pontos (treinado aqui) |
| `mp_final` | 4/4/16 | 8/8/16 | 25 | final após recalibração + seleção por CV (pesos congelados) — **fora da calibração** |

`u16`, `u8`, `kkt`, `mp_v1` e `mp_b10k` calibram o modelo de custo (`kkt_pf1` fica como evidência fora do
domínio). `mp_final` é mantida fora do ajuste para testar o modelo recalibrado.

hls4ml `io_parallel`, `Strategy=Latency`, `ReuseFactor=1`, acumuladores com precisão `auto`
(sem overflow). Part `xcvu13p-flga2577-2-e`, clock 240 MHz (6 × 40 MHz).
`PF` = `ParallelizationFactor` da Conv1D: 25 → as 25 posições de saída em paralelo (600
multiplicações, o que o estimador assume); 1 → 24 multiplicadores reaproveitados 25 vezes.

## Como rodar

```bash
python3.12 -m venv .venv
.venv/bin/pip install "tensorflow==2.20.0" "tf_keras==2.20.*" "qkeras==0.9.0" "hls4ml==1.3.0" \
    "quantizers<2" pandas tables scikit-learn matplotlib pyyaml
source scripts/env.sh                         # Vitis/Vivado 2026.1.1 + venv

python model/train.py --data <dir>/mc21_13p6TeV.Zee.JF17.2sigma.5M.et5_eta0.h5
python hls4ml/build_hls.py --configs u16 u8 kkt kkt_pf1 --impl
python scripts/compare.py

# pesos treinados no cnn-optimization: congelar antes de sintetizar (ele sobrescreve best_model.keras a cada execução)
python scripts/import_weights.py --config mp_final --from ../cnn-optimization/results/<...>/saved_models/best_model.keras

# teste do modelo calibrado: congelar previsões ANTES de sintetizar, comparar depois
python scripts/check_cost_model.py predict --tag round3
python hls4ml/build_hls.py --configs <nova_cfg> --impl
python scripts/check_cost_model.py compare --tag round3

# realimentar a calibração do cnn-optimization (lista explícita: o que fica de fora é validação)
python scripts/export_calibration.py --out ../cnn-optimization/src/calibration/vu13p_hls4ml_synthesis.json \
    --configs u16 u8 kkt kkt_pf1 mp_v1 mp_b10k
```

## Notas de compatibilidade (hls4ml 1.3 + TF 2.20)

* **Despacho Keras 2/3**: o hls4ml escolhe o frontend pela versão do pacote `keras` (3.x com TF
  2.20); modelos QKeras são `tf_keras`. `build_hls.py::keras_v2_dispatch` força o frontend Keras-2.
* **`quantizers<2`**: o `quantizers 2.0` removeu `get_fixed_quantizer_np`, usado pelo hls4ml 1.3.
* **Entrada unsigned**: o hls4ml traduz `quantized_bits(b, 0, keep_negative=False)` como
  `fixed<b,1>` com sinal (perde 1 LSB). Forçamos `ufixed<b,0>`; com isso a emulação C++ dá o
  mesmo SP do QKeras (resíduo ≤ 0,005 só na tabela da sigmoide).
* **Modelos `.keras` do cnn-optimization**: o QKeras não desserializa `QActivation` desse formato
  (`TypeError` em `from_config`); só os pesos são lidos (`load_weights`). Por isso `import_weights.py`
  reconstrói a arquitetura e congela os pesos. O exportador do cnn-optimization agora também grava `.h5`.
* **`multiplier_limit` da Conv1D**: o template não escala o limite de multiplicadores com o
  `ParallelizationFactor`; com PF=25 o Vitis HLS compartilhava 24 multiplicadores entre 600
  produtos (II = 9 em u8, 20 em u16). `patch_conv_multiplier_limit` multiplica o limite por PF.

## Resultados (VU13P, 240 MHz, pós place&route OOC)

| config | DSP est → real | LUT est → real | FF est → real | BRAM est → real | latência / II | SP holdout (hls4ml C++) |
|---|---|---|---|---|---|---|
| u16 | 23 → **1.241** (54×) | 1.221 → **29.571** (24×) | 14.444 → **17.629** | 13.072 bits → **1 RAMB18** | 12 ciclos (50 ns) / 1 | 97,91 % |
| u8 | 0 → **0** | 4.892 → **23.309** (4,8×) | 7.224 → **9.002** | 6.536 bits → **1 RAMB18** | 12 ciclos (50 ns) / 1 | 97,87 % |
| kkt | 0 → **0** | 2.268 → **6.923** (3,1×) | 4.704 → **2.984** | 4.016 bits → **1 RAMB18** | 9 ciclos (37,5 ns) / 1 | 97,75 % |
| kkt_pf1 | 0 → **0** | 2.268 → **14.589** (6,4×) | 4.704 → **15.792** | 4.016 bits → **1 RAMB18** | 41 ciclos (171 ns) / 25 | 97,75 % |
| mp_v1 | 13 → **7** | 1.147 → **7.785** (6,8×) | 4.224 → **2.962** | 3.344 bits → **1 RAMB18** | 9 ciclos (37,5 ns) / 1 | 97,65 % |
| mp_b10k | 13 → **10** | 1.456 → **12.011** (8,2×) | 5.808 → **3.870** | 4.928 bits → **1 RAMB18** | 9 ciclos (37,5 ns) / 1 | 97,85 % |
| **mp_final** | 13 → **7** | 1.499 → **6.783** (4,5×) | 4.448 → **2.585** | 3.376 bits → **1 RAMB18** | 9 ciclos (37,5 ns) / 1 | 97,64 % |

"est" nesta tabela é sempre o estimador **original** (histórico). O modelo calibrado que o substituiu é
avaliado à parte, abaixo.

Todas fecham timing a 240 MHz (WNS > 0). Detalhes por camada e tabelas completas em
[`results/validation_summary.md`](results/validation_summary.md); gráfico em
`results/validation_resources.png`.

Leitura:

* **DSP**: o modelo `⌈MAC/64⌉` subestima em 54×. Com `ReuseFactor=1` cada multiplicação de 16 b com
  peso não trivial ocupa ~1 DSP48E2 (conv 472 de 600, dense_1 758 de 768, saída 11).
* **LUT**: `MAC·w·a/18` subestima 3–24×. O modelo conta só multiplicadores; na prática dominam as
  árvores de soma/acumuladores (larguras `auto` de 13–24 b), ReLU, pooling e registradores de pipeline.
* **BRAM**: em `io_parallel` os pesos são constantes embutidas na lógica; o único bloco de RAM é a
  tabela da sigmoide (1 RAMB18), independente de `w_l`. A restrição `BRAM_bits = P·w` não existe fisicamente.
* **Valores dos pesos importam**: no `kkt` os 24 pesos da Conv1D valem só {0, 1} e o Vivado reduz a camada
  a fios e somadores. O estimador não vê zeros nem potências de 2.
* **Paralelismo (PF/RF) importa tanto quanto bits**: com os mesmos bits, `kkt_pf1` gasta 2,1× LUT e
  5,3× FF a mais que `kkt` (buffers do laço de 25 iterações), além de II = 25.
* **Direção certa, magnitude errada**: `kkt` vs `u8` economiza 70 % de LUT na realidade (estimado: 54 %),
  mas `kkt` real (6.923 LUT) **viola** o orçamento de 2.500 LUT imposto no notebook, e a ordem
  estimada u16 < kkt < u8 em LUT não se confirma (real: kkt < u8 < u16).

## Modelo de custo calibrado (cnn-optimization) contra a síntese

Previsões congeladas **antes** de cada síntese (`scripts/check_cost_model.py`, pós-rota):

| rodada | modelo | config (fora da amostra) | LUT previsto → real | FF previsto → real | DSP |
|---|---|---|---|---|---|
| 1 | somadores, 3 pontos (u16, u8, kkt) | mp_v1 | 5.072 → 7.785 (−35 %) | 1.875 → 2.962 (−37 %) | 8 → 7 |
| 1 | idem | mp_b10k | 10.251 → 12.011 (−15 %) | 3.739 → 3.870 (−3 %) | 8 → 10 |
| 2 | somadores + bits de ReLU, 5 pontos | **mp_final** | 7.893 → 6.783 (**+16 %**) | 2.728 → 2.585 (+6 %) | 8 → 7 |

A rodada 1 mostrou que a lógica de ReLU + pooling (~3–5 mil LUT) não encolhe com a rede e era ignorada.
Com o termo de ReLU e os dois pontos novos na calibração, a previsão da configuração final (mantida fora do
ajuste) errou para o lado conservador. Detalhes em `results/cost_model_check_round{1,2}.md`.

**Configuração final da otimização (`mp_final`, w = 4/4/16, a = 8/8/16):** 6.783 LUT, 7 DSP, 2.585 FF,
1 RAMB18; WNS +0,95 ns a 240 MHz; 9 ciclos (37,5 ns); II = 1; SP da emulação C++ igual ao do QKeras
(97,64 % no holdout). Usa 3,4× menos LUT que o u8 e 0,4 % das LUTs da VU13P.
