# Gradient sensitivity visualizations (2026-11-02)

Protocol: 4 understanding samples (Janus) / 4 COCO-ShareGPT4V samples (InternVL2).
Hooks: `o_proj` / `down_proj` (Janus), `attention.wo` / `feed_forward.w2` (InternVL2).
Masks:
- A_unfixed: all-text caption (system+question+answer), includes padding; fallback labels.
- B_pad_only: all-text caption, padding excluded; fallback labels.
- C_labels_fixed: assistant-only caption/labels; padding excluded.
- internvl2: official assistant-only labels/caption.

## Directory layout

```
grad_ratio_20261102/
  README.md
  compare_distributions.png          # overall vis/cap distribution, A/B/C/InternVL2
  compare_layer_medians.png          # per-layer median curve
  A_unfixed/
    heatmap/                         # per-layer raw vis/cap gradient heatmaps (sample 0)
    summary/                         # aggregate layer x channel ratio heatmap + aggregate dist
  B_pad_only/  (same)
  C_labels_fixed/ (same)
  internvl2/ (same)
  by_layer/
    L00/ ... L31/
      dist_<variant>_<type>.png      # per-layer distribution (hist+KDE+normal fit)
      compare_attn.png               # A/B/C/InternVL2 KDE overlay, attention projections
      compare_mlp.png                # A/B/C/InternVL2 KDE overlay, MLP projections
    stats.csv                        # per-layer median/p10/p90/skew/kurt/Shapiro
  npz/                               # raw [layers, channels] ratio arrays per variant
  summary/                           # reserved
```

Per-layer raw heatmaps: left panel = vis token x channel, right = cap token x channel;
channels sorted by vis per-channel mean; color = log10 |grad_output|.
`profile_*.png` in the same folder = per-channel mean gradient (vis vs cap).

## Key numbers

| variant | attention type | median | p10 | p90 | skew(log10) | kurt(log10) | Shapiro p |
|---|---|---:|---:|---:|---:|---:|---:|
| A_unfixed | o_proj | 0.0300 | 0.0098 | 0.0429 | -4.52 | 20.3 | 1e-83 |
| B_pad_only | o_proj | 0.0294 | 0.0096 | 0.0421 | -4.52 | 20.3 | 1e-83 |
| C_labels_fixed | o_proj | 0.0159 | 0.0060 | 0.0221 | -4.64 | 21.1 | 1e-83 |
| InternVL2 | wo | 0.0402 | 0.0122 | 0.0799 | -4.39 | 20.1 | 1e-81 |

- log10(vis/cap) is **not normal and not log-normal**: strong left skew (~-4.5),
  heavy tails (kurtosis ~20), Shapiro p ~1e-83. Use median/quantiles/bootstrap, not t-test.
- A/B/C/InternVL2 have nearly identical distribution shapes; C is shifted left vs InternVL2
  (median 0.016 vs 0.040); padding alone (A→B) is negligible for understanding.
- The last layer (L29 Janus / L31 InternVL2) has zero gradient for non-supervised vision tokens
  (causal LM: no loss at those positions and no next layer to propagate); its distribution is
  degenerate and should be excluded from stats if needed.
- Effective gradient weight (token count × per-token mean |grad|, relative to caption):
  Janus vis 0.69x, gen_img 1.29x, t2i_text 0.16x; InternVL2 vis 0.90x.

## Shift analysis (`analysis/`)

- A→B (padding only): median Δlog10 = **−0.008** → padding alone negligible for understanding.
- A→C (labels/caption): median Δlog10 = **−0.291** (C ≈ 0.51× A), per-layer range **[−0.701, +0.330]**;
  L0 is positive, almost all other layers negative → **layer-dependent shift**, not a global rescale.
- Per-channel rank correlation A vs C: median **0.236** (range 0.086–0.271) → the fix changes
  **which channels are prioritized**, not just the overall scale.
- Per-channel rank correlation C vs InternVL2: **≈0.01** → per-channel vis/cap patterns are
  essentially unrelated across models; only aggregate distributions / effective weights are comparable.
- L29 is degenerate (non-supervised vision tokens have zero gradient); excluded from the shift stats.
- Plots: `analysis/layer_shift_delta.png`, `analysis/layer_correlation.png`,
  `analysis/scatter_A_vs_C.png`, `analysis/normalized_dist_A_vs_C.png`;
  numbers in `analysis/layer_shift.csv`.

**Interpretation**: the fixed wrapper does not merely rescale the reweighting coefficients;
it changes the channel structure and is layer-dependent. Therefore a fixed-wrapper 8-config
ablation is a genuine test (not a duplicate), but the overall ratio remains small
(median ~0.016), so the expected performance effect is still modest.