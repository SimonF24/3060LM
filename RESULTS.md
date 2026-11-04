# Speedrun Results

Each run trains for a fixed **45 minutes** of wall-clock time on the RTX 3060, then reports bits-per-byte
(BPB) on fixed held-out val text: 2M tokens each of FineWeb-Edu and FineMath-4+, at 512 context.
**Score = weighted BPB** (70% FineWeb-Edu / 30% FineMath, the same as the training mix). Lower is better.
Full configs and metrics for every run are in `results.jsonl`.

| # | Change | Weighted BPB | FineWeb-Edu | FineMath | Params | Tokens | tok/s |
|---|--------|-------------:|------------:|---------:|-------:|-------:|------:|
| 00 | Baseline: original model and recipe (AdamW 1e-4 constant, bs 16×512) | 1.3709 | 1.3860 | 1.3358 | 57.7M | 86M | 31.9k |
| 01 | `torch.compile` | 1.3274 | 1.3464 | 1.2831 | 57.7M | 113M | 41.9k |
| 02 | LR schedule: 1e-3 peak, 200-step warmup, linear decay over last 40%, β2 = 0.95 | 1.3241 | 1.3351 | 1.2985 | 57.7M | 114M | 42.3k |
| 02b | Same as 02, seed 1338 | 1.3180 | 1.3308 | 1.2880 | 57.7M | 115M | 42.6k |
| 02c | Same as 02, peak LR 5e-4 | 1.3199 | 1.3318 | 1.2922 | 57.7M | 114M | 42.1k |
| 03a | 02c with effective batch 32×512 (grad accumulation) | 1.2809 | 1.2984 | 1.2401 | 57.7M | 116M | 43.1k |
| 03b | 02c with batch 8×512 | 1.4023 | 1.4087 | 1.3873 | 57.7M | 101M | 37.5k |

## Notes

- **00 → 01:** the baseline ran while dataset downloads were using the CPU and disk. Uncompiled runs without
  downloads reached about 33.2k tok/s, so the clean compile speedup is closer to 26% than 31%.
- **Noise:** 02 and 02b differ only by seed and are 0.0061 apart (0.0043 on FineWeb-Edu, 0.0105 on FineMath).
  That's one pair of runs, so a rough estimate. Working rule: a single-run gain under **0.01** is not treated as real.
- **02:** the 1e-3 seeds average 1.3211, about 0.006 better than 01, which is inside the noise margin. The schedule
  is kept because it costs nothing, but its gain is unproven. FineMath looking worse in 02 is also within noise.
- **02c:** 1e-3 and 5e-4 can't be told apart. 5e-4 is carried forward as the safer choice for longer runs.
- **03:** batch size is the first change after compile to clear the noise margin. BPB by effective batch at the
  same LR and token count: 8 → 1.4023, 16 → 1.3199, 32 → 1.2809.
