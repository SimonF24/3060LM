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
| 03c | 02c with effective batch 64×512 | 1.2812 | 1.3013 | 1.2344 | 57.7M | 110M | 40.6k |
| 03d | 02c with effective batch 128×512 | 1.3096 | 1.3287 | 1.2652 | 57.7M | 110M | 40.6k |
| 03e | 03c with peak LR 1e-3 | **1.2604** | 1.2816 | 1.2109 | 57.7M | 110M | 40.6k |
| 03f | 03c with peak LR 2e-3 | 1.2622 | 1.2832 | 1.2134 | 57.7M | 109M | 40.4k |
| 04a | 03e + Muon on hidden layers, LR 2e-3 | **1.2358** | 1.2592 | 1.1811 | 57.7M | 102M | 37.6k |
| 04b | 03e + Muon on hidden layers, LR 6e-3 | 1.3776 | 1.3898 | 1.3491 | 57.7M | 101M | 37.4k |
| 04c | 03e + Muon on hidden layers, LR 1e-3 | **1.2308** | 1.2545 | 1.1754 | 57.7M | 102M | 37.6k |
| 04d | 03e + Muon on hidden layers, LR 3e-3 | 1.2491 | 1.2715 | 1.1968 | 57.7M | 102M | 37.9k |

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
  Continuing at LR 5e-4: 64 → 1.2812 (tied with 32), 128 → 1.3096. With only about 1.7k steps, 128 likely has too
  few updates at this LR, and its 200-step warmup takes 12% of them. LR needs rechecking at larger batches.
  All four batch-64+ runs (03c-f) ran about 6% slower than 03a (40.5k vs 43.1k tok/s). Since it repeats, it looks
  systematic rather than random, but the cause is unknown and accumulation shouldn't cost this much.
- **03e/03f:** at batch 64, raising the peak LR from 5e-4 to 1e-3 gains 0.0208 (0.0205 over 03a, the previous best).
  1e-3 and 2e-3 tie. Batch 64 at 1e-3 is the new base recipe.
- **04:** Muon gains 0.0246 over AdamW despite about 8% lower throughput (37.6k vs 40.6k tok/s). LR 6e-3 is far too
  high. LRs are in `match_rms_adamw` units, and AdamW stays on the embeddings and norm scales at 1e-3.  By Muon LR: 1e-3 → 1.2308, 2e-3 → 1.2358 (within noise), 3e-3 → 1.2491, 6e-3 → 1.3776. 5e-4 is untested.
