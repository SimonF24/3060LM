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
| 04e | 03e + Muon on hidden layers, LR 5e-4 | 1.2393 | 1.2628 | 1.1847 | 57.7M | 104M | 38.6k |
| 05a | 04c + QK-norm | 1.2232 | 1.2489 | 1.1632 | 57.7M | 100M | 37.2k |
| 05b | 04c + zero-init output projections | 1.2294 | 1.2535 | 1.1731 | 57.7M | 104M | 38.5k |
| 05c | 04c + untied embeddings | 1.2464 | 1.2682 | 1.1955 | 74.5M | 103M | 38.0k |
| 04f | 04c, seed 1338 | 1.2301 | 1.2546 | 1.1731 | 57.7M | 104M | 38.6k |
| 05d | 05a (QK-norm), seed 1338 | 1.2201 | 1.2466 | 1.1583 | 57.7M | 102M | 37.7k |
| 06a | 04c + GQA, 2 KV heads | 1.2316 | 1.2550 | 1.1772 | 53.0M | 108M | 40.1k |
| 06b | 04c + GQA, 4 KV heads | 1.2282 | 1.2519 | 1.1728 | 54.5M | 107M | 39.7k |
| 07a | QK-norm + ReLU² | 1.2193 | 1.2448 | 1.1599 | 48.2M | 113M | 42.0k |
| 07b | QK-norm + MLP width 1365 | 1.2215 | 1.2472 | 1.1616 | 54.5M | 103M | 38.2k |
| 07c | QK-norm + factorized embedding and head, 256 dims | 1.2285 | 1.2554 | 1.1655 | 49.6M | 113M | 41.9k |
| 07d | QK-norm + looped: 8 unique layers, 12 effective | 1.2305 | 1.2554 | 1.1723 | 44.0M | 105M | 38.7k |
| 07e | QK-norm + looped: 12 unique layers, 20 effective | 1.2526 | 1.2767 | 1.1965 | 57.7M | 73M | 27.1k |
| 08a | 07a (QK-norm + ReLU²), seed 1338 | 1.2136 | 1.2405 | 1.1506 | 48.2M | 118M | 43.8k |
| 08b | 07a + MLP width 2048 | 1.2156 | 1.2413 | 1.1557 | 54.5M | 109M | 40.3k |
| 08c | 07a + GQA, 4 KV heads | 1.2164 | 1.2418 | 1.1571 | 45.1M | 119M | 44.2k |
| 09a | 07a + logit softcap 15 | 1.2154 | 1.2416 | 1.1543 | 48.2M | 113M | 41.8k |
| 09b | 07a + windows start at document boundaries | 1.3059 | 1.3143 | 1.2863 | 48.2M | 116M | 43.0k |
| 09c | 07a + LR floor at 10% of peak | 1.2148 | 1.2408 | 1.1542 | 48.2M | 118M | 43.8k |
| 09d | 09c + EMA over the last 10%, decay 0.99 | 1.2143 | 1.2402 | 1.1537 | 48.2M | 118M | 43.8k |
| 10a | 07a + softcap, LR floor, EMA | 1.2132 | 1.2392 | 1.1524 | 48.2M | 114M | 42.3k |
| 10b | 10a, seed 1338 | 1.2107 | 1.2378 | 1.1476 | 48.2M | 115M | 42.7k |
| 10c | 07a + sequence length 128, then 256 at 30%, 512 at 60% | 1.2124 | 1.2392 | 1.1499 | 48.2M | 118M | 43.6k |
| 10d | 07a + effective batch 32, then 64 at 40%, 128 at 70% | 1.2086 | 1.2352 | 1.1464 | 48.2M | 117M | 43.4k |
| 10e | 07a + embedding skip connections | 1.2154 | 1.2413 | 1.1549 | 48.2M | 117M | 43.5k |
| 11a | 10d (batch schedule), seed 1338 | 1.2085 | 1.2359 | 1.1445 | 48.2M | 116M | 43.1k |
| 11b | 10d + softcap, LR floor, EMA | 1.2074 | 1.2343 | 1.1447 | 48.2M | 112M | 41.6k |
| 11c | 10d + sequence length schedule | 1.2048 | 1.2321 | 1.1411 | 48.2M | 118M | 43.7k |
| 11d | 07a + Muon momentum 0.97 | 1.2159 | 1.2418 | 1.1557 | 48.2M | 118M | 43.5k |
| 11e | 07a + hashed bigram embedding, 65,536 rows | 1.2216 | 1.2470 | 1.1623 | 81.8M | 112M | 41.5k |
| 12a | 11c (batch and sequence schedules), seed 1338 | 1.2100 | 1.2366 | 1.1481 | 48.2M | 112M | 41.6k |
| 12b | 10d + sampled softmax, 2,048 random tokens, until 70% | 1.1916 | 1.2196 | 1.1262 | 48.2M | 144M | 53.3k |
| 12c | 10d + untie the head at 67% | 1.2144 | 1.2405 | 1.1535 | 48.2M | 111M | 41.2k |
| 12d | 10d + 16k vocabulary | 1.2080 | 1.2393 | 1.1351 | 39.9M | 135M | 49.8k |
| 12e | 07a + effective batch 16, then 32 at 25%, 64 at 50%, 128 at 75% | 1.2169 | 1.2429 | 1.1562 | 48.2M | 107M | 39.7k |
| 13a | 12b (sampled softmax), seed 1338 | 1.1922 | 1.2209 | 1.1253 | 48.2M | 143M | 53.0k |
| 13b | 12b with 512 random tokens | 1.1910 | 1.2193 | 1.1248 | 48.2M | 144M | 53.3k |
| 13c | 12b with 8,192 random tokens | 1.1969 | 1.2245 | 1.1326 | 48.2M | 134M | 49.4k |
| 13d | 12b, sampling until 85% | 1.1882 | 1.2167 | 1.1217 | 48.2M | 153M | 56.8k |
| 13e | 12b + 16k vocabulary | 1.1979 | 1.2295 | 1.1241 | 39.9M | 154M | 57.1k |
| 14a | 13d (sampling until 85%), seed 1338 | 1.1882 | 1.2172 | 1.1206 | 48.2M | 153M | 56.8k |
| 14b | 12b, sampling until 95% | 1.1964 | 1.2248 | 1.1302 | 48.2M | 158M | 58.5k |
| 14c | 12b, sampling throughout | 1.2556 | 1.2853 | 1.1861 | 48.2M | 162M | 59.8k |
| 14d | 13d with 1 random token | 1.1874 | 1.2159 | 1.1210 | 48.2M | 159M | 59.0k |
| 14e | 12b + Muon momentum 0.85 to 0.95 over 300 steps | 1.1901 | 1.2182 | 1.1244 | 48.2M | 148M | 54.9k |

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
  high. LRs are in `match_rms_adamw` units, and AdamW stays on the embeddings and norm scales at 1e-3.
  By Muon LR: 1e-3 → 1.2308, 2e-3 → 1.2358 (within noise), 3e-3 → 1.2491, 6e-3 → 1.3776. 5e-4 is untested.
- **04e:** Muon LR is bracketed, with 5e-4 → 1.2393 on the low side of 1e-3. Muon at 1e-3 (04c) stays the base recipe.
- **05:** compared against 04c (1.2308). QK-norm's 0.0076 gain is under the noise margin and needs a second seed.
  Zero-init is a tie (0.0014). Untied embeddings are 0.0156 worse despite 16.8M more parameters, so rejected.
- **QK-norm accepted:** paired by seed it gains 0.0076 and 0.0100, averaging 1.2217 vs 1.2305 for the base recipe, and
  improves both sources in both seeds. The base recipe's two seeds are only 0.0007 apart here, against 0.0061 in the
  AdamW pair, so noise varies by recipe. QK-norm becomes part of the base recipe, with a gain of about 0.009.
- **06:** GQA is a tie on score against the base recipe's 1.2305, despite 4-6% more tokens and 5-8% fewer parameters.
  It stays off for now and may be worth revisiting for sizing or faster generation. GQA with QK-norm is untested.
- **07:** compared against QK-norm's 1.2217. ReLU² and MLP width 1365 tie on score but are cheaper, ReLU² by a lot
  (16% fewer parameters, 11% more tokens). The factorized head is 0.0068 worse without being cheaper than ReLU².
- **Looped transformers:** equal compute (07d) is 0.0088 worse with no gain anywhere, and more depth (07e) is 0.0309
  worse because it sees about 28% fewer tokens. Both dropped for pretraining.
- **ReLU² accepted:** paired by seed it gains 0.0039 and 0.0065, averaging 1.2165 vs 1.2217, and it's cheaper. Identical
  recipes also varied in throughput (42.0k vs 43.8k tok/s), which likely contributes to the gap between seeds.
- **08b/08c:** both tie 07a, the same-seed comparison, gaining 0.0037 and 0.0029. A wider MLP and GQA stay off for now.
  Both are worth another look when choosing model shape on longer runs.
- **09:** none clears the margin against the base recipe's 2-seed average of 1.2165. Comparing against 07a alone would
  flatter them, because it ran slower (42.0k tok/s). 09c and 09d ran at 08a's throughput and don't beat 08a's 1.2136.
- **09b:** far worse, but not a fair test of the speedrun's technique. Every window starts at a document start here,
  so the model rarely sees text past the first 512 tokens of a document, unlike the mid-document windows in val.
  The speedrun aligns only the start of a stream that then runs through documents in order.
- **Throughput:** identical recipes vary between about 42k and 44k tok/s, and 09b's 41.8k interval overlapped a CPU
  spike from other work, costing about 0.4% of its tokens.
- **10a/10b:** the bundle averages 1.2120, a gain of 0.0046 that's positive in both seeds (0.0061 and 0.0029) but well
  short of the 0.01 set beforehand, so it's not accepted. EMA also adds about 0.7GB of memory.
- **10c-e:** against 1.2165, the accumulation schedule gains 0.0079, which needs a second seed. The sequence length
  schedule (0.0041) and embedding skips (0.0011) tie.
- **Batch schedule accepted:** effective batch 32, then 64 at 40% and 128 at 70% scores 1.2086 and 1.2085, averaging
  0.0080 better than 1.2165. Paired gains are 0.0107 and 0.0051, it improves both sources in both seeds, and it's free.
- **11b:** the bundle adds only 0.0011 to the batch schedule while costing memory and speed, so it's dropped.
- **11c:** 0.0037 better than the batch schedule alone, the lowest score yet, but it needs a second seed.
- **11d/11e:** higher Muon momentum ties. The bigram table is 0.0051 worse despite 70% more parameters, so it's rejected
  at this scale, though it may need more tokens to fill such a large table.
- **12a:** the sequence schedule is 0.0015 worse on this seed, after being 0.0037 better on the other. It averages 1.2074,
  a tie with the batch schedule alone, so 11c's lead was noise and it's rejected.
- **12b:** the largest gain since Muon, 0.0170 better than 1.2086, with 23% more tokens because most of the head isn't
  computed for the first 70%. BPB is still scored with the full softmax. It's being repeated on a second seed.
- **12c:** untying late is 0.0058 worse and adds 67MB to the checkpoint, so it's rejected at this length.
- **12d:** a tie on BPB at 1.2080, with 17% fewer parameters, 15% more tokens per second and 4.5GB peak memory instead of
  6.5GB. It scores better on FineMath (1.1351 vs 1.1464) and worse on FineWeb-Edu (1.2393 vs 1.2352). The score covers
  approximately the same val text as other runs, since the evaluated prefix is set in tokens.
- **12e:** the wider batch schedule is 0.0083 worse, so 32, 64 and 128 stays.
- **Sampled softmax accepted:** 1.1916 and 1.1922, averaging 1.1919, with paired gains of 0.0170 and 0.0163 on both
  sources.
- **13b/13c:** 512 random tokens ties 2,048 at the same speed, probably because each batch's own targets dominate the
  candidate set. 8,192 is 0.0053 worse and slower.
- **13d:** sampling until 85% gains 0.0034 on the same seed, which needs a second seed.
- **13e:** with sampling, 16k is 0.0063 worse than 32k, losing on FineWeb-Edu (1.2295 vs 1.2196) and tying on FineMath.
  The head is already cheap, so 16k's speed advantage shrinks to 7%.
- **Sampling until 85% accepted:** 1.1882 on both seeds, 0.0037 better than 1.1919 on average (paired 0.0034 and
  0.0040), better on both sources in both seeds, and faster. It's under 0.01, but like the batch schedule it's
  consistent and free, unlike the 10a bundle, which also cost memory and speed.
- **14b/14c:** sampling until 95% is 0.0082 worse, and sampling throughout is 0.0674 worse, so the model needs a
  full softmax phase of roughly 15% to calibrate across the whole vocabulary.
- **14d:** sampling only each batch's targets plus one random token ties 2,048 (1.1874 vs 1.1882) at 4% more tokens per
  second. It's a simplification rather than a gain, so 2,048 stays for now.
- **14e:** momentum warmup ties (0.0015 better). The ramp is from memory of the speedrun, not checked against its code.

## Benchmarks

Scored by total log-probability of each option with `3060lm/evaluate.py`, as accuracy ± standard error.

| Benchmark | Chance | 00 baseline | 08a | 12b sampled softmax |
|---|---:|---:|---:|---:|
| HellaSwag (n=10,042) | 25.0% | 25.95 ± 0.44% | 26.37 ± 0.44% | 26.35 ± 0.44% |
| PIQA (n=1,838) | 50.0% | 55.39 ± 1.16% | 56.96 ± 1.15% | 56.69 ± 1.16% |
| ARC-Easy (n=2,376) | 25.0% | 34.26 ± 0.97% | 41.16 ± 1.01% | 42.21 ± 1.01% |
| ARC-Challenge (n=1,172) | 25.0% | 16.81 ± 1.09% | 18.86 ± 1.14% | 17.75 ± 1.12% |

Only the gain on ARC-Easy from baseline to 08a is clearly larger than its error, and 12b's lower BPB doesn't show up on
any benchmark. Summed log-probability favors short options, and on
ARC-Challenge always picking the shortest option scores 22.7% and the longest 27.8%, which partly explains the
below-chance score. The models score below even the shortest-option rule, so length isn't the full explanation.
The bias applies to both models, so compare them with each other rather than reading these as absolute ability.

