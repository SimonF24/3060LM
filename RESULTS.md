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
| 15a | 13d + value residual | 1.1770 | 1.2065 | 1.1081 | 48.2M | 151M | 56.0k |
| 15b | 13d, 120 minutes | 1.1202 | 1.1531 | 1.0435 | 48.2M | 410M | 56.9k |
| 15c | 07a, 120 minutes | 1.1417 | 1.1722 | 1.0704 | 48.2M | 312M | 43.3k |
| 16a | 15a (value residual), seed 1338 | 1.1759 | 1.2061 | 1.1053 | 48.2M | 153M | 56.8k |
| 16b | 15a at 16×640, 10-minute probe | 1.4393 | 1.4544 | 1.4040 | 83.9M | 18M | 30.4k |
| 16c | 15a at 16×704, 10-minute probe | 1.5106 | 1.5200 | 1.4886 | 101.0M | 14M | 23.0k |
| 17a | 15a + training mix 80/20, then 55/45 at 60% | 1.1757 | 1.2068 | 1.1031 | 48.2M | 156M | 57.7k |
| 17b | 15a at 16×704, 60 minutes | 1.1861 | 1.2151 | 1.1184 | 101.0M | 94M | 26.0k |
| 17c | 17b with LR 7e-4 | 1.1983 | 1.2269 | 1.1317 | 101.0M | 89M | 24.7k |
| 17d | 17b with LR 1.4e-3 | 1.1904 | 1.2191 | 1.1236 | 101.0M | 89M | 24.6k |
| 19a | 15a with BatchedMuon (same update, same-shape matrices orthogonalized together) | 1.1732 | 1.2031 | 1.1035 | 48.2M | 158M | 58.4k |

## Long Run

`long-704`: 16×704 with the 15a recipe at LR 1e-3 for **24 hours**, on the full token set (`data/tokenized-full`, same
val text). It ran with no restarts, and the supervisor and watchdog never triggered.

| Run | Weighted BPB | FineWeb-Edu | FineMath | Params | Tokens | tok/s |
|---|---:|---:|---:|---:|---:|---:|
| long-704 | **0.9838** | 1.0254 | 0.8868 | 101.0M | 2,154M | 24.9k |

Val BPB during the run: 1.3259 at 1h, 1.2041 at 4h, 1.1538 at 9h, 1.0910 at 16h, 1.0571 at 20h, 0.9922 at 21h,
0.9838 at the end. The checkpoints in the middle aren't finished models: the warmdown starts at 60%, and the drop of
0.065 between 20h and 21h comes from switching to the full softmax at 85%. Before that switch the model's predictions
are miscalibrated across the full vocabulary, so the mid-run scores understate it.


## Chat Fine-Tuning (SFT)

`sft-704` fine-tuned long-704 for one epoch on smol-smoltalk plus GSM8K train ×4: 281M tokens, of which 218M are
assistant tokens, packed into 1024-token rows. It ran 4,513 steps of 64 rows at peak LR 3e-4 with linear decay, taking
228.5 min at 21.6k tok/s and 9.4GB peak memory, with no restarts.

| Step | 0 | 1000 | 2000 | 3000 | 4000 | 4513 |
|---|---:|---:|---:|---:|---:|---:|
| Masked val loss | 2.2177 | 1.6033 | 1.5316 | 1.4967 | 1.4797 | 1.4769 |

- **GSM8K test (n=1,319, greedy, strict `#### N`):** 0.0%. No reply used the `####` format, even though about 30k
  GSM8K answers (6% of conversations) were trained on. The model answers math in smol-smoltalk's "To find ..., we
  need to ..." style instead. Taking the last number in the reply as the answer scores 4/300 (1.3%).
- **Samples:** the replies follow the format and stop at the end token, and the code example is correct (a recursive
  factorial). Arithmetic and facts are wrong: "17 * 23 = 23^2", "Pride and Prejudice ... attributed to Jean-Paul Sartre",
  and the sky is blue because of the atmosphere's temperature. Longer replies repeat themselves.

### Math stage with a calculator (`sft-math`)

`sft-math` continued from sft-704 for one epoch on GSM8K train ×3 with its `<<48/2=24>>` annotations kept as calculator
calls (the inserted result isn't trained on), plus 10,000 smol-smoltalk conversations as replay. That's 10.3M tokens in
316 steps of 32 rows at peak LR 1e-4, taking 10 min. Both benchmarks are scored greedily with the calculator available.
SVAMP (all 1,000 problems) is never trained on. "Lenient" takes the last number in the reply.

| Model | Chat val loss | GSM8K strict | GSM8K lenient | GSM8K `####` | SVAMP strict | SVAMP lenient | SVAMP `####` | Calls per reply |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sft-704 | 1.4769 | 0.0% | 2.05% | 0% | 0.0% | 3.4% | 0% | 0 |
| sft-math | 1.4833 | 1.82 ± 0.37% | 2.27% | 64% | 2.9 ± 0.53% | 3.9% | 85% | 4.2 / 3.3 |

- The stage taught the format and the tool: about two-thirds of GSM8K replies and 85% of SVAMP replies end in `#### N`,
  with three to four calculator calls each, and chat val loss barely moves (+0.006).
- Accuracy hardly improves. Lenient scores rise only 0.2 to 0.5 points, within error, so arithmetic wasn't the
  bottleneck: the model sets up the wrong calculations (16*3 for "eats three of 16 eggs", doubling instead of halving).
  Replies without `####` mostly loop until the 320-token limit.
- The strict scores are no longer zero, so RL has some reward signal to start from, but not much.

**pass@k** (sft-math, 16 samples per problem at temperature 1.0, top-k 50, first 200 problems of each, strict scoring):

| Benchmark | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | Problems by correct samples (of 16) |
|---|---:|---:|---:|---:|---:|---|
| GSM8K | 1.1% | 2.2% | 4.3% | 8.4% | 16.0% | 0: 168, 1: 29, 2: 3 |
| SVAMP | 2.6% | 5.0% | 9.6% | 17.8% | 30.5% | 0: 139, 1: 43, 2: 16, 3: 1, 4: 1 |

pass@k grows almost linearly with k, and nearly every solved problem is solved by just one or two samples, so correct
answers are rare draws, not problems the model reliably knows. With groups of 16 samples, an RL step would get reward
signal on 16% of GSM8K problems and 30% of SVAMP problems.

**KV cache:** generation now keeps each layer's keys and values and runs only the newest token per step, as a CUDA
graph (Python couldn't issue the step's small kernels as fast as the GPU ran them). On the real model, cached and
uncached logits match to 3e-5 in fp32. In bf16 both are 0.35 from fp32, so greedy replies can differ at near-ties
(35 of 64 identical); rescored with the cache, sft-math gets GSM8K 1.90% and SVAMP 3.3%, within error of the scores
above. The GSM8K eval went from 18 min to 93 s, and the pass@k run above, which took 89 min, would take about 8.

## RL (`rl-math`)

`rl-math` ran on-policy policy gradient with group baselines (`3060lm/rl.py`) from sft-math: 300 steps of 32 GSM8K
training problems × 16 replies sampled at temperature 1 from the full distribution, with the calculator. Reward is 1
for a correct `#### N` answer. AdamW at 2e-5, 151.6 min, 10.2GB peak, no restarts. GSM8K test and SVAMP are held out.

| Model | GSM8K strict | GSM8K lenient | GSM8K `####` | SVAMP strict | SVAMP lenient | SVAMP `####` | Calls per reply |
|---|---:|---:|---:|---:|---:|---:|---:|
| sft-math (cached rescore) | 1.90% | 2.35% | 63% | 3.3% | 4.3% | 85% | 4.2 / 3.2 |
| rl-math | 3.94 ± 0.54% | 3.94% | 98% | 12.5 ± 1.05% | 12.5% | 99% | 1.7 / 1.8 |

Periodic evals (first 400 problems of each, greedy):

| Step | 0 | 50 | 100 | 150 | 200 | 250 | 300 |
|---|---:|---:|---:|---:|---:|---:|---:|
| GSM8K | 2.75% | 3.75% | 3.50% | 3.00% | 3.75% | 4.00% | 6.75% |
| SVAMP | 3.75% | 14.00% | 11.50% | 11.75% | 10.75% | 10.75% | 11.75% |

- SVAMP, which RL never saw, nearly quadruples, and GSM8K doubles. The lenient scores rise just as much, so it isn't only
  the `####` format: replies that used to loop until the token limit now finish, at about 80-100 tokens instead of 170,
  with fewer calculator calls.
- Almost all of the gain arrives in the first 50 steps. After that the training reward stays flat at 1-4% and the
  evals move within noise (the step-300 GSM8K 6.75% on 400 problems is 3.94% on all 1,319). RL fixed how the model
  finishes answers, but with so few correct samples it found little signal to improve the reasoning itself.

## Distillation (chain 18)

Does distilling from a stronger model beat training directly, when the teacher's forward passes count against the
same 45-minute budget? The teachers are SmolLM2-360M and SmolLM2-1.7B. The student is the current speedrun recipe
(12×512) with SmolLM2's 49k tokenizer, which matching the teacher's distribution requires, so it has 56.6M parameters
instead of 48.2M. Distilled runs use loss = 0.5 × KL(teacher ‖ student) + 0.5 × cross-entropy at temperature 1, with
the full softmax (sampled softmax can't produce the student's full distribution) and micro-batch 8 (16 used 13.7GB
with the 1.7B teacher).

| Run | Teacher | Weighted BPB | FineWeb-Edu | FineMath | Tokens | tok/s | Peak memory |
|---|---|---:|---:|---:|---:|---:|---:|
| 16a value residual (our tokenizer) | none | 1.1759 | 1.2061 | 1.1053 | 153M | 56.8k | 6.6GB |
| 18a direct | none | **1.1883** | 1.2082 | 1.1418 | 129M | 47.8k | 9.5GB |
| 18b distilled | SmolLM2-360M | 1.2529 | 1.2645 | 1.2257 | 40M | 14.9k | 10.2GB |
| 18c distilled | SmolLM2-1.7B | 1.4247 | 1.4278 | 1.4172 | 16M | 5.7k | 12.9GB |

Teacher BPB on the same val text (first 150k tokens of each source): SmolLM2-360M 0.8145, SmolLM2-1.7B 0.6743. These
may be flattered, since SmolLM2 trained on FineWeb-Edu and FineMath and may have seen these documents.

- Within the budget, direct training wins clearly. Each teacher forward pass costs more than the student's training
  step, so the 360M teacher cuts the student to 31% of the direct run's tokens and the 1.7B teacher to 12%, and the
  teacher's signal doesn't make up for it. The 1.7B run also peaked at 12.9GB, past the card's 12GB, so it may have
  spilled into system memory and run slower than it otherwise would.
- Per token, the teacher helps a lot, though only later in the run. At about 17M tokens (step 1000 of each, both
  at full LR) the 360M-distilled student's cross-entropy was still worse (4.61 vs 4.24), but by the end of its
  schedule it's well ahead. 18d, 18b's configuration without the teacher with a 15.5-minute budget, reached 1.3444
  (FineWeb-Edu 1.3563, FineMath 1.3167) on 31.8M tokens at 34.2k tok/s. Between 18d and 18a, direct training gains
  about 0.077 BPB per doubling of tokens, which puts it at about 1.32 at 18b's 40.3M tokens, 0.065 behind 18b's
  1.2529. Matching 1.2529 directly would take about 72M tokens, so a 360M-distilled token is worth about 1.8 direct
  tokens, while the teacher cuts throughput 3.2x. Teacher outputs computed elsewhere, or reused across runs, would
  win; computed on this GPU during training, they lose.
- Our 32k tokenizer beats SmolLM2's 49k one at this budget (1.1759 vs 1.1883): the smaller vocab is faster
  (56.8k vs 47.8k tok/s) and puts fewer parameters in the embedding.

## TinyGSM pilot (`sft-tinygsm`)

SFT from sft-704 on TinyGSM's first shard (GPT-3.5-written Python solutions to synthetic grade-school problems) plus
20,000 smol-smoltalk conversations as replay: 691k conversations, 109.6M tokens, 1,692 steps, 86.4 min, same settings
as sft-704. Each solution's docstring (a copy of the question) is removed, the function becomes `solution`, and
`3060lm/pytool.py` runs it for the answer. The reply is the code block, `>>> solution()`, the tool's output (not
trained on) and `#### N`. 3.6% of solutions fail to run or don't return a number and are dropped, and 0.15% of
problems share a 10-word sequence with a GSM8K test or SVAMP problem and are removed. TinyGSM's solutions aren't
verified, so some are wrong.

| Model | Chat val loss | GSM8K | SVAMP (held out) | `####` GSM8K / SVAMP | Tool calls per reply |
|---|---:|---:|---:|---:|---:|
| sft-math | 1.4833 | 1.90% | 3.3% | 63% / 85% | 4.2 calculator |
| rl-math | | 3.94 ± 0.54% | 12.5 ± 1.05% | 98% / 99% | 1.7 calculator |
| sft-tinygsm | 1.5316 | **10.92 ± 0.86%** | **26.6 ± 1.4%** | 93% / 99% | 0.94 / 1.0 Python |

- Scale was what sft-math lacked: with about 100x the problems, GSM8K is 5.7x sft-math's and 2.8x rl-math's, and
  SVAMP, which isn't in TinyGSM's style, is 2.1x rl-math's, without any RL yet.
- Chat val loss rises 0.055, so the 20k replay conversations don't fully protect general chat.
- The run's own SVAMP score (3.9%, in sft_results.jsonl) is wrong: the Python tool capped its worker's memory at
  1GB, but a worker started from a script that imports torch already maps more than that, so calls began failing
  with SystemError partway through the eval. The cap is now the worker's current size plus 1GB, and the scores above
  are a rescore with the fixed tool (in benchmarks.jsonl). The training data wasn't affected (96.4% of solutions run
  with the fix vs 96.3%).

### RL from the TinyGSM pilot (`rl-tinygsm`)

`rl.py` from sft-tinygsm with the Python tool: 400 steps of 32 GSM8K training problems × 16 replies, otherwise as
rl-math. 103 min (generation 9-19 s per step with continuous batching), 9.7GB peak, no restarts.

| Model | GSM8K | SVAMP (held out) | Training reward |
|---|---:|---:|---|
| sft-tinygsm (rescored with continuous batching) | 10.54% | 27.2% | |
| rl-tinygsm | **16.60 ± 1.02%** | **29.5 ± 1.4%** | 8% at step 50, 12-25% from step 100 |

Periodic evals (first 400 problems of each, greedy):

| Step | 0 | 50 | 100 | 150 | 200 | 250 | 300 | 350 | 400 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GSM8K | 12.25% | 13.00% | 14.50% | 18.25% | 16.50% | 17.00% | 17.00% | 18.75% | 18.50% |
| SVAMP | 27.25% | 29.25% | 27.50% | 31.00% | 27.50% | 28.25% | 26.00% | 31.00% | 28.75% |

- With about 10% of samples correct, 8-17 of 32 groups per step carry signal (vs 1-5 for rl-math), and RL now
  improves the reasoning rather than just the format: GSM8K rises 6 points, about 5.6 standard errors.
- SVAMP rises 2.3 points, under 2 standard errors. Most of RL's gain stays on the problems it trains on (GSM8K's
  training split, in GSM8K's style).
- Replies stay short (72-87 tokens) with exactly one Python call each.

## TinyGSM, 8 shards (`sft-tinygsm8`)

SFT from sft-704 on TinyGSM shards 0-7 (5.37M problems, docstrings removed, decontaminated) plus 170k smol-smoltalk
conversations as replay (about 12% of tokens): 5.54M conversations, 865.6M tokens, 13,381 steps of 64 rows at the
sft-704 settings, 658.5 min (11.0 h) at 21.9k tok/s, no restarts. Greedy scores with the Python tool:

| Model | TinyGSM problems | Chat val loss | GSM8K | SVAMP (held out) |
|---|---:|---:|---:|---:|
| sft-tinygsm (pilot, 1 shard) | 671k | 1.5316 | 10.54% | 27.2% |
| rl-tinygsm (pilot + RL) | 671k | 1.5958 | 16.60% | 29.5% |
| sft-tinygsm8 | 5.37M | 1.5291 | **30.63 ± 1.27%** | **48.6 ± 1.6%** |

- 8x the problems roughly triples GSM8K (10.5% to 30.6%) and nearly doubles held-out SVAMP (27.2% to 48.6%), so the
  extra data improves math reasoning in general, not only GSM8K's style.
- Chat val loss ends slightly below the pilot's (1.5291 vs 1.5316) despite 8x more math: the larger replay share
  (12% vs 2%) holds general chat steady. It rose to 1.565 by step 2,000, then fell for the rest of the run.

### Math RL from sft-tinygsm8 (`rl-tinygsm8`)

`rl.py` from sft-tinygsm8, as rl-tinygsm (400 steps × 32 GSM8K training problems × 16 replies). 138.4 min (2.3 h),
10.0GB peak, no restarts.

| Model | GSM8K | SVAMP (held out) | Training reward |
|---|---:|---:|---|
| sft-tinygsm8 | 30.63 ± 1.27% | 48.6 ± 1.6% | |
| rl-tinygsm8 | **35.33 ± 1.32%** | 46.9 ± 1.6% | 26% at step 50, 38-43% from step 300 |

| Step | 0 | 50 | 100 | 150 | 200 | 250 | 300 | 350 | 400 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GSM8K (first 400) | 33.50% | 35.00% | 34.25% | 36.75% | 39.00% | 37.75% | 39.00% | 38.50% | 36.75% |
| SVAMP (first 400) | 45.25% | 52.25% | 44.25% | 44.25% | 45.75% | 46.50% | 44.50% | 46.75% | 42.75% |

- GSM8K rises 4.7 points (about 2.6 standard errors of the difference), and the training reward climbs from 26% to
  about 40%, with 13-22 of 32 groups carrying signal per step.
- SVAMP doesn't improve (-1.7 points, within error): as with rl-tinygsm, RL on GSM8K's training split sharpens
  GSM8K-style problems rather than math in general.
- Mid-run, replies briefly used 2-3 Python calls each (steps 150-250), then went back to one.

### Cost to general chat

Masked loss on 2,000 held-out smol-smoltalk conversations, and the log-probability benchmarks from `evaluate.py`
(standard errors about 0.45% HellaSwag, 1.1% PIQA, 1.0% ARC-Easy, 1.2% ARC-Challenge):

| Model | Chat val loss | HellaSwag | PIQA | ARC-Easy | ARC-Challenge |
|---|---:|---:|---:|---:|---:|
| long-704 (base) | | 28.62% | 62.13% | 51.09% | 21.25% |
| sft-704 | 1.4769 | 29.02% | 61.26% | 47.35% | 22.44% |
| sft-math | 1.4833 | 29.06% | 61.59% | 47.39% | 22.61% |
| rl-math | 1.5492 | 29.12% | 61.32% | 45.58% | 22.95% |
| sft-tinygsm | 1.5316 | 29.02% | 60.72% | 44.74% | 22.18% |
| rl-tinygsm | 1.5958 | 29.08% | 60.17% | 45.20% | 22.53% |

- Chat val loss rises 0.119 from sft-704 to rl-tinygsm (8%): 0.055 from TinyGSM SFT and 0.064 from RL. Each RL run
  adds about 0.065, because RL narrows the model's distribution toward its math replies.
- The benchmarks barely move: HellaSwag and ARC-Challenge are flat, PIQA drops about 1 point (within error) and
  ARC-Easy 2-2.6 points (about 2 standard errors). Chat SFT itself cost more ARC-Easy (51.1% to 47.4%).
- Sampled replies to six general questions (capital of France, study tips, black holes, a poem, viruses vs
  bacteria, reversing a list) read about the same before and after: fluent, with the same kind of factual errors,
  and no Python where it doesn't belong.

## Profile (after the TinyGSM pilot)

Measured on 16×704, micro-batch 16 × 512 for pretraining and 8 × 1024 for SFT:

| Piece | Time |
|---|---:|
| Pretraining micro-batch, full softmax | 345 ms (23.9k tok/s) |
| Pretraining micro-batch, sampled softmax | 269 ms (30.5k tok/s) |
| Body only (no output head) | 255 ms |
| get_batch | 9.7 ms |
| AdamW step | 13.6 ms |
| **Muon step** | **162.5 ms** |
| SFT micro-batch | 357 ms (23.0k tok/s), 80.5% of tokens trained |
| RL decode step, 64 rows | 7.5 ms (256 cache entries) to 10.1 ms (512); 10.2 ms wall |

- Muon is the largest single target. Its cost is per optimizer step, not per token: about 23% of step time at
  accumulation 2, 14% at 4 and 6-8% at 8, or about 15% of a run with the current schedule. PyTorch's version already
  runs Newton-Schulz in bf16 but handles the 96 matrices one at a time, 5 iterations each.
- The full softmax head and loss add 90 ms per micro-batch (26%), and the sampled version 14 ms. The softmax kernels
  run on a materialized fp32 copy of the logits, which a fused loss would avoid.
- RL generation keeps only 33% of its batch busy: replies average 83 tokens, but each 64-row batch runs until its
  longest reply ends. Python overhead per step is small (10.2 ms wall vs 7.5-10.1 ms of GPU replay).
- The GPU wasn't throttling: 1,942 of 2,145 MHz, 128 of 170 W, 72°C.

## Search and abstention (setup)

The search tool is BM25 (tantivy, `3060lm/search.py`) over the DPR Wikipedia passages: the December 2018 English
Wikipedia split into 21,015,324 disjoint 100-word passages (4.7GB download, 13GB index, built in 16 min). Queries
drop common stopwords, which cuts search time from 0.3-0.5 s to about 0.1 s without hurting recall. The tool
returns the top 3 passages.

Answer recall (an accepted answer appears in the top k passages, searching with the question; first 1,000 validation
questions of each):

| | Top-1 | Top-3 | Top-5 | Top-20 |
|---|---:|---:|---:|---:|
| TriviaQA (unfiltered) | 37.4% | 49.7% | 54.8% | 66.1% |
| NQ (open) | 25.0% | 37.7% | 43.9% | 60.4% |

Closed-book baseline: replies to the questions with no search, scored with the RL reward (+1 correct, 0 "I don't
know", -1 wrong; a reply without `####` counts as wrong). "Lenient" counts any accepted answer appearing anywhere in
the reply.

| Model | TriviaQA (first 2,000) score | lenient | NQ (3,610) score | lenient |
|---|---:|---:|---:|---:|
| sft-704 | -1.000 | 7.3% | -1.000 | 2.9% |
| rl-tinygsm | -0.999 | 6.2% | -0.999 | 2.4% |

Neither model ever abstains, and nearly every reply is a confident invention ("Jamie Lee Curtis is a space rock",
"the last time anyone was on the moon was on December 30, 2022").

## Search and abstention (results)

**Search SFT (`sft-search`):** from rl-tinygsm8 on 47.0M tokens: 40k search demonstrations (20k TriviaQA, 20k NQ
training questions), 100k TinyGSM problems and 20k smol-smoltalk conversations as replay. 721 steps at LR 1e-4,
35.6 min. The model searches for 98-99% of questions but abstains 87-88% of the time, about equally whether the
answer is in the retrieved passages (88% of 181 questions) or not (86% of 219): it learned abstention as a default,
not from reading the passages. TriviaQA score -0.089 (2.0% correct, 10.8% wrong), NQ -0.101. GSM8K 31.8%, SVAMP
48.8%, chat val loss 1.524.

**Mixed RL with +1 / 0 / -1 (`rl-mixed`, stopped):** half GSM8K and half TriviaQA/NQ problems per step. The QA
samples went from 45% abstaining to 100% within 19 steps, with correct answers at 0-7%, after which the QA groups
carried no signal. Stopped at step 25 (17 min, development compute).

**Curriculum RL (`rl-curriculum`):** from sft-search, wrong QA answers rewarded 0 for the first 60% of 380 steps,
then -1. Evals score +1 / 0 / -1 throughout. 201.1 min (3.4 h), 14.8GB peak (past the card's 12GB, so it spilled
into system memory).

| Step | Wrong reward | TriviaQA correct | abstain | wrong | score | NQ score | GSM8K | SVAMP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | | 1.2% | 87.0% | 11.8% | -0.105 | -0.092 | 34.00% | 44.00% |
| 100 | 0 | 8.0% | 6.8% | 85.2% | -0.772 | -0.848 | 36.75% | 43.25% |
| 200 | 0 | 8.8% | 1.0% | 90.2% | -0.815 | -0.917 | 35.50% | 46.25% |
| 250 | -1 | 6.2% | 40.8% | 53.0% | -0.468 | -0.507 | 36.00% | 45.50% |
| 300 | -1 | 0.0% | 99.5% | 0.5% | -0.005 | 0.000 | 38.00% | 48.25% |
| 380 (full sets) | -1 | 0.0% | 98.2% | 1.8% | -0.018 | -0.001 | 33.59% | **51.60%** |

(Steps 0-300 are the first 400 questions of each benchmark; the last row is all of them: TriviaQA 11,313, NQ 3,610,
GSM8K 1,319, SVAMP 1,000.)

- The bottleneck is reading, not the reward. Even when wrong answers cost nothing and the model answered nearly
  every question, only about 9% of its answers were right, although about half of TriviaQA questions have an
  accepted answer in the top 3 passages. Under +1 / 0 / -1, answering pays only above 50% confidence, so always
  abstaining is the best policy this model can reach, and RL found it within about 20 steps of the penalty starting.
- What it does achieve is the abstention half of the idea: confident wrong answers fall from about 100% (closed-book)
  to 1.8%. But the model abstains almost regardless of the question, rather than answering the ones it can.
- The math half of the mix kept improving: SVAMP 51.6% is the best so far, GSM8K 33.6%.

## Reading pilot (development compute, toward the second model)

**`sft-reading`:** SFT from sft-704 on 79.1M tokens: all 87,599 SQuAD v1.1 training questions as reading
demonstrations (the question's own paragraph shuffled among 2 BM25 passages from other articles, formatted as search
results; `qa.py --reading`), 40k TriviaQA/NQ search demonstrations and 20k smol-smoltalk conversations. 1,285 steps,
63.9 min.

The go/no-go check (`qa.evaluate_reading`): the reply is started with a search call and results that contain an
answer, and the model writes only the answer. "Precision" is correct / (correct + wrong).

| Model | Passages handed to it | Correct | Abstain | Wrong | Precision |
|---|---|---:|---:|---:|---:|
| sft-search | SQuAD validation paragraph + 2 distractors (2,000) | 1.2% | 88.3% | 10.6% | 10% |
| sft-reading | same | **39.6%** | 4.8% | 55.7% | 42% |
| sft-search | TriviaQA/NQ validation, real BM25 top 3 containing an answer (874) | 2.4% | 88.4% | 9.2% | 21% |
| sft-reading | same | **18.5%** | 63.5% | 18.0% | **51%** |

On the open TriviaQA eval (first 2,000) sft-reading gets 16.0% correct, 63.6% abstain, 20.5% wrong (score -0.045,
precision 44%), against 2.0% / 87.3% / 10.8% for sft-search; NQ 4.4% correct. Some "wrong" answers are near misses
under exact match ("Levi's Stadium in the San Francisco Bay Area" for "Levi's Stadium").

**`rl-reading-test`:** QA-only RL from sft-reading with +1 / 0 / -1 at temperature 1, 80 steps (64.9 min). It
collapsed to abstaining again: 50% abstain at step 1, 86% at step 8, 98% by step 16, 100% from step 32, ending at
99.8% on TriviaQA. Sampled at temperature 1, the model's answers are much worse than its greedy ones (9% correct vs
41% wrong at step 1, about 18% precision, against 44-51% greedy), so RL mostly sees wrong answers.

**`rl-reading-kl`:** the same at temperature 0.6 with a KL penalty toward sft-reading (`kl_beta=0.1`, k3 estimator on
the trained sequences), 50 steps, 27.3 min. Sampled precision started at 25-43%, and abstention rose more slowly,
but in the same direction:

| Step | TriviaQA correct | abstain | wrong | score | precision | NQ score |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 11.2% | 68.2% | 20.5% | -0.092 | 35% | -0.168 |
| 25 | 4.5% | 90.2% | 5.2% | -0.007 | 46% | -0.020 |
| 50 | 3.5% | 92.0% | 4.5% | -0.010 | 44% | -0.007 |

(first 400 questions of each)

- Reading data works: extraction from a handed paragraph goes from 1% to 40%, and greedy precision on real search
  results reaches about 50%. It belongs in the second model.
- QA RL with +1 / 0 / -1 doesn't. The score rises only because answering below 50% precision loses reward on
  average, so RL answers less: wrong answers fall from 20.5% to 4.5%, but two-thirds of the correct answers go too
  (11.2% to 3.5%), and precision barely moves. It shifts the answering threshold rather than improving the model's
  judgment of which questions it can answer, so QA RL is left out of the second model.

## Second model (`sft-v2` → `rl-v2`)

Post-training from long-704 again, with the first model's lessons: one combined SFT instead of separate chat and math
stages, SQuAD reading before search, and more TinyGSM. The SFT data is 8.41M conversations, 1.404B tokens: half of
smol-smoltalk (230 of 460 groups), TinyGSM shards 0-11, all 87,599 SQuAD reading demonstrations and 40k TriviaQA/NQ
search demonstrations. 21,685 steps at the sft-704 settings, 1,027.5 min (17.1 h) at about 22.7k tok/s; then math RL
as rl-tinygsm8 (400 steps × 32 GSM8K problems × 16), 137.1 min (2.3 h). No restarts. The data build ran out of memory
after writing the train set (it kept those arrays while building val); val was copied from sft-tinygsm8, which uses
the same 2,000 conversations, and prepare_sft.py now frees each split first.

| | rl-curriculum (first model) | sft-v2 | rl-v2 |
|---|---:|---:|---:|
| GSM8K | 33.59% | 36.39% | **41.17 ± 1.36%** |
| SVAMP | 51.6% | 52.0% | **57.5 ± 1.56%** |
| Reading, SQuAD paragraph handed (2,000): correct / abstain / wrong | 0.0 / 99.5 / 0.6% | 39.4 / 4.5 / 56.2% | 37.4 / 3.4 / 59.3% |
| Reading, real BM25 top 3 containing an answer (874) | 0.0 / 100 / 0.0% | 16.0 / 67.4 / 16.6% | 17.3 / 59.6 / 23.1% |
| TriviaQA (first 2,000): correct / abstain / wrong | 0.0 / 99.2 / 0.8% | 13.5 / 66.8 / 19.8% | 12.8 / 59.1 / 28.2% |
| TriviaQA score (+1 / 0 / -1) | -0.008 | -0.063 | -0.154 |
| NQ (3,610): correct / abstain / wrong | 0.0 / 99.9 / 0.1% | 3.9 / 79.1 / 17.1% | 4.3 / 69.1 / 26.6% |
| Chat val loss | 1.5871 | 1.6015 | 1.6481 |
| HellaSwag / PIQA / ARC-Easy / ARC-Challenge | 28.79 / 59.85 / 41.04 / 22.70% | 28.81 / 59.25 / 42.47 / 20.73% | 28.68 / 59.36 / 41.84 / 21.16% |
| Post-training GPU time | 21.0 h | 17.1 h | 19.4 h |

Math RL periodic evals (first 400): GSM8K 39.25% at step 0, 40.25-41.75% from step 50; SVAMP 50.50% at step 0,
49.75-55.25% after.

- More TinyGSM (12 shards vs 8) in a single combined SFT, then the same math RL, gives GSM8K +7.6 and SVAMP +5.9
  points over the first model with 1.6 h less post-training. Unlike rl-tinygsm8's, this math RL also lifts SVAMP.
- Reading carries over from the pilot: about 37-39% extraction from a handed paragraph, and 12.8% of TriviaQA answered
  correctly from the model's own searches, against 0% for the first model, which abstains on nearly everything.
- Math RL never sees QA, but it makes the model answer QA more and abstain less: wrong TriviaQA answers rise from
  19.8% to 28.2% and the +1 / 0 / -1 score falls from -0.063 to -0.154.
- Chat val loss is 0.06 above the first model's, with chat a smaller share of the data (about 10% of tokens).
  The log-probability benchmarks are flat between the two models.

### Abstention SFT from rl-v2 (`sft-abstain`)

Targets from the model's own answers (`qa.py --relabel`): rl-v2 answers 25k random TriviaQA and 25k NQ training
questions greedily with search. The ones it gets right keep its reply as the target (11.4% of TriviaQA, 4.8% of NQ),
and the rest get its search call and results followed by `#### I don't know`. In a 500-question pilot per dataset, the
greedy label agreed with the majority of 8 samples at T=0.6 on 97% of questions, so labels come from the greedy reply
only (a ninth of the cost). SFT from rl-v2 on these 50k with the search stage's replay (20k smol-smoltalk, 100k TinyGSM),
lr 1e-4, 801 steps. GPU time: 0.14 h pilot, 0.71 h for a relabel run that crashed after its first dataset, 1.62 h
relabelling, 0.87 h SFT with evals, 3.3 h in all (rl-v2's lineage is now at 22.7 of 24 h). A bug in the first full run
labelled NQ from one T=0.6 sample instead of the greedy reply; the 491 affected labels (2%) were repaired from the saved
replies before the SFT.

| | rl-v2 | sft-abstain |
|---|---:|---:|
| GSM8K | 41.17% | 35.41% |
| SVAMP | 57.5% | 52.1% |
| TriviaQA (first 2,000): correct / abstain / wrong | 12.8 / 59.1 / 28.2% | 5.6 / 90.4 / 4.0% |
| TriviaQA score (+1 / 0 / -1) | -0.154 | **+0.016** |
| NQ (3,610): correct / abstain / wrong | 4.3 / 69.1 / 26.6% | 0.8 / 96.9 / 2.3% |
| NQ score | -0.223 | -0.015 |
| Chat val loss | 1.6481 | **1.5924** |

- Wrong answers fall 7× on TriviaQA (28.2% → 4.0%) and 12× on NQ, while correct ones fall about 2× and 5×. Abstaining
  at random would cut both by the same factor, so the model learned something about which questions it can answer:
  when it answers TriviaQA it's right 58% of the time, against 31% for rl-v2. That is the first positive TriviaQA score.
- About half of rl-v2's correct TriviaQA answers are lost to abstention, more than the labels ask for (rl-v2 itself
  answered 11.4% of the training questions correctly).
- GSM8K and SVAMP fall about 6 points, back to sft-v2's level: the SFT undoes the math RL's gain.
- Chat val loss improves by 0.056, below sft-v2's, from the smol-smoltalk replay.

## Notes

- **19a:** BatchedMuon computes torch.optim.Muon's update but runs Newton-Schulz on same-shape matrices as one
  batched matmul: 99.6 ms per step instead of 173.5 ms on 16×704's matrices, and as close to an fp64 reference as
  torch's version (1.06-1.09% relative difference vs 1.09-1.36%, both bf16 rounding). At 12×512 it gives 58.4k tok/s
  vs 56.0-56.8k for 15a and 16a, and 1.1732 vs their 1.1765 average. The update is the same, so the gain is free
  speed and the BPB difference is mostly noise plus 3-4% more tokens. Generation now refills finished rows with the
  next prompt (continuous batching): 512 RL-style samples take 8.9 s instead of 20.8 s, and greedy GSM8K and SVAMP
  scores are unchanged within error.

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
- **Value residual accepted:** 0.0112 better than 1.1882, better on both sources, past the 0.01 margin for a single
  run. Every layer after the first blends the first layer's attention values into its own, with a learned weight per
  layer that starts at 0.5 (ResFormer). A second seed is running as a check.
- **At 120 minutes:** the 13d recipe leads 07a's by 0.0215, against 0.0283 at 45 minutes (1.1882 vs a 1.2165 average),
  so about three quarters of the gain holds at 2.7 times the length. 13d processed 31% more tokens.
- **16a:** value residual repeats on the second seed, 0.0123 better than 14a. The two seeds average 1.1765.
- **16b/16c:** probes for the long run, measuring speed and memory only. Peak memory is 8.4GB at 16×640 and 9.2GB at
  16×704, both inside 12GB. Their speeds include recompiles at each phase change, so they slightly understate a long run.
- **17a:** mix annealing ties at 0.0013 better, and all of it comes from FineMath (1.1031 vs 1.1081) while FineWeb-Edu
  is flat (1.2068 vs 1.2065). It shifts the model toward the source it sees more of at the end rather than improving it
  overall, so it's rejected. A first attempt hung on the GPU for 2.5 hours and was rerun from scratch.
- **17b-d:** at 16×704 the current LR of 1e-3 (AdamW and Muon) is best. 0.7× is 0.0122 worse and 1.4× is 0.0043
  worse, so the LR carries over from 12×512. Speed varied from 24.6k to 26.0k tok/s between identical shapes, so part
  of 1.4×'s gap is its 5% fewer tokens.

## Benchmarks

Scored by total log-probability of each option with `3060lm/evaluate.py`, as accuracy ± standard error.

| Benchmark | Chance | 00 baseline | 08a | 12b sampled softmax | long-704 (24h) |
|---|---:|---:|---:|---:|---:|
| HellaSwag (n=10,042) | 25.0% | 25.95 ± 0.44% | 26.37 ± 0.44% | 26.35 ± 0.44% | 28.62 ± 0.45% |
| PIQA (n=1,838) | 50.0% | 55.39 ± 1.16% | 56.96 ± 1.15% | 56.69 ± 1.16% | 62.13 ± 1.13% |
| ARC-Easy (n=2,376) | 25.0% | 34.26 ± 0.97% | 41.16 ± 1.01% | 42.21 ± 1.01% | 51.09 ± 1.03% |
| ARC-Challenge (n=1,172) | 25.0% | 16.81 ± 1.09% | 18.86 ± 1.14% | 17.75 ± 1.12% | 21.25 ± 1.19% |

Among the 45-minute runs, only the gain on ARC-Easy from baseline to 08a is clearly larger than its error, and 12b's lower
BPB doesn't show up on any benchmark. The 24-hour run clears its error on HellaSwag, PIQA and ARC-Easy; ARC-Challenge is
still below chance. Summed log-probability favors short options, and on
ARC-Challenge always picking the shortest option scores 22.7% and the longest 27.8%, which partly explains the
below-chance score. The models score below even the shortest-option rule, so length isn't the full explanation.
The bias applies to both models, so compare them with each other rather than reading these as absolute ability.

