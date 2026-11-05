# Research Notes

The plan and the ideas behind it. Measured scores are in [RESULTS.md](RESULTS.md). When a backlog item gets run, move its outcome there and remove it here.

## Speedrun Backlog

In rough priority order. Each item gets a 45-minute run against the current best recipe, and a single-run gain under 0.01 BPB needs a second seed before it's accepted.

| Idea | Why try it | Source |
|---|---|---|
| Value skip connections | Part of a large record jump, though embedding skips alone tied for us (10e) | NanoGPT speedrun #9 |
| Momentum warmup | Same bundle as above | NanoGPT speedrun #9 |
| Multi-token prediction | Extra training signal per token | NanoGPT speedrun #53 |
| Untie the head at 2/3 of training | Untied from the start lost for us, but this may get the benefit without the early cost | NanoGPT speedrun #53 |
| Sampled softmax during training | The head is about 26% of forward FLOPs | NanoGPT speedrun |
| Smaller vocabulary (8k-16k) | Cuts head cost directly, but means more tokens per byte and needs retokenizing | Parameter Golf |
| Profile the throughput gap | Effective batch 32 ran at 43.1k tok/s and 64 at 40.5k, for an unknown reason | Our runs |
| Profile real GPU time per component | The compute shares here are FLOP estimates, not measurements | Our runs |
| NVIDIA ClimbMix dataset | Cut nanochat's time-to-GPT-2 from 2.76 to 2.02 hours, but needs its own comparison | nanochat |

**Revisit on longer runs, since 45 minutes can't settle them:** model size and shape, untied embeddings, GQA, wider MLPs.

**Low priority:**
- **Linear attention and SSMs:** attention itself is only about 10% of forward FLOPs at 512 context, and lookup precision matters for the search idea. Worth another look if contexts grow well beyond 512, likely as a hybrid.
- **Test-time looping:** looped layers lost in pretraining (07d, 07e). Could still be tested for RL, where more thinking per token might help.

## Long-Run Plan

A 24-hour run sees about 30 times the tokens of a speedrun, and speedruns favor whatever learns fastest early on.

**Open decision, after the backlog:** two 24-hour runs at 16×640 and 12×512, or one at 16×640.

| Shape | Parameters | Estimated tokens in 24h | Tokens per parameter |
|---|---:|---:|---:|
| 12×512 | 48M | 3.8B | 79 |
| 16×640 | 84M | 2.2B | 26 |
| 16×704 | 101M | 1.8B | 18 |

Token counts are estimated from FLOPs and need checking, and memory for the larger shapes is untested. The Chinchilla rule of thumb suggests about 20 tokens per parameter.

Ready to use: checkpoint and resume, periodic val BPB, and 4.17B tokens in `data/tokenized-full` with the same val sets as the speedruns.

## Post-Training Plan

1. **SFT:** teach the chat format and tool syntax.
2. **Tools on math:** calculator or restricted Python on GSM8K, the most proven part of the plan.
3. **Search with abstention (exploratory):** see below.
4. **Live web:** a demo only, once the behavior works offline.

### Search With Abstention

The idea: give the model search and code tools, then use RL that rewards correct answers, penalizes wrong ones and gives partial credit for "I don't know". This could limit hallucination and reduce how many facts the weights must store.

**The open question:** nothing we found tests whether a model too small to store many facts can make up for it with retrieval. The smallest models in that work are 1-2B parameters, and ours would be about 0.1B. The experiment to run is closed-book vs. open-book accuracy as model size shrinks. This came from a few searches, so it isn't proof that nobody has tried it.

**Design decisions so far:**
- **Search:** use an offline Wikipedia index with BM25 for RL, because live search is slow, rate-limited and not reproducible.
- **Rewards:** start with short-answer datasets that can be checked (TriviaQA, Natural Questions, HotpotQA). Free-form claims would need a claim checker.
- **Confidence threshold:** with +1 correct and 0 abstain, a penalty of t/(1−t) for a wrong answer means the model should answer only above confidence t. So −1 gives 50%, −3 gives 75% and −9 gives 90%.
- **Failure modes:** start with a mild penalty to avoid collapse into always abstaining, and add a small cost per search to prevent spam. Include decline examples, because RL can erode abstention.
- **Retrieved text:** mask it during training for stability, as Search-R1 does.
- **Code execution:** sandboxed, with no network or file access.
- **Metrics:** precision when answering, coverage, and closed-book vs. open-book accuracy.
- **Early go/no-go check:** after SFT, measure how often the model extracts the right answer when handed the correct passage. If that's poor, RL won't save it.

### Expectations for RL

Models much larger than ours still score low on GSM8K: about 16.5% at 1.38B parameters and 2.2% at 1.12B, per their model cards, which we haven't verified. Treat RL at about 100M parameters as an experiment that could fail. Single-GPU RL also reportedly diverged with linear LR decay and needed cosine, so start with cosine.

## Prior Art

Reviewed on 2026-11-04. The field doesn't move fast enough to need a re-check for every experiment. Only the first three were read directly, and the rest are from search excerpts.

**Speedruns and challenges**
- [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt): 8×H100 speedrun. Its record table is the best list of techniques, but many records bundle several changes.
- [Parameter Golf](https://github.com/openai/parameter-golf/): 16MB size limit, scored in BPB.
- [nanochat](https://github.com/karpathy/nanochat): full pipeline, including a Python tool module and the CORE benchmark.

**Not applicable to us:** FP8 and Flash Attention 3 (need newer GPUs), multi-GPU communication, quantization (for Golf's size limit), and test-time training (helps compression scores, not a chat model).

**Small models on limited hardware**
- [Cramming](https://proceedings.mlr.press/v202/geiping23a/geiping23a.pdf): one consumer GPU for one day, but a masked model, not a causal one.
- [Marcolini nanochat-d24](https://huggingface.co/Marcolini/nanochat-d24-base-champion): 910M base model on a 12GB RTX 4070, about 465 GPU-hours.
- [Nekochu nanochat-d24](https://huggingface.co/Nekochu/nanochat-d24): 1.38B full pipeline on an RTX 5090, with GSM8K rising from 10.2% to 16.5% through RL.
- [Pretraining and RL study](https://arxiv.org/pdf/2504.07912.pdf): from-scratch 150M and 1B models used to study RL on math.
- [mathLM](https://github.com/samitmohan/mathLM): 40M parameters with pretraining, SFT and GRPO. Closest to our size.
- [TinyMathReason-1B](https://huggingface.co/himanshunakrani9/TinyMathReason-1B-grpo): 1.12B, with GSM8K reaching 2.2% and mode collapse in its RL traces.

**RL for search and abstention**
- [To Answer or to Abstain](https://arxiv.org/pdf/2607.10738): abstention-aware RL for search agents. Closest to our idea.
- [TruthRL](https://www.emergentmind.com/papers/2509.25760): GRPO with +1, 0 and −1 rewards.
- [MASH](https://arxiv.org/pdf/2510.01152v1): charges for each search, so searching tracks what the model doesn't know.
- [Small Models Learn to Call](https://neurips.cc/virtual/2026/171213): at 1-2B, RL improved tool calling but eroded abstention, which decline examples and a light call penalty largely fixed.
- [Search-R1](https://arxiv.org/abs/2503.09516): the established recipe for search RL.
- [Behaviorally Calibrated RL](https://arxiv.org/html/2512.19920v1): trains across risk thresholds, and reports smaller models calibrating well.
- [KARL](https://dev.to/gentic_news/karl-rl-framework-cuts-llm-hallucinations-without-accuracy-loss-414c): estimates what the model knows, so abstention is only rewarded where appropriate. Secondary write-up.
- [Why Language Models Hallucinate](https://openai.com/blog/why-language-models-hallucinate/): the case that scoring which rewards guessing drives hallucination.
