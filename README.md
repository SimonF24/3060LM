# 3060LM

The goal of this repo is to train the best language model possible on a home desktop with a single NVIDIA RTX 3060, using at most about one day of training. Inspiration includes <a href="https://github.com/karpathy/nanochat/tree/master">nanochat</a>, <a href="https://github.com/KellerJordan/modded-nanogpt">the NanoGPT speedrun</a>, and <a href="https://github.com/openai/parameter-golf/">OpenAI Parameter Golf</a>.

The target is a small general-purpose chat model with a STEM bent, chosen because math and code give us tasks where answers can be checked automatically during RL post-training.

The original pipeline was written by hand. Since then, experiments have been run with Claude Code, starting from that hand-written model as the baseline and measuring every change against it.

## Status

Pretraining is done. The 24-hour run (16×704, 101M parameters, 2.15B tokens) reached **0.9838 bits per byte**, with
HellaSwag 28.6%, PIQA 62.1%, ARC-Easy 51.1% and ARC-Challenge 21.3%. Chat fine-tuning and RL haven't been done yet.

The recipe came from 45-minute speedruns, which took the score from a baseline of 1.3709 to **1.1765** (lower is better):

| Change | Weighted BPB |
|---|---:|
| Baseline: original model and recipe | 1.3709 |
| `torch.compile` | 1.3274 |
| LR schedule (gain unproven, within noise) | about 1.32 |
| Effective batch 32×512 | 1.2809 |
| Effective batch 64×512, peak LR 1e-3 | 1.2604 |
| Muon on hidden layers (average of 2 seeds) | 1.2305 |
| QK-norm (average of 2 seeds) | 1.2217 |
| ReLU² replaces SwiGLU (average of 2 seeds) | 1.2165 |
| Effective batch schedule (average of 2 seeds) | 1.2086 |
| Sampled softmax for the first 70% (average of 2 seeds) | 1.1919 |
| Sampled softmax until 85% (average of 2 seeds) | 1.1882 |
| Value residual (average of 2 seeds) | 1.1765 |

See [RESULTS.md](RESULTS.md) for every run, including the ones that didn't help, and `results.jsonl` for full configs and metrics.

[RESEARCH.md](RESEARCH.md) has the experiment backlog, the plans for long runs and RL, and notes on related work.

## How Experiments Work

Each experiment trains for a fixed **45 minutes** of wall-clock time, so speedups count toward the score, then reports its score on fixed held-out text.

- **Score:** weighted bits per byte (BPB) on the first 2M tokens of each source's validation set at 512 context, weighted 70% FineWeb-Edu and 30% FineMath. BPB stays comparable if the tokenizer changes, which raw loss doesn't.
- **Noise:** the same recipe with a different seed moved the score by 0.0061, so a single-run gain under 0.01 isn't treated as real without a repeat. That margin is a rough estimate from one pair of runs.
- **Limitation:** 45 minutes is only about 100M tokens, which favors whatever learns fastest early on. Model size in particular shouldn't be chosen from these scores alone, so the top recipes will get a longer check before the final run.

## Current Recipe

- **Model:** 12 layers, 512 dimensions, 8 heads and 48.2M parameters, with RoPE, ReLU² MLPs, RMSNorm, QK-norm and tied embeddings.
- **Optimizers:** Muon (LR 1e-3) on the hidden-layer weights, AdamW (LR 1e-3) on the embeddings and norm scales.
- **Schedule:** 200-step warmup, then a linear decay to zero over the last 40% of the time budget.
- **Batch:** effective 32×512, then 64×512 at 40% of the budget and 128×512 at 70%, as 16×512 with accumulation.

## Dataset

Experiments use a mix of general educational web text and math, sampled 70/30:

| Source | Train tokens used by speedruns |
|---|---:|
| <a href="https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu">FineWeb-Edu</a> | 770M |
| <a href="https://huggingface.co/datasets/HuggingFaceTB/finemath">FineMath-4+</a> | 306M |

The first 5,000 documents of each source are held out for validation. Text is tokenized with a custom 32k BPE tokenizer that puts each digit in its own token, which is intended to help with arithmetic.

The project originally used the <a href="https://huggingface.co/datasets/armanc/scientific_papers">scientific papers</a> dataset (arXiv and PubMed). It moved to this mix to better suit a general chat model and RL on math.

## Setup

Requires Python 3.12 and a CUDA GPU. `torch.compile` also needs the Python headers (`sudo apt install python3.12-dev` on Ubuntu).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd 3060lm

python download_data.py  # speedrun shards, about 3 GB (--all fetches about 10 GB for the final run)
python prepare_data.py   # trains the tokenizer, tokenizes every shard on disk

python train.py --name=my-run --compile=true --warmup_steps=200 --warmdown_frac=0.4 --beta2=0.95 \
    --batch_size=16 '--accum_schedule=[[0,2],[0.4,4],[0.7,8]]' --lr=1e-3 --muon_lr=1e-3 --qk_norm=true --activation=relu2 \
    --sampled_k=2048 --sampled_until=0.85 --value_residual=true
```

Any field of `Config` in `train.py` can be set with `--field=value`. Results are appended to `results.jsonl`, and logs and the final model go to `training_runs/<name>/`.

## Project Structure

```
3060lm/
├── download_data.py  - Download dataset shards
├── prepare_data.py   - Train the tokenizer and tokenize the data
├── model.py          - Model architecture, with experiment flags
├── train.py          - Speedrun training harness and BPB evaluation
├── evaluate.py       - HellaSwag, PIQA and ARC benchmarks for a checkpoint
│
│   Original arXiv/PubMed pipeline, not used by the current experiments:
├── eda.ipynb         - Exploratory data analysis
├── preprocessing.py  - Clean the raw papers
├── tokenizer.py      - Original tokenizer (model.py still imports its VOCAB_SIZE)
├── dataset.py        - Original data loader
└── pretrain.py       - Original training loop
```

## Roadmap

1. ~~**Choose model size and shape**~~: 16×704, with LR checked at that size.
2. ~~**Final pretraining run**~~: 24 hours, 0.9838 BPB.
3. **Chat fine-tuning (SFT):** teach the chat format and tool use.
4. **RL with verifiable rewards:** start with math, such as GSM8K with a calculator or restricted Python.
5. **RL with search and abstention (exploratory):** reward correct answers, penalize wrong ones, and give partial credit for "I don't know", to test whether a model too small to store many facts can make up for it by looking them up.

## System Specs

CPU: AMD Ryzen 5 3600  
GPU: NVIDIA GeForce RTX 3060 (12 GB)  
RAM: 32 GB (4 x TEAMGROUP-UD4-3200), with about 15 GB visible under WSL2 by default  
SSD: SPCC M.2 PCIe SSD (2 TB)
