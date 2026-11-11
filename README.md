# 3060LM

The goal of this repo is to train the best language model possible on a home desktop with a single NVIDIA RTX 3060, using at most about one day of training. Inspiration includes <a href="https://github.com/karpathy/nanochat/tree/master">nanochat</a>, <a href="https://github.com/KellerJordan/modded-nanogpt">the NanoGPT speedrun</a>, and <a href="https://github.com/openai/parameter-golf/">OpenAI Parameter Golf</a>.

The target is a small general-purpose chat model with a STEM bent, chosen because math and code give us tasks where answers can be checked automatically during RL post-training.

The original pipeline was written by hand. Since then, experiments have been run with Claude Code, starting from that hand-written model as the baseline and measuring every change against it.

## Status

Pretraining is done. The 24-hour run (16×704, 101M parameters, 2.15B tokens) reached **0.9838 bits per byte**, with
HellaSwag 28.6%, PIQA 62.1%, ARC-Easy 51.1% and ARC-Challenge 21.3%.

Chat fine-tuning (SFT, 3.8 hours) took masked val loss from 2.218 to 1.477. The model follows the chat format
and writes fluent replies, but facts and arithmetic are often wrong, and GSM8K is 0% (1.3% when the last number is
taken as the answer). A short math stage with calculator calls taught the `#### N` format and tool use, which gives
GSM8K 1.8% and SVAMP 2.9%, but the model still mostly sets up the wrong calculations. RL (2.5 hours) then raised GSM8K to 3.9% and
held-out SVAMP to 12.5%, mostly by making replies finish instead of looping. SFT on 109M tokens of TinyGSM's Python
solutions, with a sandboxed Python tool, then reached GSM8K 10.9% and SVAMP 26.6% without RL, and RL
from there reached GSM8K 16.6% and SVAMP 29.5%.

Distilling from SmolLM2-360M or -1.7B loses to direct training at the same 45-minute budget (1.2529 and 1.4247
vs 1.1883 BPB), because the teacher's forward passes leave the student with 31% and 12% of the tokens.

The recipe came from 45-minute speedruns, which took the score from a baseline of 1.3709 to **1.1732** (lower is better):

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
| Batched Muon: the same update, 1.7x faster (one run; free speed) | 1.1732 |

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

Chat fine-tuning (SFT) uses smol-smoltalk plus GSM8K's training split, at 1024 context, with loss only on the assistant's replies.
A second, short stage trains on GSM8K with its `<<48/2=24>>` annotations as calculator calls: the model writes `<<48/2=`,
generation appends `24>>`, and that inserted part isn't trained on. SVAMP (all 1,000 problems, never trained on)
checks math beyond GSM8K's style:

```bash
python prepare_sft.py                        # tokenizes and packs data/sft/ (needs data/raw/smol-smoltalk and data/raw/gsm8k)
python sft.py --name=sft-704                 # from training_runs/long-704/final.pt, results go to sft_results.jsonl
python prepare_sft.py --out=sft-math --calculator=1 --gsm8k_repeats=3 --smoltalk_groups=10   # math stage data
python sft.py --name=sft-math --init=../training_runs/sft-704/final.pt --data_dir=../data/sft-math \
    --grad_accum=4 --lr=1e-4 --muon_lr=1e-4 --warmup_steps=30   # math stage: GSM8K with calculator calls
python chat.py ../training_runs/sft-math/final.pt                   # chat in the terminal
python chat.py ../training_runs/sft-math/final.pt --gsm8k --svamp   # math accuracy, with the calculator
python prepare_sft.py --out=sft-tinygsm --tinygsm_shards=1 --gsm8k_repeats=0 --smoltalk_groups=20   # TinyGSM, Python tool
python sft.py --name=sft-tinygsm --init=../training_runs/sft-704/final.pt --data_dir=../data/sft-tinygsm
python rl.py --name=rl-math --problems=32   # RL on GSM8K train from sft-math, results go to rl_results.jsonl
```

Distillation from a Hugging Face teacher (its forward passes count against the time budget). The data has to use the
teacher's tokenizer:

```bash
python prepare_data.py --speedrun-shards --tokenizer=data/teachers/SmolLM2-360M/tokenizer.json --out=data/tokenized-smollm2
python train.py --name=distill-360m <recipe flags without sampled_k> --data_dir=../data/tokenized-smollm2 --vocab_size=49152 \
    --teacher=data/teachers/SmolLM2-360M --distill_alpha=0.5 --batch_size=8 '--accum_schedule=[[0,4],[0.4,8],[0.7,16]]'
```

## Project Structure

```
3060lm/
├── download_data.py  - Download pretraining dataset shards
├── prepare_data.py   - Train the tokenizer and tokenize the pretraining data
├── model.py          - Model architecture, with experiment flags
├── train.py          - Pretraining harness and BPB evaluation
├── evaluate.py       - HellaSwag, PIQA and ARC benchmarks for a checkpoint
├── prepare_sft.py    - Tokenize and pack the chat fine-tuning data
├── sft.py            - Chat fine-tuning
├── chat.py           - Chat format, tools, cached generation, terminal chat and math evals
├── pytool.py         - Restricted Python executor for the Python tool
└── rl.py             - RL with verifiable rewards on GSM8K
```

The original arXiv/PubMed pipeline (EDA notebook, preprocessing, tokenizer, data loader and training loop) is on the
`main` branch and in git history.

## Roadmap

1. ~~**Choose model size and shape**~~: 16×704, with LR checked at that size.
2. ~~**Final pretraining run**~~: 24 hours, 0.9838 BPB.
3. ~~**Chat fine-tuning (SFT)**~~: chat format and calculator calls learned; GSM8K 1.8%, SVAMP 2.9%.
4. ~~**RL with verifiable rewards**~~: after TinyGSM SFT with a Python tool, GSM8K 16.6%, SVAMP 29.5%.
5. **RL with search and abstention (exploratory):** reward correct answers, penalize wrong ones, and give partial credit for "I don't know", to test whether a model too small to store many facts can make up for it by looking them up.

## System Specs

CPU: AMD Ryzen 5 3600  
GPU: NVIDIA GeForce RTX 3060 (12 GB)  
RAM: 32 GB (4 x TEAMGROUP-UD4-3200), with about 15 GB visible under WSL2 by default  
SSD: SPCC M.2 PCIe SSD (2 TB)
