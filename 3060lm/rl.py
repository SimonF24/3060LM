"""
RL with verifiable rewards on GSM8K's training split: on-policy policy gradient with group baselines (GRPO-style).

Each step samples `group` replies to each of `problems` training problems, with the calculator, at temperature 1 from
the full distribution. A reply's reward is 1 if it ends with '#### <correct answer>' and 0 otherwise, and its
advantage is its reward minus its group's mean, so groups where every reply scored the same give no signal. The loss
raises the log-probability of the tokens in better-than-average replies and lowers it in worse ones. Only tokens the
model sampled are trained (plus <|im_end|> when it chose to stop); the prompt and calculator results aren't. One
gradient step per batch keeps it on-policy, so there's no importance ratio or clipping.

Held-out GSM8K test and SVAMP problems are scored greedily every eval_every steps, and in full at the end, appended to
rl_results.jsonl.

Usage: python rl.py --name=rl-math [--init=../training_runs/sft-math/final.pt] [--any_config_field=value ...]
       python rl.py --name=rl-math --resume=true   # continues from training_runs/rl-math/latest.pt
"""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from chat import END, GSM8K_DIR, PAD, TOKENIZER_PATH, evaluate_math, extract_number, generate, prompt_ids
from model import Transformer, TransformerConfig
from train import build_optimizers, parse_args

ROOT = Path(__file__).parent.parent


@dataclass
class RLConfig:
    name: str = "rl"
    init: str = str(ROOT / "training_runs/sft-math/final.pt")
    steps: int = 300
    problems: int = 16  # training problems per step
    group: int = 16  # replies sampled per problem
    max_new_tokens: int = 320
    temperature: float = 1.0
    lr: float = 2e-5  # AdamW, constant after warmup
    muon_lr: float = 0.0  # 0 puts every parameter on AdamW; Muon's fixed-size updates suit noisy RL gradients poorly
    weight_decay: float = 0.0
    beta2: float = 0.95
    muon_momentum: float = 0.95
    warmup_steps: int = 10
    grad_clip: float = 1.0
    micro_batch: int = 8  # sequences per forward/backward pass
    gen_batch: int = 64  # sequences per generation batch
    eval_every: int = 50
    eval_limit: int = 400  # problems per benchmark in the periodic evals; the final eval scores all of them
    resume: bool = False  # continue from training_runs/<name>/latest.pt
    seed: int = 1337


def train_problems() -> list[tuple[str, float]]:
    rows = pq.read_table(GSM8K_DIR / "train-00000-of-00001.parquet").to_pylist()
    return [(r["question"], float(extract_number(r["answer"]))) for r in rows]


def reward(reply_text: str, gold: float) -> float:
    answer = extract_number(reply_text)
    return float(answer is not None and math.isclose(float(answer), gold, abs_tol=1e-4))


def policy_loss(model, batch: list[tuple[list[int], list[int], float]], n_tokens: int) -> torch.Tensor:
    """
    batch holds (token ids, trained flags per id, advantage). Returns the summed advantage-weighted negative
    log-probability of the trained tokens, divided by n_tokens (the trained tokens in the whole step).
    """
    width = max(len(ids) for ids, _, _ in batch)
    x = torch.full((len(batch), width), PAD, dtype=torch.long)
    weight = torch.zeros((len(batch), width))
    for r, (ids, trained, advantage) in enumerate(batch):
        x[r, :len(ids)] = torch.tensor(ids)
        weight[r, :len(ids)] = torch.tensor(trained, dtype=torch.float32) * advantage
    x, weight = x.cuda(), weight.cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(x[:, :-1])
    logp = -F.cross_entropy(logits.float().flatten(0, 1), x[:, 1:].flatten(), reduction="none")
    return -(logp * weight[:, 1:].flatten()).sum() / n_tokens


def main():
    cfg = parse_args(RLConfig)
    run_dir = ROOT / "training_runs" / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    ckpt = torch.load(cfg.init, map_location="cuda")
    model_cfg = TransformerConfig(**{k: v for k, v in ckpt["model_config"].items() if k != "head_dim"})
    model = Transformer(model_cfg).cuda()
    model.load_state_dict(ckpt["model"])
    print(f"{cfg.name}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters from {cfg.init}")
    optimizers = build_optimizers(model, cfg)

    problems = train_problems()
    rng = np.random.default_rng(cfg.seed)
    epochs = math.ceil(cfg.steps * cfg.problems / len(problems))
    order = np.concatenate([rng.permutation(len(problems)) for _ in range(epochs)])
    print(f"{len(problems):,} training problems, {cfg.steps} steps of {cfg.problems} x {cfg.group} replies "
          f"({cfg.steps * cfg.problems / len(problems):.2f} epochs)", flush=True)

    def evaluate(step: int, limit: int) -> dict:
        model.eval()
        scores = {}
        for benchmark in ["gsm8k", "svamp"]:
            result = evaluate_math(model, tokenizer, benchmark, limit=limit)
            scores[benchmark] = {k: v for k, v in result.items() if k != "examples"}
            writer.add_scalar(f"eval/{benchmark}", result["accuracy"], step)
        print(f"step {step} | eval (first {limit or 'all'}) | " + " | ".join(
            f"{b} {s['accuracy']:.2%} (lenient {s['lenient']:.2%}, formatted {s['formatted']:.0%})"
            for b, s in scores.items()), flush=True)
        return scores

    ckpt_path = run_dir / "latest.pt"
    start_step, prior_s, history = 0, 0.0, []
    if cfg.resume:
        state = torch.load(ckpt_path, map_location="cuda", weights_only=False)
        model.load_state_dict(state["model"])
        for (optimizer, _), opt_state in zip(optimizers, state["optimizers"]):
            optimizer.load_state_dict(opt_state)
        torch.cuda.set_rng_state(state["cuda_rng"])
        start_step, prior_s, history = state["step"], state["train_s"], state["history"]
        print(f"Resumed from step {start_step}, {prior_s / 60:.1f} min", flush=True)
    else:
        torch.manual_seed(cfg.seed)
        history.append({"step": 0, **evaluate(0, cfg.eval_limit)})

    heartbeat = run_dir / "heartbeat"
    t_start = time.perf_counter() - prior_s
    for step in range(start_step, cfg.steps):
        heartbeat.touch()
        lr_scale = min(1.0, (step + 1) / cfg.warmup_steps)
        for optimizer, base_lr in optimizers:
            for group in optimizer.param_groups:
                group["lr"] = base_lr * lr_scale
        t0 = time.perf_counter()
        batch = [problems[i] for i in order[step * cfg.problems:(step + 1) * cfg.problems]]
        prompts = [prompt_ids(tokenizer, [{"role": "user", "content": q}]) for q, _ in batch for _ in range(cfg.group)]
        model.eval()
        replies = generate(model, prompts, cfg.max_new_tokens, temperature=cfg.temperature, top_k=0,
                           batch_size=cfg.gen_batch, tokenizer=tokenizer, details=True)
        t_gen = time.perf_counter() - t0

        rewards = torch.tensor([reward(tokenizer.decode(r["tokens"]), batch[i // cfg.group][1])
                                for i, r in enumerate(replies)]).view(cfg.problems, cfg.group)
        advantages = (rewards - rewards.mean(1, keepdim=True)).flatten().tolist()
        sequences = []
        for prompt, reply, advantage in zip(prompts, replies, advantages):
            if advantage == 0:
                continue
            ids = prompt + reply["tokens"] + [END] * reply["ended"]
            trained = [0] * len(prompt) + reply["sampled"] + [1] * reply["ended"]
            sequences.append((ids, trained, advantage))

        model.train()
        n_tokens = sum(sum(trained) for _, trained, _ in sequences)
        loss_sum = 0.0
        for b in range(0, len(sequences), cfg.micro_batch):
            loss = policy_loss(model, sequences[b:b + cfg.micro_batch], n_tokens)
            loss.backward()
            loss_sum += loss.item()
        if sequences:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            for optimizer, _ in optimizers:
                optimizer.step()
        for optimizer, _ in optimizers:
            optimizer.zero_grad(set_to_none=True)

        mean_reward = rewards.mean().item()
        signal = int((rewards.std(1) > 0).sum())
        lengths = [len(r["tokens"]) for r in replies]
        calls = sum(tokenizer.decode(r["tokens"]).count("<<") + tokenizer.decode(r["tokens"]).count("\n>>> ")
                    for r in replies) / len(replies)
        writer.add_scalar("train/reward", mean_reward, step + 1)
        writer.add_scalar("train/groups_with_signal", signal, step + 1)
        writer.add_scalar("train/reply_tokens", sum(lengths) / len(lengths), step + 1)
        print(f"step {step + 1}/{cfg.steps} | reward {mean_reward:.3f} | groups with signal {signal}/{cfg.problems} | "
              f"tokens {sum(lengths) / len(lengths):.0f} | calls {calls:.1f} | ended {sum(r['ended'] for r in replies) / len(replies):.0%} | "
              f"loss {loss_sum:+.4f} | gen {t_gen:.0f}s train {time.perf_counter() - t0 - t_gen:.0f}s | "
              f"{(time.perf_counter() - t_start) / 60:.1f} min", flush=True)

        if (step + 1) % cfg.eval_every == 0 or step + 1 == cfg.steps:
            history.append({"step": step + 1, **evaluate(step + 1, cfg.eval_limit)})
            state = {"model": model.state_dict(), "optimizers": [o.state_dict() for o, _ in optimizers],
                     "cuda_rng": torch.cuda.get_rng_state(), "step": step + 1,
                     "train_s": time.perf_counter() - t_start, "history": history}
            torch.save(state, ckpt_path.with_suffix(".tmp"))
            ckpt_path.with_suffix(".tmp").replace(ckpt_path)
    train_min = (time.perf_counter() - t_start) / 60

    torch.save({"model": model.state_dict(), "model_config": asdict(model_cfg), "config": asdict(cfg)},
               run_dir / "final.pt")
    t0 = time.perf_counter()
    final = evaluate(cfg.steps, 0)
    result = {"name": cfg.name, "init": cfg.init, **final, "history": history, "steps": cfg.steps,
              "train_min": round(train_min, 1), "eval_s": round(time.perf_counter() - t0),
              "peak_mem_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2), "config": asdict(cfg)}
    with open(ROOT / "rl_results.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
