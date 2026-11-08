"""
Supervised fine-tuning (SFT): teaches a pretrained checkpoint the chat format from chat.py.

Trains on the packed rows from prepare_sft.py with loss only on assistant replies, at a longer context than
pretraining (RoPE positions past 512 are new to the model, and SFT has to teach them). Reports masked val loss
before and after, then GSM8K test accuracy, and appends the result to sft_results.jsonl.

Usage: python sft.py --name=sft-704 [--init=../training_runs/long-704/final.pt] [--any_config_field=value ...]
"""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time

import numpy as np
from tokenizers import Tokenizer
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from chat import TOKENIZER_PATH, evaluate_gsm8k
from model import Transformer, TransformerConfig
from train import build_optimizers, parse_args

ROOT = Path(__file__).parent.parent


@dataclass
class SFTConfig:
    name: str = "sft"
    init: str = str(ROOT / "training_runs/long-704/final.pt")
    data_dir: str = str(ROOT / "data/sft")
    batch_size: int = 8  # rows of seq_len + 1 tokens per micro-batch
    grad_accum: int = 8  # 64 rows of 1024, about 65k tokens per step, the same as the end of pretraining
    epochs: float = 1.0
    lr: float = 3e-4  # peak, for AdamW (embeddings, head, norms)
    muon_lr: float = 3e-4  # peak, for Muon (hidden matrices)
    weight_decay: float = 0.0
    beta2: float = 0.95
    muon_momentum: float = 0.95
    warmup_steps: int = 100  # then linear decay to zero over the remaining steps
    grad_clip: float = 1.0
    compile: bool = True
    log_every: int = 50
    eval_every: int = 1000  # steps between val loss evals
    gsm8k_limit: int = 0  # 0 scores the whole GSM8K test set
    seed: int = 1337


@torch.no_grad()
def val_loss(model, ids: np.ndarray, mask: np.ndarray, batch_size: int) -> float:
    """Mean loss per assistant token on the val rows."""
    model.eval()
    total, count = 0.0, 0
    for b in range(0, len(ids), batch_size):
        x = torch.from_numpy(ids[b:b + batch_size].astype(np.int64)).cuda()
        m = torch.from_numpy(mask[b:b + batch_size, 1:].astype(np.float32)).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x[:, :-1])
        loss = F.cross_entropy(logits.float().flatten(0, 1), x[:, 1:].flatten(), reduction="none")
        total += (loss * m.flatten()).sum().item()
        count += m.sum().item()
    model.train()
    return total / count


def main():
    cfg = parse_args(SFTConfig)
    torch.manual_seed(cfg.seed)
    run_dir = ROOT / "training_runs" / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))

    data = Path(cfg.data_dir)
    meta = json.loads((data / "meta.json").read_text())
    train_ids, train_mask = np.load(data / "train_ids.npy", mmap_mode="r"), np.load(data / "train_mask.npy", mmap_mode="r")
    val_ids, val_mask = np.load(data / "val_ids.npy"), np.load(data / "val_mask.npy")
    seq_len = meta["seq_len"]

    ckpt = torch.load(cfg.init, map_location="cuda")
    model_config = {k: v for k, v in ckpt["model_config"].items() if k != "head_dim"}
    model_config["max_seq_len"] = seq_len  # RoPE tables aren't saved, so they're rebuilt at the new length
    model_cfg = TransformerConfig(**model_config)
    model = Transformer(model_cfg).cuda()
    model.load_state_dict(ckpt["model"])
    print(f"{cfg.name}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters from {cfg.init}")
    optimizers = build_optimizers(model, cfg)
    train_model = torch.compile(model) if cfg.compile else model

    rows_per_step = cfg.batch_size * cfg.grad_accum
    total_steps = int(cfg.epochs * len(train_ids) / rows_per_step)
    rng = np.random.default_rng(cfg.seed)
    order = np.concatenate([rng.permutation(len(train_ids)) for _ in range(math.ceil(cfg.epochs))])
    print(f"{len(train_ids):,} train rows, {total_steps:,} steps of {rows_per_step} rows", flush=True)

    initial_val = val_loss(model, val_ids, val_mask, cfg.batch_size)
    print(f"step 0 | val loss {initial_val:.4f}", flush=True)
    writer.add_scalar("val/loss", initial_val, 0)

    heartbeat, last_beat = run_dir / "heartbeat", 0.0
    t_start = time.perf_counter()
    for step in range(total_steps):
        lr_scale = min(1.0, (step + 1) / cfg.warmup_steps) * (1 - step / total_steps)
        for optimizer, base_lr in optimizers:
            for group in optimizer.param_groups:
                group["lr"] = base_lr * lr_scale
        step_rows = order[step * rows_per_step:(step + 1) * rows_per_step]
        # Normalize by the assistant tokens in the whole step, so every trained token counts equally
        n_trained = int(train_mask[np.sort(step_rows)][:, 1:].sum())
        step_loss = 0.0
        for micro in range(cfg.grad_accum):
            rows = np.sort(step_rows[micro * cfg.batch_size:(micro + 1) * cfg.batch_size])
            x = torch.from_numpy(train_ids[rows].astype(np.int64)).pin_memory().cuda(non_blocking=True)
            m = torch.from_numpy(train_mask[rows, 1:].astype(np.float32)).pin_memory().cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = train_model(x[:, :-1])
            loss = F.cross_entropy(logits.float().flatten(0, 1), x[:, 1:].flatten(), reduction="none")
            loss = (loss * m.flatten()).sum() / n_trained
            loss.backward()
            step_loss += loss.detach()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        for optimizer, _ in optimizers:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if time.time() - last_beat > 30:
            heartbeat.touch()
            last_beat = time.time()
        if (step + 1) % cfg.log_every == 0:
            elapsed = time.perf_counter() - t_start
            writer.add_scalar("train/loss", step_loss.item(), step + 1)
            print(f"step {step + 1}/{total_steps} | loss {step_loss.item():.4f} | {elapsed / 60:.1f} min | "
                  f"{(step + 1) * rows_per_step * seq_len / elapsed:,.0f} tok/s", flush=True)
        if (step + 1) % cfg.eval_every == 0:
            v = val_loss(model, val_ids, val_mask, cfg.batch_size)
            writer.add_scalar("val/loss", v, step + 1)
            print(f"step {step + 1} | val loss {v:.4f}", flush=True)
    train_min = (time.perf_counter() - t_start) / 60

    final_val = val_loss(model, val_ids, val_mask, cfg.batch_size)
    print(f"final val loss {final_val:.4f}", flush=True)
    torch.save({"model": model.state_dict(), "model_config": asdict(model_cfg), "config": asdict(cfg)},
               run_dir / "final.pt")
    model.eval()
    t0 = time.perf_counter()
    gsm8k = evaluate_gsm8k(model, Tokenizer.from_file(str(TOKENIZER_PATH)), limit=cfg.gsm8k_limit)
    print(json.dumps(gsm8k, indent=2), flush=True)
    result = {"name": cfg.name, "init": cfg.init, "val_loss_before": round(initial_val, 4),
              "val_loss": round(final_val, 4), "gsm8k": {k: v for k, v in gsm8k.items() if k != "examples"},
              "steps": total_steps, "train_min": round(train_min, 1), "gsm8k_eval_s": round(time.perf_counter() - t0),
              "peak_mem_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2), "config": asdict(cfg)}
    with open(ROOT / "sft_results.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
