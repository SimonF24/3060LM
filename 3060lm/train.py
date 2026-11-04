"""
Speedrun training harness.

Trains for a fixed wall-clock budget on the STEM mix (see prepare_data.py), then reports
bits-per-byte on the fixed held-out val set of each source. Every experiment is judged by
the weighted val BPB this prints at the end, and the result is appended to results.jsonl.

Usage: python train.py --name=baseline --time_budget_min=45 [--any_config_field=value ...]
"""
from dataclasses import asdict, dataclass, field, fields
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from model import Transformer, TransformerConfig

ROOT = Path(__file__).parent.parent


@dataclass
class Config:
    name: str = "debug"
    time_budget_min: float = 45.0  # training wall-clock budget, excludes evaluation
    data_dir: str = str(ROOT / "data/tokenized")
    mix: dict = field(default_factory=lambda: {"fineweb-edu": 0.7, "finemath": 0.3})
    vocab_size: int = 32_768
    # model
    model_dim: int = 512
    num_layers: int = 12
    num_heads: int = 8
    num_kv_heads: int = 0  # 0 means standard multi-head attention
    qk_norm: bool = False
    zero_init_residual: bool = False
    tie_embeddings: bool = True
    # optimization
    batch_size: int = 16
    seq_len: int = 512
    lr: float = 1e-4
    weight_decay: float = 0.01
    grad_accum: int = 1  # micro-batches per optimizer step, the effective batch is batch_size * grad_accum
    grad_clip: float = 1.0
    beta2: float = 0.999
    muon_lr: float = 0.0  # if > 0, 2D hidden layer weights use Muon at this LR, everything else stays on AdamW
    muon_momentum: float = 0.95
    warmup_steps: int = 0
    warmdown_frac: float = 0.0  # fraction of the time budget spent linearly decaying the LR to 0
    # speed
    compile: bool = False
    # eval
    eval_seq_len: int = 512  # fixed across experiments so BPB numbers stay comparable
    eval_batch_size: int = 32
    eval_max_tokens: int = 2_000_000  # per source
    log_every: int = 50
    seed: int = 1337


def parse_args() -> Config:
    cfg = Config()
    types = {f.name: f.type for f in fields(Config)}
    for arg in sys.argv[1:]:
        key, value = arg.removeprefix("--").split("=", 1)
        if key not in types:
            raise ValueError(f"Unknown config field {key}")
        current = getattr(cfg, key)
        setattr(cfg, key, json.loads(value) if isinstance(current, (dict, list, bool)) else type(current)(value))
    return cfg


class MixedSource:
    """Samples random windows from several memmapped token files with fixed mixture weights."""

    def __init__(self, data_dir: str, mix: dict, seed: int):
        self.names = list(mix)
        self.weights = np.array([mix[n] for n in self.names], dtype=np.float64)
        self.weights /= self.weights.sum()
        self.data = {n: np.memmap(Path(data_dir) / n / "train.bin", dtype=np.uint16, mode="r") for n in self.names}
        self.rng = np.random.default_rng(seed)

    def get_batch(self, batch_size: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        sources = self.rng.choice(len(self.names), size=batch_size, p=self.weights)
        rows = []
        for s in sources:
            arr = self.data[self.names[s]]
            i = self.rng.integers(0, len(arr) - seq_len - 1)
            rows.append(arr[i:i + seq_len + 1])
        batch = torch.from_numpy(np.stack(rows).astype(np.int64)).pin_memory()
        batch = batch.to("cuda", non_blocking=True)
        return batch[:, :-1], batch[:, 1:]


@torch.no_grad()
def evaluate_bpb(model, cfg: Config) -> dict:
    """
    Exact bits-per-byte on each source's val set (or its first eval_max_tokens tokens).
    Windows overlap by one token so every token after the first is predicted exactly once,
    which means summed loss in bits divided by the val text's UTF-8 byte count is the true BPB.
    """
    model.eval()
    results = {}
    for name in cfg.mix:
        meta = json.loads((Path(cfg.data_dir) / name / "meta.json").read_text())
        val = np.memmap(Path(cfg.data_dir) / name / "val.bin", dtype=np.uint16, mode="r")
        n = min(len(val), cfg.eval_max_tokens)
        # bytes covered by the evaluated prefix (proportional estimate when truncated)
        n_bytes = meta["val_bytes"] * (n / len(val))
        L = cfg.eval_seq_len
        starts = list(range(0, n - 1, L))
        total_nats = 0.0
        for b in range(0, len(starts), cfg.eval_batch_size):
            chunk = starts[b:b + cfg.eval_batch_size]
            ends = [min(s + L + 1, n) for s in chunk]
            width = max(e - s for s, e in zip(chunk, ends))
            x = torch.zeros(len(chunk), width, dtype=torch.long)
            mask = torch.zeros(len(chunk), width - 1, dtype=torch.bool)
            for j, (s, e) in enumerate(zip(chunk, ends)):
                x[j, :e - s] = torch.from_numpy(val[s:e].astype(np.int64))
                mask[j, :e - s - 1] = True
            x, mask = x.cuda(), mask.cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x[:, :-1])
            loss = F.cross_entropy(logits.float().flatten(0, 1), x[:, 1:].flatten(), reduction="none")
            total_nats += (loss * mask.flatten()).sum().item()
        results[name] = total_nats / math.log(2) / n_bytes
    weights = {n: w / sum(cfg.mix.values()) for n, w in cfg.mix.items()}
    results["weighted"] = sum(weights[n] * results[n] for n in cfg.mix)
    model.train()
    return results


def build_optimizers(model, cfg: Config) -> list[tuple[torch.optim.Optimizer, float]]:
    """Returns (optimizer, base LR) pairs. Embeddings, the output projection and the 1D RMSNorm scales always use AdamW."""
    adamw_params, muon_params = [], []
    for name, p in model.named_parameters():
        use_muon = (cfg.muon_lr > 0 and p.ndim == 2
                    and "token_embedding" not in name and "vocab_projection" not in name)
        (muon_params if use_muon else adamw_params).append(p)
    optimizers = [(torch.optim.AdamW(adamw_params, lr=cfg.lr, weight_decay=cfg.weight_decay,
                                     betas=(0.9, cfg.beta2), fused=True), cfg.lr)]
    if muon_params:
        # match_rms_adamw scales the update to 0.2 * lr per entry, so LRs are on a similar scale to AdamW
        optimizers.append((torch.optim.Muon(muon_params, lr=cfg.muon_lr, weight_decay=0.0,
                                            momentum=cfg.muon_momentum, adjust_lr_fn="match_rms_adamw"),
                           cfg.muon_lr))
    print(f"AdamW: {sum(p.numel() for p in adamw_params) / 1e6:.1f}M params, "
          f"Muon: {sum(p.numel() for p in muon_params) / 1e6:.1f}M params")
    return optimizers


def lr_schedule(step: int, progress: float, cfg: Config) -> float:
    """LR multiplier: linear warmup over steps, then constant, then linear decay to 0 over the
    last warmdown_frac of the wall-clock budget (time based, since runs have a fixed time budget)"""
    warmup = min(1.0, (step + 1) / cfg.warmup_steps) if cfg.warmup_steps > 0 else 1.0
    if cfg.warmdown_frac > 0:
        warmdown = min(1.0, (1.0 - progress) / cfg.warmdown_frac)
    else:
        warmdown = 1.0
    return warmup * max(warmdown, 0.0)


def main():
    cfg = parse_args()
    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    run_dir = ROOT / "training_runs" / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))

    model_cfg = TransformerConfig(vocab_size=cfg.vocab_size, model_dim=cfg.model_dim,
                                  num_layers=cfg.num_layers, num_heads=cfg.num_heads, num_kv_heads=cfg.num_kv_heads,
                                  max_seq_len=max(cfg.seq_len, cfg.eval_seq_len),
                                  qk_norm=cfg.qk_norm, zero_init_residual=cfg.zero_init_residual,
                                  tie_embeddings=cfg.tie_embeddings)
    model = Transformer(model_cfg).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{cfg.name}: {n_params / 1e6:.1f}M parameters")
    optimizers = build_optimizers(model, cfg)
    train_model = torch.compile(model) if cfg.compile else model
    data = MixedSource(cfg.data_dir, cfg.mix, cfg.seed)

    budget_s = cfg.time_budget_min * 60
    step, tokens, train_time = 0, 0, 0.0
    loss_ema = None
    torch.cuda.synchronize()
    while train_time < budget_s:
        t0 = time.perf_counter()
        lr_scale = lr_schedule(step, train_time / budget_s, cfg)
        for optimizer, base_lr in optimizers:
            for group in optimizer.param_groups:
                group["lr"] = base_lr * lr_scale
        for _ in range(cfg.grad_accum):
            inputs, targets = data.get_batch(cfg.batch_size, cfg.seq_len)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = train_model(inputs)
                loss = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten())
            (loss / cfg.grad_accum).backward()
            tokens += inputs.numel()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        for optimizer, _ in optimizers:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        step += 1
        if step % cfg.log_every == 0:
            loss_val = loss.item()  # syncs, so timing below is accurate
            loss_ema = loss_val if loss_ema is None else 0.9 * loss_ema + 0.1 * loss_val
            writer.add_scalar("train/loss", loss_val, tokens)
            writer.add_scalar("train/grad_norm", grad_norm.item(), tokens)
        else:
            loss_val = None
        train_time += time.perf_counter() - t0
        if loss_val is not None and step % (cfg.log_every * 10) == 0:
            print(f"step {step} | {train_time / 60:.1f}/{cfg.time_budget_min:.0f} min | "
                  f"loss {loss_ema:.4f} | {tokens / train_time:,.0f} tok/s", flush=True)

    t0 = time.perf_counter()
    bpb = evaluate_bpb(model, cfg)
    eval_time = time.perf_counter() - t0
    result = {
        "name": cfg.name, "val_bpb": bpb, "params_M": round(n_params / 1e6, 1),
        "steps": step, "tokens_M": round(tokens / 1e6, 1),
        "tok_per_s": round(tokens / train_time), "train_min": round(train_time / 60, 1),
        "eval_s": round(eval_time), "peak_mem_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "config": asdict(cfg),
    }
    for k, v in bpb.items():
        writer.add_scalar(f"val_bpb/{k}", v, tokens)
    print(json.dumps({k: v for k, v in result.items() if k != "config"}, indent=2))
    with open(ROOT / "results.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")
    torch.save({"model": model.state_dict(), "model_config": asdict(model_cfg), "config": asdict(cfg)},
               run_dir / "final.pt")


if __name__ == "__main__":
    main()
