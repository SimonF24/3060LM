"""
Scores a checkpoint on multiple-choice benchmarks, by likelihood: each option is appended to the
question, and the model's answer is the option with the highest total log-probability.

Results are appended to benchmarks.jsonl. Small pretrained models score near chance on these, so
compare the standard errors before reading anything into a difference.

Usage: python evaluate.py ../training_runs/<name>/final.pt [more checkpoints ...]
"""
import json
import math
from pathlib import Path
import re
import sys
import time

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
from tokenizers import Tokenizer
import torch
import torch.nn.functional as F

from model import Transformer, TransformerConfig

ROOT = Path(__file__).parent.parent
BENCHMARK_DIR = ROOT / "data/benchmarks"
TOKENIZER_PATH = ROOT / "tokenizer/stem32k.json"
BOS_ID = 0  # <|endoftext|>, which also starts every training document
BATCH_SIZE = 64

# (Hugging Face repo, file, revision). ARC uses its test split because its validation split is
# small, and PIQA's test labels aren't public
FILES = {
    "hellaswag": ("rowan/hellaswag", "data/validation-00000-of-00001.parquet", "main"),
    "piqa": ("ybisk/piqa", "plain_text/validation/0000.parquet", "refs/convert/parquet"),
    "arc-easy": ("allenai/ai2_arc", "ARC-Easy/test-00000-of-00001.parquet", "main"),
    "arc-challenge": ("allenai/ai2_arc", "ARC-Challenge/test-00000-of-00001.parquet", "main"),
}


def load_rows(name: str) -> list[dict]:
    repo, filename, revision = FILES[name]
    path = hf_hub_download(repo, filename, repo_type="dataset", revision=revision,
                           local_dir=BENCHMARK_DIR / name)
    return pq.read_table(path).to_pylist()


def clean_hellaswag(text: str) -> str:
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def load_benchmark(name: str) -> list[tuple[str, list[str], int]]:
    """Returns (context, options, index of the correct option) for every question."""
    questions = []
    for row in load_rows(name):
        if name == "hellaswag":
            context = f"{row['activity_label']}: {row['ctx_a']} {row['ctx_b'].capitalize()}"
            questions.append((clean_hellaswag(context),
                              [" " + clean_hellaswag(e) for e in row["endings"]], int(row["label"])))
        elif name == "piqa":
            questions.append((f"Question: {row['goal']}\nAnswer:",
                              [" " + row["sol1"], " " + row["sol2"]], int(row["label"])))
        else:
            options = [" " + text for text in row["choices"]["text"]]
            answer = row["choices"]["label"].index(row["answerKey"])
            questions.append((f"Question: {row['question']}\nAnswer:", options, answer))
    return questions


@torch.no_grad()
def option_scores(model, sequences: list[tuple[list[int], list[int]]]) -> list[float]:
    """Total log-probability of each option's tokens given its context, batched by length for speed."""
    max_len = model.config.max_seq_len
    order = sorted(range(len(sequences)), key=lambda i: len(sequences[i][0]) + len(sequences[i][1]))
    scores = [0.0] * len(sequences)
    for b in range(0, len(order), BATCH_SIZE):
        chunk = order[b:b + BATCH_SIZE]
        rows, option_starts = [], []
        for i in chunk:
            context, option = sequences[i]
            option = option[:max_len - 1]
            # The left of a long context is dropped, keeping the text closest to the option
            context = context[-(max_len - len(option)):]
            rows.append(context + option)
            option_starts.append(len(context))
        width = max(map(len, rows))
        x = torch.zeros(len(rows), width, dtype=torch.long)
        for j, row in enumerate(rows):
            x[j, :len(row)] = torch.tensor(row)
        x = x.cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x[:, :-1])
        log_probs = F.log_softmax(logits.float(), dim=-1)
        for j, i in enumerate(chunk):
            # Position p of the logits predicts token p + 1, so the option starts at start - 1
            positions = torch.arange(option_starts[j] - 1, len(rows[j]) - 1).cuda()
            scores[i] = log_probs[j, positions, x[j, positions + 1]].sum().item()
    return scores


def score_benchmark(model, tokenizer, name: str) -> dict:
    questions = load_benchmark(name)
    sequences, spans = [], []
    for context, options, _ in questions:
        context_ids = [BOS_ID] + tokenizer.encode(context).ids
        spans.append((len(sequences), len(sequences) + len(options)))
        sequences.extend((context_ids, tokenizer.encode(o).ids) for o in options)
    scores = option_scores(model, sequences)
    correct = 0
    chance = 0.0
    for (start, end), (_, options, answer) in zip(spans, questions):
        chosen = max(range(start, end), key=lambda i: scores[i]) - start
        correct += chosen == answer
        chance += 1 / len(options)
    n = len(questions)
    accuracy = correct / n
    return {"accuracy": round(accuracy, 4), "std_err": round(math.sqrt(accuracy * (1 - accuracy) / n), 4),
            "chance": round(chance / n, 4), "n": n}


def load_model(checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location="cuda")
    # head_dim is derived in TransformerConfig.__post_init__, so it isn't a constructor argument
    model_config = {k: v for k, v in ckpt["model_config"].items() if k != "head_dim"}
    model = Transformer(TransformerConfig(**model_config)).cuda()
    model.load_state_dict(ckpt["model"])
    return model.eval()


if __name__ == "__main__":
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    for checkpoint_path in sys.argv[1:]:
        model = load_model(checkpoint_path)
        result = {"checkpoint": str(Path(checkpoint_path).resolve().relative_to(ROOT))}
        t0 = time.perf_counter()
        for name in FILES:
            result[name] = score_benchmark(model, tokenizer, name)
            print(name, result[name], flush=True)
        result["eval_s"] = round(time.perf_counter() - t0)
        print(result["checkpoint"], f"in {result['eval_s']} s", flush=True)
        with open(ROOT / "benchmarks.jsonl", "a") as f:
            f.write(json.dumps(result) + "\n")
        del model
        torch.cuda.empty_cache()
