"""
Chat format, generation and the GSM8K eval for fine-tuned models.

A conversation is encoded as
    <|endoftext|> <|system|> ... <|im_end|> <|user|> ... <|im_end|> <|assistant|> ... <|im_end|> ...
and training computes loss only on assistant replies, including the <|im_end|> that ends each one.

Usage:
    python chat.py ../training_runs/<name>/final.pt            # chat in the terminal
    python chat.py ../training_runs/<name>/final.pt --gsm8k    # GSM8K test accuracy, appended to benchmarks.jsonl
"""
import json
import math
from pathlib import Path
import re
import sys
import time

import pyarrow.parquet as pq
from tokenizers import Tokenizer
import torch

ROOT = Path(__file__).parent.parent
TOKENIZER_PATH = ROOT / "tokenizer/stem32k.json"
GSM8K_DIR = ROOT / "data/raw/gsm8k/main"
BOS, PAD, END = 0, 1, 3  # <|endoftext|>, <pad>, <|im_end|>
ROLES = {"system": 4, "user": 5, "assistant": 6}


def encode_conversation(tokenizer, messages: list[dict], max_tokens: int) -> tuple[list[int], list[int]]:
    """
    Returns token ids and a loss mask (1 where the token is part of an assistant reply). Messages that would
    take the total past max_tokens are dropped, along with everything after them, so cuts fall between turns.
    """
    ids, mask = [BOS], [0]
    for message in messages:
        body = tokenizer.encode(message["content"]).ids + [END]
        if len(ids) + 1 + len(body) > max_tokens:
            break
        trained = int(message["role"] == "assistant")
        ids += [ROLES[message["role"]]] + body
        mask += [0] + [trained] * len(body)
    return ids, mask


def prompt_ids(tokenizer, messages: list[dict]) -> list[int]:
    """The conversation so far, ending with the token that starts the assistant's reply."""
    ids = [BOS]
    for message in messages:
        ids += [ROLES[message["role"]]] + tokenizer.encode(message["content"]).ids + [END]
    return ids + [ROLES["assistant"]]


def gsm8k_answer(solution: str) -> str:
    """GSM8K's solutions, without the <<48/2=24>> calculator annotations (kept for the tool-use stage)."""
    return re.sub(r"<<[^>]*>>", "", solution)


@torch.no_grad()
def generate(model, prompts: list[list[int]], max_new_tokens: int, temperature: float = 0.0,
             top_k: int = 50, batch_size: int = 32) -> list[list[int]]:
    """
    Generates a reply for each prompt, stopping at <|im_end|>. There's no KV cache, so every step reruns the
    whole sequence; rows are right-padded, which causal attention makes harmless.
    """
    max_len = model.config.max_seq_len
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    outputs = [None] * len(prompts)
    for b in range(0, len(order), batch_size):
        chunk = order[b:b + batch_size]
        seqs = [list(prompts[i][-(max_len - 1):]) for i in chunk]
        replies = [[] for _ in chunk]
        done = [False] * len(chunk)
        for _ in range(max_new_tokens):
            active = [j for j in range(len(chunk)) if not done[j]]
            if not active:
                break
            width = max(len(seqs[j]) for j in active)
            x = torch.full((len(active), width), PAD, dtype=torch.long)
            for r, j in enumerate(active):
                x[r, :len(seqs[j])] = torch.tensor(seqs[j])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x.cuda())
            last = torch.tensor([len(seqs[j]) - 1 for j in active], device="cuda")
            logits = logits[torch.arange(len(active), device="cuda"), last].float()
            if temperature > 0:
                values, indices = logits.topk(top_k, dim=-1)
                picks = torch.multinomial((values / temperature).softmax(-1), 1)
                next_ids = indices.gather(1, picks).squeeze(1)
            else:
                next_ids = logits.argmax(-1)
            for r, j in enumerate(active):
                token = next_ids[r].item()
                if token == END or len(seqs[j]) + 1 >= max_len:
                    done[j] = True
                if token != END:
                    seqs[j].append(token)
                    replies[j].append(token)
        for j, i in enumerate(chunk):
            outputs[i] = replies[j]
    return outputs


def extract_number(text: str) -> str | None:
    """The number after the final '####', with commas and a trailing period removed."""
    match = re.search(r"####\s*(-?[\d,]*\.?\d+)", text)
    return match.group(1).replace(",", "").rstrip(".") if match else None


def evaluate_gsm8k(model, tokenizer, limit: int = 0, max_new_tokens: int = 320) -> dict:
    """Greedy accuracy on the GSM8K test set. A reply counts only if it ends with '#### <correct number>'."""
    rows = pq.read_table(GSM8K_DIR / "test-00000-of-00001.parquet").to_pylist()
    if limit:
        rows = rows[:limit]
    prompts = [prompt_ids(tokenizer, [{"role": "user", "content": r["question"]}]) for r in rows]
    replies = generate(model, prompts, max_new_tokens)
    correct, formatted = 0, 0
    examples = []
    for row, reply in zip(rows, replies):
        text = tokenizer.decode(reply)
        predicted, gold = extract_number(text), extract_number(row["answer"])
        formatted += predicted is not None
        ok = predicted is not None and float(predicted) == float(gold)
        correct += ok
        if len(examples) < 3:
            examples.append({"question": row["question"], "reply": text, "correct": ok})
    n = len(rows)
    accuracy = correct / n
    return {"accuracy": round(accuracy, 4), "std_err": round(math.sqrt(accuracy * (1 - accuracy) / n), 4),
            "formatted": round(formatted / n, 4), "n": n, "examples": examples}


def main():
    from evaluate import load_model
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    checkpoint_path = sys.argv[1]
    model = load_model(checkpoint_path)
    if "--gsm8k" in sys.argv:
        t0 = time.perf_counter()
        result = evaluate_gsm8k(model, tokenizer)
        result["eval_s"] = round(time.perf_counter() - t0)
        print(json.dumps(result, indent=2))
        record = {"checkpoint": str(Path(checkpoint_path).resolve().relative_to(ROOT)),
                  "gsm8k": {k: v for k, v in result.items() if k != "examples"}}
        with open(ROOT / "benchmarks.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        return
    messages = []
    print("Chatting with", checkpoint_path, "(empty line to reset the conversation, Ctrl-D to quit)")
    while True:
        try:
            text = input("\nyou> ").strip()
        except EOFError:
            break
        if not text:
            messages = []
            print("(conversation reset)")
            continue
        messages.append({"role": "user", "content": text})
        reply = generate(model, [prompt_ids(tokenizer, messages)], 400, temperature=0.7)[0]
        reply_text = tokenizer.decode(reply)
        print("model>", reply_text)
        messages.append({"role": "assistant", "content": reply_text})


if __name__ == "__main__":
    main()
