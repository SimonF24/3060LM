"""
Chat format, generation and the GSM8K eval for fine-tuned models.

A conversation is encoded as
    <|endoftext|> <|system|> ... <|im_end|> <|user|> ... <|im_end|> <|assistant|> ... <|im_end|> ...
and training computes loss only on assistant replies, including the <|im_end|> that ends each one.

Replies can use a calculator with GSM8K's annotation syntax: the model writes "<<48/2=" and generation appends the
result and ">>" ("24>>"), which training excludes from the loss since the model doesn't produce it.

Usage:
    python chat.py ../training_runs/<name>/final.pt            # chat in the terminal
    python chat.py ../training_runs/<name>/final.pt --gsm8k --svamp   # math accuracy, appended to benchmarks.jsonl
"""
import ast
import json
import math
import operator
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
SVAMP_DIR = ROOT / "data/raw/svamp/data"
BOS, PAD, END = 0, 1, 3  # <|endoftext|>, <pad>, <|im_end|>
ROLES = {"system": 4, "user": 5, "assistant": 6}
CALL = re.compile(r"<<([\d\s.,+\-*/()%^]+)=$")  # an open calculator call at the end of the text
OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
             ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv}


def calculate(expression: str) -> str | None:
    """Evaluates arithmetic, rounded to 4 decimals with trailing zeros dropped. Returns None if it isn't valid."""
    def value(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return -value(node.operand) if isinstance(node.op, ast.USub) else value(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
            left, right = value(node.left), value(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 64:
                raise ValueError("exponent too large")
            return OPERATORS[type(node.op)](left, right)
        raise ValueError("not arithmetic")
    try:
        result = round(value(ast.parse(expression.replace(",", "").replace("^", "**"), mode="eval").body), 4)
    except Exception:  # syntax errors, division by zero, overflow
        return None
    if not math.isfinite(result):
        return None
    return str(int(result)) if result == int(result) else f"{result:.4f}".rstrip("0")


def segments(text: str) -> list[tuple[str, int]]:
    """
    Splits a reply into (text, trained) pieces: each calculator result and its ">>" is untrained, since generation
    inserts it. The result is recomputed, so training sees exactly what the calculator returns.
    """
    pieces, start = [], 0
    for match in re.finditer(r"<<([^<>=]*)=([^<>]*)>>", text):
        result = calculate(match.group(1))
        if result is None:
            continue
        pieces += [(text[start:match.start(2)], 1), (result + ">>", 0)]
        start = match.end()
    return pieces + [(text[start:], 1)]


def encode_conversation(tokenizer, messages: list[dict], max_tokens: int) -> tuple[list[int], list[int]]:
    """
    Returns token ids and a loss mask (1 where the token is part of an assistant reply). Messages that would
    take the total past max_tokens are dropped, along with everything after them, so cuts fall between turns.
    """
    ids, mask = [BOS], [0]
    for message in messages:
        assistant = message["role"] == "assistant"
        body, body_mask = [], []
        # Pieces are tokenized separately, matching generation, which appends calculator results as their own tokens
        for text, trained in segments(message["content"]) if assistant else [(message["content"], 0)]:
            piece = tokenizer.encode(text).ids
            body += piece
            body_mask += [trained * assistant] * len(piece)
        body, body_mask = body + [END], body_mask + [int(assistant)]
        if len(ids) + 1 + len(body) > max_tokens:
            break
        ids += [ROLES[message["role"]]] + body
        mask += [0] + body_mask
    return ids, mask


def prompt_ids(tokenizer, messages: list[dict]) -> list[int]:
    """The conversation so far, ending with the token that starts the assistant's reply."""
    ids = [BOS]
    for message in messages:
        ids += [ROLES[message["role"]]] + tokenizer.encode(message["content"]).ids + [END]
    return ids + [ROLES["assistant"]]


def gsm8k_answer(solution: str, calculator: bool = False) -> str:
    """GSM8K's solutions, with or without the <<48/2=24>> calculator calls."""
    return solution if calculator else re.sub(r"<<[^>]*>>", "", solution)


@torch.no_grad()
def generate(model, prompts: list[list[int]], max_new_tokens: int, temperature: float = 0.0,
             top_k: int = 50, batch_size: int = 32, tokenizer=None) -> list[list[int]]:
    """
    Generates a reply for each prompt, stopping at <|im_end|>. There's no KV cache, so every step reruns the
    whole sequence; rows are right-padded, which causal attention makes harmless. With a tokenizer, calculator
    calls are answered: when a reply ends with "<<expression=", the result and ">>" are appended.
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
                if tokenizer is not None and not done[j]:
                    call = CALL.search(tokenizer.decode(replies[j][-48:]))
                    result = calculate(call.group(1)) if call else None
                    if result is not None:
                        inserted = tokenizer.encode(result + ">>").ids
                        seqs[j] += inserted
                        replies[j] += inserted
                        done[j] = len(seqs[j]) + 1 >= max_len
        for j, i in enumerate(chunk):
            outputs[i] = replies[j]
    return outputs


def extract_number(text: str) -> str | None:
    """The number after '####', with commas and a trailing period removed."""
    match = re.search(r"####\s*(-?[\d,]*\.?\d+)", text)
    return match.group(1).replace(",", "").rstrip(".") if match else None


def last_number(text: str) -> str | None:
    """The last number in the text, for a lenient score that doesn't require the '####' format."""
    numbers = re.findall(r"-?\d[\d,]*\.?\d*|-?\.\d+", text)
    return numbers[-1].replace(",", "").rstrip(".") if numbers else None


def gsm8k_problems() -> list[tuple[str, float]]:
    rows = pq.read_table(GSM8K_DIR / "test-00000-of-00001.parquet").to_pylist()
    return [(r["question"], float(extract_number(r["answer"]))) for r in rows]


def svamp_problems() -> list[tuple[str, float]]:
    """All 1,000 SVAMP problems (its train and test splits): none are trained on, so all can be scored."""
    rows = [r for split in ["train", "test"]
            for r in pq.read_table(SVAMP_DIR / f"{split}-00000-of-00001.parquet").to_pylist()]
    return [(r["Body"].rstrip(" .") + ". " + r["Question"], float(r["Answer"])) for r in rows]


BENCHMARKS = {"gsm8k": gsm8k_problems, "svamp": svamp_problems}


def evaluate_math(model, tokenizer, benchmark: str, limit: int = 0, max_new_tokens: int = 320) -> dict:
    """
    Greedy accuracy with the calculator available. 'accuracy' counts a reply only if it ends with '#### <answer>';
    'lenient' takes the last number in the reply instead.
    """
    problems = BENCHMARKS[benchmark]()
    if limit:
        problems = problems[:limit]
    prompts = [prompt_ids(tokenizer, [{"role": "user", "content": q}]) for q, _ in problems]
    replies = generate(model, prompts, max_new_tokens, tokenizer=tokenizer)
    correct, lenient, formatted, calls = 0, 0, 0, 0
    examples = []
    for (question, gold), reply in zip(problems, replies):
        text = tokenizer.decode(reply)
        strict, loose = extract_number(text), last_number(text)
        formatted += strict is not None
        calls += text.count("<<")
        ok = strict is not None and math.isclose(float(strict), gold, abs_tol=1e-4)
        correct += ok
        try:
            lenient += loose is not None and math.isclose(float(loose), gold, abs_tol=1e-4)
        except ValueError:
            pass
        if len(examples) < 3:
            examples.append({"question": question, "reply": text, "correct": ok})
    n = len(problems)
    accuracy = correct / n
    return {"accuracy": round(accuracy, 4), "std_err": round(math.sqrt(accuracy * (1 - accuracy) / n), 4),
            "lenient": round(lenient / n, 4), "formatted": round(formatted / n, 4),
            "calls_per_reply": round(calls / n, 2), "n": n, "examples": examples}


def main():
    from evaluate import load_model
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    checkpoint_path = sys.argv[1]
    model = load_model(checkpoint_path)
    benchmarks = [b for b in BENCHMARKS if f"--{b}" in sys.argv]
    if benchmarks:
        record = {"checkpoint": str(Path(checkpoint_path).resolve().relative_to(ROOT))}
        for benchmark in benchmarks:
            t0 = time.perf_counter()
            result = evaluate_math(model, tokenizer, benchmark)
            result["eval_s"] = round(time.perf_counter() - t0)
            print(benchmark, json.dumps(result, indent=2), flush=True)
            record[benchmark] = {k: v for k, v in result.items() if k != "examples"}
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
        reply = generate(model, [prompt_ids(tokenizer, messages)], 400, temperature=0.7, tokenizer=tokenizer)[0]
        reply_text = tokenizer.decode(reply)
        print("model>", reply_text)
        messages.append({"role": "assistant", "content": reply_text})


if __name__ == "__main__":
    main()
