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
    python chat.py ../training_runs/<name>/final.pt --gsm8k --svamp --pass_k=16 --limit=200   # pass@k from sampling
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

from model import KVCache

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


BUCKET = 128  # decode graphs attend to a multiple of this many cache entries


def decode_graphs(model, batch_size: int) -> tuple:
    """
    CUDA graphs of one decode step, one per attention length (a multiple of BUCKET), sharing one KV cache. A step
    launches hundreds of small kernels, which Python issues more slowly than the GPU runs them; a graph replays them
    with one launch. Built once per model and batch size, and stored on the model.
    """
    graphs = model.__dict__.setdefault("_decode_graphs", {})
    if batch_size not in graphs:
        max_len = model.config.max_seq_len
        kv = model.make_cache(batch_size)
        tokens = torch.full((batch_size, 1), PAD, dtype=torch.long, device="cuda")
        positions = torch.zeros((batch_size, 1), dtype=torch.long, device="cuda")
        steps, pool = {}, None
        for used in sorted({min(u, max_len) for u in range(BUCKET, max_len + BUCKET, BUCKET)}):
            stream = torch.cuda.Stream()  # warm up off the default stream, as capture requires
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream), torch.autocast("cuda", dtype=torch.bfloat16):
                for _ in range(2):
                    model(tokens, cache=kv, positions=positions, used=used)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool), torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(tokens, cache=kv, positions=positions, used=used)
            pool = graph.pool()
            steps[used] = (graph, logits)
        graphs[batch_size] = (kv, tokens, positions, steps)
    return graphs[batch_size]


@torch.no_grad()
def generate(model, prompts: list[list[int]], max_new_tokens: int, temperature: float = 0.0,
             top_k: int = 50, batch_size: int = 64, tokenizer=None, cache: bool = True,
             details: bool = False) -> list:
    """
    Generates a reply of up to max_new_tokens sampled tokens for each prompt, stopping at <|im_end|>. With a
    tokenizer, calculator calls are answered: when a reply ends with "<<expression=", the result and ">>" are
    appended, one token per step. With the cache, prompts are prefilled and each step runs only the newest token,
    as a CUDA graph; cache=False reruns the whole sequence every step instead, which is slow but simple, for checks.
    top_k=0 samples from the full distribution. With details, each reply is a dict with its tokens, which of them
    the model sampled (rather than the calculator), and whether it ended with <|im_end|> (which isn't in tokens).
    """
    max_len = model.config.max_seq_len
    batch_size = min(batch_size, len(prompts))
    if cache:
        kv, token_buffer, position_buffer, steps = decode_graphs(model, batch_size)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    outputs = [None] * len(prompts)
    for b in range(0, len(order), batch_size):
        chunk = order[b:b + batch_size]
        n = batch_size if cache else len(chunk)  # graphs have a fixed batch, so a short last chunk is padded
        seqs = [list(prompts[i][-(max_len - 1):]) for i in chunk] + [[BOS]] * (n - len(chunk))
        replies = [[] for _ in range(n)]
        forced = [[] for _ in range(n)]  # calculator output still to be fed in
        sampled = [0] * n
        chosen = [[] for _ in range(n)]  # per reply token: 1 if sampled, 0 if calculator output
        ended = [False] * n
        active = [j < len(chunk) for j in range(n)]
        if cache:  # prefill the prompts, right-padded
            lengths = torch.tensor([len(s) for s in seqs], device="cuda")
            x = torch.full((n, int(lengths.max())), PAD, dtype=torch.long)
            for j, s in enumerate(seqs):
                x[j, :len(s)] = torch.tensor(s)
            positions = torch.arange(x.size(1), device="cuda").expand(n, -1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x.cuda(), cache=kv, positions=positions, select=lengths - 1, used=x.size(1))
            logits = logits[:, 0].float()
        while any(active):
            if not cache:
                rows = [j for j in range(n) if active[j]]
                x = torch.full((n, max(len(seqs[j]) for j in rows)), PAD, dtype=torch.long)
                for j in rows:
                    x[j, :len(seqs[j])] = torch.tensor(seqs[j])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(x.cuda())
                last = torch.tensor([len(seqs[j]) - 1 if active[j] else 0 for j in range(n)], device="cuda")
                logits = logits[torch.arange(n, device="cuda"), last].float()
            if temperature > 0 and not top_k:
                next_ids = torch.multinomial((logits / temperature).softmax(-1), 1).squeeze(1).tolist()
            elif temperature > 0:
                values, indices = logits.topk(top_k, dim=-1)
                picks = torch.multinomial((values / temperature).softmax(-1), 1)
                next_ids = indices.gather(1, picks).squeeze(1).tolist()
            else:
                next_ids = logits.argmax(-1).tolist()
            fed, fed_positions = [PAD] * n, [0] * n  # finished rows feed padding at position 0, which is harmless
            for j in range(n):
                if not active[j]:
                    continue
                was_forced = bool(forced[j])
                token = forced[j].pop(0) if was_forced else next_ids[j]
                sampled[j] += not was_forced
                if token == END or len(seqs[j]) + 1 >= max_len:
                    active[j] = False
                    ended[j] = token == END
                    continue
                seqs[j].append(token)
                replies[j].append(token)
                chosen[j].append(int(not was_forced))
                if tokenizer is not None and not was_forced:
                    call = CALL.search(tokenizer.decode(replies[j][-48:]))
                    result = calculate(call.group(1)) if call else None
                    if result is not None:
                        forced[j] = tokenizer.encode(result + ">>").ids
                if sampled[j] >= max_new_tokens and not forced[j]:
                    active[j] = False
                    continue
                fed[j], fed_positions[j] = token, len(seqs[j]) - 1
            if cache and any(active):
                needed = max(fed_positions) + 1
                graph, graph_logits = steps[min(max_len, -(-needed // BUCKET) * BUCKET)]
                token_buffer.copy_(torch.tensor(fed)[:, None])
                position_buffer.copy_(torch.tensor(fed_positions)[:, None])
                graph.replay()
                logits = graph_logits[:, -1].float()
        for j, i in enumerate(chunk):
            outputs[i] = {"tokens": replies[j], "sampled": chosen[j], "ended": ended[j]} if details else replies[j]
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


def pass_at_k(model, tokenizer, benchmark: str, k: int, limit: int = 0, temperature: float = 1.0,
              max_new_tokens: int = 320) -> dict:
    """
    Samples k replies per problem and reports pass@j (the chance at least one of j samples is right, with the
    unbiased estimator from the Codex paper) for j = 1, 2, 4, ... up to k. Answers are scored strictly.
    """
    problems = BENCHMARKS[benchmark]()
    if limit:
        problems = problems[:limit]
    torch.manual_seed(0)
    prompts = [prompt_ids(tokenizer, [{"role": "user", "content": q}]) for q, _ in problems for _ in range(k)]
    replies = generate(model, prompts, max_new_tokens, temperature=temperature, tokenizer=tokenizer)
    counts = []
    for i, (_, gold) in enumerate(problems):
        answers = [extract_number(tokenizer.decode(r)) for r in replies[i * k:(i + 1) * k]]
        counts.append(sum(a is not None and math.isclose(float(a), gold, abs_tol=1e-4) for a in answers))
    result = {"n": len(problems), "k": k, "temperature": temperature}
    j = 1
    while j <= k:
        estimates = [1 - math.comb(k - c, j) / math.comb(k, j) for c in counts]
        result[f"pass@{j}"] = round(sum(estimates) / len(estimates), 4)
        j *= 2
    result["solved_counts"] = {c: counts.count(c) for c in sorted(set(counts))}  # problems by number of correct samples
    return result


def main():
    from evaluate import load_model
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    checkpoint_path = sys.argv[1]
    model = load_model(checkpoint_path)
    benchmarks = [b for b in BENCHMARKS if f"--{b}" in sys.argv]
    if benchmarks:
        record = {"checkpoint": str(Path(checkpoint_path).resolve().relative_to(ROOT))}
        args = dict(a.removeprefix("--").split("=", 1) for a in sys.argv[2:] if "=" in a)
        for benchmark in benchmarks:
            t0 = time.perf_counter()
            if "pass_k" in args:
                result = pass_at_k(model, tokenizer, benchmark, int(args["pass_k"]), int(args.get("limit", 0)),
                                   float(args.get("temperature", 1.0)))
                benchmark = f"{benchmark}_pass@k"
            else:
                result = evaluate_math(model, tokenizer, benchmark, int(args.get("limit", 0)))
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
