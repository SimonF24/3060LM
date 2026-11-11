"""
Chat format, generation and the GSM8K eval for fine-tuned models.

A conversation is encoded as
    <|endoftext|> <|system|> ... <|im_end|> <|user|> ... <|im_end|> <|assistant|> ... <|im_end|> ...
and training computes loss only on assistant replies, including the <|im_end|> that ends each one.

Replies can use two tools, whose output training excludes from the loss since the model doesn't produce it:
- a calculator with GSM8K's annotation syntax: the model writes "<<48/2=" and generation appends "24>>"
- Python (pytool.py): the model writes a ```python block defining a function, then ">>> solution()" on its own line,
  and generation runs the block and appends the result line ("6\n")
- search (search.py): the model writes '>>> search("query")' on its own line, and generation appends the top
  Wikipedia passages, one line each ("[1] title: text\n")

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
from pytool import PythonTool

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


SEARCH_CALL = re.compile(r'>>> search\("([^"\n]*)"\)\n$')  # an open search call at the end of the text
SEARCH_OUTPUT = re.compile(r'>>> search\("[^"\n]*"\)\n((?:\[\d+\] [^\n]*\n|\(no results\)\n)+)')
PYTHON_OUTPUT = re.compile(r"```\n>>> [A-Za-z_]\w*\(\)\n([^\n]*\n)")  # the line after a Python call is its output
python_tool = PythonTool()


def python_call(text: str) -> tuple[str, str] | None:
    """If the text ends with a ```python block followed by ">>> name()", returns (code, name)."""
    start = text.rfind("```python\n")
    if start < 0 or not text.endswith("()\n"):
        return None
    match = re.fullmatch(r"```python\n(.*)```\n>>> ([A-Za-z_]\w*)\(\)\n", text[start:], re.S)
    return (match.group(1), match.group(2)) if match else None


def segments(text: str) -> list[tuple[str, int]]:
    """
    Splits a reply into (text, trained) pieces: tool output is untrained, since generation inserts it. Calculator
    results are recomputed, so training sees exactly what the calculator returns; Python output is used as written
    (prepare_sft.py writes it by running the code).
    """
    spans = []  # (start, end, untrained replacement)
    for match in re.finditer(r"<<([^<>=]*)=([^<>]*)>>", text):
        result = calculate(match.group(1))
        if result is not None:
            spans.append((match.start(2), match.end(), result + ">>"))
    for match in PYTHON_OUTPUT.finditer(text):
        spans.append((match.start(1), match.end(1), match.group(1)))
    for match in SEARCH_OUTPUT.finditer(text):
        spans.append((match.start(1), match.end(1), match.group(1)))
    pieces, start = [], 0
    for begin, end, output in sorted(spans):
        pieces += [(text[start:begin], 1), (output, 0)]
        start = end
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


class _Reply:
    """One reply in progress: the sequence so far, and tool output still to be fed in."""
    def __init__(self, index: int, prompt: list[int]):
        self.index, self.seq = index, list(prompt)
        self.tokens, self.chosen = [], []  # reply tokens, and 1 for each the model sampled (0 for tool output)
        self.forced = []  # tool output still to be fed in, one token per step
        self.sampled, self.ended = 0, False

    def advance(self, sampled_token: int, max_new_tokens: int, max_len: int, tokenizer) -> bool:
        """Takes the next token (pending tool output first) and returns whether the reply continues."""
        was_forced = bool(self.forced)
        token = self.forced.pop(0) if was_forced else sampled_token
        self.sampled += not was_forced
        if token == END or len(self.seq) + 1 >= max_len:
            self.ended = token == END
            return False
        self.seq.append(token)
        self.tokens.append(token)
        self.chosen.append(int(not was_forced))
        if tokenizer is not None and not was_forced:
            tail = tokenizer.decode(self.tokens[-48:])
            call = CALL.search(tail)
            result = calculate(call.group(1)) if call else None
            if result is not None:
                self.forced = tokenizer.encode(result + ">>").ids
            elif tail.endswith("()\n") and (code := python_call(tokenizer.decode(self.tokens))):
                self.forced = tokenizer.encode(python_tool(*code) + "\n").ids
            elif tail.endswith('")\n') and (query := SEARCH_CALL.search(tail)):
                from search import search_tool  # imported here, so chat.py works without the index
                self.forced = tokenizer.encode(search_tool(query.group(1))).ids
        return self.sampled < max_new_tokens or bool(self.forced)

    def output(self, details: bool):
        return {"tokens": self.tokens, "sampled": self.chosen, "ended": self.ended} if details else self.tokens


def _sample(logits: torch.Tensor, temperature: float, top_k: int) -> list[int]:
    if temperature > 0 and not top_k:
        return torch.multinomial((logits / temperature).softmax(-1), 1).squeeze(1).tolist()
    if temperature > 0:
        values, indices = logits.topk(top_k, dim=-1)
        picks = torch.multinomial((values / temperature).softmax(-1), 1)
        return indices.gather(1, picks).squeeze(1).tolist()
    return logits.argmax(-1).tolist()


@torch.no_grad()
def generate(model, prompts: list[list[int]], max_new_tokens: int, temperature: float = 0.0,
             top_k: int = 50, batch_size: int = 64, tokenizer=None, cache: bool = True,
             details: bool = False) -> list:
    """
    Generates a reply of up to max_new_tokens sampled tokens for each prompt, stopping at <|im_end|>. With a
    tokenizer, tool calls are answered, with the output fed in one token per step: when a reply ends with
    "<<expression=", the calculator appends the result and ">>", and when it ends with a ```python block and
    ">>> name()", the Python tool appends the result line.

    With the cache, batch_size slots decode together as a CUDA graph that runs only the newest token, and a slot
    whose reply ends takes the next prompt (prefilled in groups of a quarter of the slots), so the batch stays busy.
    cache=False reruns every whole sequence each step instead, which is slow but simple, for checks.
    top_k=0 samples from the full distribution. With details, each reply is a dict with its tokens, which of them the
    model sampled (rather than a tool), and whether it ended with <|im_end|> (which isn't in tokens).
    """
    max_len = model.config.max_seq_len
    outputs = [None] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    if not cache:
        for b in range(0, len(order), batch_size):
            active = [_Reply(i, prompts[i][-(max_len - 1):]) for i in order[b:b + batch_size]]
            while active:
                x = torch.full((len(active), max(len(r.seq) for r in active)), PAD, dtype=torch.long)
                for j, r in enumerate(active):
                    x[j, :len(r.seq)] = torch.tensor(r.seq)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(x.cuda())
                last = torch.tensor([len(r.seq) - 1 for r in active], device="cuda")
                next_ids = _sample(logits[torch.arange(len(active), device="cuda"), last].float(), temperature, top_k)
                still = []
                for r, token in zip(active, next_ids):
                    if r.advance(token, max_new_tokens, max_len, tokenizer):
                        still.append(r)
                    else:
                        outputs[r.index] = r.output(details)
                active = still
        return outputs

    batch_size = min(batch_size, len(prompts))
    kv, token_buffer, position_buffer, steps = decode_graphs(model, batch_size)
    queue = list(reversed(order))  # popped from the end, shortest prompt first
    slots = [None] * batch_size
    logits = torch.zeros((batch_size, model.config.vocab_size), device="cuda")

    def refill(free: list[int]):
        """Prefills the next prompts into free slots and sets their logits."""
        new = [_Reply(i, prompts[i][-(max_len - 1):]) for i in [queue.pop() for _ in range(min(len(free), len(queue)))]]
        rows = free[:len(new)]
        width = max(len(r.seq) for r in new)
        x = torch.full((len(new), width), PAD, dtype=torch.long)
        for j, r in enumerate(new):
            x[j, :len(r.seq)] = torch.tensor(r.seq)
        fresh = model.make_cache(len(new), length=width)
        lengths = torch.tensor([len(r.seq) for r in new], device="cuda")
        positions = torch.arange(width, device="cuda").expand(len(new), -1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prefill = model(x.cuda(), cache=fresh, positions=positions, select=lengths - 1, used=width)
        index = torch.tensor(rows, device="cuda")
        for slot in range(len(kv.keys)):
            kv.keys[slot][index, :, :width] = fresh.keys[slot]
            kv.values[slot][index, :, :width] = fresh.values[slot]
        logits[index] = prefill[:, 0].float()
        for row, r in zip(rows, new):
            slots[row] = r

    refill(list(range(batch_size)))
    while any(slots):
        next_ids = _sample(logits, temperature, top_k)
        # Empty slots feed padding at the last cache position, which no sequence reaches, so it's never read
        fed, fed_positions = [PAD] * batch_size, [max_len - 1] * batch_size
        decoding = []
        for j, r in enumerate(slots):
            if r is None:
                continue
            if r.advance(next_ids[j], max_new_tokens, max_len, tokenizer):
                fed[j], fed_positions[j] = r.seq[-1], len(r.seq) - 1
                decoding.append(j)
            else:
                outputs[r.index] = r.output(details)
                slots[j] = None
        if decoding:
            needed = max(fed_positions[j] for j in decoding) + 1
            graph, graph_logits = steps[min(max_len, -(-needed // BUCKET) * BUCKET)]
            token_buffer.copy_(torch.tensor(fed)[:, None])
            position_buffer.copy_(torch.tensor(fed_positions)[:, None])
            graph.replay()
            logits = graph_logits[:, -1].float()
        free = [j for j, r in enumerate(slots) if r is None]
        if queue and free and (len(free) >= max(1, batch_size // 4) or not decoding):
            refill(free)
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
        calls += text.count("<<") + text.count("\n>>> ")
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
