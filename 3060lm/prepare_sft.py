"""
Builds the SFT data: smol-smoltalk conversations plus GSM8K's training split, in the chat format from chat.py,
packed into rows of SEQ_LEN + 1 tokens (inputs plus shifted targets) with a loss mask for assistant replies.

Writes data/<out>/{train,val}_{ids,mask}.npy. The val set is the first VAL_CONVERSATIONS of smol-smoltalk's test split.
GSM8K's test split is never trained on; chat.py --gsm8k scores it.

Usage: python prepare_sft.py [--seq_len=1024] [--gsm8k_repeats=4]
       python prepare_sft.py --out=sft-math --calculator=1 --gsm8k_repeats=3 --smoltalk_groups=10
           # the math stage: GSM8K with calculator calls, plus 10,000 smol-smoltalk conversations as replay
       python prepare_sft.py --out=sft-tinygsm --tinygsm_shards=1 --gsm8k_repeats=0 --smoltalk_groups=20
           # TinyGSM's Python solutions (pytool.py runs them), decontaminated against GSM8K test and SVAMP
       python prepare_sft.py --out=sft-search --qa=1 --qa_chunks=10 --tinygsm_shards=1 --tinygsm_groups=100 \
           --gsm8k_repeats=0 --smoltalk_groups=20
           # search stage: 40k search demonstrations (from qa.py --demos) with TinyGSM and chat replay
"""
import json
import re
from multiprocessing import Pool
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

from chat import PAD, TOKENIZER_PATH, encode_conversation, gsm8k_answer, gsm8k_problems, svamp_problems
from pytool import execute, limit_memory
from qa import DEMOS

ROOT = Path(__file__).parent.parent
RAW = ROOT / "data/raw"
ARGS = dict(a.removeprefix("--").split("=", 1) for a in sys.argv[1:])
OUT = ROOT / "data" / ARGS.get("out", "sft")
SEQ_LEN = int(ARGS.get("seq_len", 1024))
GSM8K_REPEATS = int(ARGS.get("gsm8k_repeats", 4))  # GSM8K is small, so it's repeated to be a visible share of the mix
CALCULATOR = ARGS.get("calculator", "0") == "1"  # keep GSM8K's <<48/2=24>> annotations as calculator calls
SMOLTALK_GROUPS = int(ARGS.get("smoltalk_groups", -1))  # 1,000-row groups of smol-smoltalk to use; negative uses all
TINYGSM_SHARDS = int(ARGS.get("tinygsm_shards", 0))  # TinyGSM parquet files to use, about 700,000 problems each
QA = ARGS.get("qa", "0") == "1"  # search demonstrations for TriviaQA's and NQ's training questions
QA_CHUNK = 2_000  # questions per job
QA_CHUNKS = int(ARGS.get("qa_chunks", -1))  # QA_CHUNK-question chunks per dataset to use; negative uses all
TINYGSM_GROUPS = int(ARGS.get("tinygsm_groups", -1))  # 1,000-problem row groups per TinyGSM shard; negative uses all
VAL_CONVERSATIONS = 2_000
NGRAM = 10  # a training problem sharing this many consecutive words with a benchmark problem is dropped


def ngrams(text: str) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(words[i:i + NGRAM]) for i in range(len(words) - NGRAM + 1)}


_benchmark_ngrams = None
def contaminated(question: str) -> bool:
    """Whether the question shares a NGRAM-word sequence with a GSM8K test or SVAMP problem."""
    global _benchmark_ngrams
    if _benchmark_ngrams is None:
        _benchmark_ngrams = set().union(*(ngrams(q) for q, _ in gsm8k_problems() + svamp_problems()))
    return not ngrams(question).isdisjoint(_benchmark_ngrams)


def tinygsm_reply(code: str) -> str | None:
    """
    A TinyGSM solution as a reply that calls the Python tool: the docstring (a copy of the question) and blank lines
    are removed, simple_math_problem becomes solution, and the code is run for its result. None if it doesn't run
    or doesn't return a number.
    """
    code = re.sub(r'\n[ \t]*("""|\'\'\')(.*?)\1', "", code, count=1, flags=re.S)
    code, renamed = re.subn(r"def simple_math_problem\(\)\s*(->[^:]*)?:", "def solution():", code)
    if not renamed:
        return None
    code = re.sub(r"\n\s*\n", "\n", code).strip()
    result = execute(code, "solution")
    if result.startswith("Error"):
        return None
    return f"```python\n{code}\n```\n>>> solution()\n{result}\n#### {result}"

def load_conversations(source: str, part: int) -> list[list[dict]]:
    """
    One part of a source: a 1,000-row group of a smol-smoltalk file, or all of GSM8K's train split.
    Workers load their own parts, so no process holds all the text at once.
    """
    if source.startswith("qa/"):
        with open(DEMOS / f"demos_{source.removeprefix('qa/')}.jsonl") as f:  # written by qa.py --demos
            lines = [line for i, line in enumerate(f) if part * QA_CHUNK <= i < (part + 1) * QA_CHUNK]
        return [[{"role": "user", "content": d["question"]}, {"role": "assistant", "content": d["reply"]}]
                for d in map(json.loads, lines)]
    if source.startswith("tinygsm/"):
        rows = pq.ParquetFile(RAW / "tinygsm/data" / source.removeprefix("tinygsm/")).read_row_group(part).to_pylist()
        conversations = []
        for r in rows:
            if contaminated(r["question"]):
                continue
            reply = tinygsm_reply(r["code"])
            if reply is not None:
                conversations.append([{"role": "user", "content": r["question"]},
                                      {"role": "assistant", "content": reply}])
        return conversations
    if source == "gsm8k":
        rows = pq.read_table(RAW / "gsm8k/main/train-00000-of-00001.parquet").to_pylist()
        return [[{"role": "user", "content": r["question"]},
                 {"role": "assistant", "content": gsm8k_answer(r["answer"], CALCULATOR)}] for r in rows]
    table = pq.ParquetFile(RAW / "smol-smoltalk/data" / source).read_row_group(part, columns=["messages"])
    return [r["messages"] for r in table.to_pylist()]


_tokenizer = None
def _encode(job):
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    out = []
    for messages in load_conversations(*job):
        ids, mask = encode_conversation(_tokenizer, messages, SEQ_LEN + 1)
        out.append((np.array(ids, dtype=np.uint16), np.array(mask, dtype=np.uint8)))  # arrays keep RAM use low
    return out


def encode_all(jobs: list[tuple[str, int]]) -> list[tuple[np.ndarray, np.ndarray]]:
    # Workers that run TinyGSM's code have their memory capped (pytool.execute sets the time limit)
    capped = [j for j in jobs if j[0].startswith("tinygsm/")]
    other = [j for j in jobs if not j[0].startswith("tinygsm/")]
    encoded = []
    for group, init, args in [(capped, limit_memory, (2 << 30,)), (other, None, ())]:
        if group:
            with Pool(6, initializer=init, initargs=args, maxtasksperchild=20) as pool:
                encoded += [c for batch in pool.imap(_encode, group) for c in batch]
    return encoded


def pack(encoded, seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Packs conversations into rows with best fit: longest first, each into the row with the least space left that
    still fits it, then pads the end of each row. Training shuffles rows, so the order here doesn't matter.
    """
    kept = [(ids, mask) for ids, mask in encoded if mask.any()]
    width = SEQ_LEN + 1
    order = sorted(np.random.default_rng(seed).permutation(len(kept)), key=lambda i: -len(kept[i][0]))
    # Rows grouped by exact space left, with a bitmask of which groups are non-empty, so the best fit (the
    # smallest space that still fits) is one bit operation instead of a search through every row
    buckets = [[] for _ in range(width + 1)]
    nonempty = 0
    used = []  # tokens used in each row
    placement = []  # (conversation, row, offset)
    for i in order:
        n = len(kept[i][0])
        fits = nonempty >> n
        if fits:
            space = n + (fits & -fits).bit_length() - 1
            row = buckets[space].pop()
            if not buckets[space]:
                nonempty &= ~(1 << space)
        else:
            row = len(used)
            used.append(0)
        placement.append((i, row, used[row]))
        used[row] += n
        left = width - used[row]
        if left > 0:
            buckets[left].append(row)
            nonempty |= 1 << left
    n_rows = len(used)
    ids_array = np.full((n_rows, width), PAD, dtype=np.uint16)
    mask_array = np.zeros((n_rows, width), dtype=np.uint8)
    for i, r, offset in placement:
        ids, mask = kept[i]
        ids_array[r, offset:offset + len(ids)] = ids
        mask_array[r, offset:offset + len(mask)] = mask
    n_tokens = int(sum(len(ids) for ids, _ in kept))
    stats = {"conversations": len(encoded), "kept": len(kept), "rows": n_rows, "tokens": n_tokens,
             "assistant_tokens": int(mask_array.sum()), "fill": round(n_tokens / ids_array.size, 3)}
    return ids_array, mask_array, stats


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train_jobs = []
    for f in sorted((RAW / "smol-smoltalk/data").glob("train-*.parquet")):
        train_jobs += [(f.name, g) for g in range(pq.ParquetFile(f).num_row_groups)]
    if SMOLTALK_GROUPS >= 0:
        train_jobs = train_jobs[:SMOLTALK_GROUPS]
    train_jobs += [("gsm8k", 0)] * GSM8K_REPEATS
    if QA:
        for name in ["triviaqa", "nq"]:
            n = pq.ParquetFile(ROOT / "data/raw" / ("triviaqa/unfiltered.nocontext" if name == "triviaqa" else "nq_open/nq_open")
                               / "train-00000-of-00001.parquet").metadata.num_rows
            parts = range(-(-n // QA_CHUNK))
            train_jobs += [(f"qa/{name}", part) for part in (parts[:QA_CHUNKS] if QA_CHUNKS >= 0 else parts)]
    for f in sorted((RAW / "tinygsm/data").glob("train-*.parquet"))[:TINYGSM_SHARDS]:
        groups = range(pq.ParquetFile(f).num_row_groups)
        train_jobs += [(f"tinygsm/{f.name}", g) for g in (groups[:TINYGSM_GROUPS] if TINYGSM_GROUPS >= 0 else groups)]
    val_jobs = [("test-00000-of-00001.parquet", g) for g in range(VAL_CONVERSATIONS // 1000)]
    print(f"{len(train_jobs)} train jobs, GSM8K x {GSM8K_REPEATS}, val {VAL_CONVERSATIONS:,}", flush=True)

    meta = {"seq_len": SEQ_LEN, "gsm8k_repeats": GSM8K_REPEATS, "calculator": CALCULATOR,
            "smoltalk_groups": SMOLTALK_GROUPS, "tinygsm_shards": TINYGSM_SHARDS, "tinygsm_groups": TINYGSM_GROUPS,
            "qa": QA, "qa_chunks": QA_CHUNKS}
    for split, jobs in [("train", train_jobs), ("val", val_jobs)]:
        ids, mask, stats = pack(encode_all(jobs), seed=0)
        np.save(OUT / f"{split}_ids.npy", ids)
        np.save(OUT / f"{split}_mask.npy", mask)
        meta[split] = stats
        print(split, stats, flush=True)
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
