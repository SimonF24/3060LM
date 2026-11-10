"""
Builds the SFT data: smol-smoltalk conversations plus GSM8K's training split, in the chat format from chat.py,
packed into rows of SEQ_LEN + 1 tokens (inputs plus shifted targets) with a loss mask for assistant replies.

Writes data/<out>/{train,val}_{ids,mask}.npy. The val set is the first VAL_CONVERSATIONS of smol-smoltalk's test split.
GSM8K's test split is never trained on; chat.py --gsm8k scores it.

Usage: python prepare_sft.py [--seq_len=1024] [--gsm8k_repeats=4]
       python prepare_sft.py --out=sft-math --calculator=1 --gsm8k_repeats=3 --smoltalk_groups=10
           # the math stage: GSM8K with calculator calls, plus 10,000 smol-smoltalk conversations as replay
"""
import bisect
import json
from multiprocessing import Pool
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

from chat import PAD, TOKENIZER_PATH, encode_conversation, gsm8k_answer

ROOT = Path(__file__).parent.parent
RAW = ROOT / "data/raw"
ARGS = dict(a.removeprefix("--").split("=", 1) for a in sys.argv[1:])
OUT = ROOT / "data" / ARGS.get("out", "sft")
SEQ_LEN = int(ARGS.get("seq_len", 1024))
GSM8K_REPEATS = int(ARGS.get("gsm8k_repeats", 4))  # GSM8K is small, so it's repeated to be a visible share of the mix
CALCULATOR = ARGS.get("calculator", "0") == "1"  # keep GSM8K's <<48/2=24>> annotations as calculator calls
SMOLTALK_GROUPS = int(ARGS.get("smoltalk_groups", 0))  # 1,000-row groups of smol-smoltalk to use; 0 uses all
VAL_CONVERSATIONS = 2_000

def load_conversations(source: str, part: int) -> list[list[dict]]:
    """
    One part of a source: a 1,000-row group of a smol-smoltalk file, or all of GSM8K's train split.
    Workers load their own parts, so no process holds all the text at once.
    """
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
    with Pool(6, maxtasksperchild=20) as pool:
        return [c for batch in pool.imap(_encode, jobs) for c in batch]


def pack(encoded, seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Packs conversations into rows with best fit: longest first, each into the row with the least space left that
    still fits it, then pads the end of each row. Training shuffles rows, so the order here doesn't matter.
    """
    kept = [(ids, mask) for ids, mask in encoded if mask.any()]
    width = SEQ_LEN + 1
    order = sorted(np.random.default_rng(seed).permutation(len(kept)), key=lambda i: -len(kept[i][0]))
    free_space, free_rows = [], []  # sorted by space left, with the matching row index
    used = []  # tokens used in each row
    placement = []  # (conversation, row, offset)
    for i in order:
        n = len(kept[i][0])
        k = bisect.bisect_left(free_space, n)
        if k < len(free_space):
            row = free_rows.pop(k)
            free_space.pop(k)
        else:
            row = len(used)
            used.append(0)
        placement.append((i, row, used[row]))
        used[row] += n
        left = width - used[row]
        if left > 0:
            k = bisect.bisect_left(free_space, left)
            free_space.insert(k, left)
            free_rows.insert(k, row)
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
    if SMOLTALK_GROUPS:
        train_jobs = train_jobs[:SMOLTALK_GROUPS]
    train_jobs += [("gsm8k", 0)] * GSM8K_REPEATS
    val_jobs = [("test-00000-of-00001.parquet", g) for g in range(VAL_CONVERSATIONS // 1000)]
    print(f"{len(train_jobs)} train jobs, GSM8K x {GSM8K_REPEATS}, val {VAL_CONVERSATIONS:,}", flush=True)

    meta = {"seq_len": SEQ_LEN, "gsm8k_repeats": GSM8K_REPEATS, "calculator": CALCULATOR,
            "smoltalk_groups": SMOLTALK_GROUPS}
    for split, jobs in [("train", train_jobs), ("val", val_jobs)]:
        ids, mask, stats = pack(encode_all(jobs), seed=0)
        np.save(OUT / f"{split}_ids.npy", ids)
        np.save(OUT / f"{split}_mask.npy", mask)
        meta[split] = stats
        print(split, stats, flush=True)
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
