"""
Builds the pretraining data for the general STEM-leaning mix.

1. Trains a 32k byte-level BPE tokenizer on a sample of each source
   (GPT-4 style pre-tokenization, but digits are split one per token, which helps arithmetic)
2. Tokenizes every source into data/tokenized/<source>/{train,val}.bin (uint16)
   The first VAL_DOCS documents of each source are held out as a fixed val set, and the
   number of UTF-8 bytes in the val text is saved so we can report bits-per-byte (BPB),
   which stays comparable even if we change the tokenizer later.
"""
import json
from multiprocessing import Pool
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
RAW = ROOT / "data/raw"
# Pass --out=<path under the repo root> to keep a larger tokenized set apart from the speedrun data
OUT = ROOT / next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")), "data/tokenized")
# Pass --vocab=<size> to train a differently sized tokenizer
VOCAB_SIZE = int(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--vocab=")), 32_768))
TOKENIZER_PATH = ROOT / f"tokenizer/stem{VOCAB_SIZE // 1024}k.json"

SOURCES = {
    "fineweb-edu": sorted((RAW / "fineweb-edu/sample/10BT").glob("*.parquet")),
    "finemath": sorted((RAW / "finemath/finemath-4plus").glob("*.parquet")),
}
if "--speedrun-shards" in sys.argv:  # only the shards speedruns use, even if more have been downloaded
    SOURCES = {"fineweb-edu": SOURCES["fineweb-edu"][:1], "finemath": SOURCES["finemath"][:2]}
TOKENIZER_SAMPLE_BYTES = 400_000_000  # per source, keeps tokenizer training RAM modest
VAL_DOCS = 5_000

END_OF_TEXT = "<|endoftext|>"
SPECIAL_TOKENS = [END_OF_TEXT, "<pad>", "<|im_start|>", "<|im_end|>",
                  "<|system|>", "<|user|>", "<|assistant|>"]
# GPT-4 (cl100k) split pattern, with \p{N}{1,3} changed to \p{N} so every digit is its own token
SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)


def iter_docs(files):
    for f in files:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(columns=["text"], batch_size=1024):
            yield from batch.column(0).to_pylist()


def train_tokenizer():
    def sample():
        for files in SOURCES.values():
            n = 0
            for doc in iter_docs(files):
                yield doc
                n += len(doc)
                if n > TOKENIZER_SAMPLE_BYTES:
                    break

    tok = Tokenizer(models.BPE(byte_fallback=False))
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(SPLIT_PATTERN), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=SPECIAL_TOKENS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  show_progress=True)
    tok.train_from_iterator(sample(), trainer=trainer)
    TOKENIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(TOKENIZER_PATH))


_tok = None
def _encode(docs):
    global _tok
    if _tok is None:
        _tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    eot = _tok.token_to_id(END_OF_TEXT)
    out = []
    for ids in _tok.encode_batch_fast(docs):
        # Each document is prefixed with <|endoftext|> so it acts as a BOS token
        out.append(np.array([eot] + ids.ids, dtype=np.uint16))
    return np.concatenate(out), sum(len(d.encode("utf-8")) for d in docs)


def chunks(it, n):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


def tokenize_source(name, files):
    (OUT / name).mkdir(parents=True, exist_ok=True)
    docs = iter_docs(files)
    val_docs = [next(docs) for _ in range(VAL_DOCS)]
    val_ids, val_bytes = _encode(val_docs)
    val_ids.tofile(OUT / name / "val.bin")
    meta = {"val_tokens": len(val_ids), "val_bytes": val_bytes}

    n_train = 0
    with open(OUT / name / "train.bin", "wb") as f, Pool(10) as pool:
        for ids, _ in tqdm(pool.imap(_encode, chunks(docs, 2048)), desc=name, unit=" chunks"):
            ids.tofile(f)
            n_train += len(ids)
    meta["train_tokens"] = n_train
    (OUT / name / "meta.json").write_text(json.dumps(meta, indent=2))
    print(name, meta)


if __name__ == "__main__":
    if not TOKENIZER_PATH.exists():
        train_tokenizer()
    for name, files in SOURCES.items():
        tokenize_source(name, files)
