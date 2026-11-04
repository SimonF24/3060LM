"""
Downloads the pretraining shards from the Hugging Face Hub into data/raw/.

By default this fetches only the shards the speedruns use (about 3 GB).
Pass --all for the full set (about 10 GB), which is meant for the long final run.
"""
from pathlib import Path
import sys

from huggingface_hub import hf_hub_download

RAW = Path(__file__).parent.parent / "data/raw"


def fineweb(i: int) -> tuple[str, str]:
    return "HuggingFaceFW/fineweb-edu", f"sample/10BT/{i:03d}_00000.parquet"


def finemath(i: int) -> tuple[str, str]:
    return "HuggingFaceTB/finemath", f"finemath-4plus/train-{i:05d}-of-00064.parquet"


SPEEDRUN_SHARDS = [fineweb(0), finemath(0), finemath(1)]
EXTRA_SHARDS = [fineweb(1), fineweb(2), fineweb(3), *[finemath(i) for i in range(2, 7)]]

if __name__ == "__main__":
    shards = SPEEDRUN_SHARDS + (EXTRA_SHARDS if "--all" in sys.argv else [])
    for repo, filename in shards:
        path = hf_hub_download(repo, filename, repo_type="dataset", local_dir=RAW / repo.split("/")[1])
        print("done", path, flush=True)
