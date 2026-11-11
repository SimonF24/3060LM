"""
BM25 search over the DPR Wikipedia passages (Dec 2018 English Wikipedia, split into 21M disjoint 100-word passages),
for the search tool. The index is tantivy's, kept on disk, so it doesn't need the passages in RAM.

Usage: python search.py --build              # indexes data/raw/wiki_dpr/psgs_w100.tsv.gz into data/search/wiki_dpr
       python search.py "who wrote pride and prejudice"
"""
import csv
import gzip
from pathlib import Path
import re
import sys
import time

import tantivy

ROOT = Path(__file__).parent.parent
PASSAGES = ROOT / "data/raw/wiki_dpr/psgs_w100.tsv.gz"
INDEX_DIR = ROOT / "data/search/wiki_dpr"


def schema() -> tantivy.Schema:
    builder = tantivy.SchemaBuilder()
    builder.add_text_field("title", stored=True, tokenizer_name="en_stem")
    builder.add_text_field("text", stored=True, tokenizer_name="en_stem")
    return builder.build()


def build():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index = tantivy.Index(schema(), path=str(INDEX_DIR))
    writer = index.writer(heap_size=2_000_000_000, num_threads=8)
    t0 = time.time()
    csv.field_size_limit(1 << 24)
    with gzip.open(PASSAGES, "rt", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # header: id, text, title
        for n, (_, text, title) in enumerate(reader, 1):
            writer.add_document(tantivy.Document(title=title, text=text))
            if n % 1_000_000 == 0:
                print(f"{n / 1e6:.0f}M passages, {(time.time() - t0) / 60:.1f} min", flush=True)
    writer.commit()
    writer.wait_merging_threads()
    print(f"indexed {n:,} passages in {(time.time() - t0) / 60:.1f} min", flush=True)


# Common English words, dropped from queries: they barely affect BM25's ranking but make every query score most of
# the index (Lucene's English stopword list plus question words)
STOPWORDS = set("""a an and are as at be but by for if in into is it no not of on or such that the their then there these
they this to was will with who whom whose what which when where why how does did do has have had were been being am
i you he she we me my your his her its our them""".split())


class Searcher:
    """Top-k BM25 passages for a query, as (title, text, score)."""
    def __init__(self):
        self.index = tantivy.Index.open(str(INDEX_DIR))
        self.searcher = self.index.searcher()

    def __call__(self, query: str, k: int = 3) -> list[tuple[str, str, float]]:
        # Keeping only words drops the query syntax characters, so any text is a valid query
        words = [w for w in re.findall(r"\w+", query) if w.lower() not in STOPWORDS] or re.findall(r"\w+", query)
        if not words:
            return []
        parsed = self.index.parse_query(" ".join(words), ["title", "text"])
        results = []
        for score, address in self.searcher.search(parsed, k).hits:
            doc = self.searcher.doc(address)
            results.append((doc["title"][0], doc["text"][0], score))
        return results


RESULTS = 3  # passages the search tool returns


def format_results(results: list[tuple[str, str, float]]) -> str:
    """The search tool's output: one line per passage, '[k] title: text', or '(no results)'."""
    if not results:
        return "(no results)\n"
    return "".join(f"[{k}] {title}: {' '.join(text.split())}\n" for k, (title, text, _) in enumerate(results, 1))


_searcher = None
def search_tool(query: str) -> str:
    global _searcher
    if _searcher is None:
        _searcher = Searcher()
    return format_results(_searcher(query, RESULTS))


if __name__ == "__main__":
    if "--build" in sys.argv:
        build()
    else:
        search = Searcher()
        t0 = time.time()
        for title, text, score in search(" ".join(sys.argv[1:]), 5):
            print(f"{score:6.2f} | {title} | {text[:150]}")
        print(f"{(time.time() - t0) * 1000:.0f} ms")
