"""
Open-domain question answering with the search tool and abstention: TriviaQA (unfiltered, no context) and Natural
Questions (open), answered against the DPR Wikipedia passages (search.py).

A reply ends with '#### <answer>' or '#### I don't know'. Grading uses the usual normalization (lowercase, no
punctuation, no articles, single spaces) and exact match against any accepted answer. The score is the RL reward
averaged over questions: +1 correct, 0 for abstaining and -1 wrong by default, so answering pays only when the
model is more likely right than wrong. The training splits are for SFT and RL; the validation splits are held out.

Usage: python qa.py --recall [--limit=2000]   # how often BM25's top passages contain an answer
       python qa.py --demos                    # search demonstrations for the training questions, cached in data/qa
"""
import json
from multiprocessing import Pool
from pathlib import Path
import re
import string
import sys

import pyarrow.parquet as pq

ROOT = Path(__file__).parent.parent
FILES = {
    ("triviaqa", "train"): ROOT / "data/raw/triviaqa/unfiltered.nocontext/train-00000-of-00001.parquet",
    ("triviaqa", "validation"): ROOT / "data/raw/triviaqa/unfiltered.nocontext/validation-00000-of-00001.parquet",
    ("nq", "train"): ROOT / "data/raw/nq_open/nq_open/train-00000-of-00001.parquet",
    ("nq", "validation"): ROOT / "data/raw/nq_open/nq_open/validation-00000-of-00001.parquet",
}
DEMOS = ROOT / "data/qa"
ABSTAIN = "I don't know"
REWARDS = {"correct": 1.0, "abstain": 0.0, "wrong": -1.0}


def normalize(text: str) -> str:
    text = "".join(c for c in text.lower() if c not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def load_questions(name: str, split: str, start: int = 0, stop: int | None = None) -> list[tuple[str, list[str]]]:
    """(question, accepted answers) pairs."""
    table = pq.read_table(FILES[(name, split)], columns=["question", "answer"])
    rows = table.slice(start, (stop or table.num_rows) - start).to_pylist()
    if name == "triviaqa":
        return [(r["question"], list(dict.fromkeys([r["answer"]["value"]] + r["answer"]["aliases"]))) for r in rows]
    return [(r["question"][0].upper() + r["question"][1:] + "?", r["answer"]) for r in rows]


def final_answer(reply: str) -> str | None:
    """The text after the last '####', or None if there isn't one."""
    return reply.rsplit("####", 1)[1].strip() if "####" in reply else None


def grade(reply: str, answers: list[str]) -> str:
    """'correct', 'wrong' or 'abstain'. A reply without '####' counts as wrong (it never committed to an answer)."""
    answer = final_answer(reply)
    if answer is None:
        return "wrong"
    if normalize(answer) == normalize(ABSTAIN):
        return "abstain"
    return "correct" if normalize(answer) in {normalize(a) for a in answers} else "wrong"


def find_answer(passages: str, answers: list[str]) -> str | None:
    """An accepted answer as it's written in the passages (original casing), or None if none appears."""
    for answer in sorted(answers, key=len, reverse=True):
        if not normalize(answer):
            continue
        match = re.search(r"(?<!\w)" + re.escape(answer) + r"(?!\w)", passages, re.IGNORECASE)
        if match:
            return match.group(0)
    return None


def search_query(question: str) -> str:
    return question.rstrip("?").replace('"', "").strip()


def search_demo(question: str, answers: list[str], search_tool) -> str:
    """
    A training reply that searches with the question, then answers with an accepted answer if one appears in the
    results (copied as written there) and abstains otherwise.
    """
    query = search_query(question)
    results = search_tool(query)
    answer = find_answer(results, answers)
    return f'>>> search("{query}")\n{results}#### {answer if answer else ABSTAIN}'


def _demos(job: tuple[str, int, int]) -> list[dict]:
    from search import search_tool
    name, start, stop = job
    return [{"question": q, "reply": search_demo(q, answers, search_tool)}
            for q, answers in load_questions(name, "train", start, stop)]


def build_demos(chunk: int = 2_000):
    """Writes data/qa/demos_<name>.jsonl: a search demonstration for every training question, in order."""
    DEMOS.mkdir(parents=True, exist_ok=True)
    for name in ["triviaqa", "nq"]:
        n = pq.ParquetFile(FILES[(name, "train")]).metadata.num_rows
        jobs = [(name, s, min(s + chunk, n)) for s in range(0, n, chunk)]
        answered = 0
        with Pool(6) as pool, open(DEMOS / f"demos_{name}.jsonl", "w") as f:
            for demos in pool.imap(_demos, jobs):
                for d in demos:
                    f.write(json.dumps(d) + "\n")
                    answered += not d["reply"].endswith(ABSTAIN)
        print(f"{name}: {n:,} demonstrations, {answered / n:.1%} answered (the rest abstain)", flush=True)


def evaluate_qa(model, tokenizer, name: str, limit: int = 2000, max_new_tokens: int = 64) -> dict:
    """Greedy replies to the first `limit` validation questions, graded and scored with REWARDS."""
    from chat import generate, prompt_ids
    questions = load_questions(name, "validation", 0, limit or None)
    prompts = [prompt_ids(tokenizer, [{"role": "user", "content": q}]) for q, _ in questions]
    replies = [tokenizer.decode(r) for r in generate(model, prompts, max_new_tokens, tokenizer=tokenizer)]
    outcomes = [grade(r, answers) for r, (_, answers) in zip(replies, questions)]
    n = len(questions)
    rates = {k: outcomes.count(k) / n for k in REWARDS}
    # Lenient: an accepted answer appears anywhere in the reply, for models that don't use the #### format
    lenient = sum(any(normalize(a) and f" {normalize(a)} " in f" {normalize(r)} " for a in answers)
                  for r, (_, answers) in zip(replies, questions)) / n
    return {"score": round(sum(REWARDS[o] for o in outcomes) / n, 4),
            **{k: round(v, 4) for k, v in rates.items()}, "lenient": round(lenient, 4),
            "searched": round(sum('>>> search("' in r for r in replies) / n, 4), "n": n,
            "examples": [{"question": q, "reply": r, "outcome": o}
                         for (q, _), r, o in list(zip(questions, replies, outcomes))[:5]]}


def recall(limit: int):
    """How often an accepted answer appears in BM25's top k passages, searching with the question."""
    from search import Searcher, format_results
    searcher = Searcher()
    for name in ["triviaqa", "nq"]:
        questions = load_questions(name, "validation", 0, limit)
        hits = {k: 0 for k in [1, 3, 5, 20]}
        for question, answers in questions:
            results = searcher(search_query(question), 20)
            for k in hits:
                hits[k] += find_answer(format_results(results[:k]), answers) is not None
        print(name, f"n={len(questions)}", " ".join(f"top-{k} {v / len(questions):.1%}" for k, v in hits.items()),
              flush=True)


if __name__ == "__main__":
    args = dict(a.removeprefix("--").split("=", 1) if "=" in a else (a.removeprefix("--"), "") for a in sys.argv[1:])
    if "recall" in args:
        recall(int(args.get("limit", 2000)))
    if "demos" in args:
        build_demos()
