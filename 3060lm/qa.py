"""
Open-domain question answering with the search tool and abstention: TriviaQA (unfiltered, no context) and Natural
Questions (open), answered against the DPR Wikipedia passages (search.py).

A reply ends with '#### <answer>' or '#### I don't know'. Grading uses the usual normalization (lowercase, no
punctuation, no articles, single spaces) and exact match against any accepted answer. The score is the RL reward
averaged over questions: +1 correct, 0 for abstaining and -1 wrong by default, so answering pays only when the
model is more likely right than wrong. The training splits are for SFT and RL; the validation splits are held out.

Reading comprehension (SQuAD v1.1) uses the same format, with the question's own paragraph as one of the search
results, so the model learns to find an answer in passages it's handed. evaluate_reading is the go/no-go check for
search: accuracy when the right passage is in the results.

Usage: python qa.py --recall [--limit=2000]   # how often BM25's top passages contain an answer
       python qa.py --demos                    # search demonstrations for the training questions, cached in data/qa
       python qa.py --reading                  # SQuAD reading demonstrations (train) and eval prefixes (validation)
       python qa.py --relabel=<checkpoint> [--per_dataset=25000 --samples=0 --temperature=0.6]
           # abstention targets from the model's own sampled answers, in data/qa/relabel_<name>.jsonl
"""
import random
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
    ("squad", "train"): ROOT / "data/raw/squad/plain_text/train-00000-of-00001.parquet",
    ("squad", "validation"): ROOT / "data/raw/squad/plain_text/validation-00000-of-00001.parquet",
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


def load_squad(split: str, start: int = 0, stop: int | None = None) -> list[dict]:
    """SQuAD rows: question, answers (unique, in order), title and context (the paragraph the answers come from)."""
    table = pq.read_table(FILES[("squad", split)], columns=["question", "answers", "title", "context"])
    rows = table.slice(start, (stop or table.num_rows) - start).to_pylist()
    return [{"question": r["question"].strip(), "answers": list(dict.fromkeys(r["answers"]["text"])),
             "title": r["title"].replace("_", " "), "context": r["context"]} for r in rows]


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


def reading_prefix(row: dict, searcher, results: int = 3) -> str:
    """
    A search call whose results hold the row's own paragraph at a random position among BM25 passages from other
    articles, so exactly one result is meant to contain the answer.
    """
    from search import format_results
    query = search_query(row["question"])
    others = [r for r in searcher(query, results + 5) if r[0] != row["title"]][:results - 1]
    passages = others[:]
    passages.insert(random.Random(row["question"]).randrange(len(others) + 1), (row["title"], row["context"], 0.0))
    return f'>>> search("{query}")\n{format_results(passages)}'


def _reading(job: tuple[str, int, int]) -> list[dict]:
    import search
    if search._searcher is None:
        search._searcher = search.Searcher()
    split, start, stop = job
    out = []
    for row in load_squad(split, start, stop):
        prefix = reading_prefix(row, search._searcher)
        out.append({"question": row["question"], "answers": row["answers"], "prefix": prefix,
                    "reply": f"{prefix}#### {row['answers'][0]}"})
    return out


def build_reading(chunk: int = 2_000):
    """Writes data/qa/reading_squad_<split>.jsonl: a reading demonstration (train) or eval prefix (validation) per row."""
    DEMOS.mkdir(parents=True, exist_ok=True)
    for split in ["train", "validation"]:
        n = pq.ParquetFile(FILES[("squad", split)]).metadata.num_rows
        with Pool(6) as pool, open(DEMOS / f"reading_squad_{split}.jsonl", "w") as f:
            for rows in pool.imap(_reading, [(split, s, min(s + chunk, n)) for s in range(0, n, chunk)]):
                for r in rows:
                    f.write(json.dumps(r) + "\n")
        print(f"squad {split}: {n:,} rows", flush=True)


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


def relabel(checkpoint: str, per_dataset: int, samples: int = 0, temperature: float = 0.6, out: str = "relabel"):
    """
    Abstention targets from the model's own answers: samples `samples` replies (with search) to each of
    `per_dataset` random training questions per dataset, plus a greedy one. A question the model gets right in at
    least half its samples gets one of its own correct replies as the target; any other gets its search call and
    results followed by '#### I don't know'. samples=0 labels by the greedy reply alone: in a 500-question pilot per
    dataset, it agreed with the majority of 8 samples at T=0.6 on 97% of questions, at a ninth of the cost. Writes
    data/qa/<out>_<name>.jsonl with the target ('reply') and the outcomes, so the threshold can be changed later
    without sampling again.
    """
    import time
    import numpy as np
    from tokenizers import Tokenizer
    from chat import TOKENIZER_PATH, generate, prompt_ids
    from evaluate import load_model
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    model = load_model(checkpoint)
    for name in ["triviaqa", "nq"]:
        questions = load_questions(name, "train")
        picked = np.random.default_rng(0).permutation(len(questions))[:per_dataset]
        questions = [questions[i] for i in sorted(picked)]
        prompts = [prompt_ids(tokenizer, [{"role": "user", "content": q}]) for q, _ in questions]
        t0 = time.time()
        greedy = [tokenizer.decode(r) for r in generate(model, prompts, 64, tokenizer=tokenizer)]
        per_question = samples or 1  # samples=0: the greedy reply decides (answer if it's correct)
        sampled = greedy if not samples else [
            tokenizer.decode(r) for r in generate(model, [p for p in prompts for _ in range(samples)], 64,
                                                  temperature=temperature, top_k=0, tokenizer=tokenizer)]
        seconds = time.time() - t0
        counts = {"answer": 0, "greedy_correct": 0, "greedy_wrong": 0, "greedy_correct_labelled_abstain": 0,
                  "greedy_wrong_labelled_answer": 0}
        with open(DEMOS / f"{out}_{name}.jsonl", "w") as f:
            for i, (question, answers) in enumerate(questions):
                replies = sampled[i * per_question:(i + 1) * per_question]
                outcomes = [grade(r, answers) for r in replies]
                correct = [r for r, o in zip(replies, outcomes) if o == "correct"]
                if len(correct) * 2 >= per_question:
                    # the most common correct answer, as the model wrote it
                    by_answer = {}
                    for r in correct:
                        by_answer.setdefault(normalize(final_answer(r)), []).append(r)
                    reply = max(by_answer.values(), key=len)[0]
                else:
                    prefix = greedy[i].rsplit("####", 1)[0] if "####" in greedy[i] else ""
                    if not prefix.startswith('>>> search("'):
                        from search import search_tool
                        query = search_query(question)
                        prefix = f'>>> search("{query}")\n{search_tool(query)}'
                    reply = f"{prefix}#### {ABSTAIN}"
                greedy_outcome = grade(greedy[i], answers)
                answer = not reply.endswith(ABSTAIN)
                counts["answer"] += answer
                counts["greedy_correct"] += greedy_outcome == "correct"
                counts["greedy_wrong"] += greedy_outcome == "wrong"
                counts["greedy_correct_labelled_abstain"] += greedy_outcome == "correct" and not answer
                counts["greedy_wrong_labelled_answer"] += greedy_outcome == "wrong" and answer
                f.write(json.dumps({"question": question, "reply": reply, "greedy": final_answer(greedy[i]),
                                    "greedy_outcome": greedy_outcome, "sampled": [final_answer(r) for r in replies],
                                    "outcomes": outcomes}) + "\n")
        n = len(questions)
        print(f"{name}: {n:,} questions x {samples + 1} replies in {seconds / 60:.1f} min; "
              + ", ".join(f"{k} {v / n:.1%}" for k, v in counts.items()), flush=True)


def evaluate_qa(model, tokenizer, name: str, limit: int = 2000, max_new_tokens: int = 64) -> dict:
    """Greedy replies to the first `limit` validation questions, graded and scored with REWARDS."""
    from chat import generate, prompt_ids
    questions = load_questions(name, "validation", 0, limit or None)
    prompts = [prompt_ids(tokenizer, [{"role": "user", "content": q}]) for q, _ in questions]
    replies = [tokenizer.decode(r) for r in generate(model, prompts, max_new_tokens, tokenizer=tokenizer)]
    outcomes = [grade(r, answers) for r, (_, answers) in zip(replies, questions)]
    n = len(questions)
    rates = {k: outcomes.count(k) / n for k in REWARDS}
    # Lenient: an accepted answer appears anywhere in the model's own text (not the search results), for models
    # that don't use the #### format
    own = [re.sub(r"^(\[\d+\] .*|\(no results\))$", "", r, flags=re.M) for r in replies]
    lenient = sum(any(normalize(a) and f" {normalize(a)} " in f" {normalize(r)} " for a in answers)
                  for r, (_, answers) in zip(own, questions)) / n
    return {"score": round(sum(REWARDS[o] for o in outcomes) / n, 4),
            **{k: round(v, 4) for k, v in rates.items()}, "lenient": round(lenient, 4),
            "searched": round(sum('>>> search("' in r for r in replies) / n, 4), "n": n,
            "examples": [{"question": q, "reply": r, "outcome": o}
                         for (q, _), r, o in list(zip(questions, replies, outcomes))[:5]]}


def evaluate_reading(model, tokenizer, which: str = "squad", limit: int = 2000) -> dict:
    """
    The go/no-go check: the reply is started with a search call and its results, which contain an answer, and the
    model only has to write the answer. 'squad' hands it the question's own SQuAD paragraph among 2 distractors;
    'retrieved' uses TriviaQA and NQ validation questions whose real BM25 top 3 contains an accepted answer.
    """
    from chat import generate, prompt_ids, segments
    if which == "squad":
        with open(DEMOS / "reading_squad_validation.jsonl") as f:
            items = [json.loads(line) for _, line in zip(range(limit), f)]
    else:
        from search import search_tool
        items = []
        for name in ["triviaqa", "nq"]:
            for q, answers in load_questions(name, "validation", 0, limit):
                query = search_query(q)
                results = search_tool(query)
                if find_answer(results, answers):
                    items.append({"question": q, "answers": answers, "prefix": f'>>> search("{query}")\n{results}'})
    prompts = []
    for item in items:  # tokenized piece by piece, as training and generation tokenize tool calls and output
        ids = prompt_ids(tokenizer, [{"role": "user", "content": item["question"]}])
        for piece, _ in segments(item["prefix"]):
            ids += tokenizer.encode(piece).ids if piece else []
        prompts.append(ids)
    replies = [tokenizer.decode(r) for r in generate(model, prompts, 32, tokenizer=tokenizer)]
    outcomes = [grade(r, item["answers"]) for r, item in zip(replies, items)]
    n = len(items)
    return {**{k: round(outcomes.count(k) / n, 4) for k in REWARDS}, "n": n,
            "examples": [{"question": i["question"], "answers": i["answers"][:3], "reply": r, "outcome": o}
                         for i, r, o in list(zip(items, replies, outcomes))[:5]]}


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
    if "reading" in args:
        build_reading()
    if "relabel" in args:
        relabel(args["relabel"], int(args.get("per_dataset", 25_000)), int(args.get("samples", 0)),
                float(args.get("temperature", 0.6)), args.get("out", "relabel"))
