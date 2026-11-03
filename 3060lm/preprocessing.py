"""
Preprocessing for the arxiv/pubmed datasets.

Cleans LaTeX-converter artifacts out of the raw article text so the language
model trains on coherent prose. Tags that carry sentence structure (math
expressions, citations, cross-references) are replaced with placeholder tokens
rather than deleted, so sentences stay grammatical. Articles that are mostly
artifacts even after substitution are dropped. See eda.ipynb for the artifact
survey this is based on.

This will take a while, it takes a little over 3.5 hours for both datasets on my machine.
This could definitely be optimized (cleaned up and parallelized) but seeing as
I've already run it on my dataset there isn't really a point.
Recommend testing with FIRST_N before running on the full datasets.
"""
import json
from pathlib import Path
import re
from tqdm import tqdm

file_dir = Path(__file__).parent
datasets_to_process = [
    file_dir / "../data/pretraining/raw/arxiv-dataset/",
    file_dir / "../data/pretraining/raw/pubmed-dataset/"
]
output_folder = file_dir / "../data/pretraining/preprocessed/"

FIRST_N: int | None = None # for testing, set to a number N to only process the first N articles
# For testing, around 50 is reasonable. In the first 50 we can see that some artifacts survive
# but the text is much cleaned up and should be good enough for our purposes, although
# we can't expect perfection from the final LM

# --- Placeholder tokens -----------------------------------------------------
MATH = "<math>"
CITE = "<cite>"
REF = "<ref>"
PLACEHOLDERS = {MATH, CITE, REF}

# --- Article quality thresholds -----------------------------------------------------
# An article that is mostly artifacts makes poor LM data even after
# placeholder substitution, so drop it outright rather than dropping individual
# sentences (which would leave the surrounding ones incoherent)
MAX_PLACEHOLDER_RATIO = 0.15  # max fraction of article tokens allowed to be placeholders
MIN_ARTICLE_WORDS = 50        # drop degenerate / near-empty articles

# --- Artifact patterns ------------------------------------------------------
# (compiled pattern, replacement) applied in order to every sentence.
SUBSTITUTIONS = [
    # @xmath53 etc. — a math expression stood here; the digits are macro IDs
    (re.compile(r'@xmath\d+'), f' {MATH} '),
    # @xcite — an inline citation
    (re.compile(r'@xcite'), f' {CITE} '),
    # @noop — a do-nothing LaTeX macro; drop it entirely
    (re.compile(r'@noop'), ' '),
    # [ 4 ] , [ 9 , 10 ] — numeric citations
    (re.compile(r'\[ \d[\d ,]*\]'), f' {CITE} '),
    # [ figure 2 ] , [ table 1 ] — bracketed cross-references
    (re.compile(r'\[ [a-zA-Z][^\]]{0,50}\]'), f' {REF} '),
    # _ italic _ and * bold * — keep the inner words, strip the markers
    # Non-greedy so adjacent spans don't get merged into one match
    (re.compile(r'_ (\S.*?\S|\S) _'), r'\1'),
    (re.compile(r'\* (\S.*?\S|\S) \*'), r'\1'),
    # Stray newlines left in the pubmed text, especially important since the final format
    # is one article per line. Replace with space to avoid merging words together
    (re.compile(r'\n'), ' '),
]

_WHITESPACE = re.compile(r'\s+')
# A sentence "ends in punctuation" if its last char is neither word nor space.
_ENDS_IN_PUNCT = re.compile(r'[^\w\s]$')


def clean_sentence(sentence: str) -> str:
    """Replace LaTeX-converter artifacts in a single sentence."""
    for pattern, replacement in SUBSTITUTIONS:
        sentence = pattern.sub(replacement, sentence)
    return _WHITESPACE.sub(' ', sentence).strip()


def join_sentences(sentences: list[str]) -> str:
    """
    Join cleaned sentences into one article string.

    A sentence that ends in punctuation gets a space before the next one
    and one that doesn't is joined directly. This should make the text uniform
    since the original documents don't include a space if the previous sentence
    ends in punctuation. See eda.ipynb for details
    """
    out = ""
    for sentence in sentences:
        if not sentence:
            continue
        if out and _ENDS_IN_PUNCT.search(out):
            out += " "
        out += sentence
    return out


def is_low_quality(article: str) -> bool:
    """True if the article is too short or too artifact-dense to train on."""
    tokens = article.split()
    if len(tokens) < MIN_ARTICLE_WORDS:
        return True
    placeholders = sum(token in PLACEHOLDERS for token in tokens)
    return placeholders / len(tokens) > MAX_PLACEHOLDER_RATIO


def preprocess_file(file_path: Path, output_folder: Path, first_n: int | None = None) -> None:
    """
    Clean one raw split file and write the result to data/processed/.

    Output is plain text, one cleaned article per line. Only article_text is
    used, abstract_text and its <S> and </S> tags are dropped.
    """
    dataset_name = file_path.parent.name
    out_path = output_folder / dataset_name / file_path.name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept = dropped = 0
    with open(file_path) as f_in, open(out_path, "w") as f_out:
        desc = f"{dataset_name}/{file_path.name}"
        for line in tqdm(f_in, desc=desc, unit=" articles"):
            article = json.loads(line)
            sentences = [clean_sentence(s) for s in article.get("article_text", [])]
            text = join_sentences(sentences)
            if is_low_quality(text):
                dropped += 1
                continue
            f_out.write(text + "\n")
            kept += 1
            if first_n is not None and kept >= first_n:
                break

    print(f"{dataset_name}/{file_path.name}: kept {kept:,}, dropped {dropped:,}")


if __name__ == "__main__":
    for dataset in datasets_to_process:
        for file in ['train.txt', 'val.txt', 'test.txt']:
            preprocess_file(dataset / file, output_folder, first_n=FIRST_N)
            if FIRST_N is not None:
                break # break after the first file for testing
        if FIRST_N is not None:
            break # break after the first dataset for testing
