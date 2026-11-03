'''
This largely follows the guide here:
https://huggingface.co/learn/llm-course/chapter6/8#building-a-bpe-tokenizer-from-scratch

Training the tokenizer takes around 6 minutes on my machine.
Running the tokenizer on the datasets takes around an hour on my machine.
'''
import numpy as np
from pathlib import Path
from tokenizers import decoders, models, pre_tokenizers, processors, Tokenizer, trainers
from tqdm import tqdm

from preprocessing import PLACEHOLDERS

file_dir = Path(__file__).parent

DATASET_NAMES = ["arxiv-dataset", "pubmed-dataset"]
PREPROCESSED_DATASET_FOLDER = file_dir.parent / "data/pretraining/preprocessed/"
TRAIN_TOKENIZER: bool = True # set to False to skip training and just load an existing tokenizer from disk
# and apply it to the preprocessed datasets
TOKENIZER_SAVE_PATH = file_dir.parent / "tokenizer/tokenizer.json" # where to save the trained tokenizer
VOCAB_SIZE = 24_576 # Since our model is small we go smaller than the commonly used value of 32,768.
# Our dataset being a single domain (scientific articles) also means we can get away with a smaller
# vocab size than the more general-purpose models, which need to cover a wider range of language and topics.

# We include the ChatML format tokens in anticipation of future chat fine-tuning
# For more information on that format see here: https://gist.github.com/edwardzjl/8df07c1f7140c9a3e2f48d33a8032090
CHATML = ["<|im_start|>", "<|im_end|>"] # im stands for "instant message", so named for historical im chats afaik
END_OF_TEXT = "<|endoftext|>" # This is used to denote the end of an article
PAD = "<pad>" # This is used for padding sequences to the same length in a batch
SPECIAL_TOKENS = [
    END_OF_TEXT,
    PAD,
    *CHATML,
    *PLACEHOLDERS
]

def train_tokenizer(dataset_folder: Path, dataset_names: list[str], tokenizer_save_path: Path,
                    vocab_size: int = VOCAB_SIZE, special_tokens: list[str] = SPECIAL_TOKENS):
    """
    Trains the tokenizer on the preprocessed dataset and saves it to output_path
    """
    tokenizer = Tokenizer(models.BPE())
    # The text is already normalized as part of the preprocessing,
    # so we don't need to add a normalizer here
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    
    # Forming an iterator over the dataset files for the trainer
    def _dataset_iterator():
        for dataset_name in dataset_names:
            file_path = dataset_folder / dataset_name / "train.txt"
            with open(file_path, 'r') as f:
                for line in f:
                    yield line
    
    tokenizer.train_from_iterator(_dataset_iterator(), trainer=trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.save(str(tokenizer_save_path))

def tokenize_datasets(dataset_folder: Path, dataset_names: list[str], tokenizer_path: Path, output_path: Path):
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos_id = tokenizer.token_to_id(END_OF_TEXT)

    splits = ["train", "val", "test"]
    for split in splits:
        out_file = output_path / f"{split}.bin"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "wb") as f_out:
            for dataset_name in dataset_names:
                input_path = dataset_folder / dataset_name / f"{split}.txt"
                with open(input_path) as f_in:
                    for line in tqdm(f_in, desc=f"{dataset_name}/{split}"):
                        ids = tokenizer.encode(line.rstrip("\n")).ids
                        ids.append(eos_id)
                        np.array(ids, dtype=np.uint16).tofile(f_out)

        tokens_written = out_file.stat().st_size // 2
        print(f"{split}.bin: {tokens_written:,} tokens")
    
if __name__ == "__main__":
    if TRAIN_TOKENIZER:
        train_tokenizer(PREPROCESSED_DATASET_FOLDER, DATASET_NAMES, TOKENIZER_SAVE_PATH)
    print(f"Loading tokenizer from {TOKENIZER_SAVE_PATH}")
    tokenize_datasets(PREPROCESSED_DATASET_FOLDER, DATASET_NAMES, TOKENIZER_SAVE_PATH,
                      file_dir.parent / "data/tokenized/")