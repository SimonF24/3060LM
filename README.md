# 3060LM
The goal of this repo is to try to train the best language model I can on my home desktop with an NVIDIA 3060 GPU. Inspiration for this repo includes <a href="https://github.com/karpathy/nanochat/tree/master">nanochat</a>, <a href="https://github.com/KellerJordan/modded-nanogpt">the NanoGPT speedrun</a>, and <a href="https://github.com/openai/parameter-golf/">OpenAI Parameter Golf</a>.

## Dataset

The dataset I'm using for this project is the <a href="https://huggingface.co/datasets/armanc/scientific_papers">scientific papers</a> dataset from the paper "A Discourse-Aware Attention Model for Abstractive Summarization of Long Documents" (downloaded from <a href="https://github.com/armancohan/long-summarization">here</a>). This dataset consists of two zip files, arxiv-dataset.zip and pubmed-dataset.zip, which when unzipped contain test.txt, train.txt, val.txt, and vocab files.

## Project Structure
```
3060lm/
├── eda.ipynb         - Exploratory data analysis
├── preprocessing.py  - Preprocess data for tokenization
├── tokenizer.py      - Tokenize data for training
├── model.py          - Model architecture
├── train.py          - Training loop
└── evaluate.py       - Model evaluation
```

### NOTE TO SELF (DELETE BEFORE PUSHING):

If we want a question-answer version (maybe for chat finetuning?) one is available here: https://huggingface.co/datasets/armanc/ScienceQA.

Here's more information on the ChatML format we've prepared: https://gist.github.com/edwardzjl/8df07c1f7140c9a3e2f48d33a8032090

## Full System Specs

CPU: AMD Ryzen 5 3600  
GPU: NVIDIA GeForce RTX 3600  
RAM: 32 GB (4 x TEAMGROUP-UD4-3200)  
SDD: SPCC M.2 PCIe SSD (2 TB)