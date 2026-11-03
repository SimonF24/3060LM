Long Term:
- Implement the dataset get_item function
- Implement pretraining (include a training log for inclusion in the GitHub)
    - We can also include tricks like starting with a smaller context window size and expanding later in a fine tune. There are probably other tricks as well we can find from the nanoGPT speedrun or parameter golf repos.
- Implement Chat Fine-Tuning with the <a href="https://huggingface.co/datasets/armanc/ScienceQA">ScienceQA dataset</a>.
Short Term:
- Read the <a href="https://arxiv.org/pdf/2203.15556">Chincilla scaling laws</a> paper and maybe <a href="https://arxiv.org/pdf/2403.08540">this paper</a> to figure out the optimal model size for our token budget. We don't have to be exactly optimal but should at least note somewhere what optimal would be.