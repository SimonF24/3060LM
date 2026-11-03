import numpy as np
from pathlib import Path
import torch

class ScientificPapersDataset:
    
    def __init__(self, directory: str, mode: str = 'train'):
        self.directory = Path(directory)
        self.mode = mode
        self.data_path = self.directory / f"{mode}.bin"
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset file not found at {self.data_path}")
        
        self.data_array = np.memmap(self.data_path, dtype=np.uint16, mode='r')
        self.num_tokens = self.data_array.shape[0]
        
    def get_batch(self, batch_size: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly samples batch_size sequences of length seq_len from the dataset
        """
        indices = np.random.randint(0, len(self.data_array) - seq_len, size=batch_size)
        inputs = np.stack([self.data_array[i:i + seq_len] for i in indices])
        targets = np.stack([self.data_array[i + 1:i + seq_len + 1] for i in indices])
        return torch.from_numpy(inputs.astype(np.int64)), torch.from_numpy(targets.astype(np.int64))
    
    def get_batch_deterministic(
        self, start_idx: int, batch_size: int, seq_len: int
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a deterministic batch of sequences starting from start_idx
        """
        if start_idx + batch_size * seq_len + 1 > len(self.data_array):
            raise IndexError("Batch exceeds dataset bounds")
        
        inputs = np.stack(
            [self.data_array[start_idx + i*seq_len:start_idx + (i + 1)*seq_len]
                for i in range(batch_size)]
        )
        targets = np.stack(
            [self.data_array[start_idx + i*seq_len + 1:start_idx + (i + 1)*seq_len + 1]
                for i in range(batch_size)]
        )
        return torch.from_numpy(inputs.astype(np.int64)), torch.from_numpy(targets.astype(np.int64))