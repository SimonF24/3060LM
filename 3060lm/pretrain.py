from dataclasses import dataclass, field
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from dataset import ScientificPapersDataset
from model import TransformerConfig, Transformer

@dataclass
class TrainingConfig:
    batch_size: int = 16
    dataset_directory: str = "/home/simon/github/3060LM/data/pretraining/tokenized/"
    device: torch.device = torch.device("cuda")  # or "cpu"
    # You can check torch.cuda.is_available() if you want to fall back to CPU when CUDA is not available
    epoch_tokens: int = 10_000_000 # Number of tokens to consider as one epoch
    learning_rate: float = 1e-4
    log_dir: str = "/home/simon/github/3060LM/training_runs"
    loss_function: str = "cross_entropy"
    max_val_batches: int = 200
    num_epochs: int = 100
    # Note that the total number of tokens seen is epoch_tokens * num_epochs
    resume_from_latest: bool = False # Whether to resume training from the latest checkpoint
    resume_from: str | None = None # Specific checkpoint to resume training from
    save_checkpoints_every: int = 5 # Save a checkpoint every N epochs
    seq_len_schedule: list[int] = field(default_factory=lambda: [512] * 80 + [1024] * 20)
    # First 80 epochs with seq_len=512, next 20 with seq_len=1024
    # The length of seq_len_schedule must match num_epochs
    val_seq_len: int = 512
    # Val is kept at a single seq_len so numbers are comparable across epochs
    weight_decay: float = 0.01
    
    def __post_init__(self):
        if self.loss_function not in ["cross_entropy"]:
            raise ValueError(
                f"Invalid loss function: {self.loss_function}. "
                "Currently only cross entropy is supported."
            )
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss()
        if self.resume_from_latest and (self.resume_from is not None):
            raise ValueError(
                "Both resume_from_latest and resume_from are set. ",
                "Only one should be set at a time"
            )
        if len(self.seq_len_schedule) < self.num_epochs:
            raise ValueError(
                "seq_len_schedule has fewer than num_epochs elements. "
                "It should have one element for each epoch"
            )
        

def train_model(
    model: nn.Module, train_ds: ScientificPapersDataset, val_ds: ScientificPapersDataset,
    training_config: TrainingConfig, optimizer: torch.optim.Optimizer, start_epoch: int,
    training_tokens_seen: int, writer: SummaryWriter) -> None:
    
    for epoch in range(start_epoch, training_config.num_epochs + 1):
        print(f"Epoch {epoch}/{training_config.num_epochs}")
        
        # Training loop
        epoch_seq_len = training_config.seq_len_schedule[epoch - 1]
        total_train_loss = 0.0
        total_grad_norm = 0.0
        epoch_batches = training_config.epoch_tokens // (training_config.batch_size * epoch_seq_len)
        model.train()
        for i in tqdm(range(epoch_batches), desc=f"Training Epoch {epoch}"):
            
            # Get a batch from the dataset
            inputs, targets = train_ds.get_batch(training_config.batch_size, epoch_seq_len)
            inputs, targets = inputs.to(training_config.device), targets.to(training_config.device)

            # Run the model, compute the loss, clip the gradients, and perform the update
            with torch.autocast(device_type=training_config.device.type, dtype=torch.bfloat16):
                # Note that grad scaling isn't necessary since we use bfloat16, not float16.
                # bfloat16 can represent the same range of values as float32, but with less precision.
                outputs = model(inputs)
                loss = training_config.loss_fn(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss.backward()
            batch_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            total_grad_norm += batch_grad_norm.item()
            optimizer.step()
            optimizer.zero_grad()

            total_train_loss += loss.item()
        epoch_tokens_seen = epoch_batches * training_config.batch_size * epoch_seq_len
        training_tokens_seen += epoch_tokens_seen
        average_loss = total_train_loss / epoch_batches
        writer.add_scalar("Loss/train", average_loss, training_tokens_seen)
        average_grad_norm = total_grad_norm / epoch_batches
        writer.add_scalar("GradNorm/train", average_grad_norm, training_tokens_seen)
        
        # Validation loop
        total_val_loss = 0.0
        val_batches = min(
            training_config.max_val_batches,
            val_ds.num_tokens // (training_config.batch_size * training_config.val_seq_len)
        )
        model.eval()
        for i in tqdm(range(val_batches), desc=f"Validation Epoch {epoch}"):
            
            start_index = i * training_config.batch_size * training_config.val_seq_len
            inputs, targets = val_ds.get_batch_deterministic(
                start_index, training_config.batch_size, training_config.val_seq_len)
            inputs, targets = inputs.to(training_config.device), targets.to(training_config.device)

            with torch.no_grad():
                with torch.autocast(device_type=training_config.device.type, dtype=torch.bfloat16):
                    outputs = model(inputs)
                    loss = training_config.loss_fn(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            total_val_loss += loss.item()
        average_loss = total_val_loss / val_batches
        writer.add_scalar("Loss/val", average_loss, training_tokens_seen)
        
        # Save checkpoint
        if epoch % training_config.save_checkpoints_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "training_tokens_seen": training_tokens_seen
            }, f"{training_config.log_dir}/checkpoint_epoch_{epoch}.pt"
            )
    return
    
    
if __name__ == "__main__":
    
    model_config = TransformerConfig()
    training_config = TrainingConfig()
    
    train_ds = ScientificPapersDataset(directory=training_config.dataset_directory, mode="train")
    val_ds = ScientificPapersDataset(directory=training_config.dataset_directory, mode="val")
    test_ds = ScientificPapersDataset(directory=training_config.dataset_directory, mode="test")
    
    model = Transformer(model_config).to(training_config.device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=training_config.learning_rate, weight_decay=training_config.weight_decay)
    
    writer = SummaryWriter(log_dir=training_config.log_dir)
    
    if training_config.resume_from:
        checkpoint = torch.load(training_config.resume_from)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        training_tokens_seen = checkpoint["training_tokens_seen"]
    elif training_config.resume_from_latest:
        checkpoints = list(Path(training_config.log_dir).glob("checkpoint_epoch_*.pt"))
        if len(checkpoints) == 0:
            raise ValueError(
                "training_config.resume_from_latest was set to True "
                "but no checkpoints were found in training_config.log_dir"
            )
        latest_checkpoint_path = max(checkpoints, key=lambda path: int(path.stem.split("_epoch_")[-1]))
        latest_checkpoint = torch.load(latest_checkpoint_path)
        model.load_state_dict(latest_checkpoint["model_state_dict"])
        optimizer.load_state_dict(latest_checkpoint["optimizer_state_dict"])
        start_epoch = latest_checkpoint["epoch"] + 1
        training_tokens_seen = latest_checkpoint["training_tokens_seen"]
    else:
        start_epoch = 1
        training_tokens_seen = 0
    
    print(f"Model size: {sum(p.numel() for p in model.parameters())} parameters")
    train_model(model, train_ds, val_ds, training_config, optimizer, start_epoch, training_tokens_seen, writer)