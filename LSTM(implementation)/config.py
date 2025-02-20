import torch
class Config:
    # Model
    n_embd = 256
    n_hidden = 512
    n_layers = 3  # Reduce depth slightly
    dropout = 0.1  # Reduce dropout
    
    # Training
    batch_size = 128  # Reduce batch size for better stability
    max_steps = 50000  # Reduce for faster testing
    lr = 0.0005  # Lower learning rate
    weight_decay = 1e-4  # Reduce weight decay
    clip_norm = 5.0  # Increase gradient clipping
    checkpoint_interval = 2500  # Save more frequently
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Data
    dataset_path = "cleaned_dataset.txt"
