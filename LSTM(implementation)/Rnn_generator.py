import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from Prepro import Data_orig  
from config import Config

# LSTM Model (Sequence-to-Sequence)
class NameGenerator(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, Config.n_embd)
        self.lstm = nn.LSTM(
            Config.n_embd, Config.n_hidden,
            num_layers=Config.n_layers,
            dropout=Config.dropout,
            batch_first=True
        )
        self.fc = nn.Linear(Config.n_hidden, vocab_size)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, hidden=None):
        emb = self.embedding(x)  # (B, T, C)
        lstm_out, hidden = self.lstm(emb, hidden)  # (B, T, H)
        last_output = lstm_out[:, -1, :]  # Use LAST timestep: (B, H)
        logits = self.fc(last_output)  # (B, vocab_size)
        return logits, hidden

# Helper functions
def create_dataloaders(dataset, batch_size):
    train_data, val_data = train_test_split(dataset, test_size=0.1, random_state=42)
    
    train_loader = DataLoader(
        train_data, batch_size=batch_size,
        shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size,
        shuffle=False, pin_memory=True
    )
    return train_loader, val_loader

def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(Config.device), y.to(Config.device)
            logits, _ = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
    return total_loss / len(val_loader)

# Training
def main():
    # Load and prepare data
    dataset = Data_orig(Config.dataset_path)
    assert dataset.X.shape[0] == dataset.Y.shape[0], "X and Y must be aligned sequences"
    
    train_loader, val_loader = create_dataloaders(
        TensorDataset(dataset.X, dataset.Y),
        Config.batch_size
    )

    # Initialize model
    model = NameGenerator(dataset.vocab_size).to(Config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.lr,
        weight_decay=Config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=2, factor=0.5, verbose=True
    )
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_val_loss = float("inf")
    step = 0
    while step < Config.max_steps:
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(Config.device)
            y_batch = y_batch.to(Config.device)

            # Forward pass
            logits, _ = model(x_batch)  # logits: (B, vocab_size)
            loss = criterion(logits, y_batch)
            

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.clip_norm)
            optimizer.step()

            # Validation and checkpointing
            if step % Config.checkpoint_interval == 0:
                val_loss = validate(model, val_loader, criterion)
                scheduler.step(val_loss)
                
                print(f"Step {step:06d} | "
                      f"Train Loss: {loss.item():.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"LR: {optimizer.param_groups[0]['lr']:.2e}")
                model.train()
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), "best_model.pth")

            step += 1
            if step >= Config.max_steps:
                break

    print("Training completed. Best model saved to 'best_model.pth'")

if __name__ == "__main__":
    main()