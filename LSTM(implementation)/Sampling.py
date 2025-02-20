import torch
import torch.nn as nn
from Prepro import Data_orig  # Ensure this matches your preprocessing class
from config import Config

# LSTM Model (must match training architecture)
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

    def forward(self, x, hidden=None):
        emb = self.embedding(x)  # (B, T, C)
        lstm_out, hidden = self.lstm(emb, hidden)  # (B, T, H)
        logits = self.fc(lstm_out)  # (B, T, vocab_size)
        return logits, hidden

# Sampling Function
def sample(model, dataset, start_token="<sos>", max_length=20, temperature=1.0):
    model.eval()
    with torch.no_grad():
        # Initialize hidden state and input
        hidden = None
        x = torch.tensor([[dataset.stoi[start_token]]], dtype=torch.long).to(Config.device)

        # Generate sequence
        name = []
        for _ in range(max_length):
            logits, hidden = model(x, hidden)  # (1, 1, vocab_size)
            probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)  # (1, vocab_size)
            next_token = torch.multinomial(probs, num_samples=1).item()  # Sample next token
            if next_token == dataset.stoi["<eos>"]:  # Stop if end token is generated
                break
            name.append(dataset.itos[next_token])
            x = torch.tensor([[next_token]], dtype=torch.long).to(Config.device)

        return "".join(name)

# Load Model and Dataset
def load_model_and_dataset(model_path, dataset_path):
    # Load dataset for vocabulary
    dataset = Data_orig(dataset_path, block_size=3)

    # Initialize model
    model = NameGenerator(dataset.vocab_size).to(Config.device)

    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=Config.device))
    model.eval()

    return model, dataset

# Main Inference Function
def main():
    # Paths
    model_path = r"C:\Users\Rick\Name_generator\best_model.pth"  # Path to your trained model
    dataset_path = r'C:\Users\Rick\Name_generator\cleaned_dataset.txt'  # Path to your dataset

    # Load model and dataset
    model, dataset = load_model_and_dataset(model_path, dataset_path)

    # Generate names
    temperature = 0.8  # Adjust for more/less randomness
    for _ in range(10):  # Generate 10 names
        name = sample(model, dataset, temperature=temperature)
        print(name)

if __name__ == "__main__":
    main()