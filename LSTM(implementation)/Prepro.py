import torch

class Data_orig:
    def __init__(self, file_path, block_size=3):
        # Load data
        with open(file_path, 'r', encoding='utf-8') as f:
            words = f.read().lower().splitlines()

        # Build vocabulary
        self.chars = sorted(set(''.join(words)))  
        self.chars.extend(['<sos>', '<eos>', '␀'])  
        self.chars = sorted(self.chars)  

        self.stoi = {s: i for i, s in enumerate(self.chars)}  # char -> index
        self.itos = {i: s for s, i in self.stoi.items()}  # index -> char
        self.vocab_size = len(self.chars)

        # Generate aligned sequences
        X, Y = [], []
        for w in words:
            seq = ['␀'] * (block_size - 1) + ['<sos>'] + list(w) + ['<eos>']
            for i in range(len(seq) - block_size):
                context = [self.stoi[ch] for ch in seq[i:i+block_size]]
                target = self.stoi[seq[i+block_size]]
                X.append(context)
                Y.append(target)

        # Convert to tensors
        self.X = torch.tensor(X, dtype=torch.long)  # (num_samples, block_size)
        self.Y = torch.tensor(Y, dtype=torch.long)  # (num_samples,)

        # Verify alignment
        assert self.X.shape[0] == self.Y.shape[0], "X and Y must have the same number of samples"

if __name__ == "__main__":
    data = Data_orig(r'C:\Users\Rick\Name_generator\cleaned_dataset.txt', block_size=3)
    print(f"X shape: {data.X.shape}, Y shape: {data.Y.shape}")

    if data.X.shape[0] > 0:
        print(f"Sample:\nX: {data.X[1]} -> Y: {data.Y[1]}")
    else:
        print("No data to display.")
