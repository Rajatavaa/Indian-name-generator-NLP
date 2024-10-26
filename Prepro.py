import torch
from sklearn.model_selection import train_test_split

class Data_orig():
    def __init__(self):
         with open(r'C:\Users\Rick\lulu\.venv\cleaned_dataset.txt','r',encoding='utf-8')as f:
            words = f.read()
            words = words.lower()
            words = words.splitlines()
            
            self.chars = sorted(list(set(''.join(words))))
            self.stoi = {s: i + 1 for i, s in enumerate(self.chars)}
            self.stoi['.'] = 0
            self.itos = {i: s for s, i in self.stoi.items()}
            self.vocab_size = len(self.itos)
            self.block_size = 3

            X, Y = [], []
            for w in words:
                context = [0] * self.block_size
                for ch in w + '.':
                    ix = self.stoi[ch]
                    X.append(context)
                    Y.append(ix)
                    context = context[1:] + [ix]  # crop and append

            self.X = torch.tensor(X)
            self.Y = torch.tensor(Y)
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.Y, test_size=0.1, random_state=42)

# Move this outside the class definition
if __name__ == "__main__":
    data = Data_orig()
    print(f"Vocabulary size: {data.vocab_size}")
    print(f"Training set size: {len(data.X_train)}")