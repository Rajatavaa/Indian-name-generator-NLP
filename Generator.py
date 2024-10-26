import torch
import torch.nn.functional as F
from Prepro import Data_orig  

class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out), generator=g) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None
    
    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])

class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)
    
    def __call__(self, x):
        if self.training:
            xmean = x.mean(0, keepdim=True)  # batch mean
            xvar = x.var(0, keepdim=True)    # batch variance
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)  # normalize
        self.out = self.gamma * xhat + self.beta
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]

class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    
    def parameters(self):
        return []

# Model parameters
n_embd = 10
n_hidden1 = 500
n_hidden2 = 200
g = torch.Generator().manual_seed(2147483647)

# Load the dataset
dataset = Data_orig()  # This should return X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = dataset.X_train, dataset.X_test, dataset.y_train, dataset.y_test

# Initialize the character embedding matrix
C = torch.randn((dataset.vocab_size, n_embd), generator=g)

# Model layers
layers = [
    Linear(n_embd * dataset.block_size, n_hidden1, bias=False), BatchNorm1d(n_hidden1), Tanh(),
    Linear(n_hidden1, n_hidden2, bias=False), BatchNorm1d(n_hidden2), Tanh(),
    Linear(n_hidden2, dataset.vocab_size, bias=False), BatchNorm1d(dataset.vocab_size)
]

# Initialize parameters
with torch.no_grad():
    layers[-1].gamma *= 0.1  # Make last layer less confident
    for layer in layers[:-1]:
        if isinstance(layer, Linear):
            layer.weight *= 1.0

parameters = [C] + [p for layer in layers for p in layer.parameters()]
for p in parameters:
    p.requires_grad = True

# Training loop
max_steps = 100000
batch_size = 100
lri = []
lossi = []
stepi = []

for i in range(max_steps):
    ix = torch.randint(0, X_train.shape[0], (batch_size,))
    
    emb = C[X_train[ix]]  # Ensure minibatch size is correct
    
    x = emb.view(emb.shape[0], -1) # concatenate the vectors
    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x, y_train[ix]) # loss function
  
  # backward pass
    for layer in layers:
        layer.out.retain_grad() # AFTER_DEBUG: would take out retain_graph
    for p in parameters:
        p.grad = None
    loss.backward()
  
  # update
    lr = 0.1 if i < 150000 else 0.01 # step learning rate decay
    for p in parameters:
     p.data += -lr * p.grad

  # track stats
    if i % 10000 == 0: # print every once in a while
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
    lossi.append(loss.log10().item())
    with torch.no_grad():
        stepi.append([((lr*p.grad).std() / p.data.std()).log10().item() for p in parameters])

    
    
