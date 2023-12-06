import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import SelfAttentionHead as Head
from utils import FeedForward, MultiHeadAttention, Block
from utils import Transformer

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# class Transformer(nn.Module):
#     def __init__(self,dm,vocab_size,h=6,N=6,version='original'):
#         super().__init__()
#         # embedding_length = dm
#         self.token_embedding_table = nn.Embedding(vocab_size,dm)
#         self.position_embedding_table = nn.Embedding(block_size,dm)
#         if version=='original':
#             self.blocks = nn.Sequential(*[Block(dm,h) for _ in range(N)])
#         elif version == 'alternate':
#             self.blocks = nn.Sequential(*[Block2(dm,h) for _ in range(N)])
#         self.ln = nn.LayerNorm(dm)
#         self.lm_head = nn.Linear(dm,vocab_size)
#         self.logits_only=False
#         self.apply(self._init_weights)

#     # How does this work?
#     ####################################################
#     def _init_weights(self, module):
#         if isinstance(module, nn.Linear):
#             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
#             if module.bias is not None:
#                 torch.nn.init.zeros_(module.bias)
#         elif isinstance(module, nn.Embedding):
#             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
#     #######################################################
#     def forward(self,idx,targets=None):
#         B,T = idx.shape # batch size, context length
#         token_embed=self.token_embedding_table(idx)
#         pos_embed=self.position_embedding_table(torch.arange(T,device=device))
#         x = token_embed+pos_embed
#         x=self.blocks(x)
#         x=self.ln(x)
#         logits=self.lm_head(x)
#         if targets is None:
#             loss=None
#         else:
#             flat_logits=logits.view(-1,vocab_size)
#             flat_targets=targets.view(-1)
#             loss=F.cross_entropy(flat_logits,flat_targets)
#         if self.logits_only:
#             return logits
#         else:
#             return logits,loss
#     def generate(self,idx,max_new_tokens):
#         for _ in range(max_new_tokens):
#             context_idx=idx[:,-block_size:]
#             logits,_=self(context_idx)
#             last_logits=logits[:,-1,:] # Only care about next word prediction
#             probs=F.softmax(last_logits,dim=-1)
#             idx_next=torch.multinomial(probs,num_samples=1)
#             idx=torch.cat((idx,idx_next),dim=1)
#         return idx

# class Transformer2(nn.Module):

#     def __init__(self,dm,vocab_size,h=6,N=6,version='original'):
#         super().__init__()
#         # each token directly reads off the logits for the next token from a lookup table
#         self.token_embedding_table = nn.Embedding(vocab_size, dm)
#         self.position_embedding_table = nn.Embedding(block_size, dm)
#         if version=='original':
#             self.blocks = nn.Sequential(*[Block(dm,h) for _ in range(N)])
#         self.ln = nn.LayerNorm(dm) # final layer norm
#         self.lm_head = nn.Linear(dm, vocab_size)

#         # better init, not covered in the original GPT video, but important, will cover in followup video
#         self.apply(self._init_weights)

#     def _init_weights(self, module):
#         if isinstance(module, nn.Linear):
#             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
#             if module.bias is not None:
#                 torch.nn.init.zeros_(module.bias)
#         elif isinstance(module, nn.Embedding):
#             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

#     def forward(self, idx, targets=None):
#         B, T = idx.shape

#         # idx and targets are both (B,T) tensor of integers
#         tok_emb = self.token_embedding_table(idx) # (B,T,C)
#         pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)

#         x = tok_emb + pos_emb # (B,T,C)
#         x = self.blocks(x) # (B,T,C)
#         x = self.ln(x) # (B,T,C)
#         logits = self.lm_head(x) # (B,T,vocab_size)

#         if targets is None:
#             loss = None
#         else:
#             B, T, C = logits.shape
#             logits = logits.view(B*T, C)
#             targets = targets.view(B*T)
#             loss = F.cross_entropy(logits, targets)

#         return logits, loss

#     def generate(self, idx, max_new_tokens):
#         # idx is (B, T) array of indices in the current context
#         for _ in range(max_new_tokens):
#             # crop idx to the last block_size tokens
#             idx_cond = idx[:, -block_size:]
#             # get the predictions
#             logits, loss = self(idx_cond)
#             # focus only on the last time step
#             logits = logits[:, -1, :] # becomes (B, C)
#             # apply softmax to get probabilities
#             probs = F.softmax(logits, dim=-1) # (B, C)
#             # sample from the distribution
#             idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
#             # append sampled index to the running sequence
#             idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
#         return idx

model = Transformer(n_embd,vocab_size)
#model = Transformer2(n_embd,vocab_size)

m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
