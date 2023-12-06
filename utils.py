import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# from tqdm import tqdm
import requests
import os
import csv
import argparse

# Parameters
block_size = 256
batch_size = 64
eval_interval=500
eval_iters=500
dm = 384 # Model / embedding size
dk=64 # Head size
h=6 # Number of heads in multihead attn
lr=3e-4 # Learning rate
N=6 # Number of layers
# device=0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_itrs=5001
dropout=0.2

# Set seed
torch.manual_seed(1337)

# Download a sample text file (e.g., "The Complete Works of William Shakespeare")
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
file_path = "shakespeare.txt"

if not os.path.exists(file_path):
    response = requests.get(url)
    with open(file_path, 'w') as file:
        file.write(response.text)

# Read in text file
with open(file_path,'r',encoding='utf-8') as f:
    text = f.read()

# Get char list
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Define encoding and decoding functions
s2i = {ch:i for i,ch in enumerate(chars)}
i2s = chars

encode = lambda s: [s2i[c] for c in s]
decode = lambda l: ''.join([i2s[i] for i in l])

# Make tokenized datasets
data = torch.tensor(encode(text))
n = int(0.9*len(data))
train_data = data[:n]
test_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else test_data
    idx = torch.randint(len(data)-block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in idx])
    y = torch.stack([data[i+1:i+1+block_size] for i in idx])
    x,y=x.to(device),y.to(device)
    return x,y

class SelfAttentionHead(nn.Module):
    def __init__(self,dm,dk,dv,dropout=0.2):
        super().__init__()
        self.W_k = nn.Linear(dm,dk,bias=False)
        self.W_q = nn.Linear(dm,dk,bias=False)
        self.W_v = nn.Linear(dm,dv,bias=False)
        self.tril=torch.tril(torch.ones((block_size,block_size),device=device))
        self.dropout = nn.Dropout(dropout) # New
    def forward(self,x):
        B,T,C=x.shape # New
        k=self.W_k(x)
        q=self.W_q(x)
        v=self.W_v(x)
        wei = q@k.transpose(-2,-1)*k.shape[-1]**-0.5
        wei=wei.masked_fill(self.tril[:T,:T]==0,float('-inf')) # New
        wei=torch.softmax(wei,dim=-1)
        wei=self.dropout(wei) # New
        out=wei@v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self,dm,dk,dv,h,dropout=0.2):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttentionHead(dm,dk,dv) for i in range(h)])
        self.W_o = nn.Linear(dv*h,dm)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        concat = torch.cat([head(x) for head in self.heads],dim=-1)
        proj = self.W_o(concat)
        out = self.dropout(proj) # Like spiking?
        return out

class FeedForward(nn.Module):
    def __init__(self,dm,dropout=0.2):
        super().__init__()
        self.ffn = nn.Sequential(
        nn.Linear(dm,4*dm),
        nn.ReLU(),
        nn.Linear(4*dm,dm),
        nn.Dropout(dropout))
    def forward(self,x):
        return self.ffn(x)

class Block(nn.Module):
    def __init__(self,dm,h):
        super().__init__()
        dk = dm // h
        dv = dk
        assert(dk*h==dm) # Check the input/output size of block is same
        self.mha = MultiHeadAttention(dm,dk,dv,h)
        self.ffn = FeedForward(dm)
        self.ln1 = nn.LayerNorm(dm)
        self.ln2 = nn.LayerNorm(dm)
        # self.ln3 = nn.LayerNorm(dm)
    def forward(self,x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        # x = self.ln3(x)
        return x

class Block2(nn.Module):
    def __init__(self,dm,h):
        super().__init__()
        dk = dm // h
        dv = dk
        assert(dk*h==dm) # Check the input/output size of block is same
        self.mha = MultiHeadAttention(dm,dk,dv,h)
        self.ffn = FeedForward(dm)
        self.ln1 = nn.LayerNorm(dm)
        self.ln2 = nn.LayerNorm(dm)
        # self.ln3 = nn.LayerNorm(dm)
    def forward(self,x):
        x = self.ln1(x + self.mha(x))
        x = self.ln2(x + self.ffn(x))
        # x = self.ln3(x)
        return x


class Transformer(nn.Module): # Old

    # def __init__(self):
    def __init__(self,dm,vocab_size,h=4,N=3,version='original'):

        super().__init__(dm,vocab_size)
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd,n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)

        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

# class Transformer(nn.Module):
#     def __init__(self,dm,vocab_size,h=4,N=3,version='original'):
#         super().__init__()
#         # embedding_length = dm
#         self.token_embedding_table = nn.Embedding(vocab_size,dm,device=device)
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

# @torch.no_grad()
# def estimate_loss(model):
#     out = {}
#     model.eval()
#     losses=torch.zeros(eval_iters)
#     for k in range(eval_iters):
#         xb,yb = get_batch('test')
#         logits,loss = model(xb,yb)
#         losses[k] = loss.item()
#     model.train()
#     return torch.mean(losses)