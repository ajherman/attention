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
from utils import *
#from torch.utils.tensorboard import SummaryWriter

# Parameters
block_size = 256
batch_size = 64
eval_interval=200
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

# class SelfAttentionHead(nn.Module):
#     def __init__(self,dm,dk,dv,dropout=0.2):
#         super().__init__()
#         self.W_k = nn.Linear(dm,dk,bias=False)
#         self.W_q = nn.Linear(dm,dk,bias=False)
#         self.W_v = nn.Linear(dm,dv,bias=False)
#         self.tril=torch.tril(torch.ones((block_size,block_size),device=device))
#         self.dropout = nn.Dropout(dropout) # New
#     def forward(self,x):
#         B,T,C=x.shape # New
#         k=self.W_k(x)
#         q=self.W_q(x)
#         v=self.W_v(x)
#         wei = q@k.transpose(-2,-1)*k.shape[-1]**-0.5
#         wei=wei.masked_fill(self.tril[:T,:T]==0,float('-inf')) # New
#         wei=torch.softmax(wei,dim=-1)
#         wei=self.dropout(wei) # New
#         out=wei@v
#         return out

# class MultiHeadAttention(nn.Module):
#     def __init__(self,dm,dk,dv,h,dropout=0.2):
#         super().__init__()
#         self.heads = nn.ModuleList([SelfAttentionHead(dm,dk,dv) for i in range(h)])
#         self.W_o = nn.Linear(dv*h,dm)
#         self.dropout = nn.Dropout(dropout)
#     def forward(self,x):
#         concat = torch.cat([head(x) for head in self.heads],dim=-1)
#         proj = self.W_o(concat)
#         out = self.dropout(proj) # Like spiking?
#         return out

# class FeedForward(nn.Module):
#     def __init__(self,dm,dropout=0.2):
#         super().__init__()
#         self.ffn = nn.Sequential(
#         nn.Linear(dm,4*dm),
#         nn.ReLU(),
#         nn.Linear(4*dm,dm),
#         nn.Dropout(dropout))
#     def forward(self,x):
#         return self.ffn(x)

# class Block(nn.Module):
#     def __init__(self,dm,h):
#         super().__init__()
#         dk = dm // h
#         dv = dk
#         assert(dk*h==dm) # Check the input/output size of block is same
#         self.mha = MultiHeadAttention(dm,dk,dv,h)
#         self.ffn = FeedForward(dm)
#         self.ln1 = nn.LayerNorm(dm)
#         self.ln2 = nn.LayerNorm(dm)
#         # self.ln3 = nn.LayerNorm(dm)
#     def forward(self,x):
#         x = x + self.mha(self.ln1(x))
#         x = x + self.ffn(self.ln2(x))
#         # x = self.ln3(x)
#         return x

# class Block2(nn.Module):
#     def __init__(self,dm,h):
#         super().__init__()
#         dk = dm // h
#         dv = dk
#         assert(dk*h==dm) # Check the input/output size of block is same
#         self.mha = MultiHeadAttention(dm,dk,dv,h)
#         self.ffn = FeedForward(dm)
#         self.ln1 = nn.LayerNorm(dm)
#         self.ln2 = nn.LayerNorm(dm)
#         # self.ln3 = nn.LayerNorm(dm)
#     def forward(self,x):
#         x = self.ln1(x + self.mha(x))
#         x = self.ln2(x + self.ffn(x))
#         # x = self.ln3(x)
#         return x

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
#         return logits,loss
#     def generate(self,idx,max_new_tokens):
#         for _ in range(max_new_tokens):
#             context_idx=idx[:,-block_size:]
#             logits,_=self(context_idx)
#             last_logits=logits[:,-1,:] # Only care about next word prediction
#             probs=F.softmax(last_logits,dim=-1)
#             idx_next=torch.multinomial(probs,num_samples=1)
#             idx=torch.cat((idx,idx_next),dim=1)
#         return idx

@torch.no_grad()
# def estimate_loss(model):
#     out = {}
#     model.eval()
#     losses=torch.zeros(eval_iters)
#     for k in range(eval_iters):
#         xb,yb = get_batch('train')
#         logits,loss = model(xb,yb)
#         losses[k] = loss.item()
#     model.train()
#     return torch.mean(losses)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default='original', help='Specify the version')
    parser.add_argument('--filepath', type=str, help='Specify the file path')
    args = parser.parse_args()
    version = args.version
    filepath = args.filepath

    # Make / load model
    if os.path.exists('transformer_' + version + '.pt'):
        m = torch.load('transformer_' + version + '.pt')
    else:
        m = Transformer(dm=dm, vocab_size=vocab_size, h=h, N=N, version=version)
    # print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
    m.to(device)
    optimizer = torch.optim.AdamW(m.parameters(), lr=lr)
    
    # # Visualize model
    # m.logits_only=True
    # writer = SummaryWriter()
    # dummy_input = torch.zeros((1, block_size), device=device, dtype=torch.long)
    # writer.add_graph(m, dummy_input)
    # writer.close()
    # m.logits_only=False

    
    # Train
    m.train()
    for itr in range(n_itrs):
        if itr % eval_interval == 0:
            losses = estimate_loss(m)  # Calculate loss
            with open(filepath, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([losses[split] for split in ['train','test']])
            # idx = torch.zeros((1, block_size), device=device, dtype=torch.long)
            # idx = m.generate(idx, 500)
            # print("\nSample: \n", decode(list(idx[0])[block_size:]), '\n\n')
            print("Test loss: ", losses['test'])
            print("Train loss: ", losses['train'])
            torch.save(m, 'transformer_' + version + '.pt')
        xb, yb = get_batch('train')
        logits, loss = m(xb, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(m,'transformer.pt')
    idx=torch.zeros((1,block_size),device=device,dtype=torch.long)
    idx=m.generate(idx,5000)
    print(idx)
    print(decode(list(idx[0])))
