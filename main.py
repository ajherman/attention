import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import requests
import os

# Parameters
block_size = 128 #256
batch_size = 64
dm = 32
dk=16
lr=3e-4
device='cpu'
n_iters=5000

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

class BigramLanguageModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        embedding_length = dm
        self.token_embedding_table = nn.Embedding(vocab_size,embedding_length)
        self.position_embedding_table = nn.Embedding(block_size,embedding_length)
        self.mha = MultiHeadAttention(dk,dm//dk)
        self.lm_head = nn.Linear(embedding_length,vocab_size)
    def forward(self,idx,targets=None):
        B,T = idx.shape # batch size, context length
        token_embed=self.token_embedding_table(idx)
        try:
            pos_embed=self.position_embedding_table(torch.arange(T,device=device))
        except:
            print(T)
            assert(0)
        x = token_embed+pos_embed
        x=self.mha(x)
        logits=self.lm_head(x)
        flat_logits=logits.view(-1,vocab_size)
        if targets==None:
            loss=None
        else:
            flat_targets=targets.view(-1)
            loss=F.cross_entropy(flat_logits,flat_targets)
        return logits,loss
    def generate(self,idx,max_new_tokens):
        for t in range(max_new_tokens):
            logits,_=self(idx)
            last_logits=logits[:,-1,:] # Only care about next word prediction
            probs=F.softmax(last_logits,dim=-1)
            idx_next=torch.multinomial(probs,num_samples=1)
            idx=torch.cat((idx,idx_next),dim=1)
        return idx

class SelfAttentionHead(nn.Module):
    def __init__(self,dk):
        super().__init__()
        self.key = nn.Linear(dm,dk,bias=False)
        self.query = nn.Linear(dm,dk,bias=False)
        self.value = nn.Linear(dm,dk,bias=False)
        self.tril=torch.tril(torch.ones((block_size,block_size)))
    def forward(self,x):
        k=self.key(x)
        q=self.query(x)
        v=self.value(x)
        wei = q@k.transpose(-1,-2)
        wei=wei.masked_fill(self.tril==0,float('-inf'))
        wei=torch.softmax(wei,dim=-1)
        out=wei@v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self,dm,n_heads):
        super().__init__()
        dk = dm // n_heads
        self.heads = nn.ModuleList([SelfAttentionHead(dk) for i in range(n_heads)])
        self.proj = nn.Linear(dm,dm)
    def forward(self,x):
        out = torch.cat([head(x) for head in self.heads],dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    def __init__(self,dm):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dm,4*dm,bias=True),nn.ReLU(),nn.Linear(4*dm,dm,bias=True))
        # self.layer = nn.Linear(dm,dm)
    def forward(self,x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self,dm,n_heads):
        super().__init__()
        dk = dm // n_heads
        self.mha = MultiHeadAttention(dm,n_heads)
        self.ffn = FeedForward(dm)
        self.ln1 = nn.LayerNorm(dm)
        self.ln2 = nn.LayerNorm(dm)
    def forward(self,x):
        x = self.ln1(x)
        x = x + self.mha(x)
        x = self.ln2(x)
        x = x + self.ffn(x)
        return x

class Transformer(nn.Module):
    def __init__(self,dm,vocab_size,n_blocks=3):
        super().__init__()
        embedding_length = dm
        n_heads=4
        self.token_embedding_table = nn.Embedding(vocab_size,dm)
        self.position_embedding_table = nn.Embedding(block_size,dm)
        self.blocks = nn.Sequential(*[Block(dm,n_heads) for _ in range(n_blocks)])
        self.ln_f = nn.LayerNorm(dm)
        self.lm_head = nn.Linear(embedding_length,vocab_size)
    def forward(self,idx,targets=None):
        B,T = idx.shape # batch size, context length
        token_embed=self.token_embedding_table(idx)
        pos_embed=self.position_embedding_table(torch.arange(T,device=device))
        x = token_embed+pos_embed
        x=self.blocks(x)
        x=self.ln_f(x)
        logits=self.lm_head(x)
        flat_logits=logits.view(-1,vocab_size)
        if targets==None:
            loss=None
        else:
            flat_targets=targets.view(-1)
            loss=F.cross_entropy(flat_logits,flat_targets)
        return logits,loss
    def generate(self,idx,max_new_tokens):
        for t in range(max_new_tokens):
            context_idx=idx[:,-block_size:]
            logits,_=self(context_idx)
            last_logits=logits[:,-1,:] # Only care about next word prediction
            probs=F.softmax(last_logits,dim=-1)
            idx_next=torch.multinomial(probs,num_samples=1)
            idx=torch.cat((idx,idx_next),dim=1)
        return idx

# Train bigram model
if os.path.exists('bigram'):
    m=torch.load('bigram')
else:
    m = Transformer(dm,vocab_size)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

m.to(device)
optimizer = torch.optim.AdamW(m.parameters(),lr=lr)
for t in range(n_iters):
    xb,yb=get_batch('train')
    logits,loss = m(xb,yb)
    loss.backward()
    optimizer.step()
    if t%200==0:
        idx=torch.zeros((1,block_size),dtype=torch.long)
        idx=m.generate(idx,50)
        print("Sample: \n",decode(list(idx[0])[block_size:]))
        print("Loss: ",loss.item(),"\n")
torch.save(m,'bigram')


idx=torch.zeros((1,block_size),dtype=torch.long)
idx=m.generate(idx,50)
print(idx)
print(decode(list(idx[0])))
