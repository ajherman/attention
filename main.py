import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import requests
import os
import csv
import argparse
from utils import *
import transformers
from datasets import load_dataset
#from torch.utils.tensorboard import SummaryWriter
from tokenizers import Tokenizer
from transformers import GPT2Tokenizer, GPT2Model


# Parameters
block_size = 256
batch_size = 64
eval_interval= 200 #1000
eval_iters=500
dm = 100 #384 # Model / embedding size
dk=64 # Head size
h=2 #6 # Number of heads in multihead attn
lr=2e-4 # Learning rate
N=2 #6 # Number of layers
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_itrs=20001
dropout=0.2
vocab_size=50257

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
data = torch.tensor(encode(text),dtype=torch.long)
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

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=int, default=0, help='Specify the version')
    parser.add_argument('--filepath', type=str,default='original.csv', help='Specify the file path')
    args = parser.parse_args()
    version = args.version
    filepath = args.filepath

    # Make / load model
    if os.path.exists('transformer_' + str(version) + '.pt'):
        model = torch.load('transformer_' + str(version) + '.pt')
    else:
        model = Transformer(dm=dm, vocab_size=vocab_size,block_size=block_size, h=h, N=N, block_type=version)
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
    m=model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # # Visualize model
    # m.logits_only=True
    # writer = SummaryWriter()
    # dummy_input = torch.zeros((1, block_size), device=device, dtype=torch.long)
    # writer.add_graph(m, dummy_input)
    # writer.close()
    # m.logits_only=False

    dataset = load_dataset("nRuaif/tinystories-gpt4",split="train")
    dataloader = DataLoader(dataset, batch_size=64)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    
    # def tokenization(example):
        # return tokenizer(example["text"])



    # for i,x in enumerate(dataloader):
    #     # print(type(x['text'][0]))
    #     data = tokenizer(x['text'],padding="max_length",truncation=True,max_length=block_size,return_tensors="pt")        
    #     print(data['input_ids'].size())
    #     if i>1:
    #         assert(0)

    for itr,batch in enumerate(dataloader):
        data = tokenizer(batch['text'],padding="max_length",truncation=True,max_length=block_size,return_tensors="pt")        
        data = data['input_ids']
        xb,yb = data[:, :-1], data[:, 1:]
        # print(data['input_ids'].size())
        # assert(0)
        if itr % eval_interval == 0:
            losses = estimate_loss(model)  # Calculate loss
            with open(filepath, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([losses[split] for split in ['train','test']])
            idx = torch.zeros((1, block_size), device=device, dtype=torch.long)
            idx = m.generate(idx, 500)
            print("\nSample: \n", decode(list(idx[0])[block_size:]), '\n\n')
            print(f"step {itr}: train loss {losses['train']:.4f}, val loss {losses['test']:.4f}")
            torch.save(m, 'transformer_' + str(version) + '.pt')
        # xb, yb = get_batch('train')
        logits, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.save(m,'transformer_'+str(version)+'.pt')
    idx=torch.zeros((1,block_size),device=device,dtype=torch.long)
    idx=m.generate(idx,5000)
    print(idx)
    print(decode(list(idx[0])[block_size:]))
