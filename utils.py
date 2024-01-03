import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import requests
import os
import csv
import argparse

# Parameters
# block_size = 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# # Get char list
# chars = sorted(list(set(text)))
# vocab_size = len(chars)

# # Define encoding and decoding functions
# s2i = {ch:i for i,ch in enumerate(chars)}
# i2s = chars

# encode = lambda s: [s2i[c] for c in s]
# decode = lambda l: ''.join([i2s[i] for i in l])

# # Make tokenized datasets
# data = torch.tensor(encode(text))
# n = int(0.9*len(data))
# train_data = data[:n]
# test_data = data[n:]

# def get_batch(split):
#     data = train_data if split == 'train' else test_data
#     idx = torch.randint(len(data)-block_size,(batch_size,))
#     x = torch.stack([data[i:i+block_size] for i in idx])
#     y = torch.stack([data[i+1:i+1+block_size] for i in idx])
#     x,y=x.to(device),y.to(device)
#     return x,y

# Datasets
class ShakespeareData(Dataset):
    def __init__(self,file_path='shakespeare.txt'):
        super().__init__()
        with open(file_path,'r',encoding='utf-8') as f:
            self.text = f.read()
        chunk_size = 500
        self.data = [self.text[i:i + chunk_size] for i in range(0, len(self.text), chunk_size)]
    def __getitem__(self,idx):
        return self.data[idx]
    def __len__(self):
        return len(self.data)
    
class CharacterTokenizer:
    def __init__(self):
        self.char_list = ['\n', ' ', '!', '$', '&', "'", ',', '-', '.', '3', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        # self.char_to_id = char_to_id
        # self.unknown_id = unknown_id
        s2i = {ch:i for i,ch in enumerate(self.char_list)}
        i2s = self.char_list
        self.encode = lambda s: [s2i[c] for c in s]
        self.decode = lambda l: ''.join([i2s[i] for i in l])
        self.pad_token_id = 0

    def tokenize(self, text,padding=None,max_length=None,truncation=False):
        token_ids = self.encode(text)
        
        # Truncate if necessary
        if truncation and max_length is not None and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]

        # Pad if necessary
        if padding and max_length is not None and len(token_ids) < max_length:
            padding_length = max_length - len(token_ids)
            token_ids+=[self.pad_token_id] * padding_length

        return token_ids

    def convert_ids_to_tokens(self, ids):
        id_to_char = self.char_list # {id: char for char, id in self.char_to_id.items()}
        return [id_to_char.get(id, '?') for id in ids]  # Use '?' to represent unknown characters


    
# class TokenDataset(Dataset):
#     def __init__(self, encodings):
#         self.encodings = encodings

#     def __getitem__(self, idx):
#         return {key: val[idx] for key, val in self.encodings.items()}

#     def __len__(self):
#         return len(self.encodings.input_ids)


# Basic components
####################################################################################
class RMSNorm(nn.Module):
    def __init__(self,dm):
        super().__init__()
    def forward(self,x):
        x = x/torch.sqrt(torch.mean(x**2,dim=-1,keepdim=True))
        return x

# # Trying to make this class cover all cases...
# class SelfAttentionHead(nn.Module):
#     def __init__(self,dm,dk,dv,dropout=0.2,rectify=False,sim='sdp'):
#         super().__init__()
#         self.sim=sim
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

#         if self.sim=='sdp': # Original version
#             wei = q@k.transpose(-2,-1)*k.shape[-1]**-0.5
#             wei=wei.masked_fill(self.tril[:T,:T]==0,float('-inf')) # New
#             wei=torch.softmax(wei,dim=-1)
#         elif self.sim=='rdp': # Rectified dot product
#             q_rect = torch.relu(q)
#             k_rect = torch.relu(k)
#             wei = q_rect@k_rect.transpose(-2,-1)*k.shape[-1]**-0.5
#             wei=wei.masked_fill(self.tril[:T,:T]==0,float('-inf')) # New
#             wei=torch.softmax(wei,dim=-1)
#         elif self.sim=='log': # Should be equivalent to rectified version
#             q_expanded = torch.log(torch.relu(q.unsqueeze(-2)))
#             k_expanded = torch.log(torch.relu(k.unsqueeze(-3)))
#             q_plus_k = q_expanded + k_expanded
#             wei=torch.sum(torch.exp(q_plus_k),dim=-1)*k.shape[-1]**-0.5
#             wei=wei.masked_fill(self.tril[:T,:T]==0,float('-inf')) # New
#             wei=torch.softmax(wei,dim=-1)
#         elif self.sim=='log2': # Like above but no log and no softmax
#             q_expanded = torch.relu(q.unsqueeze(-2))
#             k_expanded = torch.relu(k.unsqueeze(-3))
#             q_plus_k = q_expanded + k_expanded
#             wei=torch.sum(torch.exp(q_plus_k),dim=-1)*k.shape[-1]**-0.5
#             wei=wei.masked_fill(self.tril[:T,:T]==0,float('-inf')) # New
#             wei=nn.functional.normalize(wei,p=1,dim=-1) # Just scale, rather than softmax

#         wei=self.dropout(wei) # New
#         out=wei@v
#         return out
class SelfAttentionHead(nn.Module):
    def __init__(self,dm,dk,dv,dropout=0.2,rectify=False,sim='sdp',block_size=256):
        super().__init__()
        if rectify:
            self.W_k = nn.Sequential(nn.Linear(dm,dk,bias=False),nn.ReLU())
            self.W_q = nn.Sequential(nn.Linear(dm,dk,bias=False),nn.ReLU())
        else:
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
    
class SelfAttentionHead2(nn.Module):
    def __init__(self,dm,dk,dv,dropout=0.2,block_size=256):
        super().__init__()
        self.key = nn.Sequential(nn.Linear(dm,dk,bias=False),nn.ReLU())
        self.query = nn.Sequential(nn.Linear(dm,dk,bias=False),nn.ReLU())
        self.value = nn.Linear(dm,dv,bias=False)
        self.tril=torch.tril(torch.ones((block_size,block_size),device=device))
        self.dropout = nn.Dropout(dropout) # New
    def forward(self,x):
        B,T,C=x.shape # New
        k=self.key(x)
        q=self.query(x)
        v=self.value(x)
        q_expanded = torch.log(q.unsqueeze(-2))
        k_expanded = torch.log(k.unsqueeze(-3))
        q_plus_k = q_expanded + k_expanded
        wei=torch.sum(torch.exp(q_plus_k),dim=-1)*k.shape[-1]**-0.5
        wei=wei.masked_fill(self.tril[:T,:T]==0,float('-inf')) # New
        wei=torch.softmax(wei,dim=-1)
        wei=self.dropout(wei) # New
        out=wei@v
        return out

class SelfAttentionHead3(nn.Module):
    def __init__(self,dm,dk,dv,dropout=0.2,block_size=256):
        super().__init__()
        self.key = nn.Sequential(nn.Linear(dm,dk,bias=False),nn.ReLU())
        self.query = nn.Sequential(nn.Linear(dm,dk,bias=False),nn.ReLU())
        self.value = nn.Linear(dm,dv,bias=False)
        self.tril=torch.tril(torch.ones((block_size,block_size),device=device))
        self.dropout = nn.Dropout(dropout) # New
    def forward(self,x):
        B,T,C=x.shape # New
        k=self.key(x)
        q=self.query(x)
        v=self.value(x)
        q_expanded = q.unsqueeze(-2)
        k_expanded = k.unsqueeze(-3)
        q_plus_k = q_expanded + k_expanded
        wei=torch.sum(torch.exp(q_plus_k),dim=-1)*k.shape[-1]**-0.5
        wei=wei.masked_fill(self.tril[:T,:T]==0,float('-inf')) # New
        # wei=torch.softmax(wei,dim=-1)
        wei=nn.functional.normalize(wei,p=1,dim=-1) # Just scale, rather than softmax
        wei=self.dropout(wei) # New
        out=wei@v
        return out

class LearnedSimilarityHead(nn.Module):
    def __init__(self,dm,dk,dv,dropout=0.2,block_size=256):
        super().__init__()
        self.W_k = nn.Linear(dm,dk,bias=False)
        self.W_q = nn.Linear(dm,dk,bias=False)
        self.W_v = nn.Linear(dm,dv,bias=False)
        self.W_h = nn.Linear(2*dk,dk)
        self.W_s = nn.Linear(dk,1)
        self.tril=torch.tril(torch.ones((block_size,block_size),device=device))
        self.dropout = nn.Dropout(dropout) # New
        self.dropout_hid = nn.Dropout(dropout) # New
    def forward(self,x):
        B,T,C=x.shape # New
        k=self.W_k(x)
        q=self.W_q(x)
        z = torch.concat([k,q],dim=-1)
        z = self.W_h(z)
        z = torch.tanh(z)
        z = self.dropout_hid(z)
        z = self.W_s(z)
        wei = z #torch.tanh(z)
        #wei = self.W_s(z)
        v=self.W_v(x)
        # wei = q@k.transpose(-2,-1)*k.shape[-1]**-0.5
        wei=wei.masked_fill(self.tril[:T,:T]==0,float('-inf')) # New
        wei=torch.softmax(wei,dim=-1)
        wei=self.dropout(wei) # New
        out=wei@v
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self,dm,dk,dv,h,dropout=0.2,project=True,attention_type='sdp',rectify=False,block_size=256):
        super().__init__()
        self.project=project
        if attention_type=='sdp':
            self.heads = nn.ModuleList([SelfAttentionHead(dm,dk,dv,rectify=rectify,block_size=block_size) for i in range(h)])
        elif attention_type=='learned':
            self.heads = nn.ModuleList([LearnedSimilarityHead(dm,dk,dv,rectify=rectify,block_size=block_size) for i in range(h)])
        elif attention_type=='log':
            self.heads = nn.ModuleList([SelfAttentionHead2(dm,dk,dv,block_size=block_size) for i in range(h)])    
        elif attention_type=='mine':
            self.heads = nn.ModuleList([SelfAttentionHead3(dm,dk,dv,block_size=block_size) for i in range(h)])
        if project:
            self.W_o = nn.Linear(dv*h,dm)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,x):
        out = torch.cat([head(x) for head in self.heads],dim=-1)
        if self.project:
            out = self.W_o(out)
        out = self.dropout(out) # Like spiking?
        return out

# class MultiHeadAttention(nn.Module):    # Trying to consolidate
#     def __init__(self,dm,dk,dv,h,dropout=0.2,project=True,attention_type='sdp',rectify=False):
#         super().__init__()
#         self.project=project
#         self.heads = nn.ModuleList([SelfAttentionHead(dm,dk,dv,sim=attention_type) for i in range(h)])
#         if project:
#             self.W_o = nn.Linear(dv*h,dm)
#         self.dropout = nn.Dropout(dropout)
        
#     def forward(self,x):
#         out = torch.cat([head(x) for head in self.heads],dim=-1)
#         if self.project:
#             out = self.W_o(out)
#         out = self.dropout(out) # Like spiking?
#         return out
    
# class AdditiveAttentionHead(nn.Module):
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
    
    
# class FixedKeyHead(nn.Module):
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
class SimpleMixingHead(nn.Module):  # This just mixes the input vectors, but does not apply a value matrix.
    def __init__(self,dm,dk,dv,dropout=0.2,block_size=256):
        super().__init__()
        # self.W_k = nn.Linear(dm,dk,bias=False)
        self.W_k_transpose = nn.Linear(dk,dm,bias=False)
        self.W_q = nn.Linear(dm,dk,bias=False)
        self.tril=torch.tril(torch.ones((block_size,block_size),device=device))
        self.dropout = nn.Dropout(dropout) # New
    def forward(self,x):
        B,T,C=x.shape # New
        # k=self.W_k(x)
        # q=self.W_q(x)
        q = self.W_k_transpose(self.W_q(x))
        # wei = q@k.transpose(-2,-1)*k.shape[-1]**-0.5
        wei = q@x.transpose(-2,-1)*x.shape[-1]**-0.5
        wei=wei.masked_fill(self.tril[:T,:T]==0,float('-inf')) # New
        wei=torch.softmax(wei,dim=-1)
        wei=self.dropout(wei) # New
        out=wei@x
        return out
class MultiHeadMixing(nn.Module): # This concatenates inputs from mixing heads and applies a project to the result
    def __init__(self,dm,dk,dv,h,dropout=0.2,rectify=False,block_size=256):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttentionHead(dm,dk,dv,rectifiy=rectify,block_size=block_size) for i in range(h)])
        self.W_o = nn.Linear(dv*h,dm)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x): # This 
        concat = torch.cat([head(x) for head in self.heads],dim=-1)
        out = self.dropout( self.W_o(concat) )
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

# Transformer blocks
####################################################################################################
class Block0(nn.Module): # Original
    def __init__(self,dm,h,block_size=256):
        super().__init__()
        dk = dm // h
        dv = dk
        assert(dk*h==dm) # Check the input/output size of block is same
        self.mha = MultiHeadAttention(dm,dk,dv,h)
        self.ffn = FeedForward(dm)
        self.ln1 = nn.LayerNorm(dm,elementwise_affine=False)
        self.ln2 = nn.LayerNorm(dm,elementwise_affine=False)
    def forward(self,x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
class Block1(nn.Module): # This block uses post layer norm
    def __init__(self,dm,h,block_size=256):
        super().__init__()
        dk = dm // h
        dv = dk
        assert(dk*h==dm) # Check the input/output size of block is same
        self.mha = MultiHeadAttention(dm,dk,dv,h)
        self.ffn = FeedForward(dm)
        self.ln1 = nn.LayerNorm(dm,elementwise_affine=False)
        self.ln2 = nn.LayerNorm(dm,elementwise_affine=False)
    def forward(self,x):
        x = self.ln1(x + self.mha(x))
        x = self.ln2(x + self.ffn(x))
        return x
    
class Block2(nn.Module): # This block takes attention without projection. 
    def __init__(self,dm,h,block_size=256):
        super().__init__()
        dk = dm // h
        dv = dk
        assert(dk*h==dm) # Check the input/output size of block is same
        self.mha = MultiHeadAttention(dm,dk,dv,h,project=False)
        self.W_o = nn.Linear(dv*h,dm)
        self.ffn = FeedForward(dm)
        self.ln1 = nn.LayerNorm(dm,elementwise_affine=False)
        self.ln2 = nn.LayerNorm(dm,elementwise_affine=False)
    def forward(self,x):
        x = x + self.W_o( self.mha( self.ln1(x) ))
        x = x + self.ffn( self.ln2(x) )
        return x
    
class Block3(nn.Module): # This block uses RMSNorm instead of layer norm
    def __init__(self,dm,h,block_size=256):
        super().__init__()
        dk = dm // h
        dv = dk
        assert(dk*h==dm) # Check the input/output size of block is same
        self.mha = MultiHeadAttention(dm,dk,dv,h)
        self.ffn = FeedForward(dm)
        self.ln1 = RMSNorm(dm)
        self.ln2 = RMSNorm(dm)
    def forward(self,x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class Block4(nn.Module): # 
    def __init__(self,dm,h,block_size=256):
        super().__init__()
        dk = dm // h
        dv = dk
        assert(dk*h==dm) # Check the input/output size of block is same
        self.mha = MultiHeadAttention(dm,dk,dv,h,attention_type='learned')
        self.ffn = FeedForward(dm)
        self.ln1 = nn.LayerNorm(dm,elementwise_affine=False)
        self.ln2 = nn.LayerNorm(dm,elementwise_affine=False)
    def forward(self,x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
    
class Block5(nn.Module): # This block uses RMSNorm instead of layer norm AND rectifies activities before layer norm
    def __init__(self,dm,h,block_size=256):
        super().__init__()
        dk = dm // h
        dv = dk
        assert(dk*h==dm) # Check the input/output size of block is same
        self.mha = MultiHeadAttention(dm,dk,dv,h,rectify=True)
        self.ffn = FeedForward(dm)
        self.ln1 = RMSNorm(dm)
        self.ln2 = RMSNorm(dm)
        
    def forward(self,x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class Block6(nn.Module): # This block uses RMSNorm instead of layer norm AND rectifies activities before layer norm
    def __init__(self,dm,h,block_size=256):
        super().__init__()
        dk = dm // h
        dv = dk
        assert(dk*h==dm) # Check the input/output size of block is same
        self.mha = MultiHeadAttention(dm,dk,dv,h,rectify=True,attention_type='log')
        self.ffn = FeedForward(dm)
        self.ln1 = RMSNorm(dm)
        self.ln2 = RMSNorm(dm)
        
    def forward(self,x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
    
class Block7(nn.Module): # This block uses RMSNorm instead of layer norm AND rectifies activities before layer norm
    def __init__(self,dm,h,block_size=256):
        super().__init__()
        dk = dm // h
        dv = dk
        assert(dk*h==dm) # Check the input/output size of block is same
        self.mha = MultiHeadAttention(dm,dk,dv,h,rectify=True,attention_type='mine')
        self.ffn = FeedForward(dm)
        self.ln1 = RMSNorm(dm)
        self.ln2 = RMSNorm(dm)
        
    def forward(self,x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
    
    # def __init__(self,dm,h):
    #     super().__init__()
    #     dk = dm // h
    #     dv = dk
    #     assert(dk*h==dm) # Check the input/output size of block is same
    #     self.mha = MultiHeadAttention(dm,dk,dv,h)
    #     self.ffn = FeedForward(dm)
    #     self.ln1 = RMSNorm(dm)
    #     self.ln2 = RMSNorm(dm)
        
    # def forward(self,x):
    #     x = x + self.mha(self.ln1(x))
    #     x = x + self.ffn(self.ln2(x))
    #     return x

'''
Next step is to create a block that uses only rectified activities.
'''

# Models
###############################################################################################
# My alternate class using RMS instead of layer norm
class Transformer(nn.Module):
    def __init__(self,dm=384,vocab_size=0,block_size=256,h=2,N=6,block_type=3,embedding_method='absolute',final_norm='rms',**kwargs):
        super().__init__()
        # self.__dict__.update(vars(kwargs))
        print("dm = ", dm) 
        print("vocab_size = ", vocab_size)
        print("block_size = ", block_size)
        print("h = ", h)
        print("N = ", N)
        print("block_type = ", block_type)
        print("embedding_method = ", embedding_method)
        print("final_norm = ", final_norm)
        self.vocab_size=vocab_size
        self.final_norm = final_norm
        self.block_size=block_size
        
        self.token_embedding_table = nn.Embedding(vocab_size,dm)
        self.position_embedding_table = nn.Embedding(block_size,dm)
        
        if block_type==0:
            self.blocks = nn.Sequential(*[Block0(dm,h,block_size=block_size) for _ in range(N)])
        elif block_type == 1:
            self.blocks = nn.Sequential(*[Block1(dm, h,block_size=block_size) for _ in range(N)])
        elif block_type == 2:
            self.blocks = nn.Sequential(*[Block2(dm,h,block_size=block_size) for _ in range(N)])
        elif block_type == 3:
            self.blocks = nn.Sequential(*[Block3(dm,h,block_size=block_size) for _ in range(N)])
        elif block_type == 4:
            self.blocks = nn.Sequential(*[Block4(dm,h,block_size=block_size) for _ in range(N)])
        elif block_type == 5:
            self.blocks = nn.Sequential(*[Block5(dm,h,block_size=block_size) for _ in range(N)])
        elif block_type == 6:   
            self.blocks = nn.Sequential(*[Block6(dm,h,block_size=block_size) for _ in range(N)])
        elif block_type == 7:   
            self.blocks = nn.Sequential(*[Block7(dm,h,block_size=block_size) for _ in range(N)])
        if final_norm == 'layer':
            self.ln = nn.LayerNorm(dm)
        elif final_norm == 'rms':
            self.ln = RMSNorm(dm)
        else:
            raise ValueError("Invalid value for final_norm. Must be 'layer' or 'rms'.")

        self.lm_head = nn.Linear(dm,vocab_size)
        self.logits_only=False
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self,idx,targets=None):
        B,T = idx.shape # batch size, context length
        token_embed=self.token_embedding_table(idx)
        pos_embed=self.position_embedding_table(torch.arange(T,device=device))
        x = token_embed+pos_embed
        x=self.blocks(x)
        x=self.ln(x)
        logits=self.lm_head(x)
        if targets is None:
            loss=None
        else:
            # print(logits.shape)
            # print(targets.shape)
            #assert(0)
            flat_logits=logits.view(-1,self.vocab_size)
            # print(flat_logits.shape) 
            flat_targets=targets.contiguous().view(-1)
            # print(flat_logits.shape) 

            loss=F.cross_entropy(flat_logits,flat_targets)
        if self.logits_only:
            return logits
        else:
            return logits,loss
    def generate(self,idx,max_new_tokens):
        for _ in range(max_new_tokens):
            context_idx=idx[:,-self.block_size:]
            logits,_=self(context_idx)
            last_logits=logits[:,-1,:] # Only care about next word prediction
            probs=F.softmax(last_logits,dim=-1)
            idx_next=torch.multinomial(probs,num_samples=1)
            idx=torch.cat((idx,idx_next),dim=1)
        return idx

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
