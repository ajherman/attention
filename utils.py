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

# # Download a sample text file (e.g., "The Complete Works of William Shakespeare")
# url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
# file_path = "shakespeare.txt"

# if not os.path.exists(file_path):
#     response = requests.get(url)
#     with open(file_path, 'w') as file:
#         file.write(response.text)

# # Read in text file
# with open(file_path,'r',encoding='utf-8') as f:
#     text = f.read()

# Datasets
# class ShakespeareData(Dataset):
#     def __init__(self,block_size=None,file_path='shakespeare.txt'):
#         super().__init__()
#         with open(file_path,'r',encoding='utf-8') as f:
#             self.text = f.read()
#         self.data = torch.tensor(encode(self.text))
#         self.block_size=block_size
#     def __getitem__(self,idx):
#         x = self.data[idx:idx+self.block_size]
#         y = self.data[idx+1:idx+1+self.block_size]
#         return x,y
#     def __len__(self):
#         return len(self.data)-self.block_size
    
class TextDataFromFile(Dataset):
    def __init__(self,block_size,filepath):
        self.block_size = block_size
        with open(file_path,'r',encoding='utf-8') as f:
            self.text = f.read()
        self.data = self.text #torch.tensor(encode(self.text),dtype=torch.long)
    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.block_size]
        return x

class CharacterTokenizer:
    def __init__(self, block_size, **kwargs):
        super().__init__(**kwargs)
        # Get char list
        self.chars = ['\n', ' ', '!', '$', '&', "'", ',', '-', '.', '3', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        s2i = {ch:i for i,ch in enumerate(self.chars)}
        i2s = self.chars
        self.encode = lambda s: [s2i[c] for c in s]
        self.decode = lambda l: ''.join([i2s[i] for i in l])
        self.block_size = block_size
    def _tokenize(self, text): # Takes string and returns list of tokens
        return encode(text)
    def __call__(self, text, **kwargs):
        batch = torch.stack([torch.tensor(self.encode(s),dtype=torch.long) for s in text])
        batch = batch.to(device)
        return batch    
    def __len__(self):
        return len(self.chars)
    
# Basic components
####################################################################################

# Similiarity functions
class sdp(nn.Module):
    def __init__(self,dropout=0.2):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout) # New
        
    def forward(self,q,k,T,block_size):
        # B,T,C = q.shape # New
        tril=torch.tril(torch.ones((block_size,block_size),device=device))
        out = torch.matmul(q,k.transpose(-2,-1))*k.shape[-1]**-0.5
        out = out.masked_fill(tril[:T,:T]==0,float('-inf')) # New
        out = self.softmax(out)
        out = self.dropout(out) # New
        return out
    
class log(nn.Module):
    def __init__(self,dropout=0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout) # New
        self.softmax = nn.Softmax(dim=-1)
    def forward(self,q,k,T,block_size):
        # B,T,C = q.shape # New
        q_expanded = torch.log(q.unsqueeze(-2))
        k_expanded = torch.log(k.unsqueeze(-3))
        q_plus_k = q_expanded + k_expanded
        out = torch.sum(torch.exp(q_plus_k),dim=-1)*k.shape[-1]**-0.5
        out = out.masked_fill(tril[:T,:T]==0,float('-inf')) # New
        out = self.softmax(out)
        out = self.dropout(out) # New
        return out
   
class RMSNorm(nn.Module):
    def __init__(self,dm):
        super().__init__()
    def forward(self,x):
        x = x/torch.sqrt(torch.mean(x**2,dim=-1,keepdim=True))
        return x

# !!!! Under construction !!!!
# class SelfAttentionHeadNew(nn.Module):
#     def __init__(self,dm,dk,dv,dropout=0.2,ActFun=nn.Identity(),Similarity=sdp(),block_size=256):
#         super().__init__()
#         self.block_size=block_size
#         self.key = nn.Sequential(nn.Linear(dm,dk,bias=False),ActFun)
#         self.query = nn.Sequential(nn.Linear(dm,dk,bias=False),ActFun)
#         self.value = nn.Linear(dm,dv,bias=False)
#         self.tril=torch.tril(torch.ones((block_size,block_size),device=device))
#         self.dropout = nn.Dropout(dropout) 
#         self.sim = Similarity # Calculate similarity scores
#     def forward(self,x):
#         B,T,C=x.shape # New
#         k=self.key(x)
#         q=self.query(x)
#         v=self.value(x)
#         wei=self.sim(q,k,T,self.block_size)
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
class MultiHeadAttention(nn.Module):
    def __init__(self,dm,dk,dv,h,dropout=0.2,attention_type='sdp',rectify=False,block_size=256):
        super().__init__()
        
        if attention_type=='sdp':
            self.heads = nn.ModuleList([SelfAttentionHead(dm,dk,dv,rectify=rectify,block_size=block_size) for i in range(h)])
        elif attention_type=='log':
            self.heads = nn.ModuleList([SelfAttentionHead2(dm,dk,dv,block_size=block_size) for i in range(h)])    
        elif attention_type=='mine':
            self.heads = nn.ModuleList([SelfAttentionHead3(dm,dk,dv,block_size=block_size) for i in range(h)])
        # elif attention_type=='new':
        #     self.heads = nn.ModuleList([SelfAttentionHeadNew(dm,dk,dv,block_size=block_size,ActFun=ActFun,Similarity=Similarity) for i in range(h)])
        
    def forward(self,x):
        out = torch.cat([head(x) for head in self.heads],dim=-1)
        # if self.project:
        #     out = self.W_o(out)
        # out = self.dropout(out) # Like spiking?
        return out

class FeedForward(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,dropout=0.2):
        super().__init__()
        self.ffn = nn.Sequential(
        nn.Linear(input_size,hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size,output_size),
        nn.Dropout(dropout))
    def forward(self,x):
        return self.ffn(x)

# Transformer blocks
####################################################################################################

class Block(nn.Module):
    def __init__(self, dm, dk, dv, h, block_size=256, norm_type='layer', post_norm=False, rectify=False, attention_type='sdp'):
        super().__init__()
        # dk = dm // h
        # dv = dk
        # assert(dk * h == dm)  # Check the input/output size of block is same
        self.mha = MultiHeadAttention(dm, dk, dv, h,rectify=rectify,attention_type=attention_type)
        # self.ffn = FeedForward(input_size=dm,hidden_size=4*dm,output_size=dm) # Original version
        self.ffn = FeedForward(input_size=dm,hidden_size=4*dm,output_size=dm) # Original version
        self.post_norm = post_norm

        if norm_type == 'layer':
            self.ln1 = nn.LayerNorm(dm, elementwise_affine=False)
            self.ln2 = nn.LayerNorm(dm, elementwise_affine=False)
        elif norm_type == 'rms':
            self.ln1 = RMSNorm(dm)
            self.ln2 = RMSNorm(dm)

        # if not project:
        self.W_o = nn.Linear(dv * h, dm)

        # From MHA
        # if project:
        #     self.W_o = nn.Linear(dv*h,dm)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.post_norm:
            x = self.ln1(x + self.W_o(self.mha(x)))
            x = self.ln2(x + self.ffn(x))
        else:
            x = x + self.W_o(self.mha(self.ln1(x)))
            # if not self.mha.project:
            # x = self.W_o(x)
            x = x + self.ffn(self.ln2(x))

        # if self.project:
        #     out = self.W_o(out)
        # out = self.dropout(out) # Like spiking?
        return x



# class Block0(nn.Module): # Original
#     def __init__(self,dm,h,block_size=256):
#         super().__init__()
#         dk = dm // h
#         dv = dk
#         assert(dk*h==dm) # Check the input/output size of block is same
#         self.mha = MultiHeadAttention(dm,dk,dv,h)
#         self.ffn = FeedForward(dm)
#         self.ln1 = nn.LayerNorm(dm,elementwise_affine=False)
#         self.ln2 = nn.LayerNorm(dm,elementwise_affine=False)
#     def forward(self,x):
#         x = x + self.mha(self.ln1(x))
#         x = x + self.ffn(self.ln2(x))
        return x
# class Block1(nn.Module): # This block uses post layer norm
#     def __init__(self,dm,h,block_size=256):
#         super().__init__()
#         dk = dm // h
#         dv = dk
#         assert(dk*h==dm) # Check the input/output size of block is same
#         self.mha = MultiHeadAttention(dm,dk,dv,h)
#         self.ffn = FeedForward(dm)
#         self.ln1 = nn.LayerNorm(dm,elementwise_affine=False)
#         self.ln2 = nn.LayerNorm(dm,elementwise_affine=False)
#     def forward(self,x):
#         x = self.ln1(x + self.mha(x))
#         x = self.ln2(x + self.ffn(x))
#         return x
    
# class Block2(nn.Module): # This block takes attention without projection. 
#     def __init__(self,dm,h,block_size=256):
#         super().__init__()
#         dk = dm // h
#         dv = dk
#         assert(dk*h==dm) # Check the input/output size of block is same
#         self.mha = MultiHeadAttention(dm,dk,dv,h,project=False)
#         self.W_o = nn.Linear(dv*h,dm)
#         self.ffn = FeedForward(dm)
#         self.ln1 = nn.LayerNorm(dm,elementwise_affine=False)
#         self.ln2 = nn.LayerNorm(dm,elementwise_affine=False)
#     def forward(self,x):
#         x = x + self.W_o( self.mha( self.ln1(x) ))
#         x = x + self.ffn( self.ln2(x) )
#         return x
    
# class Block3(nn.Module): # This block uses RMSNorm instead of layer norm
#     def __init__(self,dm,h,block_size=256):
#         super().__init__()
#         dk = dm // h
#         dv = dk
#         assert(dk*h==dm) # Check the input/output size of block is same
#         self.mha = MultiHeadAttention(dm,dk,dv,h)
#         self.ffn = FeedForward(dm)
#         self.ln1 = RMSNorm(dm)
#         self.ln2 = RMSNorm(dm)
#     def forward(self,x):
#         x = x + self.mha(self.ln1(x))
#         x = x + self.ffn(self.ln2(x))
#         return x
    
# class Block5(nn.Module): # This block uses RMSNorm instead of layer norm AND rectifies activities before layer norm
#     def __init__(self,dm,h,block_size=256):
#         super().__init__()
#         dk = dm // h
#         dv = dk
#         assert(dk*h==dm) # Check the input/output size of block is same
#         self.mha = MultiHeadAttention(dm,dk,dv,h,rectify=True)
#         self.ffn = FeedForward(dm)
#         self.ln1 = RMSNorm(dm)
#         self.ln2 = RMSNorm(dm)
        
#     def forward(self,x):
#         x = x + self.mha(self.ln1(x))
#         x = x + self.ffn(self.ln2(x))
#         return x

# class Block6(nn.Module): # This block uses RMSNorm instead of layer norm AND rectifies activities before layer norm
#     def __init__(self,dm,h,block_size=256):
#         super().__init__()
#         dk = dm // h
#         dv = dk
#         assert(dk*h==dm) # Check the input/output size of block is same
#         self.mha = MultiHeadAttention(dm,dk,dv,h,rectify=True,attention_type='log')
#         self.ffn = FeedForward(dm)
#         self.ln1 = RMSNorm(dm)
#         self.ln2 = RMSNorm(dm)
        
#     def forward(self,x):
#         x = x + self.mha(self.ln1(x))
#         x = x + self.ffn(self.ln2(x))
#         return x
    
# class Block7(nn.Module): # This block uses RMSNorm instead of layer norm AND rectifies activities before layer norm
#     def __init__(self,dm,h,block_size=256):
#         super().__init__()
#         dk = dm // h
#         dv = dk
#         assert(dk*h==dm) # Check the input/output size of block is same
#         self.mha = MultiHeadAttention(dm,dk,dv,h,rectify=True,attention_type='mine')
#         self.ffn = FeedForward(dm)
#         self.ln1 = RMSNorm(dm)
#         self.ln2 = RMSNorm(dm)
        
#     def forward(self,x):
#         x = x + self.mha(self.ln1(x))
#         x = x + self.ffn(self.ln2(x))
#         return x
    

# class Block8(nn.Module): # This block uses RMSNorm instead of layer norm
#     def __init__(self,dm,h,block_size=256,ActFun=nn.ReLU(),Similarity=sdp()):
#         super().__init__()
#         dk = dm // h
#         dv = dk
#         assert(dk*h==dm) # Check the input/output size of block is same
#         self.mha = MultiHeadAttention(dm,dk,dv,h,attention_type='new',block_size=block_size,ActFun=ActFun,Similarity=Similarity)
#         self.ffn = FeedForward(dm)
#         self.ln1 = RMSNorm(dm)
#         self.ln2 = RMSNorm(dm)
#     def forward(self,x):
#         x = x + self.mha(self.ln1(x))
#         x = x + self.ffn(self.ln2(x))
#         return x


# Models
###############################################################################################
# My alternate class using RMS instead of layer norm
class Transformer(nn.Module):
    def __init__(self,dm=384,dk=64,dv=64,vocab_size=0,block_size=256,h=2,N=6,block_type=3,embedding_method='absolute',final_norm='rms',norm_type='layer', post_norm=False,**kwargs):
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

        self.blocks = nn.Sequential(*[Block(dm,dk,dv,h,block_size=block_size,norm_type='layer', post_norm=False) for _ in range(N)])
        
        # if block_type==0:
        #     self.blocks = nn.Sequential(*[Block0(dm,h,block_size=block_size) for _ in range(N)])
        # elif block_type == 1:
        #     self.blocks = nn.Sequential(*[Block1(dm, h,block_size=block_size) for _ in range(N)])
        # elif block_type == 2:
        #     self.blocks = nn.Sequential(*[Block2(dm,h,block_size=block_size) for _ in range(N)])
        # elif block_type == 3:
        #     self.blocks = nn.Sequential(*[Block3(dm,h,block_size=block_size) for _ in range(N)])
        # elif block_type == 4:
        #     self.blocks = nn.Sequential(*[Block4(dm,h,block_size=block_size) for _ in range(N)])
        # elif block_type == 5:
        #     self.blocks = nn.Sequential(*[Block5(dm,h,block_size=block_size) for _ in range(N)])
        # elif block_type == 6:   
        #     self.blocks = nn.Sequential(*[Block6(dm,h,block_size=block_size) for _ in range(N)])
        # elif block_type == 7:   
        #     self.blocks = nn.Sequential(*[Block7(dm,h,block_size=block_size) for _ in range(N)])
        # elif block_type == 8:
        #     self.blocks = nn.Sequential(*[Block8(dm,h,block_size=block_size) for _ in range(N)])

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
            flat_logits=logits.view(-1,self.vocab_size)
            flat_targets=targets.contiguous().view(-1)
            loss=F.cross_entropy(flat_logits,flat_targets)
        if self.logits_only:
            return logits
        else:
            return logits,loss
    def generate(self,idx,max_new_tokens,beta=1.0):
        for _ in range(max_new_tokens):
            context_idx=idx[:,-self.block_size:]
            logits,_=self(context_idx)
            last_logits=logits[:,-1,:] # Only care about next word prediction
            probs=F.softmax(beta*last_logits,dim=-1)
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
