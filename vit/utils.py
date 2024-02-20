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
torch.cuda.manual_seed(1337)

# Classes
class RMSNorm(nn.Module):
    def __init__(self,dm):
        super().__init__()
    def forward(self,x):
        x = x/torch.sqrt(torch.mean(x**2,dim=-1,keepdim=True))
        return x

class SelfAttentionHead(nn.Module):
    def __init__(self,dm,dk,dv,dropout=0.2,rectify=0,block_size=256,n_fixed_keys=0):
        super().__init__()
        self.n_fixed_keys = n_fixed_keys

        if rectify:
            self.W_k = nn.Sequential(nn.Linear(dm,dk,bias=False),nn.ReLU())
            self.W_q = nn.Sequential(nn.Linear(dm,dk,bias=False),nn.ReLU())
        else:
            self.W_k = nn.Linear(dm,dk,bias=False)
            self.W_q = nn.Linear(dm,dk,bias=False)
        self.W_v = nn.Linear(dm,dv,bias=False)
        # self.tril=torch.tril(torch.ones((block_size,block_size+n_fixed_keys),device=device))
        self.dropout = nn.Dropout(dropout) # New

        # Initialize fixed keys and values
        self.fixed_k = nn.Parameter(torch.randn(n_fixed_keys, dk, requires_grad=True))
        self.fixed_v = nn.Parameter(torch.zeros(n_fixed_keys, dv, requires_grad=True))

    def forward(self,x):
        B,T,C=x.shape 
        k=self.W_k(x)
        q=self.W_q(x)
        v=self.W_v(x)

        # Concatenate self.fixed_k with k along the batch dimension
        fixed_k = self.fixed_k.unsqueeze(0).expand(B, -1, -1)
        fixed_v = self.fixed_v.unsqueeze(0).expand(B, -1, -1)
        if self.n_fixed_keys > 0:
            k = torch.cat([fixed_k, k], dim=1)
            v = torch.cat([fixed_v, v], dim=1)

        wei = q@k.transpose(-2,-1)*k.shape[-1]**-0.5
        # wei=wei.masked_fill(self.tril[:T,:T+self.n_fixed_keys]==0,float('-inf')) # Why do we need this T?
        wei=torch.softmax(wei,dim=-1)
        wei=self.dropout(wei) # Do we want this?
        out=wei@v
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self,dm,dk,dv,h,dropout=0.2,rectify=0,block_size=256,n_fixed_keys=0):
        super().__init__()
        
        self.heads = nn.ModuleList([SelfAttentionHead(dm,dk,dv,rectify=rectify,block_size=block_size,n_fixed_keys=n_fixed_keys) for i in range(h)])
      
    def forward(self,x):
        out = torch.cat([head(x) for head in self.heads],dim=-1)
        return out

class FeedForward(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,dropout=0.2):
        super().__init__()
        self.ffn = nn.Sequential(
        nn.Linear(input_size,hidden_size),
        # nn.ReLU(),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size,output_size),
        nn.Dropout(dropout))
    def forward(self,x):
        return self.ffn(x)

# Transformer blocks
####################################################################################################

class Block(nn.Module):
    def __init__(self, dm, dk, dv, h, block_size=256, norm_type='layer', post_norm=1, rectify=0,dropout_rate=0.2,block_architecture='series',n_fixed_keys=0):
        super().__init__()

        # self.mha = MultiHeadAttention(dm, dk, dv, h,rectify=rectify,n_fixed_keys=n_fixed_keys)
        self.mha = nn.MultiheadAttention(dm, h, dropout=dropout_rate) # Original version

        self.ffn = FeedForward(input_size=dm,hidden_size=4*dm,output_size=dm) # Original version
        self.post_norm = post_norm
        self.block_architecture = block_architecture

        if norm_type == 'layer':
            self.ln1 = nn.LayerNorm(dm, elementwise_affine=False)
            self.ln2 = nn.LayerNorm(dm, elementwise_affine=False)
        elif norm_type == 'rms':
            self.ln1 = RMSNorm(dm)
            self.ln2 = RMSNorm(dm)

        self.W_o = nn.Linear(dv * h, dm)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        if self.block_architecture == 'series':
            if self.post_norm:
                x = self.ln1(x + self.dropout(self.W_o(self.mha(x))))
                x = self.ln2(x + self.ffn(x))
            else:
                x = x + self.dropout(self.W_o(self.mha(self.ln1(x))))
                x = x + self.ffn(self.ln2(x))

        elif self.block_architecture == 'parallel':
            if self.post_norm:
                x = self.ln1(x + self.ffn(x) + self.dropout(self.W_o(self.mha(x))))
            else:
                y = self.ln1(x)
                x = x + self.ffn(y) + self.dropout(self.W_o(self.mha(y)))
        return x


# Models
###############################################################################################
# My alternate class using RMS instead of layer norm
class ViT(nn.Module): # Defaults here should be from Karpathy's tutorial
    def __init__(self,image_size=32,patch_size=8,dm=512, dk=64,dv=64,h=8,N=6,dropout=0.2,final_norm='rms',norm_type='layer', post_norm=1, rectify=0,block_architecture='series',pad_token_id=0,n_fixed_keys=0):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (image_size // patch_size)**2
        self.dm=dm
        dk = dm//h
        dv = dm//h
        assert(self.dm == dk*h)
        # Print parameters
        print("image size: ", (image_size, image_size,3))
        print("patch shape: ", (self.patch_size, self.patch_size, 3))
        print("dm = ", dm) 
        print("dk = ", dk)
        print("dv = ", dv)
        print("number of patches = ", self.n_patches)
        print("h = ", h)
        print("N = ", N)
        print("final_norm = ", final_norm)
        print("Dropout rate = ", dropout)
        print("norm type = ",norm_type)
        print("post norm = ",post_norm)
        print("rectify = ",rectify)
        print("block architecture = ",block_architecture)

        self.final_norm = final_norm
        # self.pad_token_id=pad_token_id
        
        # self.token_embedding_table = nn.Embedding(vocab_size,dm)
        self.get_patches = nn.Unfold(self.patch_size, stride=self.patch_size)
        self.patch_embedding = nn.Linear(3*self.patch_size**2,self.dm) # Get patches and project them to the embedding dimension
        self.position_embedding_table = nn.Embedding(self.n_patches+1,self.dm) # +1 for the cls token
        # self.cls_embedding = nn.Parameter(torch.randn(1,1,self.dm))
        self.blocks = nn.Sequential(*[Block(self.dm,dk,dv,h,block_size=self.n_patches+1,rectify=rectify,norm_type=norm_type, post_norm=post_norm, block_architecture=block_architecture,n_fixed_keys=n_fixed_keys) for _ in range(N)])

        if final_norm == 'layer':
            self.ln = nn.LayerNorm(self.dm)
        elif final_norm == 'rms':
            self.ln = RMSNorm(self.dm)
        else:
            raise ValueError("Invalid value for final_norm. Must be 'layer' or 'rms'.")

        self.lm_head = nn.Linear(self.dm,10)
        self.logits_only=False
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, image, targets=None):
        B = image.shape[0]

        # Reshape image into patches
        patches = self.get_patches(image).transpose(1, 2)
        embed = self.patch_embedding(patches)

        # Add positional embeddings
        pos_embed = self.position_embedding_table(torch.arange(self.n_patches+1, device=device))
        x = torch.cat([torch.zeros((B,1,self.dm),device=device),embed],dim=1) + pos_embed

        # Transformer
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.lm_head(x[:,0,:].view(B,-1))
        return logits
   
