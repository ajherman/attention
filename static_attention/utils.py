import torch
import torch.nn as nn

class StaticAttentionHead(nn.Module):
    def __init__(self,dm,dk,dv,N,dropout=0.2,rectify=0):
        super().__init__()

        if rectify:
            self.W_q = nn.Sequential(nn.Linear(dm,dk,bias=False),nn.ReLU()) # Project down to key dimension
            self.W_k = nn.Sequential(nn.Linear(dk,N,bias=False),nn.ReLU()) # Simulate context with fixed keys
        else:
            self.W_q = nn.Linear(dm,dk,bias=False)
            self.W_k = nn.Linear(dk,N,bias=False)

        self.W_v = nn.Linear(N,dv,bias=False)
        # self.tril=torch.tril(torch.ones((block_size,block_size+n_fixed_keys),device=device))
        self.dropout = nn.Dropout(dropout) # New

        # Initialize fixed keys and values
        # self.fixed_k = nn.Parameter(torch.randn(n_fixed_keys, dk, requires_grad=True))
        # self.fixed_v = nn.Parameter(torch.zeros(n_fixed_keys, dv, requires_grad=True))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x):
        q=self.W_q(x)
        scores = self.W_k(q)
        weights = self.softmax(scores)
        weights = self.dropout(weights)
        out = self.W_v(weights)

        return out
    
class StaticMultiHeadAttention(nn.Module):
    def __init__(self,dm,dk,dv,N,heads,dropout=0.2,rectify=0):
        super().__init__()
        self.attention_heads=nn.ModuleList([StaticAttentionHead(dm,dk,dv,N,dropout,rectify) for _ in range(heads)])
        self.W_o = nn.Linear(heads*dv,dm)

    def forward(self,x):
        attention_out = torch.cat([a(x) for a in self.attention_heads],dim=-1)
        return self.W_o(attention_out)
    
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
    
class StaticBlock(nn.Module):
    def __init__(self,dm,dk,dv,N,heads,dff,dropout=0.2,rectify=0):
        super().__init__()
        self.attention = StaticMultiHeadAttention(dm,dk,dv,N,heads,dropout,rectify)
        self.ffn = FeedForward(dm,dropout)
        self.ln1 = nn.LayerNorm(dm)
        self.ln2 = nn.LayerNorm(dm)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        x=self.ln1(x+self.dropout(self.attention(x)))
        x=self.ln2(x+self.dropout(self.ffn(x)))
        return x
    
class StaticTransformer(nn.Module):
    def __init__(self,dm,dk,dv,N,heads,layers,n_classes,dropout=0.2,rectify=0):
        super().__init__()
        self.blocks = nn.Sequential(*[StaticBlock(dm,dk,dv,N,heads,dropout,rectify) for _ in range(layers)])
        self.ln = nn.LayerNorm(dm)
        self.fc = nn.Linear(dm,n_classes)
        self.loss = nn.CrossEntropyLoss()
    def forward(self,x,targets=None):
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.fc(x)
        if targets is None:
            return logits
        else:
            loss = self.loss(logits,targets)
        return logits, loss