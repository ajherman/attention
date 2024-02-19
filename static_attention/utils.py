import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.dropout = nn.Dropout(dropout) 
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x):
        q=self.W_q(x)
        scores = self.W_k(q)
        weights = self.softmax(scores)
        weights = self.dropout(weights)
        out = self.W_v(weights)

        return out
    
class StaticAttentionHeadAlt(nn.Module): # Explicitly include keys
    def __init__(self,dm,dk,dv,N,dropout=0.2,rectify=0):
        super().__init__()

        if rectify:
            self.W_q = nn.Sequential(nn.Linear(dm,dk,bias=False),nn.ReLU()) # Project down to key dimension
            self.W_k = nn.Sequential(nn.Linear(dk,dm,bias=False),nn.ReLU()) # Simulate context with fixed keys
        else:
            self.W_q = nn.Linear(dm,dk,bias=False)
            self.W_k = nn.Linear(dk,dm,bias=False)


        self.W_v = nn.Linear(N,dv,bias=False)
        self.dropout = nn.Dropout(dropout) 
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
    
class AltStaticMultiHeadAttention(nn.Module):
    def __init__(self,dm,N,heads,dropout=0.2,rectify=0):
        super().__init__()
        self.heads=heads
        self.W_enc = nn.Linear(dm,N,bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.W_dec = nn.Linear(N,dm,bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        B = x.size(0)
        out = self.W_enc(x)
        out = out.view((B,self.heads,-1))
        out = torch.logsumexp(out) #self.softmax(out)
        out = self.dropout(out)
        out = out.view((B,-1))
        out = self.W_dec(out)
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
    
class ConvNet(nn.Module):
    def __init__(self,kernel_size=8,stride=2):
        super(ConvNet, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        dm = 3*kernel_size**2
        self.layer1 = StaticTransformer(dm,dm//8,dm//8,1000,8,1,10)

        # Get number of patches
        P_h = ((32 - 1) * stride + kernel_size - 32) / 2
        P_w = ((32 - 1) * stride + kernel_size - 32) / 2
        L = max(int(P_h), 0), max(int(P_w), 0)
        
        self.ffn = FeedForward(dm*L)
        self.fc = nn.Linear(dm*L,10)
        self.loss = nn.CrossEntropyLoss()
    def forward(self, x, targets=None):
        kernel_size = self.kernel_size
        stride = self.stride

        # Extract image patches
        patches = F.unfold(x, kernel_size=kernel_size, stride=stride, padding='SAME').transpose(1,2)
        
        # Apply static transformer to each patch (maybe replace with block?)
        out = self.layer1(patches).reshape(x.size(0),-1)
        out = self.ffn(out)
        logits = self.fc(out)
        if targets is None:
            return logits
        else:
            loss = self.loss(logits,targets)
            return logits, loss