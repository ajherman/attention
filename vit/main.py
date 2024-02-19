import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import requests
import os
import csv
import argparse
from transformers.utils import *
import transformers
from datasets import load_dataset
import torchvision
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor
#from torch.utils.tensorboard import SummaryWriter
# from tokenizers import Tokenizer
# from transformers import GPT2Tokenizer, GPT2Model, AutoTokenizer
import time
import numpy as np
from utils import *

data_cache_dir = "~/datasets" # "/ram/tmp" #
# dataset = 'stories' # This still needs to be set manually

# Set seed
torch.manual_seed(1337)

# @torch.no_grad()
# def estimate_loss(model):
#     out = {}
#     model.eval()
#     # Test loss
#     losses=[]
#     for itr,batch in enumerate(test_loader):
#         if itr==args.eval_iters:
#             break
#         if args.dataset not in ['shakespeare','ptb','cbt']:
#             batch = batch['text']
#         data = tokenizer(batch,padding="max_length",truncation=True,max_length=block_size+1,return_tensors="pt")        
#         if args.dataset not in ['shakespeare','ptb','cbt']:
#             data = data['input_ids']
#         # data = data['input_ids']
#         data = data.to(device)
#         xb,yb = data[:, :-1], data[:, 1:]
#         logits, loss = model(xb, yb)
#         losses.append(loss.item())
#     out['test'] = np.mean(losses)

#     # Train loss
#     losses=[]
#     for itr,batch in enumerate(train_loader):
#         if itr==args.eval_iters:
#             break
#         if args.dataset not in ['shakespeare','ptb','cbt']:
#             batch = batch['text']
#         data = tokenizer(batch,padding="max_length",truncation=True,max_length=block_size+1,return_tensors="pt")        
#         if args.dataset not in ['shakespeare','ptb','cbt']:
#             data = data['input_ids']
#         # data = data['input_ids']
#         data = data.to(device)
#         xb,yb = data[:, :-1], data[:, 1:]
#         logits, loss = model(xb, yb)
#         losses.append(loss.item())

#     out['train'] = np.mean(losses)
#     model.train()
#     return out


if __name__ == '__main__':
    #torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser()

    # Parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser.add_argument('--batch-size', type=int, default=64, help='Specify the batch size')
    parser.add_argument('--eval-interval', type=int, default=1000, help='Specify the evaluation interval')
    parser.add_argument('--eval-iters', type=int, default=500, help='Specify the evaluation iterations')

    parser.add_argument('--block-size', type=int, default=256, help='Specify the block size')        
    parser.add_argument('--dm', type=int, default=512, help='Specify embedding dimension')
    parser.add_argument('--dk', type=int, default=64, help='Specify dimension of key/query vectors')
    parser.add_argument('--dv', type=int, default=64, help='Specify dimension of value vectors')
    parser.add_argument('--h', type=int, default=8, help='Specify the number of heads')
    parser.add_argument('--N', type=int, default=8, help='Specify the number of layers')
    
    parser.add_argument('--norm-type', type=str, default='layer', help='Type of normalization layer to use ("layer" for LayerNorm, "rms" for RMSNorm)')
    parser.add_argument('--post-norm', type=int, default=1, help='Whether to use post layer normalization')
    parser.add_argument('--final-norm', type=str, default='layer', help='Norm to use in final layer ("layer" for LayerNorm, "rms" for RMSNorm)')
    parser.add_argument('--rectify', type=int, default=0, help='Whether to use rectified attention')
    parser.add_argument('--dropout', type=float, default=0.2, help='Specify the dropout')
    parser.add_argument('--attention-type', type=str, default='sdp', help='Type of attention to use ("sdp" for scaled dot product, "other" for other types)')
    parser.add_argument('--block-architecture', type=str, default='series', help='Type of block architecture to use ("series" for series of blocks, "parallel" for parallel blocks)')
    parser.add_argument('--n-fixed-keys', type=int, default=0, help='Number of fixed keys to use (0 for none)')

    parser.add_argument('--lr', type=float, default=3e-4, help='Specify the learning rate')
    parser.add_argument('--device', type=str, default=device, help='Specify the device')
    parser.add_argument('--n-itrs', type=int, default=20001, help='Specify the number of iterations')
    parser.add_argument('--filepath', type=str,default='original.csv', help='Specify the file path')
    parser.add_argument('--dataset', type=str,default='cifar10', help='Specify the dataset')
    parser.add_argument('--stream-data',action='store_true', help='Whether to stream data from disk')
    parser.add_argument('--version', type=int,default=0, help='For saving the model with distinct names')
    parser.add_argument('--patch-size', type=int, default=8, help='Specify the patch size')
    args = parser.parse_args()

    # version = args.block_type
    version = args.version
    block_size=args.block_size

    # Load the dataset
    if args.dataset == "cifar10":
        input_size = 3072
        image_size = 32
        # Download CIFAR10
        train_dataset = CIFAR10(root='~/datasets', train=True, download=True, transform=ToTensor())
        test_dataset = CIFAR10(root='~/datasets', train=False, download=True, transform=ToTensor())

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    filepath = args.filepath
    args_dict = {k: v for k, v in vars(args).items() if v is not None}

    # Make / load model
    if os.path.exists('transformer_' + str(version) + '.pt'):
        model = torch.load('transformer_' + str(version) + '.pt')
    else:
        model = ViT(image_size=image_size,patch_size=args.patch_size,dm=args.dm, dk=args.dk,dv=args.dv,h=args.h,N=args.N,final_norm=args.final_norm,norm_type=args.norm_type, post_norm=args.post_norm, rectify=args.rectify,dropout=args.dropout,block_architecture=args.block_architecture,n_fixed_keys=args.n_fixed_keys)
 
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
    criterion = nn.CrossEntropyLoss()
    # m=model.to(device)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # # Visualize model
    # m.logits_only=True
    # writer = SummaryWriter()
    # dummy_input = torch.zeros((1, block_size), device=device, dtype=torch.long)
    # writer.add_graph(m, dummy_input)
    # writer.close()
    # m.logits_only=False

    # Train ViT model on CIFAR10
    for epoch in range(10):
        for itr, (x, y) in enumerate(train_loader):

            x=x.to(device)
            y=y.to(device)

#            x.to(device)
#            y.to(device)
            logits = model(x)

            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            with torch.no_grad():
                predicted = torch.argmax(logits, dim=1)
                correct = (predicted == y).sum().item()
                accuracy = correct / y.size(0)

            # Print loss and accuracy after each batch
            if itr % 100 == 0:
                print(f'Epoch {epoch}, Batch Loss: {loss.item()}, Accuracy: {accuracy}')

        # Test
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for x, y in test_loader:
                x=x.to(device)
                y=y.to(device)
 
                logits = model(x)

                predicted = torch.argmax(logits, dim=1)
                correct += (predicted == y).sum().item()
                total += y.size(0)
            print(f'Test Accuracy: {correct / total}')
        torch.save(model,'transformer_'+str(args.version)+'.pt')
        model.train()    


  
