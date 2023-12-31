import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import requests
import os
import csv
import argparse
from utils import *
import transformers
from datasets import load_dataset
#from torch.utils.tensorboard import SummaryWriter
from tokenizers import Tokenizer
from transformers import GPT2Tokenizer, GPT2Model, AutoTokenizer
import time

data_cache_dir = "/ram/tmp"
dataset = 'stories' #'shakespeare'

# Set seed
torch.manual_seed(1337)

if dataset == 'shakespeare':
    @torch.no_grad()
    def estimate_loss(model):
        out = {}
        model.eval()
        # Test loss
        losses=torch.zeros(args.eval_iters)
        for itr,batch in enumerate(test_loader):
            if itr==args.eval_iters:
                break
            data = tokenizer(batch,padding="max_length",truncation=True,max_length=block_size,return_tensors="pt")        
            # data = data['input_ids']
            data = data.to(device)
            xb,yb = data[:, :-1], data[:, 1:]
            logits, loss = model(xb, yb)
            losses[itr]=loss.item()

        # losses=torch.tensor(losses)
        out['test'] = losses.mean().item()

        # Train loss
        losses=torch.zeros(args.eval_iters)
        for itr,batch in enumerate(train_loader):
            if itr==args.eval_iters:
                break
            data = tokenizer(batch,padding="max_length",truncation=True,max_length=block_size,return_tensors="pt")        
            # data = data['input_ids']
            data = data.to(device)
            xb,yb = data[:, :-1], data[:, 1:]
            logits, loss = model(xb, yb)
            losses[itr]=loss.item()
        out['train'] = losses.mean().item()
        model.train()
        return out
    
elif dataset == 'stories':
    @torch.no_grad()
    def estimate_loss(model):
        out = {}
        model.eval()
        # Test loss
        losses=torch.zeros(args.eval_iters)
        for itr,batch in enumerate(test_loader):
            if itr==args.eval_iters:
                break
            data = tokenizer(batch['text'],padding="max_length",truncation=True,max_length=block_size+1,return_tensors="pt")        
            data = data['input_ids']
            data = data.to(device)
            xb,yb = data[:, :-1], data[:, 1:]
            logits, loss = model(xb, yb)
            losses[itr]=loss.item()

        out['test'] = losses.mean().item()

        # Train loss
        losses=torch.zeros(args.eval_iters)
        for itr,batch in enumerate(train_loader):
            if itr==args.eval_iters:
                break
            data = tokenizer(batch['text'],padding="max_length",truncation=True,max_length=block_size+1,return_tensors="pt")        
            data = data['input_ids']
            data = data.to(device)
            xb,yb = data[:, :-1], data[:, 1:]
            logits, loss = model(xb, yb)
            losses[itr]=loss.item()

        # losses=torch.tensor(losses)
        out['train'] = losses.mean().item()
        # out['train'] = 0
        model.train()
        return out


if __name__ == '__main__':
    #torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser()

    # Parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser.add_argument('--block-size', type=int, default=256, help='Specify the block size')        
    parser.add_argument('--batch-size', type=int, default=64, help='Specify the batch size')
    parser.add_argument('--eval-interval', type=int, default=1000, help='Specify the evaluation interval')
    parser.add_argument('--eval-iters', type=int, default=500, help='Specify the evaluation iterations')
    parser.add_argument('--dm', type=int, default=384, help='Specify the model size')
    parser.add_argument('--dk', type=int, default=64, help='Specify the head size')
    parser.add_argument('--h', type=int, default=6, help='Specify the number of heads')
    parser.add_argument('--lr', type=float, default=2e-4, help='Specify the learning rate')
    parser.add_argument('--N', type=int, default=6, help='Specify the number of layers')
    parser.add_argument('--device', type=str, default=device, help='Specify the device')
    parser.add_argument('--n-itrs', type=int, default=20001, help='Specify the number of iterations')
    parser.add_argument('--dropout', type=float, default=0.2, help='Specify the dropout')
    parser.add_argument('--block-type', type=int, default=3, help='Specify the version')
    parser.add_argument('--filepath', type=str,default='original.csv', help='Specify the file path')
    parser.add_argument('--dataset', type=str,default='shakespeare', help='Specify the dataset')

    args = parser.parse_args()
    version = args.block_type
    block_size=args.block_size

    if args.dataset == 'shakespeare':
        # Download a sample text file (e.g., "The Complete Works of William Shakespeare")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        file_path = "datasets/shakespeare.txt"

        if not os.path.exists(file_path):
            response = requests.get(url)
            with open(file_path, 'w') as file:
                file.write(response.text)

        shakespeare_data = TextDataFromFile(block_size=block_size+1,filepath=file_path)
        N = len(shakespeare_data)
        test_set = Subset(shakespeare_data, [i for i in range(N) if i % 10 == 0])
        train_set = Subset(shakespeare_data, [i for i in range(N) if i % 10 != 0])
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        tokenizer = CharacterTokenizer(block_size=block_size+1)
    elif args.dataset == 'stories':
        # vocab_size=50258
        train_set = load_dataset("nRuaif/tinystories-gpt4",cache_dir=data_cache_dir,split='train')
        train_loader = DataLoader(train_set, batch_size=64)
        test_set = load_dataset("nRuaif/tinystories-gpt4",cache_dir=data_cache_dir,split='test')
        test_loader = DataLoader(test_set, batch_size=64)
        #tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        # tokenizer = AutoTokenizer.from_pretrained("georgeyw/TinyStories-tokenizer-10k")
        tokenizer = AutoTokenizer.from_pretrained("georgeyw/TinyStories-tokenizer-5k")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    vocab_size=len(tokenizer)
    decode = tokenizer.decode

    filepath = args.filepath
    # args_dict = vars(args)
    args_dict = {k: v for k, v in vars(args).items() if v is not None}
    args_dict['vocab_size'] = vocab_size

    # Make / load model
    if os.path.exists('transformer_' + str(version) + '.pt'):
        model = torch.load('transformer_' + str(version) + '.pt')
    else:
        # model = Transformer(dm=dm, vocab_size=vocab_size,block_size=block_size, h=h, N=N, block_type=version)
        model = Transformer(**args_dict)
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
    m=model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # # Visualize model
    # m.logits_only=True
    # writer = SummaryWriter()
    # dummy_input = torch.zeros((1, block_size), device=device, dtype=torch.long)
    # writer.add_graph(m, dummy_input)
    # writer.close()
    # m.logits_only=False

    if args.dataset == 'shakespeare':
        # Train
        # Shakespeare version that should already work
        for itr,batch in enumerate(train_loader):
            data = tokenizer(batch,padding="max_length",truncation=True,max_length=block_size,return_tensors="pt")    
            xb,yb = data[:,:-1],data[:,1:]
            logits, loss = model(xb, yb)
            if itr % args.eval_interval == 0:
                losses = estimate_loss(model)  # Calculate loss
                with open(filepath, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([losses[split] for split in ['train','test']])
                idx = torch.zeros((1, block_size), device=device, dtype=torch.long)
                idx = m.generate(idx, 500)
                print("\nSample: \n", decode(list(idx[0])[block_size:]), '\n\n')
                print(f"step {itr}: train loss {losses['train']:.4f}, val loss {losses['test']:.4f}")
                torch.save(m, 'transformer_' + str(version) + '.pt')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    elif args.dataset == 'stories':
        tic = time.time()
        # TinyStories version that I am currently working on
        for itr,batch in enumerate(train_loader):
            data = tokenizer(batch['text'],padding="max_length",truncation=True,max_length=block_size+1,return_tensors="pt")        
            data = data['input_ids']
            data = data.to(device)
            xb,yb = data[:, :-1], data[:, 1:]
            if itr % args.eval_interval == 0:
                elapsed, tic = time.time() - tic, time.time()
                print(f"step {itr}: {elapsed:.2f} seconds")
                losses = estimate_loss(model)  # Calculate loss
                with open(filepath, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([losses[split] for split in ['train','test']])
                idx = torch.zeros((1, args.block_size), device=device, dtype=torch.long)
                idx = m.generate(idx, 500)
                print("\nSample: \n", decode(list(idx[0])[args.block_size:]), '\n\n')
                print(f"step {itr}: train loss {losses['train']:.4f}, val loss {losses['test']:.4f}")
                torch.save(m, 'transformer_' + str(version) + '.pt')
            logits, loss = model(xb, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    torch.save(m,'transformer_'+str(args.version)+'.pt')
    idx=torch.zeros((1,block_size),device=device,dtype=torch.long)
    idx=m.generate(idx,5000)
    print(idx)
    print(decode(list(idx[0])[args.block_size:]))
