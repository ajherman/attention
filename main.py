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
import numpy as np

data_cache_dir = "/ram/tmp" #"~/datasets" #"/ram/tmp"
# dataset = 'stories' # This still needs to be set manually

# Set seed
torch.manual_seed(1337)

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    # Test loss
    losses=[]
    for itr,batch in enumerate(test_loader):
        if itr==args.eval_iters:
            break
        if args.dataset not in ['shakespeare','ptb','cbt']:
            batch = batch['text']
        data = tokenizer(batch,padding="max_length",truncation=True,max_length=block_size+1,return_tensors="pt")        
        if args.dataset not in ['shakespeare','ptb','cbt']:
            data = data['input_ids']
        # data = data['input_ids']
        data = data.to(device)
        xb,yb = data[:, :-1], data[:, 1:]
        logits, loss = model(xb, yb)
        losses.append(loss.item())
    out['test'] = np.mean(losses)

    # Train loss
    losses=[]
    for itr,batch in enumerate(train_loader):
        if itr==args.eval_iters:
            break
        if args.dataset not in ['shakespeare','ptb','cbt']:
            batch = batch['text']
        data = tokenizer(batch,padding="max_length",truncation=True,max_length=block_size+1,return_tensors="pt")        
        if args.dataset not in ['shakespeare','ptb','cbt']:
            data = data['input_ids']
        # data = data['input_ids']
        data = data.to(device)
        xb,yb = data[:, :-1], data[:, 1:]
        logits, loss = model(xb, yb)
        losses.append(loss.item())

    out['train'] = np.mean(losses)
    model.train()
    return out


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
    parser.add_argument('--N', type=int, default=6, help='Specify the number of layers')
    
    parser.add_argument('--norm-type', type=str, default='layer', help='Type of normalization layer to use ("layer" for LayerNorm, "rms" for RMSNorm)')
    parser.add_argument('--post-norm', type=int, default=1, help='Whether to use post layer normalization')
    parser.add_argument('--final-norm', type=str, default='layer', help='Norm to use in final layer ("layer" for LayerNorm, "rms" for RMSNorm)')
    parser.add_argument('--rectify', type=int, default=0, help='Whether to use rectified attention')
    parser.add_argument('--dropout', type=float, default=0.2, help='Specify the dropout')
    parser.add_argument('--attention-type', type=str, default='sdp', help='Type of attention to use ("sdp" for scaled dot product, "other" for other types)')
    parser.add_argument('--block-architecture', type=str, default='series', help='Type of block architecture to use ("series" for series of blocks, "parallel" for parallel blocks)')

    parser.add_argument('--lr', type=float, default=1e-3, help='Specify the learning rate')
    parser.add_argument('--device', type=str, default=device, help='Specify the device')
    parser.add_argument('--n-itrs', type=int, default=20001, help='Specify the number of iterations')
    parser.add_argument('--filepath', type=str,default='original.csv', help='Specify the file path')
    parser.add_argument('--dataset', type=str,default='stories', help='Specify the dataset')
    parser.add_argument('--stream-data',action='store_true', help='Whether to stream data from disk')
    parser.add_argument('--version', type=int,default=0, help='For saving the model with distinct names')
    args = parser.parse_args()

    # version = args.block_type
    version = args.version
    block_size=args.block_size

    # Load the dataset
    if args.dataset == 'shakespeare':
        # Download a sample text file (e.g., "The Complete Works of William Shakespeare")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        file_path = "datasets/shakespeare.txt"

        if not os.path.exists(file_path):
            response = requests.get(url)
            with open(file_path, 'w') as file:
                file.write(response.text)

        # Read in text file
        with open(file_path,'r',encoding='utf-8') as f:
            text = f.read()

        # Split up text
        n = len(text)
        test_text = text[:n//10]
        train_text = text[n//10:] 
        train_set = TextDataFromFile(text=train_text,block_size=block_size+1)
        test_set = TextDataFromFile(text=test_text,block_size=block_size+1)
    elif args.dataset == 'stories': # Working
        train_set = load_dataset("nRuaif/tinystories-gpt4",cache_dir=data_cache_dir,split='train',streaming=args.stream_data)
        test_set = load_dataset("nRuaif/tinystories-gpt4",cache_dir=data_cache_dir,split='test',streaming=args.stream_data)
    elif args.dataset == 'wikitext103': # Working
        train_set = load_dataset("wikitext",'wikitext-103-v1',cache_dir=data_cache_dir,split='train',streaming=args.stream_data)
        test_set = load_dataset("wikitext",'wikitext-103-v1',cache_dir=data_cache_dir,split='test',streaming=args.stream_data)
    elif args.dataset == "wikitext2": # Working
        train_set = load_dataset("wikitext",'wikitext-2-v1',cache_dir=data_cache_dir,split='train',streaming=args.stream_data)
        test_set = load_dataset("wikitext",'wikitext-2-v1',cache_dir=data_cache_dir,split='test',streaming=args.stream_data)
    # elif args.dataset == "simple_wiki": # There is not test split, can't get working
    #     train_test_split_percentage = 0.8  # 80% for training, 20% for testing
    #     full_set = load_dataset("wikipedia","20220301.en",trust_remote_code=True,cache_dir=data_cache_dir,split='train',streaming=args.stream_data)
    #     full_set = full_set.train_test_split(train_size=train_test_split_percentage)
    #     train_set = full_set['train']
    #     test_set = full_set['test']        
    #     # train_set = load_dataset("wikipedia","20220301.en",trust_remote_code=True,cache_dir=data_cache_dir,split='train',streaming=args.stream_data)
    #     # test_set = load_dataset("wikipedia","20220301.en",trust_remote_code=True,cache_dir=data_cache_dir,split='train',streaming=args.stream_data)
    elif args.dataset == "cbt":
        train_set = load_dataset("cbt",'CN',cache_dir=data_cache_dir,split='train',streaming=args.stream_data)
        test_set = load_dataset("cbt",'CN',cache_dir=data_cache_dir,split='test',streaming=args.stream_data)
    elif args.dataset == "ptb":
        train_set = load_dataset("ptb_text_only",'penn_treebank',cache_dir=data_cache_dir,split='train',streaming=args.stream_data)
        test_set = load_dataset("ptb_text_only",'penn_treebank',cache_dir=data_cache_dir,split='test',streaming=args.stream_data)
 
    # Make dataloaders
    train_loader = DataLoader(train_set, batch_size=args.batch_size) # shuffle=True
    test_loader = DataLoader(test_set, batch_size=args.batch_size) # shuffle=False

    # Select an appropriate tokenizer
    if args.dataset in ["wikitext2", "simple_wiki", "cbt"]:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    elif args.dataset in ["ptb","wikitext103"]:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")  # GPT-2 tokenizer works well with PTB
    elif args.dataset == "stories":
        tokenizer = AutoTokenizer.from_pretrained("georgeyw/TinyStories-tokenizer-5k")
        # tokenizer = AutoTokenizer.from_pretrained("georgeyw/TinyStories-tokenizer-10k")
    elif args.dataset == "shakespeare":
        tokenizer = CharacterTokenizer(block_size=block_size+1)
    
    if args.dataset != "shakespeare":
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    vocab_size=len(tokenizer)
    decode = tokenizer.decode
    encode = tokenizer.encode

    filepath = args.filepath
    args_dict = {k: v for k, v in vars(args).items() if v is not None}
    args_dict['vocab_size'] = vocab_size

    # Make / load model
    if os.path.exists('transformer_' + str(version) + '.pt'):
        model = torch.load('transformer_' + str(version) + '.pt')
    else:
        # model = Transformer(**args_dict)
        model = Transformer(vocab_size=vocab_size,dm=args.dm,dk=args.dk,dv=args.dv,block_size=args.block_size,h=args.h,N=args.N,final_norm=args.final_norm,norm_type=args.norm_type, post_norm=args.post_norm, rectify=args.rectify,dropout=args.dropout,block_architecture=args.block_architecture,attention_type=args.attention_type,pad_token_id=tokenizer.pad_token_id)
 

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

    # Train
    tic = time.time()
    for itr,batch in enumerate(train_loader):
        if args.dataset not in ['shakespeare','ptb','cbt']:
            batch = batch['text']
        data = tokenizer(batch,padding="max_length",truncation=True,max_length=block_size+1,return_tensors="pt")        
        if args.dataset not in ['shakespeare','ptb','cbt']:
            data = data['input_ids']
   
        data = data.to(device)
        xb,yb = data[:, :-1], data[:, 1:]

        # if itr == 0: # Something weird is going on here. It is printing a bunch of [SEP]
        #     for i in range(3):
        #         text = tokenizer.decode(xb[i])
        #         print("Example from training set: ", text)
        #         print(xb[i])
            # assert(0)
        if itr % args.eval_interval == 0:
            elapsed, tic = time.time() - tic, time.time()
            print(f"step {itr}: {elapsed:.2f} seconds")
            losses = estimate_loss(model)  # Calculate loss
            with open(filepath, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([losses[split] for split in ['train','test']])

            # Generate sample
            # cls_token_id = tokenizer.cls_token_id
            prompt = "The meaning of life is"
            prompt = encode(prompt, return_tensors="pt").to(device)
            print("\nSample: \n", decode(list(prompt[0])[args.block_size:]), '\n\n')
            assert(0)
            n = len(prompt[0])
            idx = torch.zeros((1, args.block_size), device=device, dtype=torch.long)
            idx[0,-n:] = prompt #cls_token_id # Just added
            # idx = m.generate(idx, 200) # Set beta = 2?
            idx = m.generate(idx, 200,prompt_len=n) # Set beta = 2?
            print("\nSample: \n", decode(list(idx[0])[args.block_size:]), '\n\n')
            # assert(0)
            print(f"step {itr}: train loss {losses['train']:.4f}, val loss {losses['test']:.4f}")
            torch.save(m, 'transformer_' + str(version) + '.pt')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.save(m,'transformer_'+str(args.version)+'.pt')
    idx=torch.zeros((1,block_size),device=device,dtype=torch.long)
    idx=m.generate(idx,5000)
  
