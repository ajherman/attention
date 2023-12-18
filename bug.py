import torch
from torch.utils.data import Dataset, DataLoader
import transformers
from datasets import load_dataset

data_cache_dir = "/ram/tmp" #"/home/ari/Desktop"
dataset = load_dataset("nRuaif/tinystories-gpt4",cache_dir=data_cache_dir,split='train')
dataloader = DataLoader(dataset, batch_size=64)


