import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import requests
import os

# Download a sample text file (e.g., "The Complete Works of William Shakespeare")
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
file_path = "shakespeare.txt"

if not os.path.exists(file_path):
    response = requests.get(url)
    with open(file_path, 'w') as file:
        file.write(response.text)

# Tokenizer and model initialization
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Modify the model to have N attention layers
n_attention_layers = 3  # You can adjust this value
model.config.n_layer = n_attention_layers
model.transformer.h = nn.ModuleList([model.transformer.h[i] for i in range(n_attention_layers)])

# Define a custom dataset for training
# class TextDataset(Dataset):
#     def __init__(self, file_path, tokenizer, max_length=512):
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         with open(file_path, 'r') as file:
#             self.data = file.read()
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         text_chunk = self.data[idx:idx + self.max_length]
#         inputs = self.tokenizer.encode(text_chunk, return_tensors='pt')
#         return inputs

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(file_path, 'r') as file:
            self.data = file.read()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text_chunk = self.data[idx:idx + self.max_length]
        encoding = self.tokenizer(text_chunk, return_tensors='pt', max_length=self.max_length, truncation=True)
        return encoding['input_ids'], encoding['attention_mask']


# Create dataset and dataloader
dataset = TextDataset(file_path, tokenizer)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

for epoch in range(5):  # You can adjust the number of epochs
    model.train()
    total_loss = 0

    for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
        optimizer.zero_grad()

        # Extract input_ids and attention_mask from the batch
        inputs, attention_mask = batch[0].to(model.device), batch[1].to(model.device)

        # Forward pass
        outputs = model.generate(input_ids=inputs, max_length=50, num_beams=5, temperature=0.8, attention_mask=attention_mask)

        # Calculate loss (you may adjust this depending on your specific use case)
        loss = outputs.loss

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % 100 == 0:
            # Print generated examples during training
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Generated text: {generated_text}")

    average_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch}, Average Loss: {average_loss}")

#
#     for i,batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
#         optimizer.zero_grad()
#
#         # inputs = batch.to(model.device)
#         inputs, attention_mask = batch[0].to(model.device), batch[1].to(model.device)
#
#         outputs = model(inputs, labels=inputs)
#         loss = outputs.loss
#         loss.backward()
#         optimizer.step()
#
#         total_loss += loss.item()
#
#         if i % 100 == 0:
#             # Print generated examples during training
#             sample_output = model.generate(inputs, max_length=50, num_beams=5, temperature=0.8, attention_mask=attention_mask)
#             # sample_output = model.generate(inputs, max_length=50, num_beams=5, temperature=0.8)
#
#             generated_text = tokenizer.decode(sample_output[0], skip_special_tokens=True)
#             print(f"Generated text: {generated_text}")
#
#     average_loss = total_loss / len(dataloader)
#     print(f"Epoch {epoch}, Average Loss: {average_loss}")
#
# # Save the trained model
# model.save_pretrained("autoregressive_model")
