import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from models.byo_gpt import BYOGPT
from data import get_and_preprocess_dataset
from transformers import AutoTokenizer
import random

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_file", help = "Model file path")

torch.cuda.empty_cache()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torch.profiler import profile, record_function

def train(model: nn.Module, loader: DataLoader, tokenizer, epochs: int = 20, lr: float = 1e-3):
    model.train()
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            inputs = batch['input_ids']
            targets = torch.tensor(inputs.clone().detach()[:, 1:])
            targets = torch.cat([targets, torch.full((targets.size(0), 1), tokenizer.eos_token_id).to(targets.device)], dim=1)
            
            outputs = model(inputs)

            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            # print("Loss at epoch ", epoch, ": ", loss.item())
            model.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        print("Epoch ", epoch, " done with loss ", loss.item())
        generate_sequence(model, tokenizer, inputs.shape[1])

def evaluate(model: nn.Module, loader: DataLoader, tokenizer):
    criterion = nn.CrossEntropyLoss()

    total_loss = torch.tensor(0.0)
    num_batches = 0
    for batch in loader:
        batch = { k: v.to(device) for k,v in batch.items()}
        inputs = batch['input_ids']
        targets = torch.tensor(inputs[:, 1:])
        targets = torch.cat([targets, torch.full((targets.size(0), 1), tokenizer.eos_token_id).to(targets.device)], dim=1)
        
        outputs = model(inputs)

        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        num_batches+=1
        total_loss+=loss.item()
    print("Average loss: ", total_loss/num_batches)

def generate_sequence(model: nn.Module, tokenizer, seq_len: int, k=1, temperature=1):
    start_token_id = random.randint(0, 50000)
    generated = [start_token_id]
    num_tokens = 64

    # sample most likely over and over
    for idx in range(0, num_tokens):
        input_seq = torch.tensor([generated + ([0] * (seq_len - len(generated) - 1))]).to(device)
        output = model(input_seq)
        last = output[0, idx]
        most_likely_id = torch.argmax(last)
        # TODO: temperature sampling. get the top k argmaxes, then 

        # print("Most likely ID: ", most_likely_id)
        # print("Detokenized most likely ID: ", tokenizer.decode(most_likely_id))
        generated.append(most_likely_id)
    
    print("Full sequence:\n", tokenizer.decode(generated)[:num_tokens])

def main(model_path):
    seq_len=64
    dataset_name="tiny_shakespeare"

    tokenizer = AutoTokenizer.from_pretrained("gpt2", add_prefix_space=True)
    
    dataset = get_and_preprocess_dataset(dataset_name=dataset_name, tokenizer=tokenizer, seq_len=seq_len, test_split=0.2)
    train_loader = DataLoader(dataset['train'], batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset['test'])
    
    model = BYOGPT(vocab_size=tokenizer.vocab_size, num_layers=1, print_shapes=False)

    if (model_path):
        print("Loading model from file ", model_path)
        model.load_state_dict(torch.load(model_path))

    model = model.to(device)

    train(model, loader=train_loader, tokenizer=tokenizer)
    evaluate(model, loader=test_loader, tokenizer=tokenizer)
    # save model and any experiment info 

if __name__ == '__main__':
    args = parser.parse_args()

    main(model_path=args.model_file)