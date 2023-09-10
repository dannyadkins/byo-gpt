import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from models.byo_gpt import BYOGPT
from data import get_train_inputs
from transformers import AutoTokenizer
import random

torch.cuda.empty_cache()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torch.profiler import profile, record_function

def train(model: nn.Module, loader: DataLoader, tokenizer, epochs: int = 10, lr: float = 1e-3):
    model.train()
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            inputs = batch['input_ids']
            targets = torch.tensor(inputs[:, 1:])
            targets = torch.cat([targets, torch.full((targets.size(0), 1), tokenizer.eos_token_id).to(targets.device)], dim=1)
            
            outputs = model(inputs)

            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            # print("Loss at epoch ", epoch, ": ", loss.item())
            model.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        print("Epoch ", epoch, " done with loss ", loss.item())
        generate_sequence(model, tokenizer, inputs.shape[1])

def evaluate(model: nn.Module):
    pass

def generate_sequence(model: nn.Module, tokenizer, seq_len: int):
    start_token_id = random.randint(0, 50000)
    generated = [start_token_id]
    num_tokens = 64

    # sample most likely over and over
    for idx in range(0, num_tokens):
        input_seq = torch.tensor([generated + ([0] * (seq_len - len(generated) - 1))]).to(device)
        output = model(input_seq)
        last = output[0, idx]
        most_likely_id = torch.argmax(last)
        # print("Most likely ID: ", most_likely_id)
        # print("Detokenized most likely ID: ", tokenizer.decode(most_likely_id))
        generated.append(most_likely_id)
    
    print("Full sequence:\n", tokenizer.decode(generated)[:num_tokens])

def main():
    seq_len=64
    tokenizer = AutoTokenizer.from_pretrained("gpt2", add_prefix_space=True)
    
    dataset = get_train_inputs(tokenizer=tokenizer, seq_len=seq_len)

    model = BYOGPT(vocab_size=tokenizer.vocab_size, print_shapes=False)
    # huggingface model 
    # config = GPT2Config()
    # model = GPT2LMHeadModel(config)
    
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = model.to(device)
    train(model, loader, tokenizer)
    evaluate(model)

if __name__ == '__main__':
    main()