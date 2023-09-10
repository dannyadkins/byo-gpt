import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from models.byo_gpt import BYOGPT
from data import get_train_inputs
from transformers import AutoTokenizer

torch.cuda.empty_cache()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torch.profiler import profile, record_function

def train(model: nn.Module, loader: DataLoader, tokenizer, epochs: int = 1, lr: float = 1e-3):
    model.train()
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    with profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
        for epoch in range(epochs):
            for batch in loader:
                with record_function("load_batch"):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    inputs = batch['input_ids']
                    targets = torch.tensor(inputs[:, 1:])
                    targets = torch.cat([targets, torch.full((targets.size(0), 1), tokenizer.eos_token_id).to(targets.device)], dim=1)
                
                with record_function("model_forward"):
                    outputs = model(inputs)
                with record_function("compute_loss"):
                    loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                with record_function("backward_and_optimize"):
                    model.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
    
    print(prof.key_averages().table(sort_by="cuda_time_total"))

def evaluate(model: nn.Module):
    pass

def main():
    tokenizer = AutoTokenizer.from_pretrained("gpt2", add_prefix_space=True)
    
    dataset = get_train_inputs(tokenizer=tokenizer, seq_len=64)

    model = BYOGPT(vocab_size=tokenizer.vocab_size, print_shapes=False)
    # huggingface model 
    # config = GPT2Config()
    # model = GPT2LMHeadModel(config)
    
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = model.to(device)
    train(model, loader, tokenizer)
    evaluate(model)

if __name__ == '__main__':
    main()