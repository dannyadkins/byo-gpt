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
parser.add_argument("-l", "--load", help = "Model file path")
parser.add_argument("-s", "--save")

torch.cuda.empty_cache()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torch.profiler import profile, record_function

def save_model(model, name):
    torch.save(model.state_dict(), "./weights/" + name)


def train(model: nn.Module, loader: DataLoader, tokenizer, epochs: int = 20, lr: float = 1e-3, clip_grad_norm=True):
    model.train()
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            inputs = batch['input_ids']
            padding_mask = batch["attention_mask"]

            targets = inputs.clone().detach()[:, 1:]
            targets = torch.cat([targets, torch.full((targets.size(0), 1), tokenizer.eos_token_id).to(targets.device)], dim=1)

            # targets[padding_mask == 0] = -100
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(inputs, padding_mask=padding_mask)

                loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            # print("Loss at epoch ", epoch, ": ", loss.item())
            model.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            if (clip_grad_norm):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            scaler.step(optimizer)
            scaler.update()

        print("Epoch ", epoch, " done with loss ", loss.item())
        generate_sequence(model, tokenizer, inputs.shape[1])
        save_model(model, "model1-mixedprecision")

def evaluate(model: nn.Module, loader: DataLoader, tokenizer):
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    total_loss = torch.tensor(0.0)
    num_batches = 0
    for batch in loader:
        batch = { k: v.to(device) for k,v in batch.items()}
        inputs = batch['input_ids']
        padding_mask = batch["attention_mask"]

        targets = inputs.clone().detach()[:, 1:]
        targets = torch.cat([targets, torch.full((targets.size(0), 1), tokenizer.eos_token_id).to(targets.device)], dim=1)
                
        outputs = model(inputs, padding_mask=padding_mask)

        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        num_batches+=1
        total_loss+=loss.item()
    print("Average loss: ", total_loss/num_batches)
    return (total_loss/num_batches).item()

def generate_sequence(model: nn.Module, tokenizer, seq_len: int, k=1, temperature=1):
    start_token_id = random.randint(0, 50000)
    generated = [start_token_id]
    num_tokens = seq_len

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

def run_experiment(model_func, train_func, eval_func, fixed_params, variable_params, runs_per_var=1):
    # get every combination of experiment_params 

    for param_func in variable_params.keys():
        for param_name, param_possible_values in variable_params[param_func].items():
            for param_value in param_possible_values:
                total_avg_loss = 0
                print("Running experiment: ", param_name, " = ", param_value)
                for i in range(runs_per_var):
                    fixed_params[param_func][param_name] = param_value

                    model = model_func(**fixed_params["model"]).to(device)
                    train_func(model, **fixed_params["train"])
                    avg_loss = eval_func(model, **fixed_params["eval"])
                    print("Avg_loss after run " + str(i) + ": " + str(avg_loss))
                    total_avg_loss += avg_loss 
                # save to table
                avg_avg_loss = total_avg_loss/runs_per_var 
                print("Average loss over all runs: ", avg_avg_loss)

                model_version = model.__version__()


def main(model_path="model1"):
    seq_len=128
    dataset_name="bookcorpus"
    tokenizer = AutoTokenizer.from_pretrained("gpt2", add_prefix_space=True)
    
    dataset = get_and_preprocess_dataset(dataset_name=dataset_name, tokenizer=tokenizer, seq_len=seq_len, test_split=0.2)
    train_loader = DataLoader(dataset['train'], batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset['test'])
    

    model = BYOGPT(vocab_size=len(tokenizer), num_layers=3, num_heads=8, d_model=128)
    model = model.to(device)
    print(sum(p.numel() for p in model.parameters()), " total params")

    if (model_path):
        try: 
            model.load_state_dict(torch.load("./weights/" + model_path))
        except Exception as e:
            print("Error loading model: ", e)
    
    fixed_params = {
        "train": { 
            "loader": train_loader,
            "tokenizer": tokenizer,
            "epochs": 10
        },
        "eval": {
            "loader": test_loader,
            "tokenizer": tokenizer
        },
        "model": {
            "num_layers": 1,
            "vocab_size": tokenizer.vocab_size,
            "print_shapes": False
        },
    }
    
    variable_params = {"train": { "clip_grad_norm": [True, False]}}

    # run_experiment(model_func=BYOGPT, train_func=train, eval_func=evaluate, fixed_params=fixed_params, variable_params=variable_params, runs_per_var=5)

    train(model, loader=train_loader, tokenizer=tokenizer)
    evaluate(model, loader=test_loader, tokenizer=tokenizer)
    # save model and any experiment info 

if __name__ == '__main__':
    args = parser.parse_args()

    main()