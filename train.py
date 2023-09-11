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

def train(model: nn.Module, loader: DataLoader, tokenizer, epochs: int = 20, lr: float = 1e-3, clip_grad_norm=True):
    model.train()
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            inputs = batch['input_ids']
            targets = inputs.clone().detach()[:, 1:]
            targets = torch.cat([targets, torch.full((targets.size(0), 1), tokenizer.eos_token_id).to(targets.device)], dim=1)
            
            outputs = model(inputs)

            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            # print("Loss at epoch ", epoch, ": ", loss.item())
            model.zero_grad(set_to_none=True)
            loss.backward()
            if (clip_grad_norm):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        print("Epoch ", epoch, " done with loss ", loss.item())
        # generate_sequence(model, tokenizer, inputs.shape[1])

def evaluate(model: nn.Module, loader: DataLoader, tokenizer):
    criterion = nn.CrossEntropyLoss()

    total_loss = torch.tensor(0.0)
    num_batches = 0
    for batch in loader:
        batch = { k: v.to(device) for k,v in batch.items()}
        inputs = batch['input_ids']
        targets = inputs.clone().detach()[:, 1:]
        targets = torch.cat([targets, torch.full((targets.size(0), 1), tokenizer.eos_token_id).to(targets.device)], dim=1)
        
        outputs = model(inputs)

        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        num_batches+=1
        total_loss+=loss.item()
    # print("Average loss: ", total_loss/num_batches)
    return (total_loss/num_batches).item()

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


def main(model_path):
    seq_len=64
    dataset_name="tiny_shakespeare"
    tokenizer = AutoTokenizer.from_pretrained("gpt2", add_prefix_space=True)
    
    dataset = get_and_preprocess_dataset(dataset_name=dataset_name, tokenizer=tokenizer, seq_len=seq_len, test_split=0.2)
    train_loader = DataLoader(dataset['train'], batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset['test'])
    
    model = BYOGPT(vocab_size=tokenizer.vocab_size, num_layers=1, num_heads=4)
    model = model.to(device)

    # if (model_path):
    #     print("Loading model from file ", model_path)
    #     model.load_state_dict(torch.load(model_path))


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

    main(model_path=args.model_file)