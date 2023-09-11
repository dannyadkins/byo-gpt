import datasets 
from datasets import load_dataset
from typing import Literal, Callable, Union, Any, Tuple, Dict 
import torch 
import re
import os  
# def generate_vocab(get_raw_text: Callable[[], str], vocab_size: int = 5000) -> Tuple[Dict, Dict]:
#     # read vocab from file if there is 
#     # otherwise generate vocab from data
#     raw_text = get_raw_text()
#     raw_text = preprocess_text(raw_text)

#     tokens = raw_text.split()
#     vocab = {}
#     for token in tokens:
#         if token in vocab:
#             vocab[token] += 1
#         else:
#             vocab[token] = 1

#     vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
#     vocab = vocab[:vocab_size]
#     vocab = {word: idx for idx, (word, _) in enumerate(vocab)}

#     reverse_vocab = {idx: word for word, idx in vocab.items()}
#     return vocab, reverse_vocab

# def get_hf_dataset(dataset_name, split='train'):
#     return datasets.load_dataset('tiny_shakespeare')[split]

# def preprocess_text(text: str) -> str:
#     return text.replace('\n', ' <eos> ').strip()

def get_train_inputs(tokenizer, seq_len: int):
    num_proc=12

    ### TINY SHAKESPEARE 
    # def tokenization(x):
    #     # print type of batch, using typing module
    #     return tokenizer(x['text'], truncation=True, max_length=seq_len, is_split_into_words=True)

    # dataset = load_dataset("tiny_shakespeare")['train']
    # chunks = []
    # text = re.split(r'\s+', dataset[0]['text'])
    # for i in range(0, len(text), seq_len):
    #     item = text[i:i+seq_len]
    #     # pad it if needed 
    #     if len(item) < seq_len:
    #         item += ' ' * (seq_len - len(item))
    #     chunks.append(item) 

    # dataset = datasets.Dataset.from_dict({'text': chunks})
    # dataset = dataset.map(tokenization, batched=True)

    ### BOOKCORPUS CODE 
    def tokenization(x):
        # print type of batch, using typing module
        return tokenizer(x['text'], truncation=True, max_length=64, return_overflowing_tokens=False, padding=True)

    dataset_path = './datasets/bookcorpus'

    if not os.path.exists(dataset_path):
        dataset = load_dataset("bookcorpus", num_proc=num_proc)['train']
        dataset = dataset.map(tokenization, batched=True, num_proc=num_proc)
        dataset.save_to_disk(dataset_path)
    else: 
        dataset = datasets.Dataset.load_from_disk(dataset_path)
    
    dataset = dataset.remove_columns(['text'])
    dataset.set_format(type='torch', columns=['input_ids'])
    return dataset

if __name__ == '__main__':
    dataset = get_train_inputs(seq_len=128)