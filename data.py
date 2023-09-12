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

def get_and_preprocess_dataset(dataset_name, tokenizer, seq_len: int, test_split = 0.2):
    num_proc=8
    override_cache=True

    special_tokens_dict = {"pad_token": "<PAD>"}
    num_tokens_added = tokenizer.add_special_tokens(special_tokens_dict)
    print("Added tokens: ", num_tokens_added)

    ### TINY SHAKESPEARE 
    if (dataset_name == "tiny_shakespeare"):
        def tokenization(x):
            # print type of batch, using typing module
            return tokenizer(x['text'], truncation=True, max_length=seq_len, is_split_into_words=True)

        dataset = load_dataset("tiny_shakespeare")['train']
        chunks = []
        text = re.split(r'\s+', dataset[0]['text'])
        for i in range(0, len(text), seq_len):
            item = text[i:i+seq_len]
            # pad it if needed 
            if len(item) < seq_len:
                item += ' ' * (seq_len - len(item))
            chunks.append(item) 

        dataset = datasets.Dataset.from_dict({'text': chunks})
        dataset = dataset.map(tokenization, batched=True)
    elif (dataset_name == "bookcorpus"):
        ### BOOKCORPUS CODE 
        def tokenization(x):
            # print type of batch, using typing module
            return tokenizer(x['text'], truncation=True, max_length=64, return_overflowing_tokens=False, padding="max_length", add_special_tokens=True)

        dataset_path = './datasets/bookcorpus'

        if not os.path.exists(dataset_path) or override_cache:
            dataset = load_dataset("bookcorpus", num_proc=num_proc)['train']
            print("Tokenizing dataset...")
            dataset = datasets.Dataset.from_dict(dataset[:100000]).map(tokenization, batched=True, num_proc=num_proc)
            print("Done tokenizing. Saving...")
            dataset.save_to_disk(dataset_path)
        else: 
            dataset = datasets.Dataset.load_from_disk(dataset_path)

    ### 
    dataset = dataset.remove_columns(['text'])
    
    # select test_split% 
    dataset.set_format(type='torch', columns=['input_ids', "attention_mask"])
    dataset = dataset.train_test_split(test_size=test_split)
    print("Dataset: ", dataset)
    
    # train_set = datasets.Dataset.from_dict(dataset[(int(dataset.num_rows*test_split)):])
    # print("Train_set size: ", train_set.num_rows)
    # test_set = datasets.Dataset.from_dict(dataset[:int(dataset.num_rows*test_split)])
    return dataset 

