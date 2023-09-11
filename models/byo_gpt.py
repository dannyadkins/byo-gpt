"""
My own custom base GPT implementation.
"""

import torch 
import torch.nn as nn
import numpy as np 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BYOGPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_layers=1, num_heads=1, print_shapes=False):
        print("Initializing BYOGPT with vocab size: {}".format(vocab_size))
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers 

        self.embed = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_encoding = PositionalEncodingLayer(d_model=self.d_model, print_shapes=print_shapes)

        self.attn_blocks = nn.ModuleList([
            AttentionBlock(d_model=self.d_model, num_heads=num_heads) for _ in range(self.num_layers)
        ])
        self.unembed = nn.Linear(self.d_model, self.vocab_size)

        self.print_shapes = print_shapes
    
    """
    Input: [batch_size x seq_len]
    Output: [batch_size x seq_len x vocab_size] (odds of each next word at each position)
    """
    def forward(self, x):
        x = self.embed(x) # [batch_size x seq_len]
        if self.print_shapes:
            print("Embed shape: ", x.shape)
        x = self.pos_encoding(x) # [batch_size x seq_len x vocab_size]
        if self.print_shapes:
            print("After pos_embed shape: ", x.shape)

        for attn_block in self.attn_blocks:
            x = attn_block(x)
        
        # get the logits and logprobs 
        logits = self.unembed(x)

        return logits
    
    def __version__(self):
        return "1.0-singlehead"

class PositionalEncodingLayer(nn.Module):
    def __init__(self, d_model, max_len=5000, print_shapes=False):
        super().__init__()
        self.d_model = d_model 
        self.n = 10000

        self.pe = self.get_pe(max_len)

    # TODO: optimize this calculation using PyTorch, and then potentially optimize with Triton
    def forward(self, x):
        # make an empty matrix of size seq_len x d_model 
        return x + self.pe[:x.size(1)].requires_grad_(False)

    def get_pe(self, max_len):
        pe = torch.zeros(max_len, self.d_model).to(device)

        for k in range(0, max_len):
            for i in range(0, self.d_model//2):
                theta = k/(self.n**(2*i/self.d_model))
                pe[k, 2*i] = np.sin(theta)
                pe[k, 2*i + 1] = np.cos(theta)
        
        # saves to state_dict but doesn't do anything with optimizer 
        return pe 

# Implementation of masked/causal self-attention, where each token only attends to previous tokens. 
class AttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 1, max_seq_len: int = 10000): 
        super().__init__()
        self.d_model = d_model  
        self.num_heads = num_heads 

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.d_head = d_model // num_heads

        self.softmax = nn.Softmax(dim=-1)

        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model)

        self.norm2 = nn.LayerNorm(d_model) 

        attn_mask = torch.tril(torch.ones((max_seq_len, max_seq_len))).view(1,1,max_seq_len,max_seq_len)
        self.register_buffer("attn_mask", attn_mask)

    
    def forward(self, x):
        # embed in q, k, v 
        q = self.wq(x).view(x.size(0), -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        k = self.wk(x).view(x.size(0), -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        v = self.wv(x).view(x.size(0), -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)

        # compute attention scores
        # [batch_size * seq_len * seq_len]
        # for each position's key, what is its relevance to each other position's query 
        # transpose k to be seq_len, d_head, num_heads dimension? still works yeah, basically q is multiplying dotwise for each head vs for each sample 
        scores = (q @ k.transpose(-2, -1))/torch.sqrt(torch.tensor(self.d_head)) 

        seq_len = x.size(1)

        mask = self.attn_mask == 0

        scores = scores.masked_fill(mask[:, :, :seq_len, :seq_len] == 0, float('-inf'))

        scores = self.softmax(scores)

        # for each seq, do the calculation softmax((q * kT)/sqrt(d_model))*v 
        attention = scores @ v 

        attention = attention.transpose(1,2).contiguous().view(-1, seq_len, self.num_heads*self.d_head)
        
        x = x + attention 

        # layernorm 
        x = self.norm1(x)

        # ffn 
        linear_output = self.linear1(x)

        # add back to input 
        x = x + linear_output 

        # layernorm 
        x = self.norm2(x)
        
        return x 



class SingleHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        