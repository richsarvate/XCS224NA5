"""
Originally forked from Andrej Karpathy's minGPT.

XCS224N : Homework 5

John Hewitt <johnhew@stanford.edu>
Ansh Khurana <anshk@stanford.edu>
Soumya Chatterjee <soumyac@stanford.edu>
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


def precompute_rotary_emb(dim, max_positions):
    """
    RoPE uses the following sinusoidal functions to encode positions:

    cos(t theta_i) and sin(t theta_i)
        where t is the position and
              theta_i = 1/10000^(2(i-1)/dim) for i in [1, dim/2]

    Since the maximum length of sequences is known, we can precompute
    these values to speed up training.

    Implement the precompute_rotary_emb function that returns a tensor of
    shape (max_positions, dim/2, 2) where the last dimension contains
    the cos and sin values for each position and each dimension of
    the embedding.
    """

    rope_cache = None

    ### TODO:
    ### [part h]
    ### START CODE HERE
    rope_cache = torch.zeros((max_positions, dim//2, 2))

    #TRY #1 - This works but is way too slow with the for loop

    #for t in range(max_positions):
    #    for i in range(dim//2):
    #
    #        theta_i = 1/(10000**(2*i/dim))
    #
    #        rope_cache[t][i][0] = math.cos(t * theta_i)
    #        rope_cache[t][i][1] = math.sin(t * theta_i)

    # TRY 2 with matrix math:

    #create dimension vector
    i = torch.arange(0, dim//2)
    #shape is (dim//2,)
    #looks like [0,1,2 ... ,dim//2]

    #create all the thetas at one time
    theta = 1/(10000**(2*i/dim))
    #shape is (dim//2,)
    #looks like [theta0, theta1,...,theta_dim/2]

    theta = torch.unsqueeze(theta, 0)
    #shape is (1, dim//2)
    # looks like:
    #   [[theta0, theta1, theta2,...,theta_d/2]]

    t = torch.arange(0, max_positions)
    t = torch.unsqueeze(t, 1)
    # shape is (max_positions,1)
    # looks like:
    #   [[0],
    #    [1],
    #    [2], 
    #       ...
    #    [max_positions]]

    angles = t * theta
    # shape is (max_positions,1) x (1, dim//2) = (max_positions, dim//2)
    #   [0*theta_0, 0*theta_1, ...]
    #   [1*theta_0, 1*theta_1, ...]
    #   [2*theta_0, 2*theta_1, ...]
    #   [...

    cos_matrix = torch.cos(angles)
    # shape is (max_positions, dim//2)
    #   [cos(0*theta_0), cos(0*theta_1), ...]
    #   [cos(1*theta_0), cos(1*theta_1), ...]
    #   [cos(2*theta_0), cos(2*theta_1), ...]
    #   [...

    sin_matrix = torch.sin(angles)
    # shape is (max_positions, dim//2)
    #   [sin(0*theta_0), sin(0*theta_1), ...]
    #   [sin(1*theta_0), sin(1*theta_1), ...]
    #   [sin(2*theta_0), sin(2*theta_1), ...]
    #   [...

    #stack these matrices on the last dimension to pair up 
    #each element making shape (max_positions, dim//2, 2)
    rope_cache = torch.stack([cos_matrix, sin_matrix], dim = -1)

    ### END CODE HERE
    return rope_cache


def apply_rotary_emb(x, rope_cache):
    """Apply the RoPE to the input tensor x."""
    # [part h]
    # You might find the following functions useful to convert
    # between real and complex numbers:

    # torch.view_as_real - https://pytorch.org/docs/stable/generated/torch.view_as_real.html
    # torch.view_as_complex - https://pytorch.org/docs/stable/generated/torch.view_as_complex.html

    # Note that during inference, the length of the sequence might be different
    # from the length of the precomputed values. In this case, you should use
    # truncate the precomputed values to match the length of the sequence.

    rotated_x = None
    ### TODO:
    ### [part h]
    ### START CODE HERE

    #shape of x is (batches, num heads, T (token position), head size or d)
    #shape of rope_cache is (max_positions, d//2, 2)
    #truncate cache to be the same length as x
    num_tokens = x.shape[2]

    if num_tokens < rope_cache.shape[0]:
        rope_cache = rope_cache[:num_tokens]

    #shape of x is (batches, num heads, T (token position), d)
    #shape of rope_cache is (T, d//2, 2)

    #reshape last 2 dimensions of x into d//2 x 2
    x = x.reshape((x.shape[0], x.shape[1], x.shape[2], x.shape[3]//2, 2))
    #shape of x is now (batches, num heads, T (token position), d//2, 2)

    x = x.contiguous() #the reshape made the tensor noncontiguous in memory

    #change the pairs in the last dimension into complex numbers
    x_complex = torch.view_as_complex(x)
    #shape of x is now (batches, num heads, T (token position), d//2)

    #change to complex
    rope_complex = torch.view_as_complex(rope_cache.contiguous())
    #shape of rope_complex is (T, d//2)

    #change shape to get ready for multiplication so dimensions match up
    rope_complex = rope_complex.unsqueeze(0).unsqueeze(0)
    #shape of rope_complex is (1,1,T, d//2)

    #apply the rotation
    rotated_complex = rope_complex * x_complex

    #switch back to real numbers:
    rotated_x = torch.view_as_real(rotated_complex.contiguous())

    #change rotated x back from pairs in the last dimension to just 1 column
    rotated_x = rotated_x.reshape((rotated_x.shape[0], rotated_x.shape[1], rotated_x.shape[2], rotated_x.shape[3]*2))

    ### END CODE HERE
    return rotated_x

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    I believe I could have just used torch.nn.MultiheadAttention but their documentation
    is all but absent and code ugly so I don't trust it, rolling my own here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

        self.rope = config.rope
        if self.rope:
            assert (config.n_embd % config.n_head) % 2 == 0

            ### TODO:
            # [part h] Precompute the cos and sin values for RoPE and
            # store them in rope_cache.
            # Hint: The maximum sequence length is given by config.block_size.
            rope_cache = None
            ### START CODE HERE

            #dimensions are the embedding size, divided by the number of heads
            #because we're only calculating rotations per head
            dim_size = config.n_embd // config.n_head
            rope_cache = precompute_rotary_emb(dim_size,config.block_size)
            ### END CODE HERE

            self.register_buffer("rope_cache", rope_cache)

        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        # (B x T x C) is of dimension (batch x block_size x n_embd) which is (batch x l x d) in the handout.
        # nh should be number_of_heads, and hs would then stand for n_embed (or "dimensionality" d in the handout) per head

        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if self.rope:
            pass
            ### TODO:
            # [part h] Apply RoPE to the query and key.
            ### START CODE HERE
            q = apply_rotary_emb(q,self.rope_cache)
            k = apply_rotary_emb(k,self.rope_cache)
            ### END CODE HERE

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, -1e10)
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class CausalCrossAttention(nn.Module):
    """
    Modifications over the self-attention layer to handle two inputs and perform
    cross-attention between them.
    This follows the implementation of the self attention module with
    auto-regressive masking on (key).
    Manipulation of batch-size to allow for different batch size between the 
    two inputs, with broadcasting over to the higher batch size value.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x_kv, x_q):
        Bk, Tk, Ck = x_kv.size()
        Bq, Tq, Cq = x_q.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        
        # keys of x1
        k = self.key(x_kv).view(Bk, Tk, self.n_head, Ck // self.n_head).transpose(1, 2) # (B, nh, Tk, hs)
        
        # query with x2
        q = self.query(x_q).view(Bq, Tq, self.n_head, Cq // self.n_head).transpose(1, 2) # (B, nh, Tq, hs)
        
        # values from x1
        v = self.value(x_kv).view(Bk, Tk, self.n_head, Ck // self.n_head).transpose(1, 2) # (B, nh, Tk, hs)

        # causal self-attention;  (B, nh, Tk, hs) x (B, nh, hs, Tq) -> (B, nh, Tq, Tk)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        B = max(Bk, Bq)
        
        att = att.masked_fill(self.mask[:,:,:Tq,:Tk] == 0, -1e10) 
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, Tq, Tk) x (B, nh, Tk, hs) -> (B, nh, Tq, hs)
        y = y.transpose(1, 2).contiguous().view(B, Tq, Cq) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y
