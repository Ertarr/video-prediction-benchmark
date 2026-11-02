"""
Window Transformer + kNN memory (after Memorizing Transformers, Wu et al.,
ICLR 2022), simplified.

Every frame that leaves the window leaves its keys and values from one
block (mem_layer) in a cache. In that block, each query attends to its
top-k keys in the cache, and the result is mixed with the normal
attention over the window by a learned gate per head:

    out = g * attention_over_memory + (1 - g) * attention_over_window,
    g = sigmoid(gate)

As in the paper, the cache is stored detached (no gradient). Exact top-k
search here, since the cache is small. cache_frames keeps only the most
recent frames, which is how forgetting is tested.
"""
import math
import torch
import torch.nn as nn

from ..transformer_window.core import SelfAttention, attention, TOKENS_PER_FRAME
from ..transformer_window.window_predictor import WindowTransformer


def knn_attention(q, mem_k, mem_v, topk: int):
    """
    Attention restricted, for each query, to its top-k keys in the memory.
    q: (B, heads, Nq, d), mem_k and mem_v: (B, heads, Nm, d) -> (B, heads, Nq, d)
    """
    scores = q @ mem_k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    k = min(topk, scores.shape[-1])
    kth = scores.topk(k, dim=-1).values[..., -1:]           # k-th best score per query
    scores = scores.masked_fill(scores < kth, float("-inf"))
    return scores.softmax(dim=-1) @ mem_v


class KNNAttention(SelfAttention):
    def __init__(self, d_model: int, n_heads: int, topk: int = 32):
        super().__init__(d_model, n_heads)
        self.topk = topk
        self.gate = nn.Parameter(torch.zeros(n_heads))       # g = 0.5 at init
        self.last_kv = None

    def forward(self, x, cache=None):
        q, k, v = self.project(x)
        self.last_kv = (k.detach(), v.detach())               # for the cache
        y = attention(q, k, v)
        if cache is not None and cache["k"].shape[2] > 0:
            y_mem = knn_attention(q, cache["k"], cache["v"], self.topk)
            g = torch.sigmoid(self.gate).view(1, -1, 1, 1)
            y = g * y_mem + (1 - g) * y
        return self.merge(y)


class KNNTransformer(WindowTransformer):
    def __init__(self, *args, mem_layer: int = 2, topk: int = 32,
                 cache_frames: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mem_layer = mem_layer
        self.cache_frames = cache_frames          # None = keep everything
        n_heads = self.blocks[mem_layer].attn.n_heads
        self.blocks[mem_layer].attn = KNNAttention(self.d_model, n_heads, topk)

    def memory_kv(self, segment):
        """Keys and values of the memory block for a segment of frames."""
        with torch.no_grad():
            x = self.embed(segment)
            for block in self.blocks[:self.mem_layer]:
                x = block(x)
            mem_block = self.blocks[self.mem_layer]
            _, k, v = mem_block.attn.project(mem_block.norm1(x))
        return k, v

    def trim(self, state):
        if self.cache_frames is None:
            return state
        n = self.cache_frames * TOKENS_PER_FRAME
        if n == 0:
            return {"k": state["k"][:, :, :0], "v": state["v"][:, :, :0]}
        return {"k": state["k"][:, :, -n:], "v": state["v"][:, :, -n:]}

    def init_state(self, history):
        B, L = history.shape[:2]
        attn = self.blocks[self.mem_layer].attn
        d_head = self.d_model // attn.n_heads
        ks = [history.new_zeros(B, attn.n_heads, 0, d_head)]
        vs = [history.new_zeros(B, attn.n_heads, 0, d_head)]
        start = L % self.window
        bounds = ([(0, start)] if start > 0 else []) + \
                 [(i, i + self.window) for i in range(start, L, self.window)]
        for a, b in bounds:
            k, v = self.memory_kv(history[:, a:b])
            ks.append(k)
            vs.append(v)
        return self.trim({"k": torch.cat(ks, dim=2), "v": torch.cat(vs, dim=2)})

    def run_blocks(self, x, state):
        for i, block in enumerate(self.blocks):
            x = block(x, cache=state if i == self.mem_layer else None)
        return x

    def update_state(self, state, leaving):
        # the frame leaving the window was the first one of the last pass
        k, v = self.blocks[self.mem_layer].attn.last_kv
        n = TOKENS_PER_FRAME
        return self.trim({"k": torch.cat([state["k"], k[:, :, :n]], dim=2),
                          "v": torch.cat([state["v"], v[:, :, :n]], dim=2)})
