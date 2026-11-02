"""
Shared pieces for the memory experiments.

Each frame goes through the same encoder as Compact n_layers=3 and becomes
a 64x8x8 latent. The latent is cut into 2x2 patches, which gives 16 tokens
of dimension 256 per frame, projected to d_model. A small Transformer looks
at the tokens of the last T frames and predicts the tokens of the next
frame, and the Compact decoder turns them back into an image. That image is
encoded again and enters the window: the rollout goes one frame at a time.

The memory variants (transformer_rmt, transformer_knn) only change what
happens to frames once they leave the window.
"""
import math
import torch.nn as nn

from ..compact.compact_predictor import CompactEncoder, CompactDecoder


N_LAYERS_CODEC = 3          # 64x64 frame -> 8x8 latent
LATENT_DIM = 64
PATCH = 2                   # 8x8 latent -> 4x4 grid of 2x2 patches
TOKENS_PER_FRAME = (8 // PATCH) ** 2          # 16
TOKEN_IN = LATENT_DIM * PATCH * PATCH         # 256


def attention(q, k, v):
    """
    Scaled dot-product attention: softmax(q k^T / sqrt(d)) v.
    q: (B, heads, Nq, d), k and v: (B, heads, Nk, d) -> (B, heads, Nq, d)
    """
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    return scores.softmax(dim=-1) @ v


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def project(self, x):
        """(B, N, d_model) -> q, k, v of shape (B, heads, N, d_head)"""
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, D // self.n_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        return q, k, v

    def merge(self, y):
        B, H, N, d = y.shape
        return self.out(y.transpose(1, 2).reshape(B, N, H * d))

    def forward(self, x, cache=None):
        q, k, v = self.project(x)
        return self.merge(attention(q, k, v))


class Block(nn.Module):
    """Pre-norm Transformer block: x + attn(norm(x)), then x + mlp(norm(x))."""

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.GELU(),
            nn.Linear(mlp_ratio * d_model, d_model),
        )

    def forward(self, x, cache=None):
        x = x + self.attn(self.norm1(x), cache)
        return x + self.mlp(self.norm2(x))


class FrameCodec(nn.Module):
    """Compact encoder / decoder + patch tokenization."""

    def __init__(self, d_model: int):
        super().__init__()
        self.encoder = CompactEncoder(1, LATENT_DIM, N_LAYERS_CODEC)
        self.decoder = CompactDecoder(LATENT_DIM, 1, N_LAYERS_CODEC)
        self.to_tokens = nn.Linear(TOKEN_IN, d_model)
        self.from_tokens = nn.Linear(d_model, TOKEN_IN)

    def encode(self, frames):
        """(B, T, 1, 64, 64) -> (B, T, 16, d_model)"""
        B, T = frames.shape[:2]
        z = self.encoder(frames.reshape(B * T, *frames.shape[2:]))      # (BT, 64, 8, 8)
        g = z.shape[-1] // PATCH
        z = z.reshape(B * T, LATENT_DIM, g, PATCH, g, PATCH)
        z = z.permute(0, 2, 4, 1, 3, 5).reshape(B, T, g * g, TOKEN_IN)
        return self.to_tokens(z)

    def decode(self, tokens):
        """(B, 16, d_model) -> (B, 1, 64, 64)"""
        B = tokens.shape[0]
        g = int(math.sqrt(TOKENS_PER_FRAME))
        z = self.from_tokens(tokens).reshape(B, g, g, LATENT_DIM, PATCH, PATCH)
        z = z.permute(0, 3, 1, 4, 2, 5).reshape(B, LATENT_DIM, g * PATCH, g * PATCH)
        return self.decoder(z)
