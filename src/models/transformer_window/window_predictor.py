"""
Transformer over a window of the last T frames, without memory.

This is the reference for the memory experiments: whatever left the
window is gone. Input (B, input_len, 1, 64, 64), output
(B, target_len, 1, 64, 64), like every VideoPredictor in the repo.

Prediction goes one frame at a time:
    1. the tokens of the last T frames get a spatial and a temporal
       position embedding and go through the Transformer blocks
    2. the output tokens of the last frame give the change to apply to
       that frame's tokens, which gives the next frame's tokens
    3. the decoder turns them into an image, which is encoded again and
       pushed into the window, while the oldest frame leaves it

The memory variants override three hooks: init_state (what to do with
the frames older than the window), run_blocks (how the window reads the
memory) and update_state (what to do with a frame leaving the window).
"""
import torch
import torch.nn as nn

from ..base import VideoPredictor
from .core import Block, FrameCodec, TOKENS_PER_FRAME


class WindowTransformer(VideoPredictor):
    def __init__(
        self,
        input_len: int = 50,
        target_len: int = 10,
        window: int = 10,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        mlp_ratio: int = 4,
    ):
        super().__init__(input_len=input_len, target_len=target_len)
        assert window <= input_len
        self.window = window
        self.d_model = d_model
        self.codec = FrameCodec(d_model)
        self.spatial_pos = nn.Parameter(torch.randn(TOKENS_PER_FRAME, d_model) * 0.02)
        self.time_pos = nn.Parameter(torch.randn(window, d_model) * 0.02)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, mlp_ratio) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, d_model)

    # ---- hooks for the memory variants ----

    def init_state(self, history):
        """history: tokens of the frames before the window, (B, L, 16, d)."""
        return None

    def run_blocks(self, x, state):
        for block in self.blocks:
            x = block(x)
        return x

    def update_state(self, state, leaving):
        """leaving: tokens of the frame that just left the window, (B, 1, 16, d)."""
        return state

    # ---- shared ----

    def embed(self, tokens):
        """
        (B, T, 16, d) -> (B, T*16, d). The last frame always gets the last
        temporal embedding, so positions are relative to the newest frame.
        """
        B, T, N, D = tokens.shape
        x = tokens + self.spatial_pos + self.time_pos[-T:, None]
        return x.reshape(B, T * N, D)

    def predict_tokens(self, window_tokens, state):
        x = self.run_blocks(self.embed(window_tokens), state)
        last = x[:, -TOKENS_PER_FRAME:]
        return window_tokens[:, -1] + self.head(self.norm(last))

    def start(self, x):
        """Encode the observed frames and build the window and the memory."""
        tokens = self.codec.encode(x)
        state = self.init_state(tokens[:, :-self.window])
        return tokens[:, -self.window:], state

    def step(self, window_tokens, state):
        """Predict one frame and slide the window. Returns (frame, window, state)."""
        next_tokens = self.predict_tokens(window_tokens, state)
        frame = self.codec.decode(next_tokens)                     # (B, 1, 64, 64)
        new_tokens = self.codec.encode(frame[:, None])             # (B, 1, 16, d)
        state = self.update_state(state, window_tokens[:, :1])
        window_tokens = torch.cat([window_tokens[:, 1:], new_tokens], dim=1)
        return frame, window_tokens, state

    def forward(self, x):
        window_tokens, state = self.start(x)
        frames = []
        for _ in range(self.target_len):
            frame, window_tokens, state = self.step(window_tokens, state)
            frames.append(frame)
        return torch.stack(frames, dim=1)


def state_size_bytes(state) -> int:
    """Memory kept outside the window (0 for the plain window model)."""
    if state is None:
        return 0
    if torch.is_tensor(state):
        return state.numel() * state.element_size()
    if isinstance(state, dict):
        return sum(state_size_bytes(v) for v in state.values())
    if isinstance(state, (list, tuple)):
        return sum(state_size_bytes(v) for v in state)
    return 0
