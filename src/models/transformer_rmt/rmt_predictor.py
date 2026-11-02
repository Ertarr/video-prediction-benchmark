"""
Window Transformer + recurrent memory tokens (after the Recurrent Memory
Transformer, Bulatov et al., NeurIPS 2022).

A few learned memory tokens are put in front of the window tokens. The
frames older than the window are read in segments of T frames: the
memory tokens and the segment's tokens go through the Transformer blocks,
and the memory tokens that come out are the memory for the next segment.
When predicting, the window attends to the memory tokens like to any
other token. The memory has a fixed size, so the cost per predicted
frame doesn't grow with the video. The gradient goes through all the
segments (backpropagation through time). Same tokens for reading and
writing (encoder-style variant).

memory_frames (test-time control): the memory only reads the last
memory_frames frames before the window.
"""
import torch
import torch.nn as nn

from ..transformer_window.window_predictor import WindowTransformer


class RMTTransformer(WindowTransformer):
    def __init__(self, *args, n_mem: int = 16, memory_frames: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_mem = n_mem
        self.memory_frames = memory_frames        # None = read everything
        self.mem_init = nn.Parameter(torch.randn(n_mem, self.d_model) * 0.02)
        self.mem_pos = nn.Parameter(torch.randn(n_mem, self.d_model) * 0.02)

    def with_memory(self, mem, x):
        """Run the blocks on [memory tokens; x], return (new memory, x)."""
        h = torch.cat([mem + self.mem_pos, x], dim=1)
        for block in self.blocks:
            h = block(h)
        return h[:, :self.n_mem], h[:, self.n_mem:]

    def write(self, mem, segment):
        """Read one segment of frames (B, S, 16, d) into the memory."""
        new_mem, _ = self.with_memory(mem, self.embed(segment))
        return new_mem

    def init_state(self, history):
        if self.memory_frames is not None:
            history = history[:, max(0, history.shape[1] - self.memory_frames):]
        B, L = history.shape[:2]
        mem = self.mem_init.expand(B, -1, -1)
        # segments of T frames, the first one shorter if L is not a multiple of T
        start = L % self.window
        if start > 0:
            mem = self.write(mem, history[:, :start])
        for i in range(start, L, self.window):
            mem = self.write(mem, history[:, i:i + self.window])
        return {"mem": mem, "pending": []}

    def run_blocks(self, x, state):
        _, x = self.with_memory(state["mem"], x)
        return x

    def update_state(self, state, leaving):
        # frames that left the window are written once a full segment is ready
        pending = state["pending"] + [leaving]
        if len(pending) < self.window:
            return {"mem": state["mem"], "pending": pending}
        return {"mem": self.write(state["mem"], torch.cat(pending, dim=1)), "pending": []}
