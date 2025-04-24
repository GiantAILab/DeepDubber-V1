# modified from https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/model/cfm.py

"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint

from .modules import MelSpec
from .utils import (
    default,
    exists,
    list_str_to_idx,
)


class CFM(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        odeint_kwargs: dict = dict(method="euler"),
        num_channels=None,
        mel_spec_module: nn.Module | None = None,
        mel_spec_kwargs: dict = dict(),
        vocab_char_map: dict[str:int] | None = None,
        controlnet: nn.Module | None = None,
    ):
        super().__init__()

        self.mel_spec = default(mel_spec_module, MelSpec(**mel_spec_kwargs))
        num_channels = default(num_channels, self.mel_spec.n_mel_channels)
        self.num_channels = num_channels

        self.transformer = transformer

        self.odeint_kwargs = odeint_kwargs

        self.vocab_char_map = vocab_char_map

        self.controlnet = controlnet

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def sample(
        self,
        cond,
        text,
        clip,
        duration,
        *,
        caption_emb,
        spk_emb=None,
        lens=None,
        steps=32,
        seed=None,
    ):
        self.eval()

        cond = cond.to(next(self.parameters()).dtype)

        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        text = list_str_to_idx(text, self.vocab_char_map).to(device)

        if exists(text):
            text_lens = (text != -1).sum(dim=-1)
            lens = torch.maximum(text_lens, lens)

        if isinstance(duration, int):
            duration = torch.full((batch,), duration, device=device, dtype=torch.long)

        mask = None
        step_cond = cond

        def fn(t, x):
            step_cond = cond
            controlnet_embeds = self.controlnet(x=x, text=text, clip=clip, spk_emb=spk_emb, caption=caption_emb, time=t)

            cond_pred = self.transformer(
                x=x,
                cond=step_cond,
                text=text,
                time=t,
                mask=mask,
                drop_audio_cond=[False],
                drop_text=[False],
                controlnet_embeds=controlnet_embeds,
            )

            null_pred = self.transformer(
                x=x,
                cond=step_cond,
                text=text,
                time=t,
                mask=mask,
                drop_audio_cond=[True],
                drop_text=[True],
                controlnet_embeds=None,
            )

            return null_pred + (cond_pred - null_pred) * 2

        y0 = []
        for dur in duration:
            if exists(seed):
                torch.manual_seed(seed)
            y0.append(torch.randn(dur, self.num_channels, device=self.device, dtype=step_cond.dtype))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        t_start = 0
        t = torch.linspace(t_start, 1, steps + 1, device=self.device, dtype=step_cond.dtype)

        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)

        sampled = trajectory[-1]
        out = sampled

        return out, trajectory
