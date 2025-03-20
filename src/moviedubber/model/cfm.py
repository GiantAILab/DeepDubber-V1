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

from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint

from .modules import MelSpec
from .utils import (
    default,
    exists,
    lens_to_mask,
    list_str_to_idx,
    list_str_to_tensor,
)


class CFM(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        sigma=0.0,
        odeint_kwargs: dict = dict(
            method="euler"  # 'midpoint'
        ),
        num_channels=None,
        mel_spec_module: nn.Module | None = None,
        mel_spec_kwargs: dict = dict(),
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
        vocab_char_map: dict[str:int] | None = None,
        controlnet: nn.Module | None = None,
    ):
        super().__init__()

        self.frac_lengths_mask = frac_lengths_mask

        self.mel_spec = default(mel_spec_module, MelSpec(**mel_spec_kwargs))
        num_channels = default(num_channels, self.mel_spec.n_mel_channels)
        self.num_channels = num_channels

        self.transformer = transformer
        dim = transformer.dim
        self.dim = dim

        self.sigma = sigma

        self.odeint_kwargs = odeint_kwargs

        self.vocab_char_map = vocab_char_map

        self.controlnet = controlnet

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def sample(
        self,
        cond: float["b n d"] | float["b nw"],  # noqa: F722
        text: int["b nt"] | list[str],  # noqa: F722
        clip: float["b n d"],  # noqa: F722
        duration: int | int["b"],  # noqa: F821
        *,
        caption_emb: float["b n d"] | None = None,  # noqa: F722
        spk_emb: float["b n d"] | None = None,  # noqa: F722
        lens: int["b"] | None = None,  # noqa: F821
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        seed: int | None = None,
        max_duration=4096,
        vocoder: Callable[[float["b d n"]], float["b nw"]] | None = None,  # noqa: F722
        no_ref_audio=False,
        duplicate_test=False,
        t_inter=0.1,
        edit_mask=None,
    ):
        self.eval()

        if cond.ndim == 2:
            cond = self.mel_spec(cond)
            cond = cond.permute(0, 2, 1)
            assert cond.shape[-1] == self.num_channels

        cond = cond.to(next(self.parameters()).dtype)

        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        if exists(text):
            text_lens = (text != -1).sum(dim=-1)
            lens = torch.maximum(text_lens, lens)

        cond_mask = lens_to_mask(lens)
        if edit_mask is not None:
            cond_mask = cond_mask & edit_mask

        if isinstance(duration, int):
            duration = torch.full((batch,), duration, device=device, dtype=torch.long)

        # duration = torch.maximum(lens + 1, duration)

        duration = duration.clamp(max=max_duration)
        max_duration = duration.amax()

        if duplicate_test:
            test_cond = F.pad(cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0)

        cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)
        cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
        cond_mask = cond_mask.unsqueeze(-1)
        step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

        if batch > 1:
            mask = lens_to_mask(duration)
        else:
            mask = None

        if no_ref_audio:
            cond = torch.zeros_like(cond)

        def fn(t, x):
            step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

            controlnet_embeds = self.controlnet(
                x=x,
                text=text,
                clip=clip,
                spk_emb=spk_emb,
                caption=caption_emb,
                time=t,
            )

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

        if duplicate_test:
            t_start = t_inter
            y0 = (1 - t_start) * y0 + t_start * test_cond
            steps = int(steps * (1 - t_start))

        t = torch.linspace(t_start, 1, steps + 1, device=self.device, dtype=step_cond.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)

        sampled = trajectory[-1]
        out = sampled
        out = torch.where(cond_mask, cond, out)

        if exists(vocoder):
            out = out.permute(0, 2, 1)
            out = vocoder(out)

        return out, trajectory
