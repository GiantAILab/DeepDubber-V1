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
import torch.nn.functional as F
from einops import pack
from torch import nn
from x_transformers.x_transformers import RotaryEmbedding

from moviedubber.model.modules import (
    AdaLayerNormZero_Final,
    ConvNeXtV2Block,
    ConvPositionEmbedding,
    DiTBlock,
    TimestepEmbedding,
    get_pos_embed_indices,
    precompute_freqs_cis,
)


# Text embedding


class TextEmbedding(nn.Module):
    def __init__(self, text_num_embeds, text_dim, conv_layers=0, conv_mult=2):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)  # use 0 as filler token

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 4096  # ~44s of 24khz audio
            self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)
            self.text_blocks = nn.Sequential(
                *[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)]
            )
        else:
            self.extra_modeling = False

    def forward(self, text: int["b nt"], seq_len, drop_text=False):  # noqa: F722
        text = text + 1  # use 0 as filler token. preprocess of batch pad -1, see list_str_to_idx()
        text = text[:, :seq_len]  # curtail if character tokens are more than the mel spec tokens
        batch, text_len = text.shape[0], text.shape[1]
        text = F.pad(text, (0, seq_len - text_len), value=0)

        for idx, _drop in enumerate(drop_text):  # cfg for text
            if _drop:
                text[idx] = torch.zeros_like(text[idx])

        text = self.text_embed(text)  # b n -> b n d

        # possible extra modeling
        if self.extra_modeling:
            # sinus pos emb
            batch_start = torch.zeros((batch,), dtype=torch.long)
            pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.precompute_max_pos)
            text_pos_embed = self.freqs_cis[pos_idx]
            text = text + text_pos_embed

            # convnextv2 blocks
            text = self.text_blocks(text)

        return text


# noised input audio and context mixing embedding


class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2 + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(self, x: float["b n d"], cond: float["b n d"], text_embed: float["b n d"], drop_audio_cond=False):  # noqa: F722
        for idx, _drop in enumerate(drop_audio_cond):  # cfg for cond audio
            if _drop:
                cond[idx] = torch.zeros_like(cond[idx])

        x = self.proj(torch.cat((x, cond, text_embed), dim=-1))
        x = self.conv_pos_embed(x) + x
        return x


class InputEmbeddingO(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(mel_dim + 512 + text_dim + 192 + 32, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(
        self,
        x: float["b n d"],  # noqa: F722
        text_emb: float["b n d"],  # noqa: F722
        video_emb: float["b n d"],  # noqa: F722
        spk_emb: float["b n d"],  # noqa: F722
        caption_emb: float["b n d"],  # noqa: F722
    ):
        x = self.proj(torch.cat((x, text_emb, video_emb, spk_emb, caption_emb), dim=-1))
        x = self.conv_pos_embed(x) + x
        return x


# Transformer backbone using DiT blocks


class DiT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=100,
        text_num_embeds=256,
        text_dim=None,
        conv_layers=0,
        long_skip_connection=False,
    ):
        super().__init__()

        self.time_embed = TimestepEmbedding(dim)
        if text_dim is None:
            text_dim = mel_dim
        self.text_embed = TextEmbedding(text_num_embeds, text_dim, conv_layers=conv_layers)
        self.input_embed = InputEmbedding(mel_dim, text_dim, dim)

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        self.transformer_blocks = nn.ModuleList(
            [DiTBlock(dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult, dropout=dropout) for _ in range(depth)]
        )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None

        self.norm_out = AdaLayerNormZero_Final(dim)  # final modulation
        self.proj_out = nn.Linear(dim, mel_dim)

    def forward(
        self,
        x: float["b n d"],  # nosied input audio  # noqa: F722
        cond: float["b n d"],  # masked cond audio  # noqa: F722
        text: int["b nt"],  # text  # noqa: F722
        time: float["b"] | float[""],  # time step  # noqa: F821 F722
        drop_audio_cond,  # cfg for cond audio
        drop_text,  # cfg for text
        mask: bool["b n"] | None = None,  # noqa: F722
        controlnet_embeds: float["b n d"] | None = None,  # noqa: F722
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        # t: conditioning time, c: context (text + masked cond audio), x: noised input audio
        t = self.time_embed(time)
        text_embed = self.text_embed(text, seq_len, drop_text=drop_text)
        x = self.input_embed(x, cond, text_embed, drop_audio_cond=drop_audio_cond)

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x

        for i, block in enumerate(self.transformer_blocks):
            x += controlnet_embeds[i]

            x = block(x, t, mask=mask, rope=rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x)

        return output


class ControlNetDiT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=100,
        text_num_embeds=256,
        text_dim=None,
        conv_layers=0,
        long_skip_connection=False,
        checkpoint_activations=False,
        duration_predictor=None,
    ):
        super().__init__()

        if text_dim is None:
            text_dim = mel_dim

        self.time_embed = TimestepEmbedding(dim)

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth / 2

        self.transformer_blocks1 = nn.ModuleList(
            [
                DiTBlock(dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult, dropout=dropout)
                for _ in range(self.depth)
            ]
        )

        self.text_embed = TextEmbedding(text_num_embeds, text_dim, conv_layers=conv_layers)

        self.input_embed = InputEmbeddingO(mel_dim, text_dim, dim)

        self.spk_embed_affine_layer = torch.nn.Linear(192, 192)
        self.clip_embed_affine_layer = torch.nn.Linear(768, 512)
        self.caption_embed_affine_layer = torch.nn.Linear(512, 32)

        self.zero_linear = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(12)])
        for zero_linear in self.zero_linear:
            nn.init.zeros_(zero_linear.weight)

        self.duration_predictor = duration_predictor

    def ckpt_wrapper(self, module):
        # https://github.com/chuanyangjin/fast-DiT/blob/main/models.py
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs

        return ckpt_forward

    def forward(
        self,
        x: float["b n d"],  # nosied input audio  # noqa: F722
        text: int["b nt"],  # text  # noqa: F722
        clip: float["b n d"],  # video clip # noqa: F722
        spk_emb: float["b d"],  # speaker embedding  # noqa: F722
        time: float["b"] | float[""],  # time step  # noqa: F821 F722
        caption: float["b nt"] | None = None,  # caption  # noqa: F722
        mask: bool["b n"] | None = None,  # noqa: F722
        lens: int["b"] | None = None,  # noqa: F722, F821
        return_dur: bool = False,  # return duration prediction
    ):
        batch, seq_len = x.shape[0], x.shape[1]

        if time.ndim == 0:
            time = time.repeat(batch)

        t = self.time_embed(time)

        clip_emb = F.normalize(clip, dim=-1)
        clip_emb = self.clip_embed_affine_layer(clip)

        spk_emb = F.normalize(spk_emb, dim=-1)
        spk_emb = self.spk_embed_affine_layer(spk_emb)
        spk_emb = torch.repeat_interleave(spk_emb, seq_len, dim=1)

        if caption is None:
            caption = torch.zeros(1, seq_len, 512).to(device=x.device)

        caption_emb = F.normalize(caption, dim=-1)
        caption_emb = self.caption_embed_affine_layer(caption_emb)

        text_embed = self.text_embed(text, seq_len, drop_text=[False])

        x = self.input_embed(x, text_embed, clip_emb, spk_emb, caption_emb)

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        info = []
        for i, block in enumerate(self.transformer_blocks1):
            x = block(x, t, mask=mask, rope=rope)  # 'b n 1024'

            info.append(x)

        out_info = []
        for i, linear in enumerate(self.zero_linear):
            h = linear(info[i])
            out_info.append(h)

        if return_dur and self.duration_predictor is not None:
            dur_loss = self.duration_predictor(x=x, text=clip_emb, lens=lens)

            return out_info, dur_loss

        else:
            return out_info
