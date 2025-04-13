# A unified script for inference process
# Make adjustments inside functions, and consider both gradio and cli scripts if need to change func output format

import subprocess as sp

import cv2
import torch

from src.moviedubber.model import CFM
from src.moviedubber.model.utils import get_tokenizer


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# -----------------------------------------

target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
win_length = 1024
n_fft = 1024
mel_spec_type = "bigvgan"
target_rms = 0.1
cross_fade_duration = 0.15
ode_method = "euler"
nfe_step = 32
speed = 1.0
fix_duration = None


# load vocoder
def load_vocoder(local_path, device=device):
    from src.third_party.BigVGAN import bigvgan

    vocoder = bigvgan.BigVGAN.from_pretrained(local_path, use_cuda_kernel=False)

    vocoder.remove_weight_norm()
    vocoder = vocoder.eval().to(device)
    return vocoder


# load model checkpoint for inference


def load_checkpoint(model, ckpt_path, device: str, dtype=torch.float32):
    model = model.to(dtype)

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    del checkpoint
    torch.cuda.empty_cache()

    return model.to(device)


# load model for inference


def load_model(
    model_cls,
    model_cfg,
    ckpt_path,
    controlnet=None,
    mel_spec_type=mel_spec_type,
    vocab_file="",
    ode_method=ode_method,
    device=device,
):
    tokenizer = "custom"

    print("\nvocab : ", vocab_file)
    print("token : ", tokenizer)
    print("model : ", ckpt_path, "\n")

    vocab_char_map, vocab_size = get_tokenizer(vocab_file, tokenizer)

    if controlnet is not None:
        controlnet = controlnet(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels)

    model = CFM(
        transformer=model_cls(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
        mel_spec_kwargs=dict(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        ),
        odeint_kwargs=dict(
            method=ode_method,
        ),
        vocab_char_map=vocab_char_map,
        controlnet=controlnet,
    ).to(device)

    model = load_checkpoint(model, ckpt_path, device)

    return model


def merge_video_audio(video_path, audio_path, output_path, start_time, duration):
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_time),
        "-t",
        str(duration),
        "-i",
        video_path,
        "-i",
        audio_path,
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-shortest",
        "-strict",
        "experimental",
        output_path,
    ]

    try:
        sp.run(command, check=True, stdout=sp.DEVNULL, stderr=sp.DEVNULL, stdin=sp.DEVNULL)
        print(f"Successfully merged audio and video into {output_path}")
        return output_path
    except sp.CalledProcessError as e:
        print(f"Error merging audio and video: {e}")
        return None


def get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fps = cap.get(cv2.CAP_PROP_FPS)

    duration = total_frames / fps
    return duration
