# Prediction interface for Cog ⚙️
# https://cog.run/python

import logging
import os
import os.path as osp
import sys
from uuid import uuid4

import cog
import numpy as np
import onnxruntime
import soundfile
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from cog import BasePredictor, Input, Path
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf
from transformers import AutoTokenizer, T5EncoderModel


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
    datefmt="%m/%d/%Y %H:%M:%S",
)

sys.path.append("src/third_party/BigVGAN")

from src.moviedubber.infer.utils_infer import (
    get_video_duration,
    load_model,
    load_vocoder,
    merge_video_audio,
    nfe_step,
)
from src.moviedubber.infer.video_preprocess import VideoFeatureExtractor
from src.moviedubber.model import ControlNetDiT, DiT


device = "cuda" if torch.cuda.is_available() else "cpu"


def get_spk_emb(audio_path, ort_session):
    audio, sample_rate = torchaudio.load(str(audio_path))
    if sample_rate != 16000:
        audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)
    feat = kaldi.fbank(audio, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    embedding = (
        ort_session.run(None, {ort_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[0]
        .flatten()
        .tolist()
    )
    return embedding


def load_models(device):
    repo_local_path = "./DeepDubber-V1"

    ckpt_file = os.path.join(repo_local_path, "mmdubber.pt")
    vocab_file = os.path.join(repo_local_path, "vocab.txt")
    campplus_path = os.path.join(repo_local_path, "campplus.onnx")

    model_cfg = "src/moviedubber/configs/basemodel.yaml"
    model_cls = DiT
    model_cfg = OmegaConf.load(model_cfg).model.arch
    controlnet = ControlNetDiT

    logging.info(f"Loading model from {ckpt_file}")
    model = load_model(
        model_cls,
        model_cfg,
        ckpt_file,
        mel_spec_type="bigvgan",
        vocab_file=vocab_file,
        controlnet=controlnet,
        device=device,
    )

    logging.info("Loading vocoder...")
    vocoder = load_vocoder(local_path="./bigvgan_v2_24khz_100band_256x", device=device)
    logging.info("Vocoder loaded.")

    logging.info("Loading ONNX model...")
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    providers = ["CPUExecutionProvider"]
    ort_session = onnxruntime.InferenceSession(
        campplus_path,
        sess_options=option,
        providers=providers,
    )
    logging.info("ONNX model loaded.")
    return model, vocoder, ort_session


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")
        logging.info("Loading model...")
        self.model, self.vocoder, self.ort_session = load_models(device=device)
        logging.info("Model loaded.")

        logging.info("Loading video feature extractor...")
        self.videofeature_extractor = VideoFeatureExtractor(
            pretrained_model_name_or_path="./clip-vit-large-patch14", device=device
        )
        logging.info("Video feature extractor loaded.")

        logging.info("Loading T5 model...")
        self.tokenizer = AutoTokenizer.from_pretrained("./t5-small")
        self.t5_model = T5EncoderModel.from_pretrained("./t5-small")
        logging.info("T5 model loaded.")

    def predict(
        self,
        video: Path = Input(description="Grayscale input video"),
        subtitle: str = Input(description="Grayscale input video"),
        description: str = Input(description="Grayscale input video"),
        ref_wav: Path = Input(description="Grayscale input video", default=None),
    ) -> cog.File:
        """Run a single prediction on the model"""
        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
        # return postprocess(output)
        video = str(video)
        logging.info(f"Video path: {video}")
        v_dur = get_video_duration(video_path=video)
        duration = int(v_dur * 24000 // 256)
        v_clip = self.videofeature_extractor.extract_features(video).to(device, dtype=torch.float32)
        v_clip = F.interpolate(
            v_clip.unsqueeze(0).transpose(1, 2), size=duration, mode="linear", align_corners=False
        ).transpose(1, 2)

        desc_input_ids = self.tokenizer(description, return_tensors="pt").input_ids
        desc_outputs = self.t5_model(input_ids=desc_input_ids)
        caption_emb = desc_outputs.last_hidden_state.detach().to(device, dtype=torch.float32)
        caption_emb = F.interpolate(
            caption_emb.transpose(1, 2), size=duration, mode="linear", align_corners=False
        ).transpose(1, 2)

        cond = torch.zeros(1, duration, 100).to(device)

        if ref_wav is not None:
            spk_emb = get_spk_emb(ref_wav, self.ort_session)
        else:
            spk_emb = torch.zeros(1, 1, 192).to(device)  # do not use speaker embedding, use desc control

        with torch.inference_mode():
            generated, _ = self.model.sample(
                cond=cond,
                text=[subtitle],
                clip=v_clip,
                spk_emb=spk_emb,
                caption_emb=caption_emb,
                duration=duration,
                steps=nfe_step,
            )

            generated = generated.to(torch.float32)

            generated_mel_spec = generated.permute(0, 2, 1)
            generated_wave = self.vocoder(generated_mel_spec)
            generated_wave = generated_wave.squeeze().cpu().numpy()

        # save generated audio with tempfile
        audio_output_path = f"/tmp/generated_{Path(video).stem}.wav"
        soundfile.write(audio_output_path, generated_wave, 24000)
        logging.info(f"Generated audio saved to {audio_output_path}")

        video_output_path = f"/tmp/generated_{Path(video).stem}.mp4"
        _ = merge_video_audio(video, audio_output_path, video_output_path, 0, v_dur)

        # remove the tmp audio file
        os.remove(audio_output_path)

        return cog.File.from_path(video_output_path)
