import argparse
import os
import os.path as osp
import random
import sys
from pathlib import Path

import numpy as np
import onnxruntime
import soundfile
import tomli
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from moviepy import AudioFileClip, VideoFileClip
from omegaconf import OmegaConf
from pydub import AudioSegment
from tqdm import tqdm


src_path = Path(osp.dirname(__file__)).parent.parent
sys.path.insert(0, str(src_path))
sys.path.append(str(src_path / "src/third_party/BigVGAN"))

from src.moviedubber.infer.utils_infer import (
    cfg_strength,
    chunk_text,
    load_model,
    load_vocoder,
    mel_spec_type,
    nfe_step,
    sway_sampling_coef,
)
from src.moviedubber.infer.video_preprocess import VideoFeatureExtractor
from src.moviedubber.model import ControlNetDiT, DiT
from src.moviedubber.model.utils import convert_char_to_pinyin


def concat_movie_with_audio(wav, video_path, out_dir):
    if not os.path.exists(wav):
        raise FileNotFoundError(f"Audio file {wav} does not exist")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file {video_path} does not exist")

    try:
        with (
            AudioFileClip(str(wav)) as audio_clip,
            VideoFileClip(str(video_path)) as video_clip,
        ):
            duration = min(video_clip.duration, audio_clip.duration)

            video_subclip = video_clip.subclipped(0, duration)
            audio_subclip = audio_clip.subclipped(0, duration)

            final_video = video_subclip.with_audio(audio_subclip)

            output_path = wav.replace(".wav", ".mp4")

            final_video.write_videofile(
                str(output_path),
                codec="libx264",
                audio_codec="mp3",
                fps=25,
                logger=None,
                threads=1,
                temp_audiofile_path=out_dir,
            )

    except Exception as e:
        print(f"Error processing {wav} {video_path}: {str(e)}")

    return output_path


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


def load_models(config, device):
    model_cfg = config.get("model_cfg", "src/moviedubber/configs/basemodel.yaml")
    ckpt_file = config.get("ckpt_file", None)
    campplus_path = config.get("campplus_path", None)
    vocab_file = config.get("vocab_file", None)

    vocoder_local_path = config.get("vocoder_local_path", None)

    if ckpt_file is None or vocab_file is None or vocoder_local_path is None or campplus_path is None:
        raise ValueError("ckpt_file, vocab_file and vocoder_local_path must be specified")

    vocoder_name = config.get("vocoder_name", mel_spec_type)

    vocoder = load_vocoder(local_path=vocoder_local_path, device=device)

    model_cls = DiT
    model_cfg = OmegaConf.load(model_cfg).model.arch
    controlnet = ControlNetDiT

    ema_model = load_model(
        model_cls,
        model_cfg,
        ckpt_file,
        mel_spec_type=vocoder_name,
        vocab_file=vocab_file,
        controlnet=controlnet,
        device=device,
    )

    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    providers = ["CPUExecutionProvider"]
    ort_session = onnxruntime.InferenceSession(
        campplus_path,
        sess_options=option,
        providers=providers,
    )
    return ema_model, vocoder, ort_session


def main(config, device, chunk, gen_dir, target_dir, out_dir, idx):
    ema_model, vocoder, ort_session = load_models(config, device=device)

    videofeature_extractor = VideoFeatureExtractor(device=device)

    for it in tqdm(chunk, total=len(chunk), position=idx, desc=f"Processing {idx}"):
        wav, video, text, ref_wav = it

        with open(f"{target_dir}/{wav.split('/')[-1].split('.')[0]}.txt", "a") as f:
            f.write(text + "\n")

        if wav.endswith(".mp3"):
            audio = AudioSegment.from_mp3(wav)

            wav_file = wav.replace(".mp3", ".wav")
            audio.export(wav_file, format="wav")

        wav = Path(wav).with_suffix(".wav")
        if wav.exists() is False:
            continue

        os.system(f"cp {wav} {target_dir}/")

        gen_audio, sr = torchaudio.load(str(wav))
        resampler = torchaudio.transforms.Resample(sr, 24000)
        if sr != 24000:
            gen_audio = resampler(gen_audio)

        if gen_audio.shape[0] > 1:
            gen_audio = torch.mean(gen_audio, dim=0, keepdim=True)

        gen_video = video
        gen_clip_path = gen_video.replace(".mp4", ".clip")

        if not os.path.exists(gen_clip_path):
            gen_clip = videofeature_extractor.extract_features(gen_video)

            torch.save(gen_clip.detach().cpu(), gen_clip_path)

        else:
            gen_clip = torch.load(gen_clip_path, weights_only=True).to(device=device, dtype=torch.float32)

        if ref_wav == "None":
            use_ref_audio = False
            gen_text_ = text

            gen_clip_ = gen_clip

            ref_audio_ = gen_audio

            spk_emb = torch.zeros(1, 1, 192).to(device=device, dtype=torch.float32)

        else:
            use_ref_audio = True
            ref_audio = Path(ref_wav)

            spk_emb = get_spk_emb(ref_audio, ort_session)
            spk_emb = torch.tensor(spk_emb).to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            ref_text = ref_audio.with_suffix(".txt").read_text().strip()
            gen_text_ = ref_text + " " + text

            if ref_audio.exists() is False:
                raise Exception(f"ref_audio {ref_audio} not found")

            if ref_audio.suffix == ".mp3":
                audio = AudioSegment.from_mp3(ref_audio)

                wav_file = ref_audio.with_suffix(".wav")
                audio.export(wav_file, format="wav")

            ref_audio_, _ = torchaudio.load(str(ref_audio.with_suffix(".wav")))
            resampler = torchaudio.transforms.Resample(sr, 24000)
            if sr != 24000:
                ref_audio_ = resampler(ref_audio_)

            if ref_audio_.shape[0] > 1:
                ref_audio_ = torch.mean(ref_audio_, dim=0, keepdim=True)

            ref_video = ref_audio.with_suffix(".mp4")
            ref_clip_path = ref_video.with_suffix(".clip")

            if not ref_clip_path.exists():
                ref_clip = videofeature_extractor.extract_features(str(ref_video))

                torch.save(ref_clip.detach().cpu(), ref_clip_path)

            else:
                ref_clip = torch.load(ref_clip_path, weights_only=True).to(device=device, dtype=torch.float32)

            gen_clip_ = torch.cat([ref_clip, gen_clip], dim=0)

        gen_audio_len = gen_audio.shape[1] // 256

        if use_ref_audio:
            ref_audio_len = ref_audio_.shape[1] // 256
            duration = ref_audio_len + gen_audio_len
        else:
            duration = gen_audio_len

        gen_clip_ = gen_clip_.unsqueeze(0).to(device=device, dtype=torch.float32).transpose(1, 2)
        gen_clip_ = F.interpolate(gen_clip_, size=duration, mode="linear", align_corners=False).transpose(1, 2)

        gen_text_batches = chunk_text(gen_text_, max_chars=1024)
        final_text_list = convert_char_to_pinyin(gen_text_batches)

        with torch.inference_mode():
            generated, _ = ema_model.sample(
                cond=ref_audio_.to(device),
                text=final_text_list,
                clip=gen_clip_,
                spk_emb=spk_emb,
                duration=duration,
                steps=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
                no_ref_audio=not use_ref_audio,
            )

            generated = generated.to(torch.float32)

            if use_ref_audio:
                generated = generated[:, ref_audio_len:, :]

            generated_mel_spec = generated.permute(0, 2, 1)
            generated_wave = vocoder(generated_mel_spec)

            generated_wave = generated_wave.squeeze().cpu().numpy()

            out_path = osp.join(gen_dir, f"{wav.stem}.wav")
            soundfile.write(out_path, generated_wave, samplerate=24000)
            _ = concat_movie_with_audio(out_path, gen_video, out_dir)


if __name__ == "__main__":
    import torch.multiprocessing as mp

    parser = argparse.ArgumentParser(
        prog="python3 infer-cli.py",
        description="Commandline interface for moviedubber infer with Advanced Batch Processing.",
        epilog="Specify options above to override one or more settings from config.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="src/moviedubber/infer/basic.toml",
        help="The configuration file, default see infer/basic.toml",
    )
    parser.add_argument("-i", "--input_list", type=str, required=True, help="The val list file")
    parser.add_argument("-s", "--ref_spk_list", type=str, required=True, help="The spk list file")
    parser.add_argument("-o", "--out_dir", type=str, default="data/dubberout", help="The output directory")
    parser.add_argument("--gpuids", type=str, help="GPU ids to use, split by comma")
    parser.add_argument("--nums_workers", type=int, default=1, help="Number of workers for per gpu")

    args = parser.parse_args()

    out_dir = args.out_dir
    input_list = args.input_list
    gpu_ids = args.gpuids.split(",") if args.gpuids else ["0"]
    num_pre = args.nums_workers
    spk_ref_path = args.ref_spk_list

    config = tomli.load(open(args.config, "rb"))

    gen_lst = Path(input_list).read_text().splitlines()[1:]

    gen_pre_conf = []

    spk_lines = Path(spk_ref_path).read_text().splitlines()

    for idx, line in enumerate(gen_lst):
        if line.strip():
            mp4_path, is_correc, _, _ = line.split(",")

            wav_path = mp4_path.replace(".mp4", ".mp3")
            text = Path(wav_path.replace(".mp3", ".txt")).read_text().strip()

            if is_correc == "True":
                ref_wav = spk_lines[idx].split(",")[1].strip()
            else:
                ref_wav = random.choice(spk_lines).split(",")[-1].strip()  # Use random speaker for incorrect samples

            gen_pre_conf.append([wav_path, mp4_path, text, ref_wav])

    chunks = np.array_split(gen_pre_conf, len(gpu_ids) * num_pre)

    gen_dir = os.path.join(out_dir, "generated")
    target_dir = os.path.join(out_dir, "target")

    if os.path.exists(gen_dir) is False or os.path.exists(target_dir) is False:
        os.makedirs(gen_dir)
        os.makedirs(target_dir)

    mp.set_start_method("spawn", force=True)
    processes = []
    for idx, chunk in enumerate(chunks):
        device = gpu_ids[idx % len(gpu_ids)]

        device = f"cuda:{device}"
        p = mp.Process(target=main, args=(config, device, chunk, gen_dir, target_dir, out_dir, idx))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()

    print("All processes finished.")
