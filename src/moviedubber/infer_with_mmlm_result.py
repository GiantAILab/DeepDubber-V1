import argparse
import os
import os.path as osp
import random
import sys
from pathlib import Path

import numpy as np
import onnxruntime
import soundfile
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf
from pydub import AudioSegment
from tqdm import tqdm


src_path = Path(osp.dirname(__file__)).parent.parent
sys.path.insert(0, str(src_path))
sys.path.append(str(src_path / "src/third_party/BigVGAN"))

from src.moviedubber.infer.utils_infer import (
    load_model,
    load_vocoder,
    merge_video_audio,
    nfe_step,
)
from src.moviedubber.infer.video_preprocess import VideoFeatureExtractor
from src.moviedubber.model import ControlNetDiT, DiT
from src.moviedubber.model.utils import convert_char_to_pinyin


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
    repo_local_path = snapshot_download(repo_id="woak-oa/DeepDubber-V1")

    ckpt_file = os.path.join(repo_local_path, "mmdubber.pt")
    vocab_file = os.path.join(repo_local_path, "vocab.txt")
    campplus_path = os.path.join(repo_local_path, "campplus.onnx")

    model_cfg = "src/moviedubber/configs/basemodel.yaml"
    model_cls = DiT
    model_cfg = OmegaConf.load(model_cfg).model.arch
    controlnet = ControlNetDiT

    vocoder = load_vocoder(local_path="nvidia/bigvgan_v2_24khz_100band_256x", device=device)

    model = load_model(
        model_cls,
        model_cfg,
        ckpt_file,
        mel_spec_type="bigvgan",
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
    return model, vocoder, ort_session


def main(device, chunk, gen_dir, target_dir, out_dir, idx):
    model, vocoder, ort_session = load_models(device=device)

    videofeature_extractor = VideoFeatureExtractor(device=device)

    for it in tqdm(chunk, total=len(chunk), position=idx, desc=f"Processing {idx}"):
        wav_path, video_path, text, ref_wav = it

        _out_path = osp.join(gen_dir, f"{Path(wav_path).stem}.wav")
        if os.path.exists(_out_path):
            continue

        with open(f"{target_dir}/{Path(wav_path).stem}.txt", "a") as f:
            f.write(text + "\n")

        if wav_path.endswith(".mp3"):
            audio = AudioSegment.from_mp3(wav_path)

            wav_file = wav_path.replace(".mp3", ".wav")
            audio.export(wav_file, format="wav")

        wav_path = Path(wav_path).with_suffix(".wav")
        if wav_path.exists() is False:
            raise FileNotFoundError(f"{wav_path} not found ")

        os.system(f"cp {wav_path} {target_dir}/")
        os.system(f"cp {video_path} {target_dir}/")

        gen_audio, sr = torchaudio.load(str(wav_path))
        resampler = torchaudio.transforms.Resample(sr, 24000)
        if sr != 24000:
            gen_audio = resampler(gen_audio)

        if gen_audio.shape[0] > 1:
            gen_audio = torch.mean(gen_audio, dim=0, keepdim=True)

        clip_path = video_path.replace(".mp4", ".clip")

        if not os.path.exists(clip_path):
            gen_clip = videofeature_extractor.extract_features(video_path)

            torch.save(gen_clip.detach().cpu(), clip_path)

        else:
            gen_clip = torch.load(clip_path, weights_only=True).to(device=device, dtype=torch.float32)

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

        final_text_list = convert_char_to_pinyin([gen_text_])

        with torch.inference_mode():
            generated, _ = model.sample(
                cond=ref_audio_.to(device),
                text=final_text_list,
                clip=gen_clip_,
                spk_emb=spk_emb,
                duration=duration,
                steps=nfe_step,
                no_ref_audio=not use_ref_audio,
            )

            generated = generated.to(torch.float32)

            if use_ref_audio:
                generated = generated[:, ref_audio_len:, :]

            generated_mel_spec = generated.permute(0, 2, 1)
            generated_wave = vocoder(generated_mel_spec)

            generated_wave = generated_wave.squeeze().cpu().numpy()

            out_path = osp.join(gen_dir, f"{wav_path.stem}.wav")
            soundfile.write(out_path, generated_wave, samplerate=24000)
            merge_video_audio(video_path, out_path, osp.join(gen_dir, f"{wav_path.stem}.mp4"), out_dir)


if __name__ == "__main__":
    import torch.multiprocessing as mp

    parser = argparse.ArgumentParser(
        prog="python3 infer-cli.py",
        description="Commandline interface for moviedubber infer with Advanced Batch Processing.",
        epilog="Specify options above to override one or more settings from config.",
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
        p = mp.Process(target=main, args=(device, chunk, gen_dir, target_dir, out_dir, idx))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()

    print("All processes finished.")
