import os
import tempfile

import gradio as gr
import librosa
import soundfile
import tomli
import torch
import torch.nn.functional as F
import torchaudio
from moviepy import VideoFileClip
from pydub import AudioSegment
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from src.moviedubber.infer.utils_infer import (
    cfg_strength,
    chunk_text,
    nfe_step,
    sway_sampling_coef,
)
from src.moviedubber.infer.video_preprocess import VideoFeatureExtractor
from src.moviedubber.infer_with_mmlm_result import concat_movie_with_audio, get_spk_emb, load_models
from src.moviedubber.model.utils import convert_char_to_pinyin


def load_asr_model(model_id="openai/whisper-large-v3-turbo"):
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    return pipe


device = "cpu"
config = tomli.load(open("src/moviedubber/infer/basic.toml", "rb"))


ema_model, vocoder, ort_session = load_models(config, device=device)
asr_pipe = load_asr_model()

videofeature_extractor = VideoFeatureExtractor(device=device)


def deepdubber(video_path: str, subtitle_text: str, audio_path: str = None) -> str:
    print(f"Starting deepdubber with video_path: {video_path} and subtitle_text: {subtitle_text}")
    gen_clip = videofeature_extractor.extract_features(video_path)
    gen_text = subtitle_text

    clip = VideoFileClip(video_path)
    gen_audio_len = int(clip.duration * 24000 // 256)

    gen_clip = gen_clip.unsqueeze(0).to(device=device, dtype=torch.float32).transpose(1, 2)
    gen_clip = F.interpolate(gen_clip, size=(gen_audio_len,), mode="linear", align_corners=False).transpose(1, 2)

    ref_audio_len = None
    if audio_path is not None:
        print("reference audio is not None, dubbing with reference audio")

        if audio_path.endswith(".mp3"):
            audio = AudioSegment.from_mp3(audio_path)

            wav_file = audio_path.replace(".mp3", ".wav")
            audio.export(wav_file, format="wav")
        else:
            wav_file = audio_path

        ref_text = asr_pipe(librosa.load(wav_file, sr=16000)[0], generate_kwargs={"language": "english"})["text"]
        ref_text = ref_text.replace("\n", " ").replace("\r", " ")
        print(f"Reference text: {ref_text}")

        spk_emb = get_spk_emb(wav_file, ort_session)
        spk_emb = torch.tensor(spk_emb).to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        audio_data, sr = torchaudio.load(wav_file)
        resampler = torchaudio.transforms.Resample(sr, 24000)
        if sr != 24000:
            audio_data = resampler(audio_data)

        if audio_data.shape[0] > 1:
            audio_data = torch.mean(audio_data, dim=0, keepdim=True)

        audio_data = audio_data.to(device)

        ref_audio_len = int(audio_data.shape[-1] // 256)
        ref_clip = torch.zeros((1, ref_audio_len, 768)).to(device=device)

        gen_clip = torch.cat((gen_clip, ref_clip), dim=1)

        gen_audio_len = ref_audio_len + gen_audio_len

        gen_text = ref_text + " " + gen_text

    else:
        spk_emb = torch.zeros((1, 1, 192)).to(device=device)
        audio_data = torch.zeros((1, gen_audio_len, 100)).to(device=device)

    gen_text_batches = chunk_text(gen_text, max_chars=1024)
    final_text_list = convert_char_to_pinyin(gen_text_batches)

    with torch.inference_mode():
        generated, _ = ema_model.sample(
            cond=audio_data,
            text=final_text_list,
            clip=gen_clip,
            spk_emb=spk_emb,
            duration=gen_audio_len,
            steps=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            no_ref_audio=False,
        )

        generated = generated.to(torch.float32)

        if ref_audio_len is not None:
            generated = generated[:, ref_audio_len:, :]

        generated_mel_spec = generated.permute(0, 2, 1)
        generated_wave = vocoder(generated_mel_spec)

        generated_wave = generated_wave.squeeze().cpu().numpy()

    # using a temporary wav file to save the generated audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav_file:
        temp_wav_path = temp_wav_file.name
        soundfile.write(temp_wav_path, generated_wave, samplerate=24000)

    concated_video = concat_movie_with_audio(temp_wav_path, video_path, ".")

    # Ensure the temporary file is deleted after use
    os.remove(temp_wav_path)

    print(f"Deepdubber completed successfully, output path: {concated_video}")
    return concated_video


def process_video_dubbing(video_path: str, subtitle_text: str, audio_path: str = None) -> str:
    try:
        print(f"Processing video: {video_path}")
        if not os.path.exists(video_path):
            raise ValueError("Video file does not exist")

        if not subtitle_text.strip():
            raise ValueError("Subtitle text cannot be empty")

        output_path = deepdubber(video_path, subtitle_text, audio_path)

        return output_path

    except Exception as e:
        print(f"Error in process_video_dubbing: {e}")

        return None


def create_ui():
    with gr.Blocks(title="DeepDubber-V1") as app:
        gr.Markdown("# DeepDubber-V1\nUpload your video file and enter the text you want to dub")

        with gr.Row():
            video_input = gr.Video(label="Upload video")
            audio_input = gr.Audio(label="Upload audio", type="filepath")
            subtitle_input = gr.Textbox(label="Enter the text", placeholder="Enter the text to be dubbed...", lines=5)

        process_btn = gr.Button("Start Dubbing")

        output_video = gr.Video(label="Dubbed Video")

        process_btn.click(
            fn=process_video_dubbing, inputs=[video_input, subtitle_input, audio_input], outputs=output_video
        )

    return app


if __name__ == "__main__":
    app = create_ui()
    app.launch()
