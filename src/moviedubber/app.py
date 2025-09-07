import os
import os.path as osp
import sys
import tempfile
from uuid import uuid4

import gradio as gr
import soundfile
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, T5EncoderModel

from src.internvl.eval import load_video
from src.moviedubber.infer.utils_infer import chunk_text, get_video_duration, merge_video_audio, nfe_step
from src.moviedubber.infer.video_preprocess import VideoFeatureExtractor
from src.moviedubber.infer_with_mmlm_result import get_spk_emb, load_models
from src.moviedubber.model.utils import convert_char_to_pinyin


sys.path.insert(0, "src/third_party")
sys.path.append("src/third_party/BigVGAN")

from InternVL.internvl_chat.internvl.model.internvl_chat.modeling_internvl_chat import InternVLChatModel  # type: ignore


load_dotenv()

local_path = os.getenv("MOVIEDUBBER_LOCAL_PATH", None)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if local_path is not None and os.path.exists(local_path):
    repo_local_path = local_path
else:
    repo_local_path = snapshot_download(repo_id="woak-oa/DeepDubber-V1")
mmlm_path = osp.join(repo_local_path, "mmlm")
mmlm = InternVLChatModel.from_pretrained(
    mmlm_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=False,
)
mmlm = mmlm.eval().to(device)

tokenizer = AutoTokenizer.from_pretrained(mmlm_path, trust_remote_code=True, use_fast=False)

generation_config = dict(max_new_tokens=1024, do_sample=False)

model, vocoder, ort_session = load_models(repo_local_path, device=device)

videofeature_extractor = VideoFeatureExtractor(device=device)

t5_tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
t5_model = T5EncoderModel.from_pretrained("google-t5/t5-small")

out_dir = "./output"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


def deepdubber(video_path: str, subtitle_text: str, audio_path: str = None) -> str:
    pixel_values, num_patches_list = load_video(video_path, num_segments=8, max_num=1)
    pixel_values = pixel_values.to(torch.bfloat16).to(device)
    video_prefix = "".join([f"Frame{i + 1}: <image>\n" for i in range(len(num_patches_list))])
    question = (
        video_prefix
        + "What is the voice-over category for this video? Options: A. dialogue, B. monologue, C. narration."
        + "And given a possible gender and age of the speaker."
    )
    response = mmlm.chat(
        tokenizer,
        pixel_values,
        question,
        generation_config,
        num_patches_list=num_patches_list,
        history=None,
        return_history=False,
    )

    print(f"MMLM response: {response}")

    try:
        conclusion = response.split("<CONCLUSION>")[1].split("</CONCLUSION>")[0].strip()
    except Exception as e:
        print(f"Error: {e}, response: {response}")
        conclusion = response.strip()[0]
    if conclusion == "A":
        d_stype = "dialogue"
    elif conclusion == "B":
        d_stype = "monologue"
    elif conclusion == "C":
        d_stype = "narration"
    else:
        d_stype = ""

    gen_clip = videofeature_extractor.extract_features(video_path)
    gen_text = subtitle_text

    desc_input_ids = t5_tokenizer(d_stype, return_tensors="pt").input_ids
    desc_outputs = t5_model(input_ids=desc_input_ids)
    caption_emb = desc_outputs.last_hidden_state.detach().to(device, dtype=torch.float32)

    v_dur = get_video_duration(video_path)
    gen_audio_len = int(v_dur * 24000 // 256)

    gen_clip = gen_clip.unsqueeze(0).to(device=device, dtype=torch.float32).transpose(1, 2)
    gen_clip = F.interpolate(gen_clip, size=(gen_audio_len,), mode="linear", align_corners=False).transpose(1, 2)

    caption_emb = F.interpolate(
        caption_emb.transpose(1, 2), size=gen_audio_len, mode="linear", align_corners=False
    ).transpose(1, 2)
    if audio_path is not None:
        spk_emb = get_spk_emb(audio_path, ort_session)
        spk_emb = torch.tensor(spk_emb).to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    else:
        spk_emb = torch.zeros(1, 1, 256).to(device=device, dtype=torch.float32)

    gen_text_batches = chunk_text(gen_text, max_chars=1024)
    final_text_list = convert_char_to_pinyin(gen_text_batches)

    cond = torch.zeros(1, gen_audio_len, 100).to(device)

    with torch.inference_mode():
        generated, _ = model.sample(
            cond=cond,
            text=final_text_list,
            clip=gen_clip,
            spk_emb=spk_emb,
            duration=gen_audio_len,
            caption_emb=caption_emb,
            steps=nfe_step,
            no_ref_audio=True,
        )

        generated = generated.to(torch.float32)

        generated_mel_spec = generated.permute(0, 2, 1)
        generated_wave = vocoder(generated_mel_spec)

        generated_wave = generated_wave.squeeze().cpu().numpy()

    # using a temporary wav file to save the generated audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir="./output") as temp_wav_file:
        temp_wav_path = temp_wav_file.name
        soundfile.write(temp_wav_path, generated_wave, samplerate=24000)

    video_out_path = os.path.join(out_dir, f"dubbed_video_{uuid4().hex[:6]}.mp4")
    concated_video = merge_video_audio(
        video_path, temp_wav_path, video_out_path, 0, soundfile.info(temp_wav_path).duration
    )

    # Ensure the temporary file is deleted after use
    os.remove(temp_wav_path)

    print(f"Deepdubber completed successfully, output path: {concated_video}")
    return response, concated_video


def process_video_dubbing(
    video_path: str, subtitle_text: str, audio_path: str = None, caption_input: str = None
) -> str:
    try:
        if not os.path.exists(video_path):
            raise ValueError("Video file does not exist")

        if not subtitle_text.strip():
            raise ValueError("Subtitle text cannot be empty")

        if audio_path is None:
            audio_path = "datasets/CoTMovieDubbing/GT.wav"

        print(f"Processing video: {video_path}")

        res, output_path = deepdubber(video_path, subtitle_text, audio_path)

        return res, output_path

    except Exception as e:
        print(f"Error in process_video_dubbing: {e}")

        return None, None


def create_ui():
    with gr.Blocks(title="DeepDubber-V1") as app:
        gr.Markdown("# DeepDubber-V1\nUpload your video file and enter the subtitle you want to dub")

        with gr.Row():
            video_input = gr.Video(label="Upload video")
            subtitle_input = gr.Textbox(
                label="Enter the subtitle", placeholder="Enter the subtitle to be dubbed...", lines=5
            )
            audio_input = gr.Audio(label="Upload speech prompt (Optional)", type="filepath")
            # caption_input = gr.Textbox(label="Enter the description of Video (Optional)", lines=1)

        process_btn = gr.Button("Start Dubbing")

        with gr.Row():
            output_response = gr.Textbox(label="Response", placeholder="Response from MMLM", lines=5)
            output_video = gr.Video(label="Dubbed Video")

        # add some examples
        examples = [
            [
                "datasets/CoTMovieDubbing/demos/v01input.mp4",
                "it isn't simply a question of creating a robot who can love",
                "datasets/CoTMovieDubbing/demos/speech_prompt_01.mp3",
            ],
            [
                "datasets/CoTMovieDubbing/demos/v02input.mp4",
                "Me, I'd be happy with one who's not... fixed.",
                "datasets/CoTMovieDubbing/demos/speech_prompt_02.mp3",
            ],
            [
                "datasets/CoTMovieDubbing/demos/v03input.mp4",
                "Man, Papi. What am I gonna do?",
                "datasets/CoTMovieDubbing/demos/speech_prompt_03.mp3",
            ],
        ]

        process_btn.click(
            fn=process_video_dubbing,
            inputs=[video_input, subtitle_input, audio_input],
            outputs=[output_response, output_video],
        )

        gr.Examples(examples=examples, inputs=[video_input, subtitle_input, audio_input])

    return app


if __name__ == "__main__":
    app = create_ui()
    app.launch(allowed_paths=["./output", "./datasets"])
