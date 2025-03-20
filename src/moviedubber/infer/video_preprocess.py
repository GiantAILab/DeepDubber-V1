import argparse
import glob
import logging
import os
import os.path as osp
from pathlib import Path
from typing import Optional, Union

import cv2
import imageio
import numpy as np
import torch
import torch.multiprocessing as mp
from decord import AudioReader, VideoReader, cpu
from PIL import Image
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection


logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")


NUM_FRAMES = None  # NUM_FRAMES = 160
MAX_FRAMES = None  # MAX_FRAMES = 256
NUM_FRAMES_PER_SECOND = 10


def get_full_indices(reader: Union[VideoReader, AudioReader]) -> np.ndarray:
    if isinstance(reader, VideoReader):
        return np.linspace(0, len(reader) - 1, len(reader), dtype=int)
    elif isinstance(reader, AudioReader):
        return np.linspace(0, reader.shape[-1] - 1, reader.shape[-1], dtype=int)


def create_output_directories(output_dir):
    try:
        os.makedirs(osp.join(output_dir, "audio"), exist_ok=True)
        os.makedirs(osp.join(output_dir, "video"), exist_ok=True)
    except OSError as e:
        print(f"Error creating directories: {e}")
        raise


def frame_sample(duration, mode="uniform", num_frames=None, fps=None):
    if mode == "uniform":
        assert num_frames is not None, "Number of frames must be provided for uniform sampling."
        # NOTE: v1 version
        # Calculate the size of each segment from which a frame will be extracted
        seg_size = float(duration - 1) / num_frames

        frame_ids = []
        for i in range(num_frames):
            # Calculate the start and end indices of each segment
            start = seg_size * i
            end = seg_size * (i + 1)
            # Append the middle index of the segment to the list
            frame_ids.append((start + end) / 2)

        return np.round(np.array(frame_ids) + 1e-6).astype(int)
        # NOTE: v0 version
        # return np.linspace(0, duration-1, num_frames, dtype=int)
    elif mode == "fps":
        assert fps is not None, "FPS must be provided for FPS sampling."
        segment_len = min(fps // NUM_FRAMES_PER_SECOND, duration)
        return np.arange(segment_len // 2, duration, segment_len, dtype=int)
    else:
        raise ImportError(f"Unsupported frame sampling mode: {mode}")


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_video(video_path, processor, s=None, e=None, aspect_ratio="pad", num_frames=NUM_FRAMES):
    if isinstance(video_path, str):
        if s is not None and e is not None:
            s = s if s >= 0.0 else 0.0
            e = e if e >= 0.0 else 0.0
            if s > e:
                s, e = e, s
            elif s == e:
                e = s + 1

        # 1. Loading Video
        if os.path.isdir(video_path):
            frame_files = sorted(os.listdir(video_path))

            fps = 3
            num_frames_of_video = len(frame_files)
        elif video_path.endswith(".gif"):
            gif_reader = imageio.get_reader(video_path)

            fps = 25
            num_frames_of_video = len(gif_reader)
        else:
            try:
                vreader = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            except:  # noqa: E722
                return None

            fps = vreader.get_avg_fps()
            num_frames_of_video = len(vreader)

        # 2. Determine frame range & Calculate frame indices
        f_start = 0 if s is None else max(int(s * fps) - 1, 0)
        f_end = num_frames_of_video - 1 if e is None else min(int(e * fps) - 1, num_frames_of_video - 1)
        frame_indices = list(range(f_start, f_end + 1))

        duration = len(frame_indices)
        # 3. Sampling frame indices
        if num_frames is None:
            sampled_frame_indices = [frame_indices[i] for i in frame_sample(duration, mode="fps", fps=fps)]
        else:
            sampled_frame_indices = [
                frame_indices[i] for i in frame_sample(duration, mode="uniform", num_frames=num_frames)
            ]

        # 4. Acquire frame data
        if os.path.isdir(video_path):
            video_data = [Image.open(os.path.join(video_path, frame_files[f_idx])) for f_idx in sampled_frame_indices]
        elif video_path.endswith(".gif"):
            video_data = [
                Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB))
                for idx, frame in enumerate(gif_reader)
                if idx in sampled_frame_indices
            ]
        else:
            video_data = [Image.fromarray(frame) for frame in vreader.get_batch(sampled_frame_indices).asnumpy()]

    elif isinstance(video_path, np.ndarray):
        video_data = [Image.fromarray(f) for f in video_path]
    elif isinstance(video_path, list) and isinstance(video_path[0], np.ndarray):
        video_data = [Image.fromarray(f) for f in video_path]
    elif isinstance(video_path, list) and isinstance(video_path[0], str):
        video_data = [Image.open(f) for f in video_path]
    elif isinstance(video_path, list) and isinstance(video_path[0], Image.Image):
        video_data = video_path
    else:
        raise ValueError(f"Unsupported video path type: {type(video_path)}")

    while num_frames is not None and len(video_data) < num_frames:
        video_data.append(Image.fromarray(np.zeros((*video_data[-1].size, 3), dtype=np.uint8)))

    # MAX_FRAMES filter
    if MAX_FRAMES:
        video_data = video_data[:MAX_FRAMES]

    if aspect_ratio == "pad":
        images = [expand2square(f, tuple(int(x * 255) for x in processor.image_mean)) for f in video_data]
    else:
        images = list(video_data)
    video = processor.preprocess(images, return_tensors="pt")["pixel_values"]
    return video


class VideoFeatureExtractor:
    def __init__(
        self,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = "openai/clip-vit-large-patch14",
        device: str = "cuda",
    ):
        self.device = device

        self.processor = CLIPImageProcessor.from_pretrained(pretrained_model_name_or_path)
        self.model = CLIPVisionModelWithProjection.from_pretrained(pretrained_model_name_or_path).to(self.device).half()

    def extract_features(self, video_path):
        images = process_video(video_path, self.processor)
        if images is None:
            return None
        clip_feature = self.model(images.to(self.device).half()).image_embeds

        return clip_feature


def video_processor(item, feature_extractor, output_dir=None):
    video_path = Path(item)
    if not os.path.exists(video_path):
        return

    clip_feature = feature_extractor.extract_features(str(video_path))
    if clip_feature is None:
        return

    if output_dir is not None and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        output_path = osp.join(output_dir, f"{video_path.stem}.pt")
    else:
        output_path = video_path.with_suffix(".clip")

    torch.save(clip_feature, output_path)


def s_thread(items, id, device, output_dir):
    feature_extractor = VideoFeatureExtractor(device=device)
    for i, data in tqdm(enumerate(items), total=len(items), position=id):
        video_processor(data, feature_extractor, output_dir)


def load_tensor(file_path, map_location="cpu", weights_only=True):
    try:
        return torch.load(file_path, map_location=map_location, weights_only=weights_only)
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
    except torch.serialization.pickle.UnpicklingError:
        logging.error(f"Failed to unpickle file: {file_path}")
    except Exception as e:
        logging.error(f"An error occurred while loading {file_path}: {e}")
    return None


def post_check(directory):
    if not osp.isdir(directory):
        logging.error(f"Invalid directory: {directory}")
        return

    video_dir = osp.join(directory, "video")
    pt_files = glob.glob(f"{video_dir}/*.pt")

    for file_path in tqdm(pt_files):
        embeds = load_tensor(file_path)
        if embeds is None:
            continue

        audio_file_path = file_path.replace("video", "audio")
        audio_text_embeds = load_tensor(audio_file_path)
        if audio_text_embeds is None:
            logging.error(f"Failed to load audio file: {audio_file_path}")
            continue

        text = audio_text_embeds.get("text")
        mel = audio_text_embeds.get("mel")
        if text is None or mel is None:
            logging.error(f"Missing 'text' or 'mel' in {audio_file_path}")


def args_parse():
    args = argparse.ArgumentParser()
    args.add_argument("--data_type", "-d", type=str, default="video", help="'audio' or 'video'")
    args.add_argument("--check", action="store_true", help="post check, if any pt file was damaged")

    args.add_argument(
        "--num_threads",
        "-n",
        type=int,
        default=1,
        required=False,
        help="num_threads",
    )
    args.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="input file path",
    )
    args.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default=None,
        help="output folder path",
    )
    args.add_argument("--multi_gpu", "-m", nargs="+", type=str, default=None, required=False, help="GPU ids")

    args = args.parse_args()
    return args


if __name__ == "__main__":
    args_main = args_parse()

    if args_main.check:
        post_check(args_main.output_dir)
        exit(0)

    gpu_ids = ["cuda:0"]
    if args_main.multi_gpu is not None:
        gpu_ids = [f"cuda:{gpu}" for gpu in args_main.multi_gpu]

    output_dir = args_main.output_dir
    if output_dir is not None:
        create_output_directories(output_dir)

    rows = None
    rows = [it.strip() for it in Path(args_main.input).read_text().split("\n") if it.strip() != ""]

    chunks = np.array_split(rows, args_main.num_threads)
    chunks = [chunk.tolist() for chunk in chunks]

    processes = []
    mp.set_start_method("spawn", force=True)
    for idx, chunk in enumerate(chunks):
        device = gpu_ids[idx % len(gpu_ids)]
        p = mp.Process(target=s_thread, args=(chunk, idx, device, output_dir))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()

    # DEBUG
    # s_thread(args_main, input_dir, output_dir, chunks[0], 0, "cuda:0")

    print("process done!")
