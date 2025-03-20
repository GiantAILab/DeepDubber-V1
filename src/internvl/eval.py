import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import AutoTokenizer


sys.path.insert(0, os.path.join(str(Path(__file__).resolve().parents[2]), "src/third_party/InternVL/internvl_chat"))
from internvl.model.internvl_chat.modeling_internvl_chat import InternVLChatModel  # type: ignore


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array(
        [int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) for idx in range(num_segments)]
    )
    return frame_indices


def load_video(
    video_path,
    bound=None,
    input_size=448,
    max_num=1,
    num_segments=32,
    cache_dir="~/.cache/expcache",
):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    cache_filename = os.path.join(
        cache_dir,
        f"{video_path.split('/')[-2]}_{os.path.basename(video_path).split('.')[0]}_bound-{bound}_input_size-{input_size}_max_num-{max_num}_num_segments-{num_segments}.pt",
    )
    if os.path.exists(cache_filename) and os.path.isfile(cache_filename):
        cache = torch.load(cache_filename, weights_only=True)
        pixel_values = cache["pixel_values"]
        num_patches_list = cache["num_patches_list"]

    else:
        pixel_values_list, num_patches_list = [], []
        transform = build_transform(input_size=input_size)
        frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)

        frame_indices = np.append(0, frame_indices)  # Add 0 at the beginning of the list
        frame_indices = np.append(frame_indices, max_frame)  # Add max_frame at the end of the list

        img_folder_name = video_path.split("/")[-2] + "_" + os.path.basename(video_path).split(".")[0]
        img_save_dir = os.path.join(cache_dir, img_folder_name)
        os.makedirs(img_save_dir, exist_ok=True)

        idx = 0
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")

            img.save(os.path.join(img_save_dir, f"frame_{frame_index}_tile_{idx}.png"))

            img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)

            pixel_values = [transform(tile) for tile in img]
            pixel_values = torch.stack(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)

            idx += 1
        pixel_values = torch.cat(pixel_values_list)

        os.makedirs(cache_dir, exist_ok=True)
        torch.save({"pixel_values": pixel_values, "num_patches_list": num_patches_list}, cache_filename)

    return pixel_values, num_patches_list


def analyze_predictions(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Calculate overall accuracy
    total_samples = len(df)
    correct_predictions = df["is_correct"].value_counts().get(True, 0)
    overall_accuracy = correct_predictions / total_samples

    # Initialize metrics for each class
    classes = ["A", "B", "C"]
    class_metrics = {}

    for cls in classes:
        # Filter for samples where target is this class
        true_class = df[df["target"] == cls]
        # Filter for samples where prediction is this class
        # pred_class = df[df["predict"] == cls]

        # Calculate TP, FP, FN
        TP = len(df[(df["target"] == cls) & (df["predict"] == cls)])
        FP = len(df[(df["target"] != cls) & (df["predict"] == cls)])
        FN = len(df[(df["target"] == cls) & (df["predict"] != cls)])

        # Calculate precision, recall, F1
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Store metrics
        class_metrics[cls] = {
            "total_samples": len(true_class),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": TP,
            "false_positives": FP,
            "false_negatives": FN,
        }

    print(f"Overall Accuracy: {overall_accuracy:.4f} ({correct_predictions}/{total_samples})")
    print()
    print("Indicators for each category:")

    for cls in classes:
        metrics = class_metrics[cls]
        print(f"  Class {cls}:")
        print(f"    Total Samples: {metrics['total_samples']}")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall: {metrics['recall']:.4f}")
        print(f"    F1 Score: {metrics['f1']:.4f}")
        print(f"    True Positives: {metrics['true_positives']}")
        print(f"    False Positives: {metrics['false_positives']}")
        print(f"    False Negatives: {metrics['false_negatives']}")

    return overall_accuracy, class_metrics


def s_thread(video_dir, model_path, device, chunk, idx, queue):
    model = InternVLChatModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
    )
    model = model.eval().to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    generation_config = dict(max_new_tokens=1024, do_sample=False)

    res = []
    for line in tqdm(chunk, position=idx, desc=f"Device {device}"):
        data = json.loads(line)

        video_path = os.path.join(video_dir, data["video"])
        ques = data["conversations"][0]["value"]

        target_ans = data["conversations"][1]["value"].split("<CONCLUSION>")[1].split("</CONCLUSION>")[0].strip()

        pixel_values, num_patches_list = load_video(video_path, num_segments=8, max_num=1)
        pixel_values = pixel_values.to(torch.bfloat16).to(device)
        video_prefix = "".join([f"Frame{i + 1}: <image>\n" for i in range(len(num_patches_list))])
        question = video_prefix + f"{ques}"
        response = model.chat(
            tokenizer,
            pixel_values,
            question,
            generation_config,
            num_patches_list=num_patches_list,
            history=None,
            return_history=False,
        )

        try:
            ans = response.split("<CONCLUSION>")[1].split("</CONCLUSION>")[0].strip()
        except Exception as e:
            print(f"Error: {e}, response: {response}")
            ans = response.strip()[0]

        is_correct = False
        if ans == target_ans:
            is_correct = True

        res.append(f"{video_path},{is_correct},{target_ans},{ans}")

    queue.put(res)


if __name__ == "__main__":
    import argparse

    import torch.multiprocessing as mp

    parser = argparse.ArgumentParser(description="eval script for mmlm")
    parser.add_argument("--model_path", type=str, help="Path to the model checkpoint.")
    parser.add_argument("--test_file", type=str, help="Path to the test file.")
    parser.add_argument("--video_dir", type=str, help="Path to the test video directory.")
    parser.add_argument("--gpuids", type=str, help="GPU ids to use.")

    # python eval.py --model_path /path/to/model --test_file /path/to/test_file --video_dir /path/to/video_dir --gpuids 0,1,2,3

    args = parser.parse_args()

    model_path = args.model_path
    test_file = args.test_file
    video_dir = args.video_dir

    gpu_ids = args.gpuids.split(",") if args.gpuids else ["0"]

    cot_test = Path(test_file).read_text().splitlines()

    chunks = np.array_split(cot_test, len(gpu_ids))

    mp.set_start_method("spawn", force=True)

    queue = mp.Queue()

    processes = []
    for idx, chunk in enumerate(chunks):
        device = gpu_ids[idx % len(gpu_ids)]
        device = f"cuda:{device}"

        p = mp.Process(target=s_thread, args=(video_dir, model_path, device, chunk, idx, queue))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()

    result = []
    for _ in range(len(chunks)):
        res = queue.get()
        result.extend(res)

    res_saved = f"{'__'.join(model_path.split('/'))}_res.csv"
    with open(res_saved, "w") as f:
        f.write("video_id,is_correct,target,predict\n")
        for res in result:
            f.write(f"{res}\n")

    accuracy, metrics = analyze_predictions(res_saved)

    print("All processes finished.\n\n")
