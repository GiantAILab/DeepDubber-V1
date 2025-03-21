import argparse
import os
import string
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import torch
from evaluate import load
from pymcd.mcd import Calculate_MCD
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, Wav2Vec2FeatureExtractor, WavLMForXVector, pipeline


def convert_numbers_to_words(text):
    """Convert single digits in text to words with spaces"""
    number_word_map = {
        "0": "zero",
        "1": "one",
        "2": "two",
        "3": "three",
        "4": "four",
        "5": "five",
        "6": "six",
        "7": "seven",
        "8": "eight",
        "9": "nine",
    }

    words = text.split()
    converted_words = []

    for word in words:
        # Check if the word contains both letters and numbers (like 'j4')
        if any(c.isdigit() for c in word) and any(c.isalpha() for c in word):
            # Split the word into parts and convert digits
            new_word = ""
            for c in word:
                if c.isdigit():
                    new_word += " " + number_word_map[c]
                else:
                    new_word += c
            converted_words.append(new_word)
        # Check if the word is a single digit
        elif word.isdigit() and len(word) == 1:
            converted_words.append(number_word_map[word])
        else:
            converted_words.append(word)

    return " ".join(converted_words)


def clean_text(text):
    text = convert_numbers_to_words(text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()
    return text


def wer_pipe(gen_dir: str, target_dir: str, model_id="openai/whisper-large-v3-turbo"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
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

    gen_list = list(Path(gen_dir).glob("*.wav"))
    for line in tqdm(gen_list, desc="Processing audio files"):
        wav = line
        if not wav.exists():
            continue

        text = pipe(librosa.load(wav, sr=16000)[0], generate_kwargs={"language": "english"})["text"]
        with open(wav.with_suffix(".asrtxt"), "w") as fw:
            fw.write(text)

    wer_metric = load("wer")

    val_list = list(Path(target_dir).glob("*.txt"))

    wer = []
    for txt in tqdm(val_list, desc="Calculating WER"):
        try:
            # Since the original text is automatically transcribed and has not been manually verified, all texts will be cleaned here.

            target_text = " ".join(set(txt.read_text().splitlines()))
            target_text = clean_text(target_text)

            gen_text = " ".join(Path(os.path.join(gen_dir, txt.with_suffix(".asrtxt").name)).read_text().splitlines())
            gen_text = clean_text(gen_text)

            if target_text == "" or gen_text == "":
                continue

            wer_ = wer_metric.compute(references=[target_text], predictions=[gen_text])

        except Exception as e:
            print("Error in wer calculation: ", e)
            continue

        wer.append(wer_)

    return np.mean(wer)


def spk_sim_pipe(gen_dir, target_dir):
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-sv")
    model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-sv").cuda()

    cosine_sim = torch.nn.CosineSimilarity(dim=-1)

    val_list = list(Path(target_dir).glob("*.wav"))

    scos = []

    for target_wav in tqdm(val_list, desc="Calculating speaker similarity"):
        target = librosa.load(target_wav, sr=16000)[0]
        gen = librosa.load(os.path.join(gen_dir, target_wav.name), sr=16000)[0]

        try:
            input1 = feature_extractor(gen, return_tensors="pt", sampling_rate=16000).to("cuda")
            embeddings1 = model(**input1).embeddings

            input2 = feature_extractor(target, return_tensors="pt", sampling_rate=16000).to("cuda")
            embeddings2 = model(**input2).embeddings

            similarity = cosine_sim(embeddings1[0], embeddings2[0])

        except Exception as e:
            print(f"Error in {target_wav}, {e}")
            continue

        scos.append(similarity.detach().cpu().numpy())

    return np.mean(scos)


def calculate_mcd_for_wav(target_wav, gen_dir, mcd_toolbox_dtw, mcd_toolbox_dtw_sl):
    _mcd_dtw = mcd_toolbox_dtw.calculate_mcd(target_wav, os.path.join(gen_dir, target_wav.name))
    _mcd_dtw_sl = mcd_toolbox_dtw_sl.calculate_mcd(target_wav, os.path.join(gen_dir, target_wav.name))
    return _mcd_dtw, _mcd_dtw_sl


def mcd_pipe(gen_dir, target_dir, num_processes=16):
    mcd_toolbox_dtw = Calculate_MCD(MCD_mode="dtw")
    mcd_toolbox_dtw_sl = Calculate_MCD(MCD_mode="dtw_sl")

    val_list = list(Path(target_dir).glob("*.wav"))

    mcd_dtw = []
    mcd_dtw_sl = []

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [
            executor.submit(calculate_mcd_for_wav, target_wav, gen_dir, mcd_toolbox_dtw, mcd_toolbox_dtw_sl)
            for target_wav in val_list
        ]
        for future in tqdm(futures, desc="Calculating MCD"):
            _mcd_dtw, _mcd_dtw_sl = future.result()
            mcd_dtw.append(_mcd_dtw)
            mcd_dtw_sl.append(_mcd_dtw_sl)

    return np.mean(mcd_dtw), np.mean(mcd_dtw_sl)


def run_all_metrics(gen_dir, target_dir, whisper_model="openai/whisper-large-v3-turbo"):
    """Run all evaluation metrics and return results"""
    results = {}

    print("Running WER evaluation...")
    results["wer"] = wer_pipe(gen_dir, target_dir, model_id=whisper_model)

    print("Running speaker similarity evaluation...")
    results["speaker_similarity"] = spk_sim_pipe(gen_dir, target_dir)

    print("Running MCD evaluation...")
    mcd_dtw, mcd_dtw_sl = mcd_pipe(gen_dir, target_dir)
    results["mcd_dtw"] = mcd_dtw
    results["mcd_dtw_sl"] = mcd_dtw_sl

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio evaluation metrics")
    parser.add_argument("--gen_dir", type=str, required=True, help="Directory containing generated audio files")
    parser.add_argument("--target_dir", type=str, required=True, help="Directory containing target audio files")
    parser.add_argument(
        "--metric",
        type=str,
        default="all",
        choices=["wer", "spk_sim", "mcd", "all"],
        help="Evaluation metric to use",
    )
    parser.add_argument(
        "--whisper_model",
        type=str,
        default="openai/whisper-large-v3-turbo",
        help="Whisper model to use for WER evaluation",
    )
    # python eval.py --gen_dir path/to/generated --target_dir path/to/target
    # keep the name of gen_wav and target_wav the same
    args = parser.parse_args()

    gen_dir = args.gen_dir
    target_dir = args.target_dir

    if not os.path.exists(gen_dir):
        raise ValueError(f"Generated audio directory does not exist: {gen_dir}")
    if not os.path.exists(target_dir):
        raise ValueError(f"Target audio directory does not exist: {target_dir}")

    if args.metric == "all":
        results = run_all_metrics(gen_dir, target_dir, args.whisper_model)
        print("\nEvaluation Results:")
        print(f"WER: {results['wer']:.4f}")
        print(f"Speaker Similarity: {results['speaker_similarity']:.4f}")
        print(f"MCD (DTW): {results['mcd_dtw']:.4f}")
        print(f"MCD (DTW-SL): {results['mcd_dtw_sl']:.4f}")

    elif args.metric == "wer":
        wer = wer_pipe(gen_dir, target_dir, model_id=args.whisper_model)
        print(f"WER: {wer:.4f}")

    elif args.metric == "spk_sim":
        spk_sim = spk_sim_pipe(gen_dir, target_dir)
        print(f"Speaker Similarity: {spk_sim:.4f}")

    elif args.metric == "mcd":
        mcd_dtw, mcd_dtw_sl = mcd_pipe(gen_dir, target_dir)
        print(f"MCD (DTW): {mcd_dtw:.4f}")
        print(f"MCD (DTW-SL): {mcd_dtw_sl:.4f}")
