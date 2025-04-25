<div align="center">
<p align="center">
  <h2>DeepDubber-V1</h2>
  <a href="https://arxiv.org/abs/xxxx.xxxx">Paper</a> | <a href="https://woka-0a.github.io/DeepDubber-V1/">Webpage</a> | <a href="https://huggingface.co/spaces/woak-oa/Deepdubber-V1">Huggingface Demo</a> | <a href="https://huggingface.co/woak-oa/DeepDubber-V1/tree/main">Models</a> | <a href="https://colab.research.google.com/drive/1IiajkCoXkmpPX1Ajt59drZfu97m5Xw6A?usp=sharing">Colab Demo</a> | <a href="https://replicate.com/woka-0a/deepdubber-v1">Replicate Demo</a> 
</p>
</div>

## [DeepDubber-V1: Towards High Quality and Dialogue, Narration, Monologue Adaptive Movie Dubbing Via Multi-Modal Chain-of-Thoughts Reasoning Guidance](https://woka-0a.github.io/DeepDubber-V1/)

## Results

Objective evaluation of the initial reasoning setting. 
For speech generation setting, we use the target speaker’s speech as voice prompt if the predict scene type is correct and use random speaker’s speech as voice prompt if the predict scene type is not correct.

| Models Name  | MMLMs based    | Ave.Acc(%) ↑ | Ave.Recall(%) ↑ | A.Recall(%) ↑ | B.Recall(%) ↑ | C.Recall(%) ↑ | SPK-SIM(%) ↑ | WER(%) ↓ | MCD ↓ | MCD-SL ↓ |
|--------------|----------------|--------------|-----------------|---------------|---------------|---------------|---------------|-----------|--------|----------|
| **Qwen** | MMLM-1B  | 84.09        | 82.97           | 86.50         | 68.40         | 94.00         | 83.17         | 23.60     | 8.59   | 8.60     |
|              | MMLM-4B  | 81.73        | 80.98           | 83.33         | 75.20         | 84.40         | 83.34         | 23.41     | 8.53   | 8.53     |
| **InternLM** | MMLM-2B  | 84.18        | 81.23           | 90.50         | 59.20         | 94.00         | 82.97         | 23.20     | 8.54   | 8.54     |
|              | MMLM-8B  | 86.00        | 85.84           | 86.33         | 73.20         | 98.00         | 83.42 (+30.28%) | 23.20 (+55.70%)    | 8.54 (+0.93%)   | 8.54 (+3.94%)    |
| **Dubbing Models** | HPMDubbing | -            | -               | -             | -             | -             | 61.06         | 199.40    | 8.82   | 11.88    |
|              | Speaker2Dub | -            | -               | -             | -             | -             | 61.73         | 84.42     | 8.75   | 10.78    |
|              | StyleDubber | -            | -               | -             | -             | -             | 64.03         | 52.69     | 8.62   | 8.89     |



Videos from V2CAnimation:

https://github.com/user-attachments/assets/5d78401e-efc6-4034-b66b-047aad129338

Videos from Proposed CoTMovieDubbing Dataset:

https://github.com/user-attachments/assets/58ab9155-c088-44f8-be5e-1967e01a94bc

Videos from Grid:

https://github.com/user-attachments/assets/360893bd-6e06-470f-b6f2-0ef723cb8dc7

<!-- For more results, visit https://xxxxx.com/DeepDubber-V1/video_main.html. -->

## Environment

Our python version is 3.9 and cuda version 11.8. Both training and inference are implemented with PyTorch on NVIDIA A800 GPUs.

### Prerequisites

**1. Install prerequisite if not yet met:**

```bash
conda create -n deepdubber python=3.9 && conda activate deepdubber
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install -r requirements.txt
```

**2. Clone our repository and init submodules:**

```bash
git clone https://github.com/woka-0a/DeepDubber-V1.git

cd DeepDubber-V1/
git submodule update --init --recursive
```

**Pretrained models:**

The models are available at [huggingface.co](https://huggingface.co/woak-oa/DeepDubber-V1/tree/main)

## Inference & Evaluation

```bash
# set model path, test file path, and video directory
# test_file: see datasets/CoTMovieDubbing/filelist
# video_dir: see datasets/CoTMovieDubbing -- Data Tree

# run MMLM inference & eval
# generate xxx_res.csv file for moviedubber inference
python src/internvl/eval.py --test_file /path/to/test_file.lst --video_dir /path/to/video_dir

# MovieDubber inference
python src/moviedubber/infer_with_mmlm_result.py --input_list mmlm_res.csv --ref_spk_list datasets/CoTMovieDubbing/filelist/cot_spk_for_speech_gen.lst

# MovieDubber eval
python src/moviedubber/eval.py --gen_dir generated --target_dir target
```

## Training Datasets

DeepDubber-V1 was trained on several datasets, including [V2CAnimation](https://github.com/chenqi008/V2C), [Grid](https://paperswithcode.com/dataset/grid), and proposed [CoTMovieDubbing](https://github.com/woka-0a/DeepDubber-V1/tree/main/datasets/CoTMovieDubbing). These datasets are subject to specific licenses, which can be accessed on their respective websites. We do not guarantee that the pre-trained models are suitable for commercial use. Please use them at your own risk.

## Update Logs

- 2025-03-30: Released pre-trained models on proposed dataset and evaluation script.

## Acknowledgement and Relevant Repositories

Many thanks to:

- [F5-TTS](https://github.com/SWivid/F5-TTS) for speech synthesis backbone.
- [InternVL](https://github.com/OpenGVLab/InternVL) for MMLM Backbone.
- [BigVGAN](https://github.com/NVIDIA/BigVGAN) for high-quality speech generation.
- [V2C](https://github.com/chenqi008/V2C) for animated movie benchmark.
- [SyncNet](https://github.com/joonson/syncnet_python) for LSE-C/D evaluation.
- [Wav2Vec2-Emotion](https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim) for emotion recognition in EMO-SIM evaluation.
- [WavLM-SV](https://huggingface.co/microsoft/wavlm-base-plus-sv) for speech recognition in SPK-SIM evaluation.
- [Whisper](https://huggingface.co/openai/whisper-large-v3-turbo) for speech recognition in WER evaluation.

<!-- ## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=woka-0a/DeepDubber-V1&type=Date)](https://www.star-history.com/#woka-0a/DeepDubber-V1&Date) -->
