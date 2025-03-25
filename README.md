<div align="center">
<p align="center">
  <h2>DeepDubber-V1</h2>
  <a href="https://arxiv.org/abs/xxxx.xxxx">Paper</a> | <a href="https://woka-0a.github.io/DeepDubber-V1/">Webpage</a> | <a href="https://huggingface.co/spaces/woak-oa/Deepdubber-V1">Huggingface Demo</a> | <a href="https://huggingface.co/woak-oa/DeepDubber-V1/tree/main">Models</a> 
</p>
</div>

## [DeepDubber-V1: Towards High Quality and Dialogue, Narration, Monologue Adaptive Movie Dubbing Via Multi-Modal Chain-of-Thoughts Reasoning Guidance](https://woka-0a.github.io/DeepDubber-V1/)

## Results

Videos from V2CAnimation:

https://github.com/user-attachments/assets/5d78401e-efc6-4034-b66b-047aad129338

Videos from Proposed CoTMovieDubbing Dataset:

https://github.com/user-attachments/assets/58ab9155-c088-44f8-be5e-1967e01a94bc

Videos from Grid:

https://github.com/user-attachments/assets/360893bd-6e06-470f-b6f2-0ef723cb8dc7

For more results, visit https://xxxxx.com/DeepDubber-V1/video_main.html.

## Environment

Our python version is 3.9 and cuda version 11.8. Both training and inference are implemented with PyTorch on NVIDIA A800 GPUs.

### Prerequisites

**1. Install prerequisite if not yet met:**

```bash
conda create -n deepdubber python=3.9 && conda activate deepdubber
pip install -r requirements.txt
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
# model_path: path to the model
# test_file: see datasets/CoTMovieDubbing/filelist
# video_dir: see datasets/CoTMovieDubbing -- Data Tree

# run MMLM inference & eval
# generate xxx_res.csv file for moviedubber inference
python src/internvl/eval.py --model_path /path/to/model --test_file /path/to/test_file.lst --video_dir /path/to/video_dir

# set model params in src/moviedubber/infer/basic.toml
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
