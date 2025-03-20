<div align="center">
<p align="center">
  <h2>DeepDubber-V1</h2>
  <a href="https://arxiv.org/abs/xxxx.xxxx">Paper</a> | <a href="https://woka-0a.github.io/DeepDubber-V1/">Webpage</a> | <a href="https://huggingface.co/woak-oa/DeepDubber-V1/tree/main">Models</a> 
</p>
</div>

## [DeepDubber-V1: Towards High Quality and Dialogue, Narration, Monologue Adaptive Movie Dubbing Via Multi-Modal Chain-of-Thoughts Reasoning Guidance](https://woka-0a.github.io/DeepDubber-V1/)

<!-- [Ho Kei Cheng](https://hkchengrex.github.io/), [Masato Ishii](https://scholar.google.co.jp/citations?user=RRIO1CcAAAAJ), [Akio Hayakawa](https://scholar.google.com/citations?user=sXAjHFIAAAAJ), [Takashi Shibuya](https://scholar.google.com/citations?user=XCRO260AAAAJ), [Alexander Schwing](https://www.alexander-schwing.de/), [Yuki Mitsufuji](https://www.yukimitsufuji.com/) -->

<!-- University of xxxx, xxxx, and xxxx Corporation -->

<!-- xxxx 2025 -->

## Results

Videos from V2CAnimation:

https://github.com/user-attachments/assets/xxxx

Videos from Grid:

https://github.com/user-attachments/assets/xxxx

Videos from Proposed CoTMovieDubbing Dataset:

https://github.com/user-attachments/assets/xxxx

For more results, visit https://xxxxx.com/DeepDubber-V1/video_main.html.

## Environment

Our python version is 3.10 and cuda version 11.8. Both training and inference are implemented with PyTorch on a NVIDIA A800 GPU.

### Prerequisites

**1. Install prerequisite if not yet met:**

```bash
pip install -r requirements.txt
```

**2. Clone our repository:**

```bash
git clone https://github.com/woka-0a/DeepDubber-V1.git

cd DeepDubber-V1/src/third_party/InternVL/
git submodule update --init --recursive

cd - && cd DeepDubber-V1/src/third_party/BigvGAN/
git submodule update --init --recursive
```

**Pretrained models:**

The models are available at [huggingface.co](https://huggingface.co/woak-oa/DeepDubber-V1/tree/main)

## Inference & Evaluation

```bash
# MMLM inference & eval
python src/internvl/eval.py --model_path /path/to/model --test_file /path/to/test_file.lst --video_dir /path/to/video_dir
# MovieDubber inference
python src/moviedubber/infer_with_mmlm_result.py --input_list mmlm_res.csv --ref_spk_list datasets/CoTMovieDubbing/filelist/cot_spk_for_speech_gen.lst
# MovieDubber eval
python src/moviedubber/eval.py --gen_dir generated --target_dir target
```

## Training Datasets

DeepDubber-V1 was trained on several datasets, including [V2CAnimation](https://github.com/chenqi008/V2C), [Grid](https://paperswithcode.com/dataset/grid), and proposed [CoTMovieDubbing](https://github.com/woka-0a/DeepDubber-V1/tree/main/datasets/CoTMovieDubbing). These datasets are subject to specific licenses, which can be accessed on their respective websites. We do not guarantee that the pre-trained models are suitable for commercial use. Please use them at your own risk.

## Update Logs

- 2025-03-30: Released pre-trained models on proposed dataset and evaluation script.

<!-- ## Citation

```bibtex
@inproceedings{cheng2025taming,
  title={Taming Multimodal Joint Training for High-Quality Video-to-Audio Synthesis},
  author={Cheng, Ho Kei and Ishii, Masato and Hayakawa, Akio and Shibuya, Takashi and Schwing, Alexander and Mitsufuji, Yuki},
  booktitle={CVPR},
  year={2025}
}
``` -->

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
