# Unlocking the Hidden Potential of CLIP in Generalizable Deepfake Detection

[![arXiv Badge](https://img.shields.io/badge/arXiv-B31B1B?logo=arxiv&logoColor=FFF)](https://arxiv.org/abs/2503.19683)
[![Hugging Face Badge](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/yermandy/deepfake-detection)

This is the official repository for the paper:

**[Unlocking the Hidden Potential of CLIP in Generalizable Deepfake Detection](https://arxiv.org/abs/2503.19683)**.

### Abstract

> This paper tackles the challenge of detecting partially manipulated facial deepfakes, which involve subtle alterations to specific facial features while retaining the overall context, posing a greater detection difficulty than fully synthetic faces. We leverage the Contrastive Language-Image Pre-training (CLIP) model, specifically its ViT-L/14 visual encoder, to develop a generalizable detection method that performs robustly across diverse datasets and unknown forgery techniques with minimal modifications to the original model. The proposed approach utilizes parameter-efficient fine-tuning (PEFT) techniques, such as LN-tuning, to adjust a small subset of the model's parameters, preserving CLIP's pre-trained knowledge and reducing overfitting. A tailored preprocessing pipeline optimizes the method for facial images, while regularization strategies, including L2 normalization and metric learning on a hyperspherical manifold, enhance generalization. Trained on the FaceForensics++ dataset and evaluated in a cross-dataset fashion on Celeb-DF-v2, DFDC, FFIW, and others, the proposed method achieves competitive detection accuracy comparable to or outperforming much more complex state-of-the-art techniques. This work highlights the efficacy of CLIP's visual encoder in facial deepfake detection and establishes a simple, powerful baseline for future research, advancing the field of generalizable deepfake detection.


## Set up environment

``` bash
conda create --name dfdet python=3.12 uv
conda activate dfdet
uv pip install -r requirements.txt
```

## Minimal inference example

**❗ Important note**: sample images are already preprocessed. To get the same results as in the paper, you need to preprocess images using DeepfakeBench [preprocessing](https://github.com/SCLBD/DeepfakeBench/blob/fb6171a8e1db2ae0f017d9f3a12be31fd9e0a3fb/preprocessing/preprocess.py) pipeline.

### Minimal dependencies (torch + transformers)

This example requires only `torch` and `transformers` to run. This is an easy-to-integrate solution. The model has been traced and saved to a [`model.torchscript`](https://huggingface.co/yermandy/deepfake-detection/tree/main) file. Run:

``` bash
python inference_torchscript.py
```

Results might be a little bit different than in **precise inference** ↓

### Precise inference (full dependencies)

Read `inference.py`, it automatically downloads the model from [huggingface](https://huggingface.co/yermandy/deepfake-detection/tree/main) and runs inference on sample images.

``` bash
python inference.py
```

## Training

### Minimal example without external data

#### Run Training

You can adjust training configuration in `get_train_config` function in `run.py` or override them with command line arguments. Command line arguments have higher priority.

Example changing configurations in `get_train_config`:

1. Set `config.wandb = True` for logging to wandb

2. Set `config.devices = [2]` for using GPU number 2

``` bash
python run.py --train
```

#### Run testing (for example, on other dataset)

``` bash
python run.py --test
```

---

### Full training

#### Prepare the dataset

To fully train the model, you need to download datasets, preprocess them, and create a file with paths to the images.

For example, if you want to work with the [FaceForensics++](https://github.com/ondyari/FaceForensics) dataset, follow these steps:

1. Download the dataset first from the [official source](https://github.com/ondyari/FaceForensics)

2. Preprocess the dataset using [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench)

3. Place images in the recommended directory structure: `datasets / <dataset_name> / <source_name> / <video_name> / <frame_name>`, see `src/dataset/deepfake.py` for more details

``` bash
datasets
└── FF
    ├── DF
    │   └── 000_003
    │       ├── 025.png
    │       └── 038.png
    ├── F2F
    │   └── 000_003
    │       ├── 019.png
    │       └── 029.png
    ├── FS
    │   └── 000_003
    │       ├── 019.png
    │       └── 029.png
    ├── NT
    │   └── 000_003
    │       ├── 019.png
    │       └── 029.png
    └── real
        └── 000
            ├── 025.png
            └── 038.png
```

4. Create files with paths to images similar to the ones in `config/datasets` directory. Get inspired by this script:

``` bash
sh scripts/prepare_FF.sh
```

#### Run training

Adjust training configuration as needed before executing the command below:

``` bash
python run.py --train
```

# Results - On Training Data across varying sizes of FF++ Dataset

## Method 1  - Linear Probing 

Linear probing is a technique to evaluate the quality of pretrained representations (like those from CLIP) by freezing the backbone (e.g., ViT in CLIP) and training only a shallow linear classifier (e.g., a single-layer MLP, nn.Linear) on top of it.

Test whether the pretrained features are linearly separable for the downstream classification task (e.g., real vs. fake).

<img width="1213" height="315" alt="image" src="https://github.com/user-attachments/assets/69344bad-b34f-4e76-ac4a-5f6835350e50" />
<img width="1628" height="814" alt="image" src="https://github.com/user-attachments/assets/04e27c8c-4672-42fe-ad72-9f0e05093fad" />

## Method 2  - LN Tuning Only (LayerNorn Tuning)

LayerNorm (LN) Tuning involves unfreezing only the LayerNorm parameters in the frozen CLIP backbone and training them, in addition to the linear classifier. It enables minimal fine-tuning while preserving most of the CLIP model's knowledge.

Improve performance over plain linear probing while keeping the number of trainable parameters very low (∼0.034%).

<img width="1218" height="350" alt="image" src="https://github.com/user-attachments/assets/56c8155c-8a44-4c4d-8cf4-1f05eb6ae424" />
<img width="1622" height="810" alt="image" src="https://github.com/user-attachments/assets/df2c5c65-bef5-4b3e-ad99-2c585fff7d56" />



















