# Unlocking the Hidden Potential of CLIP in Generalizable Deepfake Detection

<img width="4968" height="88" alt="image" src="https://github.com/user-attachments/assets/61a1cd35-c4c9-41b8-87ec-e7def2ff9bae" /> 

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

Linear probing is a technique used to evaluate the quality of pretrained representations (from CLIP) by freezing the entire backbone (ViT encoder) and training only a linear classifier (a single-layer MLP like nn.Linear) on top of the extracted features ([CLS] token embedding).
This sets the baseline performance for the rest of the paper.

Test whether the pretrained features are linearly separable for the downstream classification task (e.g., real vs. fake).

<img width="1213" height="315" alt="image" src="https://github.com/user-attachments/assets/69344bad-b34f-4e76-ac4a-5f6835350e50" />
<img width="1628" height="814" alt="image" src="https://github.com/user-attachments/assets/04e27c8c-4672-42fe-ad72-9f0e05093fad" />

## Method 2  - LN Tuning Only (LayerNorn Tuning)

LayerNorm (LN) Tuning is a parameter-efficient fine-tuning (PEFT) technique where only the LayerNorm parameters inside the frozen CLIP visual encoder (ViT-L/14) are made trainable, along with the classification head. This allows the model to adapt to the task with minimal changes, while preserving the bulk of the pretrained CLIP knowledge.

Improve performance over plain linear probing while keeping the number of trainable parameters very low (∼0.034%).

<img width="1218" height="350" alt="image" src="https://github.com/user-attachments/assets/56c8155c-8a44-4c4d-8cf4-1f05eb6ae424" />
<img width="1622" height="810" alt="image" src="https://github.com/user-attachments/assets/df2c5c65-bef5-4b3e-ad99-2c585fff7d56" />

## Method 3  - LN Tuning + Norm

This method extends LN-Tuning by applying L2 normalization to the [CLS] token features (z_cls) extracted from the CLIP encoder before passing them to the linear classifier. By constraining the features to lie on the unit hypersphere, it improves class separability, stability, and generalization

This method extends LN Tuning by applying L2 normalization to the CLIP [CLS] token features before passing them to the linear classifier. This constrains the features to lie on a unit hypersphere, improving class separability and robustness.

<img width="1460" height="351" alt="image" src="https://github.com/user-attachments/assets/7a43028d-0bbc-438f-bfea-197730ea14e4" />
<img width="1632" height="816" alt="image" src="https://github.com/user-attachments/assets/7a00a4ec-e3df-4c5c-819e-c6d1784ded55" />

## Method 4  - LN Tuning + Norm + Slerp

This method builds upon LN-Tuning + L2 Normalization by introducing latent space augmentation via Spherical Linear Interpolation (SLERP).It augments the training data by interpolating between normalized feature embeddings (z_cls) of samples belonging to the same class, encouraging the model to learn more robust decision boundaries and generalize better to unseen manipulations.

<img width="6601" height="88" alt="image" src="https://github.com/user-attachments/assets/616c1e3f-2682-4cfb-a0ef-6deb341a419e" />
<img width="1632" height="816" alt="image" src="https://github.com/user-attachments/assets/d8c0bb8e-a651-407d-bdd0-6e13d97acfc5" />

# Comparing Methods Across Varying Dataset Size

<img width="923" height="461" alt="image" src="https://github.com/user-attachments/assets/217d4fde-f011-48cb-8d1d-9064adfd6c2d" />
<img width="923" height="461" alt="image" src="https://github.com/user-attachments/assets/08f02b5b-d67d-4ab9-aa9d-ebf9b54ae113" />
<img width="934" height="467" alt="image" src="https://github.com/user-attachments/assets/54f95d50-9469-4ece-9ac1-3ae00da5045d" />
<img width="932" height="466" alt="image" src="https://github.com/user-attachments/assets/b67620ad-28b6-404d-b2a7-33f036be94d7" />

# Results - On Testing Data

## AUROC Comparison Across Methods On Varying Dataset Size

<img width="949" height="569" alt="image" src="https://github.com/user-attachments/assets/829ed821-4e74-4426-94a5-e40aec5bc9e1" />
<img width="961" height="263" alt="image" src="https://github.com/user-attachments/assets/a213916d-06ed-4aba-aa73-5a54587e30fe" />

## Balanced Accuracy Comparison Across Methods On Varying Dataset Size

<img width="961" height="577" alt="image" src="https://github.com/user-attachments/assets/f1c86738-4ce4-430a-b50a-a13d8a23bdd5" />
<img width="930" height="260" alt="image" src="https://github.com/user-attachments/assets/39ba1ef3-2381-432a-a39d-be644c2b2480" />

# Cross Dataset Evaluation 

## AUROC Comparison Across Methods on Different Dataset

<img width="928" height="557" alt="image" src="https://github.com/user-attachments/assets/8bf44033-89b9-4257-ae24-4bd038f0b38b" />
<img width="936" height="266" alt="image" src="https://github.com/user-attachments/assets/2910b9d7-3af8-4df6-825b-171856e46b13" />

## Balanced Accuracy Comparison Across Methods on Different Datasets

<img width="985" height="591" alt="image" src="https://github.com/user-attachments/assets/06088102-9799-4ecb-b400-d09afac9467d" />
<img width="920" height="248" alt="image" src="https://github.com/user-attachments/assets/168b38a7-c63a-4d9d-8ca9-f6cdcb0aa2e0" />

# Observation

Highest Overall Performance: This method consistently achieves the highest balanced accuracy across most dataset sizes, peaking at ~0.85 for 100% data. This marks a substantial improvement over all prior methods, demonstrating its superior ability to extract and classify features.

Exceptional Gains for Limited Data: The most striking result is the dramatic improvement for the 25% dataset, where accuracy surges from ~0.55 (Method 2) or ~0.65 (Method 3) to ~0.76. This highlights Slerp's effectiveness in compensating for data scarcity by synthesizing meaningful latent-space variations.

For smaller datasets (25%), the training exhibits a pronounced initial lag (slow learning for first few epochs) followed by a sudden and significant acceleration in performance. This suggests Slerp's full benefits become apparent once sufficient initial feature understanding is established.

Accelerated Initial Learning for Abundant Data: Conversely, with ample data (75% and 100%), the model demonstrates extremely rapid learning from epoch 0, quickly achieving high accuracy. This indicates the potent combination of Slerp augmentation and the multi-objective loss function (Cross-Entropy + Alignment + Uniformity) efficiently guides the feature space organization.





















































