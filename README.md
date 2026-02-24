# D4C: Data-free Quantization for Contrastive  Language-Image Pre-training Models

## 1. Environment Settings

### 1.1 Create Environment

1. Install PyTorch.

```
conda create -n d4c python=3.10 scipy
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

2. Install OpenAI CLIP

```
cd CLIP
python setup.py install
cd ..
```

3. Install CLIP_benchmark

```
cd CLIP_benchmark
python setup.py install
cd ..
```

4. Install other requirements

```
pip install easydict
```

### 1.2 Prepare Dataset

If a valid ``--dataset_root`` is provided, CLIP_benchmark will automatically download the CIFAR-10 and CIFAR-100 datasets. For ImageNet-1K, you may either place the original ILSVRC ``.tar`` archive in the specified root, or reformat your existing ImageNet dataset using the following structure:

```
|-- ImageNet-1K
|   |-- train
|   |   |-- n01440764
|   |   |-- ...
|   |-- val
|   |   |-- n01440764
|   |   |-- ...
|   |-- meta.bin
```

### 1.3 Pre-trained Model Weights

We use the original pre-trained CLIP models provided by OpenAI for our experiments. Once a valid ``--model`` name is specified, CLIP will automatically download the corresponding pre-trained weights.

## 2. Run Experiments

### 2.1 Quick Start

Use the following command to perform D4C quantization:

```
python d4c/solver/test_quant.py \
--dataset <DATASET_NAME> \
--dataset_root <DATASET_ROOT> \
--model <MODEL_NAME> \
--q_config ./exp/<QCONFIG>.yaml \
--recon \
--dfq \
--gen_img
```

Here, ``<DATASET_NAME>`` and ``<DATASET_ROOT>`` refer to the name and directory of the dataset, ``<MODEL_NAME>`` specifies the encoder model, and ``<QCONFIG>.yaml`` is the corresponding quantization configuration file.

For example, to perform W6A6 quantization on ViT-B/32 evaluated on CIFAR-10, use the following command:
```
python d4c/solver/test_quant.py \
--dataset cifar10 \
--dataset_root ./your/dataset/root \
--model ViT-B/32 \
--q_config ./exp/config66.yaml \
--recon \
--dfq \
--gen_img
```

### 2.2 Related Argument Description

A detailed description of the arguments is provided below for your reference:

| Argument       | Description                                                          | Options                                                                             |
|----------------|----------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| dataset        | Dataset for evaluation.                                              | cifar10, cifar100, imagenet1k                                                       |
| dataset_root   | Directory for dataset download.                                      | NA                                                                                  |
| model          | CLIP model (image encoder).                                          | RN50, RN101, RN50x4, RN50x16, RN50x64, ViT-B/32, ViT-B/16, ViT-L/14, ViT-L/14@336px |
| fp_model       | Run FP model without quantization.                                   | NA                                                                                  |
| q_config       | Quantization configuration file.                                     | NA                                                                                  |
| recon          | Apply PTQ using reconstruction.                                      | NA                                                                                  |
| custom_file    | PGSI prompt and template.                                            | NA                                                                                  |
| dfq            | Activate DFQ model. Run PTQ without this argument.                   | NA                                                                                  |
| gen_img        | Generate pseudo images for DFQ.                                      | NA                                                                                  |
| gen_method     | Select method for pseudo image generation.                           | baseline, d4c                                                                       |
| d4c_config     | Ablation with different combination of PGSI, SCG, and PAE.           | 0, 1, 2, 3                                                                          |
| gen_batch_size | Batch size for pseudo image generation.                              | NA                                                                                  |
| gen_lr         | Learning rate for pseudo image generation.                           | NA                                                                                  |
| gen_iter       | Total iterations for pseudo image generation.                        | NA                                                                                  |
| img_path       | Give a path if you want to save and visualize the generated samples. | your path, None                                                                     |

### 2.3 Hardware Resource and Training Cost

All experiments were conducted on an RTX A6000 GPU with 48 GB of memory. However, we believe that a more commonly available GPU with 16 GB of memory is sufficient to reproduce the results reported in the paper. For reference, the reconstruction for ViT-B/32 require 6,593 sec, and the pseudo image generation time (sec) of 128 images on the RTX A6000 (with a batch size of 16) is listed below:

| Method | RN50  | RN50x16 | ViT-B/32 |    ViT-B/16   |
|--------|-------|---------|----------|---------------|
| BNS    | 1,280 | 9,515   | NA       | NA            |
| PSE    | NA    | NA      | 3,488    | 44,398 (bs=8) |
| D4C    | 1,623 | 12,491  | 1,434    | 5,346         |

## 3. Abstract

Data-Free Quantization (DFQ) offers a practical solution for model compression without requiring access to real data, making it particularly attractive in privacy-sensitive scenarios. While DFQ has shown promise for unimodal models, its extension to Vision-Language Models such as Contrastive Language-Image Pre-training (CLIP) models remains underexplored. In this work, we reveal that directly applying existing DFQ techniques to CLIP results in substantial performance degradation due to two key limitations: insufficient semantic content and low intra-image diversity in synthesized samples. To tackle these challenges, we propose D4C, a novel DFQ framework tailored for CLIP. D4C synthesizes semantically rich and structurally diverse pseudo images through three key components: (1) Prompt-Guided Semantic Injection aligns generated images with real-world semantics using text prompts; (2) Structural Contrastive Generation reproduces compositional structures of natural images by leveraging foreground-background contrastive synthesis; and (3) Perturbation-Aware Enhancement applies controlled perturbations to improve sample diversity and robustness. These components jointly empower D4C to synthesize images that are both semantically informative and structurally diverse, effectively bridging the performance gap of DFQ on CLIP. Extensive experiments validate the effectiveness of D4C, showing significant performance improvements on various bit-widths and models. For example, under the W4A8 setting with CLIP ResNet-50 and ViT-B/32, D4C achieves Top-1 accuracy improvement of 12.4% and 18.9% on CIFAR-10, 6.8% and 19.7% on CIFAR-100, and 1.4% and 5.7% on ImageNet-1K in zero-shot classification, respectively.

## Citation

If you find this repo is useful, please cite our paper. Thanks.

```bibtex
@article{zhang2025d4c,
  title={D4C: Data-free Quantization for Contrastive Language-Image Pre-training Models},
  author={Zhang, Wenlun and Zhong, Yunshan and Ding, Zihao and Li, Xinyu and Yoshioka, Kentaro},
  journal={arXiv preprint arXiv:2511.15411},
  year={2025}
}
```

## Acknowledgments

Our work builds upon [QDrop](https://github.com/wimh966/QDrop), [CLIP](https://github.com/openai/CLIP), and [CLIP_benchmark](https://github.com/LAION-AI/CLIP_benchmark). We sincerely appreciate their pioneering efforts, which provided the foundation and codebase for developing and evaluating D4C.
