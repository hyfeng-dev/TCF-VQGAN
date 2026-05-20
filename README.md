
![Banner](imgs/Banner.png)

### 🛰️ TCF-VQ GAN — Official PyTorch Implementation

[中文 (简体)](docs/README.zh.md)

### ⚙️ Installation

Install `torch` and `torchvision` following the official PyTorch instructions: https://pytorch.org/get-started/locally/. Install the remaining Python dependencies with:

```bash
pip install -r requirements.txt
```

Tested with `torch==2.6.0+cu126`. If you already have some dependencies installed, install only what is missing.

### 📦 Dataset Structure

#### SEN12MS-CR

Organize the SEN12MS-CR dataset as follows (train/val/test):

```text
SEN12MS-CR/
├── train/
│   ├── input_s1/
│   │   ├── ROIs1158_spring_s1_1_p88.tif
│   │   ├── ...
│   │
│   ├── input_s2_cloudy/
│   │   ├── ROIs1158_spring_s2_cloudy_1_p88.tif
│   │   ├── ...
│   │
│   └── label/
│       ├── ROIs1158_spring_s2_1_p88.tif
│       ├── ...
│
├── val/
│
└── test/
```

#### SMILE-CR

Download SMILE-CR from [SMILE-CR](https://www.kaggle.com/datasets/yuxiawhu/smile-cr/data) and extract it. The expected layout is:

```text
SMILE-CR/
├── TrainData/
│   ├── CloudLandsat_2020/
│   ├── Landsat-8_2020/
│   ├── Mask/
│   ├── MODIS_2020/
│   └── Sentinel-1_2020-De/
│
├── ValData/
│
└── TestData/
```

### 🚀 Training

The code is configuration-driven. Typical workflow:

1. In `configs/SEN12MS_CR` or `configs/SMILE_CR`, edit `HQD.yaml` and `VQGAN.yaml` to set `data -> params -> train|val|test -> params -> path` to your dataset paths. Verify paths before running.

Example (SEN12MS-CR):

```text
PATH_TO_DATASET/SEN12MS-CR-5K/train
PATH_TO_DATASET/SEN12MS-CR-5K/val
PATH_TO_DATASET/SEN12MS-CR-5K/test
```

Example (SMILE-CR):

```text
PATH_TO_DATASET/SMILE-CR/TrainData
PATH_TO_DATASET/SMILE-CR/ValData
PATH_TO_DATASET/SMILE-CR/TestData
```

2. Stage I: set `HQD.yaml` in `main.py` (line ~110), e.g.:

```python
config = OmegaConf.load("configs/SMILE_CR/HQD.yaml")
```

Then run:

```bash
python main.py
```

Stage I will produce a best checkpoint for Stage II.

3. Stage II: set the best `.ckpt` from Stage I into `VQGAN.yaml` at `model -> params -> ckpt_path`, then switch `main.py` to load `VQGAN.yaml`:

```python
config = OmegaConf.load("configs/SMILE_CR/VQGAN.yaml")
```

Then run:

```bash
python main.py
```

Stage II refines generation quality on top of Stage I.

4. Evaluation: refer to `test.ipynb` (cells 1–2) for example evaluation scripts and metric calculations.

### Acknowledgements

Thanks to the following projects for code and ideas:

- [RestoreFormerPlusPlus](https://github.com/wzhouxiff/RestoreFormerPlusPlus)
- [generative_inpainting](https://github.com/JiahuiYu/generative_inpainting)
