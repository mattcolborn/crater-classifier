# Crater Classifier

A binary image classifier that distinguishes cratered from non-cratered terrain using transfer learning with ResNet18 and PyTorch.

## Project Structure

```
crater-classifier/
├── src/
│   └── crater_classifier/
│       ├── __init__.py
│       ├── config.py       # all settings and hyperparameters
│       ├── data.py         # data loading, augmentation, splitting
│       ├── model.py        # ResNet18 model builder
│       ├── train.py        # training loop and fine-tuning
│       ├── evaluate.py     # test evaluation and plotting
│       ├── predict.py      # single image inference
│       └── utils.py        # save and load model
├── notebooks/
│   └── crater_classifier.ipynb
├── main.py                 # entry point
├── environment.yml         # conda environment
├── .gitignore
└── README.md
```

## Setup

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/crater-classifier.git
cd crater-classifier
```

**2. Create and activate the conda environment**
```bash
conda env create -f environment.yml
conda activate crater-classifier
```

**3. Register the environment as a Jupyter kernel**
```bash
python -m ipykernel install --user --name crater-classifier --display-name "crater-classifier"
```

## Data

Prepare your image data in the following structure:

```
data/
├── craters/
│   ├── image_001.jpg
│   └── ...
└── no_craters/
    ├── image_001.jpg
    └── ...
```

Images are automatically split 80% train / 10% validation / 10% test.

## Configuration

Edit `src/crater_classifier/config.py` to set your paths and hyperparameters:

```python
DATA_DIR   = "/path/to/your/data"
OUTPUT_DIR = "/path/to/your/output"
IMAGE_SIZE = 128
NUM_EPOCHS = 20
```

## Training

```bash
conda activate crater-classifier
python main.py
```

The best model (by validation accuracy) is saved to `OUTPUT_DIR/crater_classifier.pth`.

## Training Approach

Uses two-stage transfer learning with ResNet18 pretrained on ImageNet:

- **Stage 1** (epochs 1–10): backbone frozen, only the classifier head is trained
- **Stage 2** (epochs 11–20): last ResNet block unfrozen for fine-tuning at a lower learning rate

## Inference

To predict on a single image, uncomment and edit the `predict_image` call at the bottom of `main.py`:

```python
predict_image(
    image_path   = "/path/to/your/image.jpg",
    model        = model,
    class_to_idx = full_dataset.class_to_idx
)
```

## Requirements

- PyTorch
- torchvision
- matplotlib
- Pillow
- jupyter
