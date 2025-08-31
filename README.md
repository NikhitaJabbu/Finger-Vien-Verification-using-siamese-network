# Finger-Vein Verification with a Siamese Network

Siamese CNN for finger-vein verification. The pipeline builds balanced positive/negative pairs on the fly, applies grayscale + histogram equalization preprocessing, trains a separable-convolution Siamese model with Euclidean distance and contrastive loss, and calibrates a decision threshold for verification.

## Features
- **Streaming pair generator**: balanced positives/negatives by subject/hand/finger; caches decoded images for speed.
- **Preprocessing**: grayscale, histogram equalization, optional resize, values scaled to [0, 1].
- **Model**: twin branches with separable conv blocks → global average pooling → L2-normalized embeddings → Euclidean distance → contrastive loss.
- **Metrics & calibration**: accuracy at a configured threshold and an automatically searched best threshold; saves a small calibration JSON alongside the model.
- **Mixed precision (GPU)**: auto-enables on systems with a supported GPU.

## Repository layout
├─ data_loading_patched.py # streaming loader + preprocessing
├─ keras_siamese_MLP1_loop_patched.py # training/eval script (TensorFlow/Keras)
├─ README.md
└─ outputs_fv_siamese_fast/ # created on first run (model + logs)

## Requirements
- Python 3.9–3.11
- TensorFlow 2.15+
- Keras (standalone) 3.x
- NumPy, Pillow, scikit-image

## Quick install:
bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate
pip install "tensorflow>=2.15" "keras>=3.0.0" numpy pillow scikit-image

## Data
This project does not redistribute datasets. Place your images under data/(or any folder) and point the script to those paths.
Supported image types: .bmp, .png, .jpg, .jpeg, .tif, .tiff
Pairing keys (auto-parsed from paths):
Subject ID: numeric directory names preferred (e.g., .../001/...)
Hand: left or right detected from folder/file names
Finger: index, middle, or ring detected from folder/file names

## Working of the project
Load & index images by (subject, hand, finger).
Streaming generator yields each batch with ~50% positives and ~50% negatives, mixing same-type and cross-type negatives for balance.
Model learns L2-normalized embeddings per image; Euclidean distance is trained with contrastive loss.
Evaluate on a fixed set of pairs and report accuracy at the configured threshold and the best threshold found via a simple sweep.
