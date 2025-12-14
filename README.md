# Pix2TeX Fine-Tuning Attempts + Baseline Evaluation (Printed Math → LaTeX)

This repository documents our end-to-end attempts to fine-tune the **Pix2TeX (LaTeX-OCR)** model for **Printed Mathematical Expression Recognition**, along with a complete data pipeline and a reproducible baseline evaluation.
Although we prepared the model for partial fine-tuning (freezing/unfreezing layers + optimizer setup), we faced architectural and API-level constraints that prevented reliable **gradient-based fine-tuning** using the inference interface. Therefore, the final implementation focuses on **rigorous evaluation, analysis, and transparent reporting** of the limitations.

---

## 1) Project Goal

* Convert **printed mathematical expression images** into **LaTeX code**.
* Start from a **pre-trained Pix2TeX** model and attempt **partial fine-tuning** (e.g., decoder/head only).
* Provide a clean experimental pipeline to demonstrate:

  * Data preparation and inspection
  * Baseline model evaluation (BLEU-based)
  * Fine-tuning preparation steps (freezing/unfreezing)
  * Why full fine-tuning was not feasible under the chosen interface

---

## 2) Dataset Used

We used a ready-to-use dataset from Hugging Face:

* **Dataset**: `linxy/LaTeX_OCR` (train split)
* Data fields:

  * `image`: formula image
  * `text`: LaTeX ground truth (mapped from `formula` or `caption` if needed)

> Note: This is **not** the CROHME/ICDAR handwritten dataset. The pipeline here targets **printed / typeset** expressions.

### 2.1 Column Mapping

To standardize the dataset schema:

* If the dataset column is `formula`, it is renamed to `text`.
* If it is `caption`, it is renamed to `text`.

---

## 3) Data Preparation Pipeline (What We Implemented)

### 3.1 Transformations (for PyTorch-style pipeline)

We defined a vision transform pipeline:

* Resize to **224×224**
* Convert to tensor
* Normalize using ImageNet statistics:

  * mean: `[0.485, 0.456, 0.406]`
  * std:  `[0.229, 0.224, 0.225]`

### 3.2 Custom Dataset Class (Tensor-based)

`LaTeXOCRDataset` returns:

* `image`: tensor (after transforms)
* `latex`: string

### 3.3 Dataset Preview + Statistics

We implemented a preview tool to:

* Randomly visualize samples
* Compute LaTeX-length stats (min/max/avg/median)
* Print random LaTeX examples

### 3.4 Train/Val/Test Split

We split the dataset:

* 70% Train
* 15% Validation
* 15% Test
  Using a fixed random seed for reproducibility.

### 3.5 DataLoaders + Sanity Check

We created `DataLoader`s for each split and tested:

* image batch shapes
* correct pairing of image ↔ LaTeX text

---

## 4) Baseline Evaluation (Pre-trained Pix2TeX)

### 4.1 Why a Separate "Clean" Dataset Loader?

Pix2TeX inference (`LatexOCR()`) is designed to accept **PIL images** and applies internal preprocessing.
For baseline evaluation, we reloaded a slice of the dataset and returned **PIL images** directly using `CleanLaTeXDataset` + a `pil_collate` function.

### 4.2 Metrics

We implemented:

* **Strong LaTeX normalization** (`strong_normalize`) to remove superficial formatting differences.
* **BLEU score** using NLTK with smoothing (custom LaTeX tokenization).

### 4.3 Evaluation Protocol

* Evaluated the original pre-trained Pix2TeX on a subset (e.g., first 200 samples)
* Printed qualitative examples (Actual vs Prediction)
* Reported average BLEU

---

## 5) Fine-Tuning Attempts (What We Tried)

### 5.1 Loading the Model

We loaded:

* `baseline_model = LatexOCR()` for baseline eval
* `model = LatexOCR()` for fine-tuning preparation steps

### 5.2 Freezing / Unfreezing Strategies

We attempted multiple strategies:

1. **Freeze encoder only**
2. **Freeze everything, unfreeze output/head/proj/final layers only**

We also computed:

* total parameters
* trainable parameters
* frozen parameters

This confirms we correctly prepared the model for **partial fine-tuning** conceptually.

---

## 6) Why True Fine-Tuning Was Difficult (Core Technical Reasons)

### 6.1 Inference API is Not Training API

`LatexOCR()` provides an inference wrapper:

```python
pred = model(pil_img)
```

This returns a **string** prediction (decoded LaTeX), not differentiable outputs like logits.

**Missing components for gradient-based training:**

* logits / token probabilities
* differentiable loss tensor (e.g., CrossEntropy)
* access to internal forward pass with targets
* backpropagation path from loss → parameters

### 6.2 Training Loop Was Evaluation-Only

Our loop computed BLEU and derived a “loss” as:

* `loss = 100 - BLEU`

However, BLEU is:

* **not differentiable**
* cannot produce gradients for backprop

Therefore, even with:

* optimizer defined (Adam/SGD)
* trainable parameters selected

**no parameters can be updated** without:

```python
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

…and without a differentiable loss.

### 6.3 Practical Constraints

* Training Pix2TeX end-to-end typically requires:

  * the original training code and configs
  * correct tokenization/label preparation
  * significant compute (GPU) and stable training environment
* Using only the inference wrapper is not sufficient for full fine-tuning.

---

## 7) What This Repo Demonstrates (Final Outcome)

Even though gradient-based fine-tuning was not feasible via the inference wrapper, this repo provides:

* A complete, reproducible **data preparation** pipeline
* A rigorous **baseline evaluation** of Pix2TeX on the chosen dataset
* Transparent documentation of:

  * what we tried
  * what worked
  * what blocked true fine-tuning
* Evidence that the limitation is primarily **interface/architecture**, not lack of effort.

---

## 8) How to Run

### 8.1 Install

```bash
pip install datasets pillow torch torchvision transformers pix2tex nltk
```

### 8.2 Run Notebook

Open and run:

* `Deep_learning_project_pix2tex.ipynb`

---

## 9) Repository Structure

* `Deep_learning_project_pix2tex.ipynb` — main notebook (data prep, baseline eval, fine-tuning attempts)
* (Optional) `README.md` — this file

---

## 10) References

* Pix2TeX / LaTeX-OCR: Lukas Blecher — GitHub repository
* Hugging Face Dataset: `linxy/LaTeX_OCR`
* BLEU (NLTK) with smoothing functions

---

## Instructor Note (Summary in One Paragraph)

We prepared the Pix2TeX model for partial fine-tuning by freezing and unfreezing selected layers and setting up optimizers. However, the available Pix2TeX inference interface outputs only decoded LaTeX strings and does not expose differentiable logits/loss needed for gradient-based training. As a result, the implemented loops function as evaluation rather than true fine-tuning. The project therefore emphasizes a robust data pipeline and baseline evaluation while documenting the technical constraints transparently.
