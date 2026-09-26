# Pix2TeX Fine-Tuning Attempts & Baseline Evaluation

An experimental deep learning project exploring **Pix2TeX (LaTeX-OCR)** for printed mathematical expression recognition.

The project focuses on converting images of printed mathematical expressions into **LaTeX code**, evaluating the pre-trained Pix2TeX model, preparing a reproducible data pipeline, and investigating partial fine-tuning strategies.

## 🎯 Project Goal

The main objectives of this project were to:

- Convert printed mathematical expression images into LaTeX code
- Evaluate the pre-trained Pix2TeX model
- Build a reproducible data preparation and evaluation pipeline
- Explore partial fine-tuning through layer freezing and unfreezing
- Analyze the technical limitations encountered during fine-tuning

## 📊 Dataset

The project uses the `linxy/LaTeX_OCR` dataset from Hugging Face.

The dataset contains:

- Mathematical expression images
- Corresponding LaTeX ground-truth text

The pipeline targets **printed/typeset mathematical expressions**, rather than handwritten mathematical expressions.

## ⚙️ Data Preparation

A complete preprocessing pipeline was implemented, including:

- Image resizing to 224 × 224
- Tensor conversion
- ImageNet normalization
- Dataset inspection and visualization
- LaTeX sequence-length statistics
- Train / validation / test splitting
- PyTorch DataLoader preparation
- Image-to-LaTeX pairing validation

The dataset was split into:

- 70% Training
- 15% Validation
- 15% Testing

A fixed random seed was used to support reproducibility.

## 🧠 Baseline Evaluation

The original pre-trained Pix2TeX model was evaluated before attempting fine-tuning.

For inference, Pix2TeX accepts PIL images and internally handles its own preprocessing.

The evaluation pipeline includes:

- LaTeX normalization
- Custom LaTeX tokenization
- BLEU score calculation using NLTK with smoothing
- Qualitative comparison between ground-truth and predicted LaTeX
- Average BLEU evaluation across a subset of the dataset

## 🔬 Fine-Tuning Attempts

Several partial fine-tuning strategies were explored.

### Strategy 1 — Freeze Encoder

The encoder was frozen while other parts of the model were prepared for training.

### Strategy 2 — Selective Layer Training

Most model parameters were frozen while selected output, projection, head, or final layers were made trainable.

The experiments also included analysis of:

- Total model parameters
- Trainable parameters
- Frozen parameters
- Optimizer configuration

These experiments demonstrated how the model could be prepared conceptually for partial fine-tuning.

## ⚠️ Fine-Tuning Limitation

A key challenge was that the available `LatexOCR()` interface is designed primarily for **inference rather than training**.

A typical prediction returns decoded LaTeX text:

```python
pred = model(pil_img)
```

The output is a string rather than differentiable model outputs such as logits.

For gradient-based training, the pipeline would require access to:

- Model logits or token probabilities
- A differentiable loss function
- Target-aware internal forward passes
- A valid backpropagation path

The evaluation used BLEU to compare predicted and ground-truth LaTeX. However, BLEU is not differentiable and therefore cannot be used directly for gradient-based optimization.

As a result, although layers were successfully prepared for partial fine-tuning and optimizers were configured, the inference interface did not provide the components required for reliable gradient-based training.

## 📌 Project Outcome

Although full gradient-based fine-tuning was not completed through the available inference interface, the project demonstrates:

- A reproducible mathematical OCR data pipeline
- Dataset preprocessing and inspection
- Baseline evaluation of a pre-trained Pix2TeX model
- LaTeX normalization and BLEU-based evaluation
- Model parameter freezing and unfreezing strategies
- Investigation of partial fine-tuning approaches
- Analysis of architectural and API-level training limitations

The project highlights both the practical implementation process and the technical challenges involved in adapting an inference-oriented model interface for training.

## 🛠️ Technologies

- Python
- PyTorch
- Pix2TeX / LaTeX-OCR
- Hugging Face Datasets
- Torchvision
- NLTK
- Pillow
- Jupyter Notebook

## 📁 Repository Structure

```text
Pix2TeX-FineTuning-Attempts/
│
├── Deep_learning_project_pix2tex.ipynb
└── README.md
```

## ▶️ How to Run

Install the required dependencies:

```bash
pip install datasets pillow torch torchvision transformers pix2tex nltk
```

Then open and run:

```text
Deep_learning_project_pix2tex.ipynb
```

## 📚 References

- Pix2TeX / LaTeX-OCR by Lukas Blecher
- Hugging Face `linxy/LaTeX_OCR` Dataset
- NLTK BLEU Score with Smoothing
