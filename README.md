# Stress Detection in Social Media Text
DSA4262 – Predictive Modelling Assignment

## Overview

This project investigates automated stress detection in informal social media text using machine learning and transformer-based models. The objective is to evaluate whether contextual deep learning models outperform traditional lexical approaches in detecting psychological distress signals.

The study compares:
- Keyword-based rule baseline
- Logistic Regression (TF-IDF)
- Support Vector Machine (Linear Kernel)
- Random Forest
- Fine-tuned BERT (Transformer)

Performance is evaluated using F1-score as the primary metric, alongside Precision, Recall, Accuracy, and ROC-AUC.

---

## Experimental Design

### Data Processing
- Text cleaning and normalization
- TF-IDF vectorisation (traditional models)
- BERT tokenization (transformer model)
- Fixed random seed for reproducibility

### Models Evaluated
- **Baseline**: Keyword threshold rule
- **Logistic Regression**: Linear lexical classifier
- **SVM**: Margin-based linear classifier
- **Random Forest**: Non-linear ensemble model
- **BERT**: Contextual transformer with attention

### Evaluation Metrics
Primary metric:
- **F1-score** (balanced measure for imbalanced classification)

Secondary metrics:
- Precision
- Recall
- Accuracy
- ROC-AUC

---

## Reproducibility

All experiments are reproducible.

- Random seeds are fixed.
- No absolute file paths are used.
- The notebook runs end-to-end without manual edits.

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone git@github.com:faithyeore/stress-detection-ml.git
cd stress-detection-ml

### 2. Create Virtual Environment

python3 -m venv venv
source venv/bin/activate

### 3. Install Dependencies

pip install --upgrade pip
pip install -r requirements.txt

### 4. Running the Notebook

jupyter lab




