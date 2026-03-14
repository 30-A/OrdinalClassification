# Ordinal Classification for Avian Toxicity Prediction

Code and dataset for the paper **"Ordinal Classification for Multi-Class Avian Acute Toxicity Prediction"**. 

This project implements a Python version of the Frank and Hall (2001) ordinal decomposition. The model predicts avian oral toxicity categories (low, moderate and high) using MACCS fingerprints.

---

### Repository Structure

* `data/`: Train and test datasets (`train.csv`, `test.csv`).
* `ordinal_toxicity_model.py`: Script for fingerprint generation, feature selection and model training.

---

### Requirements

Install dependencies:

```bash
pip install numpy pandas scikit-learn scikit-fingerprints feature-engine
```

---

### Usage

Clone the repository and run the script:

```bash
git clone https://github.com/30-A/OrdinalClassification.git
cd OrdinalClassification
python ordinal_toxicity_model.py
```

The script trains the models and outputs evaluation metrics and confusion matrices.

---

### Data Attribution

The dataset and train/test splits by Iovine et al. (2025). If you use this data, please cite:

```bibtex
@article{iovine2025predicting,
  title={Predicting acute oral toxicity in Bobwhite quail: Development of QSAR models for LD50},
  author={Iovine, N. and Roncaglioni, A. and Benfenati, E.},
  journal={Environments},
  volume={12},
  number={2},
  pages={56},
  year={2025}
}
```
