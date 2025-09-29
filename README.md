# Enron Email Fraud Detection

This repository contains a supervised machine learning pipeline used to detect potentially fraudulent emails within the **Enron Email Corpus**, using a smaller labeled dataset for training.

> *“We predicted over 151,000 emails to be fraudulent”*

---

## Project Overview

This project started as an exploration into the Enron Email dataset and ended up using **supervised learning** to classify emails as fraudulent or genuine.

Rather than training a model directly on the full Enron Email Corpus (over 500K emails), I opted to:

- Train a model on a smaller labeled **fraudulent email dataset**.
- Use this trained model to classify emails in the Enron dataset.

---

## Datasets Used

1. **Training Dataset** (Supervised):
   - [Fraud Email Dataset from Kaggle](https://www.kaggle.com/datasets/llabhishekll/fraud-email-dataset)
   - Structure: `Email` (text), `Label` (0 = genuine, 1 = fraudulent)

2. **Prediction Dataset**:
   - Cleaned version of the **Enron Email Dataset**, sourced from Kaggle.
---

## Tools & Libraries

- Python
- scikit-learn
  - `Pipeline`, `TfidfVectorizer`, `RandomForestClassifier`, `SVC`
- pandas
- joblib

---

## Model Training

Two models were compared:

| Model                | Accuracy      |
|---------------------|---------------|
| SVM (SVC)           | 59.34%        |
| Random Forest       | **98.01%** ✅ |

**TF–IDF** vectorization was used to convert emails to numerical features. Pre-trained embeddings were avoided due to hardware limitations (8GB RAM, no GPU).

Here’s the pipeline used for Random Forest:

```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

clf = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])
```
## Blog

My article on the project: https://ramerasrambles.blogspot.com/2025/09/enron-part-2.html
