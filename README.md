# Essay-Subject-Classification-ML
Machine Learning model classifying essay sentences into academic subjects using hybrid TF-IDF (word + character n-grams) and LinearSVC for high macro-F1 scores. Features: text preprocessing, lemmatization, hybrid vectorization, full-dataset retraining. Outputs predictions to result.csv for competition evaluation.

# Intro

This project focuses on classifying sentences from high-school essays into their correct academic subjects using Machine Learning and NLP techniques. The goal is to automatically identify the subject each sentence belongs to, demonstrating automated text understanding in education systems.

### 🎯 Objective
- Train a machine learning model on labeled essay sentences
- Classify unseen essay sentences into subjects (e.g., Biology, History, English, etc.)
- Maximize Macro F1-Score (competition metric)
- Export final predictions as `result.csv`

---

## 📂 Dataset
| File | Description |
|------|-------------|
| train.csv | 10,000 rows — ID, Text, Subject |
| test.csv | 4,020 rows — ID, Text |
| result.csv | Output — ID, Predicted Subject |

---

## 🧠 Model & Approach

### ✅ Text Preprocessing
- Lowercasing text  
- Preserving numbers (useful for scientific/maths content)
- Tokenization & Lemmatization (NLTK)
- Minimal loss of context — stopwords mostly retained

### ✅ Feature Extraction
**Hybrid TF-IDF Vectorization**
- Word n-grams (1–3)
- Character n-grams (3–6)
- Captures vocabulary + morphological patterns

### ✅ Classification Model
- **Linear SVC** (`class_weight='balanced'`)
- Superior performance for high-dimensional text

### ✅ Evaluation
- Macro F1-Score (competition metric)
- Final model retrained on full dataset for maximum accuracy

---

## 🚀 Output
Generates `result.csv`:

| ID | Subject |
|----|---------|
| 101 | English |
| 102 | Biology |
| ... | ... |

---

## 📦 Requirements
Install dependencies:

```bash
pip install pandas numpy scikit-learn nltk seaborn matplotlib
