#  Kindle Review Intelligence | Deep Learning Rating Prediction
### NLP · LSTM · CNN+LSTM · Bidirectional LSTM | Amazon eBook Sentiment Analysis

**Skills demonstrated:** Python · TensorFlow/Keras · NLP · Text Preprocessing · Sequence Modeling · LSTM · Bidirectional LSTM · CNN · Regression Modeling · E-Commerce Analytics · Data Visualization · Business Decision-Making

---

## Project Overview

Built an end-to-end NLP pipeline — from raw Amazon Kindle customer review text to production-ready star-rating predictions — benchmarking three deep learning architectures against a naive mean-prediction baseline. Applied text cleaning, Keras tokenization, and sequence padding before training Stacked LSTM, CNN+LSTM hybrid, and Bidirectional LSTM models, each using learned word embeddings trained on the Kindle corpus. The Bidirectional LSTM achieves a test MAE of **~0.485 stars**, reducing prediction error by **37% over the baseline** and enabling automated scoring of books not yet available on the Kindle platform.

---

## Business Problem

Amazon generates millions of Kindle reviews, but the strategic value of that unstructured text goes beyond simply displaying ratings. Three high-value use cases drive this project:

| Business Opportunity | How Rating Prediction Delivers Value |
|---|---|
| **Exclusive content acquisition** | Score books discussed on Goodreads or Reddit that are not yet on Kindle — identify hidden commercial opportunities before competitors do |
| **Publisher contract negotiation** | Quantify predicted audience satisfaction to justify licensing fees and price exclusivity deals more accurately |
| **Catalog quality control** | Flag books likely to receive low ratings before they are promoted on the Kindle homepage or featured in email campaigns |

**Solution:** A deep learning NLP model that predicts star ratings (1–5) from review text alone — deployable as a batch scoring API on any book discussion corpus, published or unpublished.

---

## Data Description

- **Dataset:** Amazon Kindle Store Customer Reviews
- **Source:** [Kaggle — Amazon Kindle Reviews](https://www.kaggle.com/datasets/bharadwaj6/kindle-reviews)
- **Key columns:** `reviewText` (free-form review), `overall` (star rating 1.0–5.0), `asin` (book identifier)
- **Filter applied:** Books with more than 250 reviews retained — ensures sufficient per-title signal for learning
- **Target variable:** `overall` rating treated as a **continuous regression output** (not classification) — a predicted score of 3.7 means satisfaction level between 3 and 4, leaning toward 4
- **Class imbalance:** 48% of reviews are 5★ — stratified sampling applied to preserve rating proportions across train and test
- **Split:** 90% train · 10% test, stratified by rating value
- **Fallback:** If the dataset file is unavailable, the notebook auto-generates a statistically representative synthetic dataset and runs without interruption

---

## Methodology

- **EDA (Section 1):** Rating distribution bar chart with counts and percentages, review length histogram with maxlen=100 marker, average review length by star rating, cumulative sequence coverage curve justifying the maxlen choice, top-word frequency comparison between 5★ and 1–2★ reviews, and rating share pie chart
- **Text Preprocessing (Section 2):** Lowercase + remove non-alphabetic characters; Keras `Tokenizer` fitted on training data only (prevents data leakage); vocabulary size = 10,000; maxlen = 100 tokens; post-padding and post-truncation; stratified 90/10 split with class distribution verified in both sets
- **Naive Baseline (Section 3):** Training set mean rating predicted for every test sample — establishes the MAE floor (0.766) that all models must beat to justify ML investment
- **Model 1 — Stacked LSTM (Section 4):** Embedding(10K, 128) → LSTM(64, return_sequences=True, dropout=0.3) → LSTM(32) → Dense(1); two LSTM layers learn sequential and hierarchical review patterns; EarlyStopping, ReduceLROnPlateau, ModelCheckpoint applied
- **Model 2 — CNN + LSTM Hybrid (Section 5):** Embedding(10K, 64) → Conv1D(64 filters, kernel=5, ReLU) → MaxPooling1D(2) → LSTM(100, dropout=0.2) → Dense(1); Conv1D extracts local 5-gram phrase patterns before the LSTM processes the compressed sequential representation
- **Model 3 — Bidirectional LSTM (Section 6):** Embedding(10K, 64) → BiLSTM(100, dropout=0.2) → Dense(1); parallel forward and backward LSTMs capture full-sentence context at every token position — handles negations and sentiment reversals that confuse unidirectional models
- **Evaluation:** MAE, RMSE, MAPE on held-out test set; training vs validation loss convergence curves; predicted-vs-actual scatter plots; overlapping residual distributions; MAE bar chart across all four approaches

---

## Results

| Model | Architecture | Test MAE | MAPE | vs Baseline |
|---|---|---|---|---|
| Naive Baseline | Training mean prediction | ~0.766 | ~26.9% | — |
| Stacked LSTM | Emb → LSTM(64) → LSTM(32) → Dense | ~0.550 | ~20.0% | −28% |
| CNN + LSTM | Emb → Conv1D → Pool → LSTM(100) → Dense | ~0.522 | ~19.0% | −32% |
| **BiLSTM ⭐** | **Emb → BiLSTM(100) → Dense** | **~0.485** | **~17.0%** | **−37%** |

- A MAE of **0.485 stars** means predictions are typically within half a star of the actual rating — within the range of natural human rating variation on the same book
- The CNN + LSTM hybrid converges faster than the Stacked LSTM due to the convolutional downsampling step reducing the sequence length the LSTM processes
- BiLSTM's backward reading pass is the decisive advantage: review language often uses qualifiers ("not quite", "almost perfect", "better than expected") where the words that follow determine the overall sentiment
- EarlyStopping triggers around epoch 6–7 for all models — later epochs show mild overfitting on the 5★-skewed training distribution

---

## Business Recommendations

| Use Case | Recommended Model | Rationale |
|---|---|---|
| Scoring unpublished book candidates (acquisition) | **BiLSTM** | Highest accuracy; best handles nuanced review language |
| Real-time review moderation pipeline | CNN + LSTM | Strong accuracy with faster per-request inference |
| Large-scale nightly batch scoring (millions) | Stacked LSTM | Lower compute and memory cost at scale |
| Stakeholder proof-of-concept demo | Naive Baseline | Zero compute; quantifies the exact value of ML investment |

**ROI framing:** With 500 book acquisition candidates evaluated per quarter, BiLSTM reduces uncertain-prediction cases from ~380 (baseline) to ~240 — giving the acquisitions team 37% more confident ranking signals and lowering the cost of mis-acquired exclusive titles.

**Priority next step:** Fine-tune DistilBERT on the Kindle corpus to target MAE below 0.40 — the threshold at which model predictions become reliable enough for automated contract pricing without human review.

---

## Tools & Technologies

- **Language:** Python 3.10
- **Deep Learning:** TensorFlow 2.x · Keras — Sequential API, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- **NLP:** Keras Tokenizer · pad_sequences · Custom regex text cleaning pipeline
- **Data Processing:** Pandas · NumPy · Scikit-learn
- **Visualization:** Matplotlib · Seaborn
- **Model Persistence:** Keras `.keras` format · Pickle (tokenizer)
- **Environment:** Google Colab (GPU-accelerated) · Jupyter Notebook

---

## Project Files

```
kindle-review-intelligence/
│
├── Kindle_Review_Intelligence_Deep_Learning_Rating_Prediction.ipynb  ← Full notebook
├── README.md                                                          ← This file
│
└── Generated outputs (produced when notebook runs end-to-end)
    │
    ├── eda_kindle.png                ← 8-panel EDA: rating distribution (bar + pie),
    │                                    review length histogram with maxlen=100 marker,
    │                                    avg length by star rating, top words 5★ vs 1–2★,
    │                                    cumulative sequence coverage curve
    │
    ├── model_comparison_dashboard.png ← Training/validation loss curves (all 3 models),
    │                                    predicted-vs-actual scatter plots (3 panels),
    │                                    MAE bar chart across all 4 approaches,
    │                                    overlapping residual error distributions
    │
    ├── model_stacked_lstm.keras      ← Final Stacked LSTM weights
    ├── model_cnn_lstm.keras          ← Final CNN + LSTM weights
    ├── model_bilstm.keras            ← Final Bidirectional LSTM weights
    ├── best_stacked_lstm.keras       ← Best-epoch LSTM (EarlyStopping checkpoint)
    ├── best_cnn_lstm.keras           ← Best-epoch CNN + LSTM checkpoint
    ├── best_bilstm.keras             ← Best-epoch BiLSTM checkpoint
    ├── tokenizer.pkl                 ← Fitted Keras Tokenizer (required for inference)
    ├── model_predictions.csv         ← Test set: actual ratings + all 4 model predictions
    ├── model_comparison.csv          ← MAE / RMSE / MAPE summary for all models
    └── training_history.csv          ← Epoch-level loss and MAE for all 3 deep learning models
```

> **Dataset:** Download `kindle_reviews.xlsx` from [Kaggle](https://www.kaggle.com/datasets/bharadwaj6/kindle-reviews). If the file is absent, the notebook generates representative synthetic data automatically — no interruption to the run.

---

## How to Run

**Google Colab** *(recommended — GPU significantly speeds LSTM training)*

1. Upload the notebook at [colab.research.google.com](https://colab.research.google.com)
2. Set runtime to GPU: Runtime → Change runtime type → T4 GPU
3. Upload `kindle_reviews.xlsx` via the Colab file browser, or skip and let the built-in synthetic fallback run
4. Runtime → Run all

**Local Jupyter**
```bash
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn openpyxl
jupyter notebook Kindle_Review_Intelligence_Deep_Learning_Rating_Prediction.ipynb
```

---

## Author

**Aketch Okoth** · M.S. Business Analytics · Montclair State University  
*Actively seeking Data Analyst and Business Analyst roles in the United States.*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/your-profile)
[![GitHub](https://img.shields.io/badge/GitHub-Portfolio-181717?style=flat&logo=github&logoColor=white)](https://github.com/your-username)
[![Email](https://img.shields.io/badge/Email-Contact-EA4335?style=flat&logo=gmail&logoColor=white)](mailto:your-email@email.com)
