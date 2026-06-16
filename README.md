# 🎭 Amazon Reviews Sentiment Analysis

A comparative NLP project that analyzes sentiment in Amazon product reviews using two approaches — **VADER** (rule-based) and **RoBERTa** (transformer-based) — with an interactive Streamlit web app for live predictions.

---

## 📌 Project Overview

This project explores how well different sentiment analysis techniques align with actual star ratings from Amazon customer reviews. It compares a lightweight lexicon-based model (VADER) against a state-of-the-art pre-trained transformer (RoBERTa), providing both quantitative evaluation and an interactive demo app.

---

## 🗂️ Project Structure

```
amazon-reviews-sentiment-analysis/
├── sentiment_analysis.ipynb          # Full analysis pipeline (EDA → VADER → RoBERTa → Evaluation)
├── app.py                  # Streamlit web application
├── Reviews.csv             # Amazon Fine Food Reviews dataset
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 🔍 What's Inside the Notebook

### 1. Exploratory Data Analysis (EDA)
- Distribution of star ratings (1–5) across the dataset
- Basic text inspection and sample review exploration

### 2. NLTK Basics
- Tokenization, POS tagging, and Named Entity Recognition (NER) using NLTK
- Demonstrates foundational NLP preprocessing steps

### 3. VADER Sentiment Scoring
- Applied `SentimentIntensityAnalyzer` from NLTK to compute `pos`, `neu`, `neg`, and `compound` scores
- Ran scoring across the full dataset and visualized results against star ratings using bar plots

### 4. RoBERTa Transformer Model
- Used `cardiffnlp/twitter-roberta-base-sentiment` from HuggingFace Transformers
- Ran inference on a 5,000-review sample (with error handling for runtime issues)
- Extracted three-class probabilities: negative, neutral, positive

### 5. Model Evaluation & Comparison
- Mapped star ratings to sentiment labels: 1–2 → negative, 3 → neutral, 4–5 → positive
- Generated confusion matrices and classification reports for both VADER and RoBERTa
- Created pairplot to visually compare score distributions across both models

### 6. Edge Case Analysis
- Identified reviews with mismatched sentiment (e.g., 1-star reviews predicted as positive)
- Highlighted limitations of rule-based vs. context-aware models

### 7. HuggingFace Pipeline Demo
- Quick demo using `transformers.pipeline("sentiment-analysis")` for zero-shot predictions

---

## 🎬 See It In Action

The video below walks through the app — enter any text, and instantly get sentiment predictions from both VADER and RoBERTa with confidence scores.

https://github.com/user-attachments/assets/04230404-841e-488d-98a8-a6f50f5ed367

---

## 🌐 Streamlit App

The app (`app.py`) provides a live interface to:
- Enter any text and get real-time sentiment predictions
- View both VADER and RoBERTa scores side by side
- See sentiment label and confidence scores

**🔗 Live Demo:** [amazon-reviews-sentiment-analysis-pvbuqjvqprhwypnvkckkm5.streamlit.app](https://amazon-reviews-sentiment-analysis-pvbuqjvqprhwypnvkckkm5.streamlit.app/)

**To run locally:**

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.8+
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Key libraries used:**

| Library | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `nltk` | Tokenization, POS tagging, VADER |
| `transformers` | RoBERTa model (HuggingFace) |
| `torch` | Deep learning backend |
| `scipy` | Softmax for score normalization |
| `matplotlib`, `seaborn` | Visualization |
| `scikit-learn` | Confusion matrix & classification report |
| `streamlit` | Web application |
| `tqdm` | Progress bars |

### NLTK Downloads (run once)

```python
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
```

---

## 📊 Dataset

**Amazon Fine Food Reviews** — available on [Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)

- ~568,000 reviews
- Columns used: `Id`, `Score` (1–5 stars), `Text`
- Sampled 5,000 rows for transformer model inference (due to compute constraints)

---

## 📈 Key Findings

- **RoBERTa outperforms VADER** on capturing nuanced sentiment, especially for sarcastic or context-dependent reviews
- VADER works well for clearly positive/negative language but struggles with neutral and ambiguous text
- Both models align strongly with 5-star (positive) reviews; misclassification is highest for 1-star reviews with positive language (e.g., complaints phrased sarcastically)

---

## 🚀 Future Improvements

- [ ] Fine-tune RoBERTa on the Amazon reviews dataset for domain-specific accuracy
- [ ] Add aspect-based sentiment analysis (e.g., "packaging was bad, taste was great")
- [ ] Expand to full dataset with GPU acceleration
- [ ] Add multilingual sentiment support

---

## 🙋 Author

**Ritika Choudhary**

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).
