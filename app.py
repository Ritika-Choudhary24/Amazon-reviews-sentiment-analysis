import streamlit as st
import pandas as pd
import nltk
import torch

from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Sentiment Dashboard",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}

.title-box {
    background: linear-gradient(90deg, #9333ea, #ec4899);
    padding: 30px;
    border-radius: 18px;
    color: white;
    text-align: center;
    margin-bottom: 25px;
}

.title-box h1 {
    font-size: 42px;
    margin-bottom: 5px;
}

.title-box p {
    font-size: 18px;
}

.metric-card {
    background-color: #fce7f3;
    padding: 22px;
    border-radius: 16px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    text-align: center;
    margin-bottom: 15px;
}

.metric-card h3 {
    color: #9d174d;
    margin-bottom: 8px;
}

.metric-card h2 {
    color: #581c87;
    font-size: 30px;
}

.metric-card p {
    color: #831843;
    font-size: 16px;
    font-weight: bold;
}

.info-box {
    background-color: #f3e8ff;
    padding: 18px;
    border-left: 6px solid #4f46e5;
    border-radius: 10px;
    margin-top: 15px;
    color: #581c87;
}

.warning-box {
    background-color: #ffe4e6;
    padding: 18px;
    border-left: 6px solid #f97316;
    border-radius: 10px;
    margin-top: 15px;
    color: #881337;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="title-box">
    <h1>📊 Sentiment Analysis Dashboard</h1>
    <p>Compare VADER and RoBERTa sentiment predictions side by side</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("⚙️ Dashboard Controls")
st.sidebar.write("Use this app to test any review and compare two sentiment models.")

example = st.sidebar.selectbox(
    "Choose an example review:",
    [
        "This product is amazing. I really loved it!",
        "The product was okay, not very good but not terrible.",
        "Worst purchase ever. I am very disappointed.",
        "The quality is decent, but delivery was very slow.",
        "Absolutely fantastic! I would buy it again."
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Models Used")
st.sidebar.write("**VADER:** Rule-based sentiment model")
st.sidebar.write("**RoBERTa:** Transformer-based deep learning model")

# -----------------------------
# Download NLTK resource
# -----------------------------
nltk.download("vader_lexicon", quiet=True)

# -----------------------------
# Load models
# -----------------------------
@st.cache_resource
def load_vader():
    return SentimentIntensityAnalyzer()

@st.cache_resource
def load_roberta():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

with st.spinner("Loading models... Please wait."):
    sia = load_vader()
    tokenizer, model = load_roberta()

# -----------------------------
# Helper functions
# -----------------------------
def get_vader_label(compound):
    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    else:
        return "neutral"

def get_roberta_scores(text):
    encoded_text = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        output = model(**encoded_text)

    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    return {
        "negative": float(scores[0]),
        "neutral": float(scores[1]),
        "positive": float(scores[2])
    }

def sentiment_emoji(label):
    if label == "positive":
        return "😊"
    elif label == "negative":
        return "😞"
    else:
        return "😐"

# -----------------------------
# Input section
# -----------------------------
st.subheader("✍️ Enter Review Text")

review = st.text_area(
    "Type or edit a review below:",
    value=example,
    height=140
)

analyze_button = st.button("Analyze Sentiment 🚀")

# -----------------------------
# Main analysis
# -----------------------------
if analyze_button:

    if review.strip() == "":
        st.warning("Please enter a review before analyzing.")
    else:
        vader_scores = sia.polarity_scores(review)
        roberta_scores = get_roberta_scores(review)

        vader_label = get_vader_label(vader_scores["compound"])
        roberta_label = max(roberta_scores, key=roberta_scores.get)

        vader_confidence = max(
            vader_scores["neg"],
            vader_scores["neu"],
            vader_scores["pos"]
        )

        roberta_confidence = roberta_scores[roberta_label]

        agreement = "Yes" if vader_label == roberta_label else "No"

        # -----------------------------
        # Top metric cards
        # -----------------------------
        st.subheader("📌 Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>VADER Prediction</h3>
                <h2>{sentiment_emoji(vader_label)} {vader_label.title()}</h2>
                <p>Confidence: {vader_confidence:.2f}</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>RoBERTa Prediction</h3>
                <h2>{sentiment_emoji(roberta_label)} {roberta_label.title()}</h2>
                <p>Confidence: {roberta_confidence:.2f}</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Model Agreement</h3>
                <h2>{agreement}</h2>
                <p>Do both models predict the same sentiment?</p>
            </div>
            """, unsafe_allow_html=True)

        # -----------------------------
        # Score comparison table
        # -----------------------------
        st.subheader("📋 Side-by-side Score Comparison")

        comparison_df = pd.DataFrame({
            "Sentiment": ["Negative", "Neutral", "Positive"],
            "VADER Score": [
                vader_scores["neg"],
                vader_scores["neu"],
                vader_scores["pos"]
            ],
            "RoBERTa Score": [
                roberta_scores["negative"],
                roberta_scores["neutral"],
                roberta_scores["positive"]
            ]
        })

        st.dataframe(comparison_df, use_container_width=True)

        # -----------------------------
        # Bar chart
        # -----------------------------
        st.subheader("📈 Visual Score Comparison")

        chart_df = comparison_df.set_index("Sentiment")
        st.bar_chart(chart_df)

        # -----------------------------
        # Detailed VADER and RoBERTa outputs
        # -----------------------------
        col4, col5 = st.columns(2)

        with col4:
            st.subheader("🔹 VADER Details")

            vader_detail_df = pd.DataFrame({
                "Metric": ["Negative", "Neutral", "Positive", "Compound"],
                "Score": [
                    vader_scores["neg"],
                    vader_scores["neu"],
                    vader_scores["pos"],
                    vader_scores["compound"]
                ]
            })

            st.dataframe(vader_detail_df, use_container_width=True)

        with col5:
            st.subheader("🔹 RoBERTa Details")

            roberta_detail_df = pd.DataFrame({
                "Metric": ["Negative", "Neutral", "Positive"],
                "Score": [
                    roberta_scores["negative"],
                    roberta_scores["neutral"],
                    roberta_scores["positive"]
                ]
            })

            st.dataframe(roberta_detail_df, use_container_width=True)

        # -----------------------------
        # Interpretation
        # -----------------------------
        st.subheader("🧠 Interpretation")

        if vader_label == roberta_label:
            st.markdown(f"""
            <div class="info-box">
                Both models predicted the review as <b>{vader_label}</b>. 
                This means the rule-based model and transformer model agree on the overall sentiment.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="warning-box">
                VADER predicted <b>{vader_label}</b>, while RoBERTa predicted <b>{roberta_label}</b>. 
                This difference may happen because RoBERTa understands context better, while VADER mainly depends on sentiment words and rules.
            </div>
            """, unsafe_allow_html=True)

else:
    st.info("Enter a review and click **Analyze Sentiment** to see the dashboard.")

# -----------------------------
# Footer
# -----------------------------
st.markdown(
    """
    <hr>
    <p style='text-align: center; color: gray; font-size: 15px;'>
        Made with ❤️ by <b>Ritika Choudhary</b><br>
        Amazon Reviews Sentiment Analysis Project
    </p>
    """,
    unsafe_allow_html=True
)
