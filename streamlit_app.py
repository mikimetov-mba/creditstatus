import streamlit as st
import numpy as np
import pandas as pd
import joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# =========================
# 1. Load model + metadata
# =========================

MODEL_PATH = "loan_logreg_model.pkl"
FEAT_PATH = "loan_logreg_features.pkl"
DEFAULTS_PATH = "loan_logreg_defaults.pkl"

model = joblib.load(MODEL_PATH)
feature_cols = joblib.load(FEAT_PATH)
default_values = joblib.load(DEFAULTS_PATH)

# =========================
# 2. Sentiment & red-flag logic
# =========================

analyzer = SentimentIntensityAnalyzer()

STRONG_NEGATIVE_PHRASES = [
    # Direct refusal to pay
    "do not want to pay",
    "dont want to pay",
    "don't want to pay",
    "won't pay",
    "will not pay",
    "no intention to pay",
    "not planning to pay",
    "never pay this loan",
    "refuse to pay",
    "not going to repay",
    "not going to pay back",
    "i will just default",
    "i want to default",

    # Direct inability to pay
    "cannot pay",
    "can't pay",
    "cant pay",
    "cannot pay back",
    "can't pay back",
    "cant pay back",
    "unable to pay",
    "unable to repay",
    "i cannot pay back",
    "i can't pay back",
    "i cant pay back",
    "i cannot repay",
    "i can't repay",
    "i cant repay",
    "i will not be able to pay",
]

RISKY_PHRASES = [
    "struggling to pay",
    "struggle to pay",
    "hard to pay",
    "difficult to pay",
    "behind on payments",
    "late on payments",
    "missed payments",
    "missed some payments",
    "huge debt",
    "a lot of debt",
    "drowning in debt",
    "serious financial problems",
    "serious financial difficulty",
    "lost my job",
    "lost my income",
    "unemployed",
    "no stable income",
    "income is unstable",
]


def get_sentiment_label(text: str) -> int:
    """Return -1 (neg), 0 (neutral), 1 (pos) using VADER + phrase rules."""
    if not text or not text.strip():
        return 0

    text_lower = text.lower()

    # Strong red-flag phrases override everything
    if any(flag in text_lower for flag in STRONG_NEGATIVE_PHRASES):
        return -1

    compound = analyzer.polarity_scores(text_lower)["compound"]

    # Risky phrases push compound more negative
    if any(flag in text_lower for flag in RISKY_PHRASES):
        compound -= 0.3

    if compound > 0.1:
        return 1
    elif compound < -0.1:
        return -1
    else:
        return 0


def has_strong_red_flag(text: str) -> bool:
    if not text:
        return False
    text_lower = text.lower()
    return any(flag in text_lower for flag in STRONG_NEGATIVE_PHRASES)


def has_risky_flag(text: str) -> bool:
    if not text:
        return False
    text_lower = text.lower()
    return any(flag in text_lower for flag in RISKY_PHRASES)


# =========================
# 3. Mappings (must match training)
# =========================

home_ownership_mapping = {
    "OWN": 1,
    "MORTGAGE": 2,
    "RENT": 3,
    "OTHER": 4,
    "NONE": 5,
    "ANY": 5,
}

# =========================
# 4. Streamlit UI
# =========================

st.set_page_config(page_title="Loan Status Predictor", page_icon="💳")

st.title("💳 Loan Status Prediction App")
st.markdown(
    """
This app uses a Logistic Regression model trained on historical loan data.
It combines **traditional credit variables** with **sentiment from the borrower's description**
to predict whether a loan is likely to be **Fully Paid (1)** or **Default (0)**.
"""
)

st.sidebar.header("Input Loan Information")

loan_amnt = st.sidebar.number_input(
    "Loan Amount", min_value=0.0, step=1000.0, value=10000.0
)

term = st.sidebar.selectbox("Term (months)", [36, 60])

emp_length_years = st.sidebar.number_input(
    "Employment Length (years)", min_value=0.0, max_value=50.0, step=1.0, value=5.0
)

annual_inc = st.sidebar.number_input(
    "Annual Income", min_value=0.0, step=1000.0, value=50000.0
)

home_ownership_text = st.sidebar.selectbox(
    "Home Ownership", ["OWN", "MORTGAGE", "RENT", "OTHER", "NONE", "ANY"]
)

st.sidebar.markdown("---")
desc_text = st.sidebar.text_area(
    "Loan description / comment for sentiment analysis",
    "I need this loan to consolidate my debts and pay everything back on time.",
)

# =========================
# 5. Build model input row
# =========================

sentiment_label_num = get_sentiment_label(desc_text)
strong_red_flag = has_strong_red_flag(desc_text)
risky_flag = has_risky_flag(desc_text)

sentiment_word = (
    "positive" if sentiment_label_num == 1
    else "negative" if sentiment_label_num == -1
    else "neutral"
)

home_ownership_risk = home_ownership_mapping[home_ownership_text]

# Start from dataset medians
input_data = default_values.copy()

# Override with user inputs
input_data["id"] = 0  # dummy
input_data["loan_amnt"] = loan_amnt
input_data["term"] = float(term)
input_data["emp_length"] = emp_length_years
input_data["annual_inc"] = annual_inc
input_data["home_ownership_risk"] = home_ownership_risk
input_data["sentiment_label"] = sentiment_label_num

# Create DataFrame in correct order
input_df = pd.DataFrame([input_data])[feature_cols]

st.subheader("Input Summary (model-ready features)")
st.write(input_df)

st.markdown(
    f"**Detected sentiment from description:** {sentiment_word} (label = {sentiment_label_num})"
)

# =========================
# 6. Prediction (button)
# =========================

if st.button("Predict Loan Status"):
    # Base model prediction
    pred_class = model.predict(input_df)[0]
    prob_full = model.predict_proba(input_df)[0, 1]

    # Strong red flag override
    if strong_red_flag:
        pred_class = 0
        prob_full = min(prob_full, 0.10)
        st.warning(
            "🚨 **Strong red-flag phrase** detected in the description "
            "(e.g., clear refusal or inability to pay). "
            "For safety, this loan is treated as **high default risk**."
        )

    # Risky phrase override (softer cap)
    elif risky_flag:
        pred_class = 0
        prob_full = min(prob_full, 0.30)
        st.warning(
            "⚠️ **Risky phrase** detected in the description "
            "(e.g., 'lost my job', serious financial difficulty). "
            "This loan is treated as **default risk** even if other factors look good."
        )

    st.subheader("Prediction Result")
    if pred_class == 1:
        st.success("✅ The model predicts this loan is likely to be **Fully Paid (1)**.")
    else:
        st.error("⚠️ The model predicts this loan is likely to **Default / Charged Off (0)**.")

    st.write(f"**Probability of Fully Paid (class 1):** {prob_full:.3f}")
