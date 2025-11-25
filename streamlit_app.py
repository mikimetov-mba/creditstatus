import streamlit as st
import numpy as np
import pandas as pd
import joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# =========================
# Beautiful UI setup (colors like landing page)
# =========================

st.set_page_config(
    page_title="YES BANK – Online Loan Risk Checker",
    page_icon="💳",
    layout="wide",
)

st.markdown(
    """
<style>
/* Background gradient like online banking landing page */
.stApp {
    background: linear-gradient(135deg, #04c8ff 0%, #0072ff 40%, #f5f7fb 100%);
}

/* Main white card in the middle */
.main-card {
    background-color: rgba(255,255,255,0.98);
    border-radius: 26px;
    padding: 2rem 2.5rem;
    box-shadow: 0 20px 45px rgba(0,0,0,0.18);
    margin-top: 1.5rem;
}

/* Small inner cards */
.section-card {
    background-color: #ffffff;
    border-radius: 18px;
    padding: 1.1rem 1.4rem;
    box-shadow: 0 6px 18px rgba(0,0,0,0.06);
    margin-bottom: 1rem;
}

/* Header text styles */
.header-title {
    font-size: 2.0rem;
    font-weight: 800;
    color: #0b2e5b;
    margin-bottom: 0.4rem;
}
.header-subtitle {
    font-size: 0.95rem;
    color: #4b6a96;
}

/* Small badge "For individuals" style */
.hero-badge {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: #004a99;
    background: rgba(255,255,255,0.7);
    padding: 0.18rem 0.7rem;
    border-radius: 999px;
    display: inline-block;
}

/* Primary button style */
.stButton>button {
    background: #0052cc;
    color: white;
    border-radius: 999px;
    padding: 0.5rem 1.6rem;
    border: none;
    font-weight: 600;
}
.stButton>button:hover {
    background: #003c99;
}

/* Result & warning cards */
.card-warning {
    background-color: #FFF9E6;
    border-left: 5px solid #FFB300;
    border-radius: 14px;
    padding: 0.8rem 1.0rem;
    margin-top: 0.6rem;
}
.card-danger {
    background-color: #FFE9E9;
    border-left: 5px solid #E53935;
    border-radius: 14px;
    padding: 0.8rem 1.0rem;
    margin-top: 0.6rem;
}
.card-success {
    background-color: #E5F9EE;
    border-left: 5px solid #2E7D32;
    border-radius: 14px;
    padding: 0.8rem 1.0rem;
    margin-top: 0.6rem;
}

/* Sidebar background */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0072ff 0%, #0bb5ff 60%, #ffffff 100%);
    color: #ffffff;
}
section[data-testid="stSidebar"] h2, 
section[data-testid="stSidebar"] label {
    color: #ffffff;
}
</style>
""",
    unsafe_allow_html=True,
)

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
# 4. Layout: header + sidebar + main card
# =========================

# --- Top bar with "For individuals" and YES BANK logo ---
top_left, top_right = st.columns([4, 1])
with top_left:
    st.markdown('<span class="hero-badge">For individuals</span>', unsafe_allow_html=True)
with top_right:
    # make sure yes_bank_logo.png is in the same folder as app.py
    st.image("yes_bank_logo.png", use_column_width=True)

# --- Sidebar inputs ---
st.sidebar.header("Loan Application Inputs")

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
# 5. Main content card
# =========================

st.markdown('<div class="main-card">', unsafe_allow_html=True)

hero_left, hero_right = st.columns([2, 1.4])

with hero_left:
    st.markdown(
        '<div class="header-title">Online Banking Loan Risk Check</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="header-subtitle">'
        'YES BANK uses data & sentiment analysis to estimate the likelihood that a loan '
        'will be <b>Fully Paid</b> or <b>Default / Charged Off</b>. '
        'Adjust the inputs on the left and see how the risk changes.'
        '</div>',
        unsafe_allow_html=True,
    )

with hero_right:
    st.markdown(
        """
        <div class="section-card">
            <b>Why it matters?</b><br>
            • Faster credit decisions<br>
            • Combine numbers + text<br>
            • Highlight risky descriptions
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================
# 6. Build model input row
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

# --- Input summary section ---
st.markdown("### 🧾 Input Summary")
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.dataframe(input_df, use_container_width=True)
st.markdown(
    f"<br><b>Detected sentiment from description:</b> {sentiment_word} "
    f"(label = {sentiment_label_num})",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# =========================
# 7. Prediction (button + colored result cards)
# =========================

st.markdown("### 🔍 Prediction")

if st.button("Check Loan Status"):
    # Base model prediction
    pred_class = model.predict(input_df)[0]
    prob_full = model.predict_proba(input_df)[0, 1]

    # Strong red flag override
    if strong_red_flag:
        pred_class = 0
        prob_full = min(prob_full, 0.10)
        st.markdown(
            """
            <div class="card-danger">
            🚨 <b>Strong red-flag phrase detected in the description.</b><br>
            This loan is treated as <b>high default risk</b> even if numeric factors look good.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Risky phrase override (softer cap)
    elif risky_flag:
        pred_class = 0
        prob_full = min(prob_full, 0.30)
        st.markdown(
            """
            <div class="card-warning">
            ⚠️ <b>Risky phrase detected in the description</b> 
            (e.g. lost job, serious financial difficulty).<br>
            The model treats this loan as <b>default risk</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Final prediction card
    if pred_class == 1:
        st.markdown(
            f"""
            <div class="card-success">
            ✅ <b>Prediction:</b> This loan is likely to be <b>Fully Paid (1)</b>.<br>
            <b>Estimated probability of Fully Paid:</b> {prob_full:.1%}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="card-danger">
            ⚠️ <b>Prediction:</b> This loan is likely to <b>Default / Charged Off (0)</b>.<br>
            <b>Estimated probability of Fully Paid:</b> {prob_full:.1%}
            </div>
            """,
            unsafe_allow_html=True,
        )

# close main-card div
st.markdown("</div>", unsafe_allow_html=True)
