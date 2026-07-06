import streamlit as st
import joblib
import pandas as pd

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="ShopEase Sentiment Analyzer",
    page_icon="📊",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
model = joblib.load("models/logistic_regression_model.pkl")

label_map = {0: "negative", 1: "neutral", 2: "positive"}

# =========================
# TITLE
# =========================
st.title("ShopEase Europe — Real-Time Sentiment Analyser")
st.write(
    "Classify customer reviews instantly using the trained ML pipeline. "
    "Use Single Review or Batch Upload."
)

# =========================
# SUCCESS BANNER
# =========================
st.success("Model loaded successfully.")

# =========================
# TABS
# =========================
tab1, tab2, tab3 = st.tabs(["Single Review", "Batch Upload", "About"])

# =========================
# SINGLE REVIEW
# =========================
with tab1:
    st.subheader("Analyse a Single Review")

    user_input = st.text_area(
        "Paste a customer review below",
        placeholder="e.g. The item arrived broken and customer service refused to help."
    )

    if st.button("Analyse Sentiment"):
        if user_input.strip() == "":
            st.warning("Please enter a review.")
        else:
            X = vectorizer.transform([user_input])
            pred = model.predict(X)[0]
            label = label_map[pred]

            if label == "positive":
                st.success(f"😊 Positive Sentiment")
            elif label == "negative":
                st.error(f"😡 Negative Sentiment")
            else:
                st.info(f"😐 Neutral Sentiment")

# =========================
# BATCH UPLOAD
# =========================
with tab2:
    st.subheader("Batch Prediction")

    file = st.file_uploader("Upload CSV with 'review' column", type=["csv"])

    if file:
        df = pd.read_csv(file)

        if "review" in df.columns:
            df["prediction"] = df["review"].apply(
                lambda x: label_map[model.predict(vectorizer.transform([x]))[0]]
            )

            st.dataframe(df.head())

            st.download_button(
                "Download Results",
                df.to_csv(index=False),
                "predictions.csv",
                "text/csv"
            )

            st.bar_chart(df["prediction"].value_counts())

        else:
            st.error("CSV must contain 'review' column")

# =========================
# ABOUT TAB
# =========================
with tab3:
    st.subheader("About this App")

    st.write("""
    This sentiment analysis tool is built using:
    - TF-IDF Vectorizer
    - Logistic Regression Model
    - Streamlit for deployment

    It classifies customer reviews into Positive, Negative, or Neutral.
    """)

# =========================
# FLOATING PROFILE (BOTTOM RIGHT)
# =========================
st.markdown("""
<style>
#profile {
    position: fixed;
    bottom: 20px;
    right: 20px;
    border-radius: 50%;
}
</style>
""", unsafe_allow_html=True)

try:
    st.image("assets/profile.png", width=60)
except:
    st.write("")