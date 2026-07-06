import streamlit as st
import joblib
import pandas as pd

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="📊",
    layout="wide"
)

# =========================
# CUSTOM UI STYLING
# =========================
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}

.block-container {
    padding-top: 2rem;
}

.title {
    font-size: 38px;
    font-weight: bold;
    text-align: center;
    color: #ffffff;
}

.subtitle {
    text-align: center;
    color: #a0a0a0;
    margin-bottom: 25px;
}

.card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("<div class='title'>📊 Sentiment Analysis Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Analyze single reviews or upload CSV for batch predictions</div>", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
model = joblib.load("models/logistic_regression_model.pkl")

label_map = {0: "negative", 1: "neutral", 2: "positive"}

# =========================
# LAYOUT
# =========================
col1, col2 = st.columns(2)

# =========================
# SINGLE PREDICTION
# =========================
with col1:
    st.subheader("✍️ Single Review Analysis")

    user_input = st.text_area("Enter a review:")

    if st.button("Predict Sentiment"):
        if user_input.strip() == "":
            st.warning("Please enter a review.")
        else:
            transformed_text = vectorizer.transform([user_input])
            prediction = model.predict(transformed_text)[0]
            prediction_label = label_map[prediction]

            st.markdown("### Result")

            if prediction_label == "positive":
                st.success(f"😊 Positive Sentiment")
            elif prediction_label == "negative":
                st.error(f"😡 Negative Sentiment")
            else:
                st.info(f"😐 Neutral Sentiment")

# =========================
# BATCH PREDICTION
# =========================
with col2:
    st.subheader("📁 Batch Prediction (CSV Upload)")

    uploaded_file = st.file_uploader("Upload CSV file with 'review' column", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if "review" in df.columns:
            df["prediction"] = df["review"].apply(
                lambda x: label_map[model.predict(vectorizer.transform([x]))[0]]
            )

            st.markdown("### Preview")
            st.dataframe(df.head())

            # Download button
            st.download_button(
                "📥 Download Predictions",
                df.to_csv(index=False),
                "sentiment_predictions.csv",
                "text/csv"
            )

            # =========================
            # SUMMARY CHART
            # =========================
            st.markdown("### 📊 Sentiment Distribution")

            counts = df["prediction"].value_counts()
            st.bar_chart(counts)

        else:
            st.error("CSV must contain a column named 'review'")

# =========================
# FOOTER BRANDING PLACEHOLDER
# =========================
st.markdown("---")

col1, col2, col3 = st.columns([1, 6, 2])

with col1:
    st.image("assets/profile.png", width=60)

with col2:
    st.markdown(
        """
        ### Emmanuel Effiong  
        Data Science & NLP Enthusiast  

        🔗 [View Profile](https://github.com/emmanumo/Sentiment-Analysis-App)
        """
    )

with col3:
    st.image("assets/app_preview.png", width=100)