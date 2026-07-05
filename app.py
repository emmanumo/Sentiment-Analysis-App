import streamlit as st
import joblib

# Load model and vectorizer
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
model = joblib.load("models/logistic_regression_model.pkl")

# App title
st.title("📊 Sentiment Analysis App")
st.write("Enter a review and get predicted sentiment (Positive / Negative / Neutral)")

# User input
user_input = st.text_area("Enter your review here:")

# Predict button
if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        # Transform text
        transformed_text = vectorizer.transform([user_input])
        label_map = {0: "negative",1: "neutral",2: "positive"}
        # Predict
        prediction = model.predict(transformed_text)[0]
        prediction_label = label_map[prediction]
        # Show result
        if prediction_label == "positive":
            st.success(f"Sentiment: {prediction_label}")
        elif prediction_label == "negative":
            st.error(f"Sentiment: {prediction_label}")
        else:
            st.info(f"Sentiment: {prediction_label}")