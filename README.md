# 📊 ShopEase Europe Customer Review Sentiment Analysis

## Project Overview

This project develops an end-to-end sentiment analysis system for customer reviews collected from ShopEase Europe. The aim is to transform unstructured customer feedback into actionable business insights using Natural Language Processing (NLP), Machine Learning, Tableau dashboards, and a Streamlit web application.

The system classifies customer reviews into **Positive**, **Neutral**, and **Negative** sentiments, enabling organisations to better understand customer satisfaction and identify operational issues.

---

## Objectives

- Develop a text-processing pipeline to clean and normalize multilingual customer reviews.
- Build a sentiment classification model using both classical and transformer-based algorithms.
- Identify the most common sentiment drivers (e.g., delivery time, product quality, price fairness).
- Generate concise summaries of customer opinions and recommendations for each product category.
- Visualize insights in an interactive dashboard for the marketing and operations teams.
- Generate business recommendations from customer feedback.



---

## Dataset

The dataset contains customer reviews with the following features:

- Product Category
- Review Text
- Rating
- Country
- Timestamp
- Language
- Sentiment

---

## Project Workflow

```
Customer Reviews
        │
        ▼
Data Cleaning & Preprocessing
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Text Processing (TF-IDF)
        │
        ▼
Model Development
(Naïve Bayes, Logistic Regression, DistilBERT)
        │
        ▼
Model Evaluation
        │
        ▼
Business Insights
        │
        ▼
Tableau Dashboard
        │
        ▼
Streamlit Deployment
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Joblib
- Streamlit
- Tableau
- Git
- GitHub

---

## Machine Learning Models

- Naïve Bayes
- Logistic Regression
- DistilBERT

The Logistic Regression model was selected for deployment because it provided an excellent balance between prediction accuracy and computational efficiency.

---

## Features

- Single Review Prediction
- Batch CSV Prediction
- Download Prediction Results
- Interactive Dashboard
- Real-time Sentiment Classification

---

## Tableau Dashboard

The Tableau dashboard includes:

- Executive Overview
- Product Category Analysis
- Country Analysis
- Review Analysis
- Model Performance

---

## Streamlit Application

The deployed application allows users to:

- Analyse individual customer reviews
- Upload CSV files for batch predictions
- Download prediction results
- Visualise sentiment distribution

---

## Installation

```bash
git clone https://github.com/emmanumo/Sentiment-Analysis-App.git

cd Sentiment-Analysis-App

pip install -r requirements.txt

streamlit run app.py
```

---

## Repository Structure

```
Sentiment-Analysis-App
│
├── app.py
├── requirements.txt
├── README.md
├── models/
├── assets/
├── notebooks/
├── src/
└── data/
```

---

## Author

**Emmanuel Effiong**

Data Science & Machine Learning Enthusiast

GitHub:
https://github.com/emmanumo
https://sentiment-analysis-app-965pwtzcejbpq6m5sa22gr.streamlit.app/
---

## Future Improvements

- Deploy DistilBERT model
- Add confidence scores
- Connect to live customer reviews database
- Automated model retraining
