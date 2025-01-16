# Import Library
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import re

from wordcloud import WordCloud
from plotly import graph_objs as go
import plotly.express as px

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('movie_reviews') 
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import sklearn

from datetime import datetime
current_year = datetime.now().year

# ------------------------
st.set_page_config(
    page_title="Sentiment Analysis on Twitter Data for Cellular Service Providers | Team 5 MIKTI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Display the logo
st.image("https://mikti.id/assets/images/resources/logo-1.png", width=250)

# Title for the app
st.title("🔍 Sentiment Analysis on Twitter Data for Cellular Service Providers 🇮🇩")

# Sidebar Navigation
st.sidebar.title("Navigation")
nav_options = [
    "About",
    "How to Use?",
    "Sentiment Analyze",
    "Sentiment Prediction"
]
section = st.sidebar.radio("Go to Section", nav_options)

# Section About
if section == "About":
    st.markdown("""<hr style="border: 2px solid #00b6ff; border-radius: 5px;">""", unsafe_allow_html=True)
    st.subheader("ℹ️ About")
    st.markdown(""" 
    ### Project Overview:
    Welcome to Team 5's Sentiment Analysis Project! 🌟 Our team, made up of 5 passionate members from the Data Analyst-1 class at MIKTI, is diving deep into the world of social media to understand public sentiment.
    ### Objective:
    In this project, our mission is to uncover hidden insights from millions of Twitter posts by performing Sentiment Analysis. By classifying tweets as positive, negative, or neutral, we aim to shed light on how users feel about various cellular services. Whether it's customer satisfaction, issues, or praise, our analysis will provide valuable insights into the telecommunications industry.
    """)

# Sentiment Prediction Logic with VADER Sentiment
elif section == "Sentiment Prediction":
    st.markdown("""<hr style="border: 2px solid #00b6ff; border-radius: 5px;">""", unsafe_allow_html=True)

    # Initialize VADER Sentiment Intensity Analyzer
    sia = SentimentIntensityAnalyzer()

    st.markdown("""### 🔮 Sentiment Prediction System (Works only in Indonesian Sentences)""")
    new_text = st.text_input("Enter text for sentiment prediction:")

    if st.button("Predict Sentiment"):
        if not new_text.strip():
            st.warning("Please enter text to predict.")
        else:
            try:
                # Preprocess the text (you can keep your preprocessing function as is)
                preprocessed_text = preprocess_text(new_text)
                
                # Predict Sentiment using VADER
                sentiment_score = sia.polarity_scores(preprocessed_text)
                compound_score = sentiment_score['compound']

                # Determine Sentiment based on compound score
                if compound_score >= 0.05:
                    prediction = 'Positive'
                elif compound_score <= -0.05:
                    prediction = 'Negative'
                else:
                    prediction = 'Neutral'

                st.success(f"Predicted Sentiment: {prediction}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")

    st.markdown("""### 📂 Predict Sentiment from File (Works only in Indonesian Sentences)""")
    uploaded_file = st.file_uploader("Upload a text file (.txt):", type="txt")

    if st.button("Predict Sentiment from File"):
        if not uploaded_file:
            st.warning("Please upload a file.")
        else:
            file_content = uploaded_file.read().decode("utf-8")
            sentences = file_content.splitlines()

            if not sentences or all(s.strip() == "" for s in sentences):
                st.error("The file is empty or does not contain valid text.")
            else:
                results = []
                for idx, sentence in enumerate(sentences, start=1):
                    if sentence.strip():
                        preprocessed_text = preprocess_text(sentence)
                        sentiment_score = sia.polarity_scores(preprocessed_text)
                        compound_score = sentiment_score['compound']

                        if compound_score >= 0.05:
                            sentiment = 'Positive'
                        elif compound_score <= -0.05:
                            sentiment = 'Negative'
                        else:
                            sentiment = 'Neutral'

                        results.append((idx, sentence, sentiment))

                st.subheader("Prediction Results:")
                for idx, sentence, sentiment in results:
                    st.write(f"**Sentence {idx}:** {sentence}")
                    st.write(f"**!> Predicted Sentiment**: {sentiment}")
                    st.markdown("---")

# ------------------------
# Add Fixed Footer with dynamic year
st.markdown(f"""
    <style>
    .footer {{
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #333;
        color: white;
        text-align: center;
        padding: 10px 0;
        font-size: 14px;
    }}
    .footer a {{
        color: #00b6ff;
        text-decoration: none;
    }}
    .footer a:hover {{
        text-decoration: underline;
    }}
    </style>
    <div class="footer">
        Copyright © {current_year} All rights reserved | Application Created by <b><a href="https://bio.asharirasyid.my.id" target="_blank">Ashari Rasyid</a></b>
    </div>
""", unsafe_allow_html=True)
