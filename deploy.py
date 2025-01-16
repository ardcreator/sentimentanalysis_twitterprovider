# Import Library
# Import Library
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import re
import os

from wordcloud import WordCloud
from plotly import graph_objs as go
import plotly.express as px

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from textblob import Word
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
    "About Us",
    "How to Use?",
    "Sentiment Analyze",
    "Sentiment Prediction"
]
section = st.sidebar.radio("Go to Section", nav_options)

# Section About
if section == "About Us":
    st.markdown("""<hr style="border: 2px solid #00b6ff; border-radius: 5px;">""", unsafe_allow_html=True)
    st.subheader("ℹ️ About Us")
    st.markdown("""
    ### Team 5 Members in [MIKTI](https://mikti.id):
    - **Leader**: [MUH. ASHARI RASYID](https://www.linkedin.com/in/ardcreator/)
    - **Members**:
        - [Mega Febriani](https://www.linkedin.com/in/megafebriani-528915326/)
        - [Sri Agustin](https://www.linkedin.com/in/sriagustin/)
        - Muh. Fikri Firman
        - Rifky Agung Chrisatya Ntjali
    **Class**: Data Analyst 1 MIKTI
    **Mentor**: [Fadhan Adha](https://www.linkedin.com/in/fadhlan-adha/)
    ### Project Overview:
    Welcome to Team 5's Sentiment Analysis Project! 🌟 Our team, made up of 5 passionate members from the Data Analyst-1 class at MIKTI, is diving deep into the world of social media to understand public sentiment.
    ### Objective:
    In this project, our mission is to uncover hidden insights from millions of Twitter posts by performing Sentiment Analysis. By classifying tweets as positive, negative, or neutral, we aim to shed light on how users feel about various cellular services. Whether it's customer satisfaction, issues, or praise, our analysis will provide valuable insights into the telecommunications industry.
    """)

elif section == "How to Use?":
    st.markdown("""<hr style="border: 2px solid #00b6ff; border-radius: 5px;">""", unsafe_allow_html=True)
    st.subheader("💡 How to Use?")
    st.markdown("""
    ### How to Use this Application?
    1. **View Sentiment Overview**: Understand the general sentiment on Twitter regarding cellular service providers by exploring word frequencies and sentiment distributions.
    2. **Explore Sentiment Visualizations**: Visualize the sentiment breakdown (positive, negative, neutral) and see which words are frequently associated with each sentiment.
    3. **Predict Sentiment**:
        - **From Text**: Enter a sentence or a short paragraph in Indonesian, and the app will predict whether the sentiment is positive, negative, or neutral.
        - **From File**: Upload a `.txt` file containing multiple sentences, and the app will predict the sentiment for each sentence inside the file.
    4. **Interactive Results**: View the predicted sentiment for each input and explore detailed charts and visualizations based on the data analysis.
    """)

# Read Data
clean_data = pd.read_csv('data/twittercellular-clean-sentiment.csv')

# Section Sentiment Analyze (Gabungan dari Sentiment Overview, Sentiment Distribution, dan Sentiment Visualization)
if section == "Sentiment Analyze":
    st.markdown("""<hr style="border: 2px solid #00b6ff; border-radius: 5px;">""", unsafe_allow_html=True)
    st.subheader("📊 Sentiment Analyze")

    # 1. Sentiment Overview
    st.markdown("### 1. Sentiment Overview")
    freq = pd.Series(' '.join(clean_data['Text Tweet']).split()).value_counts()
    head_freq = freq.head(20)
    fig1 = px.bar(
        head_freq, x=head_freq.index, y=head_freq.values,
        labels={'x': 'Words', 'y': 'Frequency'},
        title="Top 20 Words by Frequency"
    )
    fig1.update_layout(
        title_font_size=20, xaxis_tickangle=-45, template="plotly_dark"
    )
    st.plotly_chart(fig1, use_container_width=True)

    # 2. Sentiment Distribution
    st.markdown("### 2. Sentiment Distribution")
    temp = clean_data.groupby('Sentiment').count()['Text Tweet'].reset_index().sort_values(by='Text Tweet', ascending=False)
    col1, col2 = st.columns(2)
    with col1:
        fig2 = px.bar(temp, x='Sentiment', y='Text Tweet', color='Sentiment',
                    title="Bar Chart of Sentiment Distribution",
                    labels={'Text Tweet': 'Count'})
        fig2.update_layout(template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        fig3 = px.pie(temp, values='Text Tweet', names='Sentiment', hole=0.4,
                    title="Sentiment Proportion")
        fig3.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig3, use_container_width=True)

    # 3. Sentiment Visualization
    for sentiment_label, emoji in zip(["Positif", "Negatif", "Netral"], ["\U0001F600", "\U0001F641", "\U0001F610"]):
        st.subheader(f"{emoji} > {sentiment_label} Sentiment Visualization")
        df_sent = clean_data[clean_data['Sentiment'] == sentiment_label]
        word_freq = pd.Series(' '.join(df_sent['Text Tweet']).split()).value_counts()
        top_words = word_freq.head(10)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(top_words, y=top_words.index, x=top_words.values, orientation='h',
                        labels={'x': 'Frequency', 'y': 'Words'},
                        title=f"Top 10 Words in {sentiment_label} Sentiment")
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            wordcloud = WordCloud(width=500, height=300, random_state=21, max_font_size=110).generate(' '.join(df_sent['Text Tweet']))
            fig_cloud, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis('off')
            ax.set_title(f"Word Cloud for {sentiment_label} Sentiment", fontsize=16)
            st.pyplot(fig_cloud)

# Section Sentiment Prediction
elif section == "Sentiment Prediction":
    st.markdown("""<hr style="border: 2px solid #00b6ff; border-radius: 5px;">""", unsafe_allow_html=True)

    # Sentiment Prediction Logic
    # Initialize Sastrawi Stemmer
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stop_words = set(stopwords.words('indonesian') + stopwords.words('english') + ["v", "h", "gak", "deh", "kok", "ga", "cug", "ya", "kah", "sih", "ht", "noh", "thu", "lho", "pejet", "dn", "cie", "dong", "aja", "itu", "hadeh", "si", "yg", "yah", "tuh", "nih", "d", "hal", "sy", "dr", "th", "lgsg", "jgn", "dgn", "krn", "yaa", "jabo", "tm", "tp", "hooq", "nya", "an", "oh", "jd", "g", "rb", "rt", "gb", "glte", "gnya", "lte", "pki", "j", "rp", "dg", "duh", "yuk", "js"])
    lemmatization_dictionary = {
        "muas": "puas",
        "lemot": "lambat",
        "lot": "lambat",
        "terimakasih": "terima kasih",
        "riah": "meriah",
    }
    def correct_lemmatization(text):
        for typo, correct_word in lemmatization_dictionary.items():
            text = text.replace(typo, correct_word)
        return text
    def preprocess_text(text):
        if not isinstance(text, str):
            raise ValueError("Input to preprocess_text must be a string.")
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[^a-z\s]', '', text)
        text = text.lower()
        words = [word for word in text.split() if word not in stop_words]
        lemmatized_words = []
        for word in words:
            if Word(word).spellcheck()[0][1] == 1.0:
                lemmatized_words.append(Word(word).lemmatize())
            else:
                lemmatized_words.append(stemmer.stem(word))
        text = ' '.join(lemmatized_words)
        text = correct_lemmatization(text)
        return text

    model = joblib.load('model/logistic_regression_CVModel.pkl')

    st.markdown("""### 🔮 Sentiment Prediction System (Works only in Indonesian Sentences)""")
    new_text = st.text_input("Enter text for sentiment prediction:")

    if st.button("Predict Sentiment"):
        if not new_text.strip():
            st.warning("Please enter text to predict.")
        else:
            try:
                preprocessed_text = preprocess_text(new_text)
                prediction = model.predict([preprocessed_text])
                st.success(f"Predicted Sentiment: {prediction[0]}")
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
                        prediction = model.predict([preprocessed_text])[0]
                        results.append((idx, sentence, prediction))

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
        Copyright © {current_year} | Apps Created by <b><a href="https://bio.asharirasyid.my.id" target="_blank">Ashari Rasyid</a></b>
    </div>
""", unsafe_allow_html=True)
