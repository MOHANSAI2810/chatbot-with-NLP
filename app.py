import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from textblob import TextBlob

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from the JSON file
file_path = os.path.abspath("./intents.json")
try:
    with open(file_path, "r") as file:
        intents = json.load(file)
except FileNotFoundError:
    st.error("The intents.json file was not found. Please check the path.")
    st.stop()
except json.JSONDecodeError:
    st.error("Error decoding the JSON file. Please ensure the file is correctly formatted.")
    st.stop()

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
def preprocess_data(intents):
    tags, patterns = [], []
    for intent in intents:
        for pattern in intent['patterns']:
            tags.append(intent['tag'])
            patterns.append(pattern)
    return patterns, tags

patterns, tags = preprocess_data(intents)
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# Function to retrain the model
def retrain_model(intents):
    global vectorizer, clf
    patterns, tags = preprocess_data(intents)
    x = vectorizer.fit_transform(patterns)
    y = tags
    clf.fit(x, y)

# Chatbot response generation
def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

# Sentiment analysis
def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# Main app logic
def main():
    st.title("Chatbot Using NLP")

    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.write("Welcome! Start chatting below:")

        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp', 'Sentiment'])

        user_input = st.text_input("You:")

        if user_input:
            response = chatbot(user_input)
            sentiment = analyze_sentiment(user_input)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input, response, timestamp, sentiment])

            st.text_area("Chatbot:", value=response, height=120)

    elif choice == "Conversation History":
        st.header("Conversation History")
        if os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)  # Skip header row
                for row in csv_reader:
                    st.text(f"User: {row[0]}")
                    st.text(f"Chatbot: {row[1]}")
                    st.text(f"Timestamp: {row[2]}")
                    st.text(f"Sentiment: {row[3]}")
                    st.markdown("---")
        else:
            st.write("No conversation history found.")

    elif choice == "About":
        st.subheader("Project Overview")
        st.write(
            """
            This chatbot is built using Natural Language Processing (NLP) and Logistic Regression. 
            The chatbot classifies user inputs into predefined intents and provides appropriate responses. 
            It features:
            - NLP preprocessing with TF-IDF vectorization
            - Logistic Regression for intent classification
            - Interactive UI built with Streamlit
            - Sentiment analysis for user input
            """
        )

if __name__ == "__main__":
    main()
