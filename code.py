import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import os

# Load the model and tokenizer
model_dir = './distilbert_model'
st.write(f"Loading model from {model_dir}...")

try:
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    st.write("Model and tokenizer loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Define sentiment mapping
sentiment_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Define a function to predict sentiment
def predict_sentiment(text):
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class = logits.argmax().item()
    return sentiment_mapping[predicted_class]

# Streamlit UI
st.title("Sentiment Analysis App")
st.write("Enter some text to analyze its sentiment:")

user_input = st.text_area("Text Input")

if st.button("Analyze"):
    if user_input:
        sentiment = predict_sentiment(user_input)
        st.success(f"The predicted sentiment is: **{sentiment}**")
    else:
        st.error("Please enter some text for analysis.")
