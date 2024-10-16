import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Load the model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained('./distilbert_model')
tokenizer = DistilBertTokenizer.from_pretrained('./distilbert_model')

# Define sentiment mapping
sentiment_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}

def predict_sentiment(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()
    
    return sentiment_mapping[predicted_class]

# Streamlit app
st.title("Sentiment Analysis with DistilBERT")
st.write("Enter text for sentiment analysis:")

# Text input from user
user_input = st.text_area("Text Input")

if st.button("Predict"):
    if user_input:
        sentiment = predict_sentiment(user_input)
        st.write(f"Predicted Sentiment: **{sentiment}**")
    else:
        st.write("Please enter some text to analyze.")

