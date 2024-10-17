import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Load the model and tokenizer
model_path = './distilbert_model'  # Change this if needed
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

# Define sentiment mapping
sentiment_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Streamlit app title
st.title("Sentiment Analysis with DistilBERT")

# Input text
user_input = st.text_area("Enter your text for sentiment analysis:")

if st.button("Analyze"):
    if user_input:
        # Tokenize and encode the input
        inputs = tokenizer(user_input, return_tensors='pt', truncation=True, padding=True, max_length=512)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
        
        # Display the result
        sentiment = sentiment_mapping[predicted_class]
        st.success(f"The sentiment is: **{sentiment}**")
    else:
        st.warning("Please enter some text for analysis.")

# Run the app with `streamlit run app.py`
