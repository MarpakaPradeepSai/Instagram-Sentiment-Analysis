import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

@st.cache(allow_output_mutation=True)
def load_model():
    model = DistilBertForSequenceClassification.from_pretrained('distilbert_model')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert_model')
    return model, tokenizer

model, tokenizer = load_model()

st.title("Sentiment Analysis with DistilBERT")

input_text = st.text_area("Enter text to analyze")

if st.button("Analyze"):
    with torch.no_grad():
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()

    sentiment = "Positive" if prediction == 2 else "Neutral" if prediction == 1 else "Negative"
    st.write(f"Sentiment: {sentiment}")
