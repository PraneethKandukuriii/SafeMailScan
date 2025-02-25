import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Load trained model and vectorizer
model = pickle.load(open("spam_classifier_model.pkl", "rb"))
vectorizer = pickle.load(open("model.pkl", "rb"))

# Streamlit UI
st.title("📧 Spam Mail Classifier (Safe Mail Scan)")
st.write("Enter an email message below to check if it's spam or not.")

# Input field
user_input = st.text_area("Type your email message here:")

if st.button("Check Spam"):  
    if user_input:
        # Preprocess input
        input_features = vectorizer.transform([user_input])
        prediction = model.predict(input_features)
        
        # Display result
        if prediction[0] == 0:
            st.error("🚨 This is a SPAM email!")
        else:
            st.success("✅ This is NOT a spam email.")
    else:
        st.warning("Please enter an email message.")