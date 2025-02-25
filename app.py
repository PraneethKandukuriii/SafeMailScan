import streamlit as st
import pickle
import numpy as np

# Load trained model and vectorizer
try:
    model = pickle.load(open("spam_classifier_model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))  # Ensure this file exists
except FileNotFoundError:
    st.error("Model or vectorizer file not found. Please check file paths.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.title("📧 Spam Mail Classifier (Safe Mail Scan)")
st.write("Enter an email message below to check if it's spam or not.")

user_input = st.text_area("Type your email message here:")

if st.button("Check Spam"):  
    if user_input:
        try:
            input_features = vectorizer.transform([user_input])
            prediction = model.predict(input_features)
            
            if prediction[0] == 0:
                st.error("🚨 This is a SPAM email!")
            else:
                st.success("✅ This is NOT a spam email.")
        except Exception as e:
            st.error(f"Error processing input: {e}")
    else:
        st.warning("Please enter an email message.")