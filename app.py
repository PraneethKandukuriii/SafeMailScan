import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("spam_classifier_model.pkl", "rb"))

st.title("📧 Spam Mail Classifier (Safe Mail Scan)")
st.write("Enter an email message below to check if it's spam or not.")

user_input = st.text_area("Type your email message here:")

if st.button("Check Spam"):  
    if user_input:
        # Convert input into a NumPy array if needed
        input_features = np.array([user_input])  # Adjust based on your model requirements
        prediction = model.predict(input_features)
        
        if prediction[0] == 0:
            st.error("🚨 This is a SPAM email!")
        else:
            st.success("✅ This is NOT a spam email.")
    else:
        st.warning("Please enter an email message.")