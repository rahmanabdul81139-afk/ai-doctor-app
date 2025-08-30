import streamlit as st
import joblib

# Load saved model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Streamlit App UI
st.title("🩺 AI Doctor - Symptom Based Disease Prediction")
st.write("Enter your symptoms below, separated by commas (e.g. fever, cough, sore throat)")

# User input
user_input = st.text_input("Symptoms:")

if st.button("Predict Disease"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some symptoms.")
    else:
        # Transform input using vectorizer
        vect = vectorizer.transform([user_input])
        # Predict disease
        prediction = model.predict(vect)[0]
        st.success(f"🧠 Predicted Disease: **{prediction}**")
