import streamlit as st
import requests


st.title("Music Genre Prediction")

uploaded_file = st.file_uploader("Upload an audio file (wav or mp3)", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    if st.button("Predict Genre"):
        response = requests.post("http://localhost:5000/predict", files={"file": uploaded_file})
        if response.status_code == 200:
            data = response.json()
            st.write(f"Predicted Genre: {data['genre']}")
            st.write(f"Probabilities: {data['probabilities']}")
        else:
            st.error("Error: Could not get prediction")

