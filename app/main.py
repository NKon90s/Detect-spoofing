import streamlit as st
import requests

st.title("GNSS Spoofing Detector")

uploaded_file = st.file_uploader("Upload a CSV file")

button = st.button("Predict")

if uploaded_file and button:

    files = {"file": uploaded_file.getvalue()}

    response = requests.post(
        "http://127.0.0.1:8000/predict-spoofing", 
        files={"file": uploaded_file}
    )
 
    st.write(response.json()) 
   