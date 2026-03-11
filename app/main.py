import streamlit as st
import requests

st.title("GNSS Spoofing Detector")

uploaded_file = st.file_uploader("Upload a CSV file")

if uploaded_file:

    files = {"file": uploaded_file.getvalue()}

    response = requests.post(
        "http://api:8000/predict-spoofing",
        files={"file": uploaded_file}
    )
    
    st.write(response.json())
