import streamlit as st
import requests

API_URL = "http://api:8000/predict-spoofing"

st.title("GNSS Spoofing Detector")
 
uploaded_file = st.file_uploader("Upload a CSV file")

button = st.button("Predict")

if uploaded_file and button:

    files = {"file": uploaded_file.getvalue()}

    response = requests.post(
        API_URL, 
        files={"file": uploaded_file}
    )


    if response.status_code == 200:
        st.write(response.json())
    else:
        st.error(f"API Error {response.status_code}")
        st.text(response.text)
   