import streamlit as st
import requests

API_URL = "http://localhost:8000/analyze"

st.set_page_config(page_title="Customer Support AI", page_icon="🤖", layout="wide")

st.title("🤖 AI Customer Support Assistant")
st.write("Enter a customer message and the system will analyze sentiment, severity, intent, and generate a response.")

text = st.text_area("Customer Message:", height=200)

if st.button("Analyze"):
    if not text.strip():
        st.error("Please enter text.")
    else:
        with st.spinner("Analyzing..."):
            response = requests.post(API_URL, json={"text": text})

        if response.status_code != 200:
            st.error(f"API Error: {response.text}")
        else:
            data = response.json()

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Sentiment", data["sentiment"])

            with col2:
                st.metric("Severity", data["severity"])

            with col3:
                st.metric("Intent", data["intent"])

            st.subheader("💬 AI Support Response")
            st.write(data["response"])
