import requests
import json
import streamlit as st

# Ollama API endpoint
OLLAMA_URL = "http://localhost:11434/api/generate"

# Function to call the Ollama API
def get_vocab_words(text):
    payload = {
        "model": "deepseek-r1:8b",
        "prompt": f"""
        Analyze the following text and extract 20 vocabulary words suitable for young students learning English. 
        For each word, provide:
        1. The word.
        2. A simple definition.
        3. One sentence preceding the word.
        4. The sentence containing the word.
        5. One sentence following the word.
        Ensure the words do not repeat.

        Text:
        {text}
        """,
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=payload)
    if response.status_code == 200:
        return response.json()["response"]
    else:
        st.error("Failed to get response from the model.")
        return None

# Streamlit app
def main():
    st.title("Vocabulary Workbook Generator")
    st.write("Upload a text file containing the lesson, and we'll generate vocabulary words for your workbook.")

    # File upload
    uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
    if uploaded_file is not None:
        text = uploaded_file.read().decode("utf-8")
        st.write("File uploaded successfully!")

        # Progress bar
        with st.spinner("Generating vocabulary words..."):
            vocab_output = get_vocab_words(text)
        
        if vocab_output:
            st.success("Vocabulary words generated!")
            st.write("### Vocabulary Words:")
            st.text(vocab_output)  # Display the raw output for now
        else:
            st.error("No vocabulary words were generated.")

if __name__ == "__main__":
    main()