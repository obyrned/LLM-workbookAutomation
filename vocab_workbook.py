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
        Extract a list of 20 unique and challenging vocabulary words that explicitly appear in the provided text. For each word:

        1. Confirm that the word exists in the text by providing the exact sentence or passage where it appears.
        2. Provide one sentence before the word for context.
        3. Provide one sentence after the word for context.
        4. Ensure each word is distinct (no repetitions of any kind).
        5. If fewer than 20 such words can be found, provide as many unique, challenging words as possible without repeating.

        Text:
        {text}

        Format your response as a JSON list, where each item is a dictionary with the following keys:
        - "word": The vocabulary word.
        - "preceding_sentence": The sentence before the word.
        - "sentence_with_word": The sentence containing the word.
        - "following_sentence": The sentence after the word.

        Important:
        - Do not infer, fabricate, or imply words or their contexts—only include words verifiably present in the text.
        - Ensure the response is a valid JSON list.
        """,
        "stream": False,
        "format": "json"
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        raw_response = response.json().get("response", "{}")
        st.write("### Debug: Raw Model Response")
        st.code(raw_response)

        # Parse and normalize JSON response
        vocab_output = json.loads(raw_response)

        if isinstance(vocab_output, dict):  # Normalize a single dictionary to a list
            vocab_output = [vocab_output]
        elif not isinstance(vocab_output, list):  # If not a list, raise an error
            st.error("The response is not a valid JSON list. Please check the model output.")
            return None

        valid_entries = []
        for entry in vocab_output:
            if all(key in entry for key in ["word", "preceding_sentence", "sentence_with_word", "following_sentence"]):
                valid_entries.append(entry)
            else:
                st.warning(f"Skipping invalid entry: {entry}")

        return valid_entries

    except requests.exceptions.RequestException as e:
        st.error(f"Error contacting the model API: {e}")
    except json.JSONDecodeError:
        st.error("Failed to parse the model's response. Ensure the response is valid JSON.")
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
            st.success(f"Generated {len(vocab_output)} vocabulary words!")
            st.write("### Vocabulary Words:")
            for entry in vocab_output:
                st.write(f"**{entry['word']}**")
                st.write(f"- **Preceding Sentence**: {entry['preceding_sentence']}")
                st.write(f"- **Sentence with Word**: {entry['sentence_with_word']}")
                st.write(f"- **Following Sentence**: {entry['following_sentence']}")
                st.write("---")
        else:
            st.error("No vocabulary words were generated. Please check the input text or the model configuration.")

if __name__ == "__main__":
    main()
