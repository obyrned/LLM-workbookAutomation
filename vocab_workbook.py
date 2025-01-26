import streamlit as st
import requests
import json
import re

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "deepseek-r1:8b"

###############################################################################
# 1. Text Chunking
###############################################################################
def chunk_text(text, max_chars=6000):
    """
    Splits the text into chunks of approximately `max_chars` characters each.
    This helps prevent the model from being overwhelmed by large inputs.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start_idx = 0
    while start_idx < len(text):
        end_idx = min(start_idx + max_chars, len(text))
        chunk = text[start_idx:end_idx]
        chunks.append(chunk)
        start_idx = end_idx
    return chunks

###############################################################################
# 2. Single-Chunk Processing
###############################################################################
def extract_vocab_from_chunk(chunk):
    """
    Sends one chunk of text to the Ollama model, asking for:
      - Up to 5 unique vocabulary words
      - Combined quote with highlighted word
    Returns a list of dict entries or None on error.
    """

    prompt = f"""
You are an advanced text analysis assistant. Return your entire output ONLY as JSON.

Instructions:
1. You have a chunk of text below.
2. Identify up to 5 unique and challenging vocabulary words found in this chunk.
3. For each word, provide:
   - "word": the exact word as it appears in the text
   - "quote": a single quote combining the preceding sentence, the sentence containing the word (highlighted with **bold**), and the following sentence, if available.
4. Ensure each word is distinct (no duplicates).
5. Return ONLY valid JSON in this format:

[
  {{
    "word": "...",
    "quote": "..."
  }},
  ...
]

CHUNKED TEXT:
{chunk}
"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        raw_response = response.json().get("response", "")
        st.write("#### Debug: Raw model response for this chunk")
        st.code(raw_response)

        # Extract JSON array from the response
        match = re.search(r"(\[\s*\{.*\}\s*\])", raw_response, re.DOTALL)
        json_str = match.group(1) if match else raw_response

        # Parse as JSON
        vocab_data = json.loads(json_str.strip())

        if not isinstance(vocab_data, list):
            st.warning("Model returned something that isn't a JSON list.")
            return []

        # Basic structure validation
        results = []
        for entry in vocab_data:
            if isinstance(entry, dict) and "word" in entry and "quote" in entry:
                results.append(entry)
        return results

    except requests.exceptions.RequestException as e:
        st.error(f"Error contacting Ollama API: {e}")
    except json.JSONDecodeError as e:
        st.error(f"JSON parsing error: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

    return []

###############################################################################
# 3. Aggregating Across All Chunks
###############################################################################
def generate_vocabulary_workbook(full_text):
    """
    1. Break the full text into smaller chunks.
    2. For each chunk, call `extract_vocab_from_chunk`.
    3. Accumulate unique words until we have up to 5.
    4. Return the aggregated list of vocabulary entries.
    """
    chunks = chunk_text(full_text, max_chars=6000)

    aggregated = {}
    for i, chunk in enumerate(chunks, start=1):
        st.info(f"Processing chunk {i}/{len(chunks)}...")
        chunk_vocab = extract_vocab_from_chunk(chunk)

        for entry in chunk_vocab:
            word_lower = entry["word"].lower()
            if word_lower not in aggregated:
                aggregated[word_lower] = entry
            if len(aggregated) >= 5:
                break

        if len(aggregated) >= 5:
            break

    return list(aggregated.values())[:5]

###############################################################################
# 4. Streamlit App
###############################################################################
def main():
    st.title("Vocabulary Workbook Generator")
    st.write("Upload a text file (lesson/chapters) to extract up to 5 challenging vocabulary words.")

    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
    if uploaded_file is not None:
        text = uploaded_file.read().decode("utf-8")
        st.success("File uploaded successfully!")

        with st.spinner("Generating vocabulary workbook..."):
            vocab_list = generate_vocabulary_workbook(text)

        if vocab_list:
            st.success(f"Generated {len(vocab_list)} vocabulary words!")
            for i, entry in enumerate(vocab_list, start=1):
                st.markdown(f"**{i}. {entry['word']}**")
                st.markdown(f"**Quote:** {entry['quote']}")
                st.write("---")
        else:
            st.error("No valid vocabulary entries were returned. The model may not have followed JSON format.")

if __name__ == "__main__":
    main()
