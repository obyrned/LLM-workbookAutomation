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
    # Remove extra whitespace
    text = text.strip()
    # If the text is already small, no need to chunk
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start_idx = 0
    while start_idx < len(text):
        end_idx = min(start_idx + max_chars, len(text))
        chunk = text[start_idx:end_idx]
        # Extend the chunk to the nearest sentence boundary so we don't cut mid-sentence
        # by looking for a period or newline. This is optional but helps preserve context.
        # We'll keep it simple for now and just cut at max_chars.
        chunks.append(chunk)
        start_idx = end_idx
    return chunks

###############################################################################
# 2. Single-Chunk Processing
###############################################################################
def extract_vocab_from_chunk(chunk):
    """
    Sends one chunk of text to the Ollama model, asking for:
      - Up to 20 unique vocabulary words
      - definition
      - preceding_sentence
      - containing_sentence
      - following_sentence
    Returns a list of dict entries or None on error.
    """

    # Prompt: ask for JSON array of objects
    prompt = f"""
You are an advanced text analysis assistant. Return your entire output ONLY as JSON.

Instructions:
1. You have a chunk of text below.
2. Identify up to 20 unique and challenging vocabulary words found in this chunk.
3. For each word, provide:
   - "word": the exact word as it appears in the text
   - "definition": a short definition suitable for a young reader
   - "preceding_sentence": the sentence immediately before the one that has the word
   - "containing_sentence": the entire sentence containing the word
   - "following_sentence": the sentence immediately after the one that has the word
4. Ensure each word is distinct (no duplicates).
5. Return ONLY valid JSON in this format:

[
  {{
    "word": "...",
    "definition": "...",
    "preceding_sentence": "...",
    "containing_sentence": "...",
    "following_sentence": "..."
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
        "format": "json"  # Tells Ollama we want raw JSON back, though it might still include extra text
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        raw_response = response.json().get("response", "")
        # Debug: show raw model response
        st.write("#### Debug: Raw model response for this chunk")
        st.code(raw_response)

        # Attempt to extract only the JSON array
        match = re.search(r"(\[\s*\{.*\}\s*\])", raw_response, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            json_str = raw_response

        # Parse as JSON
        vocab_data = json.loads(json_str.strip())

        if not isinstance(vocab_data, list):
            st.warning("Model returned something that isn't a JSON list.")
            return []

        # Basic structure validation
        results = []
        for entry in vocab_data:
            if (
                isinstance(entry, dict)
                and all(k in entry for k in [
                    "word",
                    "definition",
                    "preceding_sentence",
                    "containing_sentence",
                    "following_sentence"
                ])
            ):
                # Add the entry if it looks valid
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
    3. Accumulate unique words until we have up to 20.
    4. Return the aggregated list of vocabulary entries.
    """
    chunks = chunk_text(full_text, max_chars=6000)

    aggregated = {}
    for i, chunk in enumerate(chunks, start=1):
        st.info(f"Processing chunk {i}/{len(chunks)}...")
        chunk_vocab = extract_vocab_from_chunk(chunk)

        for entry in chunk_vocab:
            word_lower = entry["word"].lower()
            # Only add if not already in aggregated
            if word_lower not in aggregated:
                aggregated[word_lower] = entry
            # If we've reached 20, we can stop
            if len(aggregated) >= 20:
                break

        if len(aggregated) >= 20:
            break

    # Return a list of up to 20 entries
    # The order is the order we found them
    final_vocab = list(aggregated.values())[:20]
    return final_vocab


###############################################################################
# 4. Streamlit App
###############################################################################
def main():
    st.title("Vocabulary Workbook Generator")
    st.write("Upload a text file (lesson/chapters) to extract up to 20 challenging vocabulary words.")

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
                st.markdown(f"**Definition:** {entry['definition']}")
                st.markdown(f"**Preceding Sentence:** {entry['preceding_sentence']}")
                st.markdown(f"**Containing Sentence:** {entry['containing_sentence']}")
                st.markdown(f"**Following Sentence:** {entry['following_sentence']}")
                st.write("---")
        else:
            st.error("No valid vocabulary entries were returned. The model may not have followed JSON format.")

if __name__ == "__main__":
    main()
