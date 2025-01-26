import streamlit as st
import requests
import json
import re
import os

###############################################################################
# Where we left off, problems:
# 1. The if the model knows the book, it will make up words and quote from the wrong chapter.
# 2. It will sometimes select portions of words.
# 3. I would like it to prefer verbs, nouns, adjectives, adverbs, phrasal verbs, and idioms.
# 4. I would like to format the output text correctly
# 5. I would like it to pick better words. Sometimes it picks words that are too easy like door.
# 6. Output should have curly quotes and apostrophes.
# 7. read the words from the .json file and give checkboxes to select the words to include in the workbook.
# 8. Sometimes it highlights a different word than the word it picked.
# 9. Sometimes it will highlight an entire sentence instead of the word with **word**.


###############################################################################
# Configuration
###############################################################################
OLLAMA_URL = "http://localhost:11434/api/generate"  # Your Ollama endpoint
MODEL_NAME = "deepseek-r1:8b"

# Directory to save output files
SAVE_DIR = os.path.abspath("data")
os.makedirs(SAVE_DIR, exist_ok=True)

# Number of words to collect overall
MAX_VOCAB_WORDS = 20

###############################################################################
# 1. Robust JSON Parsing
###############################################################################
def robust_parse_json(raw_response: str):
    """
    Tries to parse the model's raw response into a Python list of dicts.
    We accept a JSON array or a single JSON object (which we wrap in a list).
    Returns a list of dictionaries, or an empty list if parsing fails.
    """

    # 1) Try direct parse
    try:
        parsed = json.loads(raw_response)
        # If it's a dict, wrap in a list
        if isinstance(parsed, dict):
            return [parsed]
        # If it's a list, return as is
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    # 2) Use a regex approach to isolate bracketed array
    match = re.search(r"(\[\s*\{.*\}\s*\])", raw_response, re.DOTALL)
    if match:
        array_str = match.group(1).strip()
        try:
            parsed = json.loads(array_str)
            if isinstance(parsed, dict):
                return [parsed]
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    return []

###############################################################################
# 2. Text Chunking
###############################################################################
def chunk_text(full_text: str, max_chars=6000):
    """
    Splits 'full_text' into chunks of ~max_chars to avoid overloading the model.
    Returns a list of chunk strings.
    """
    full_text = full_text.strip()
    if len(full_text) <= max_chars:
        return [full_text]

    chunks = []
    start_idx = 0
    while start_idx < len(full_text):
        end_idx = min(start_idx + max_chars, len(full_text))
        chunks.append(full_text[start_idx:end_idx])
        start_idx = end_idx
    return chunks

###############################################################################
# 3. Single-Chunk Processing
###############################################################################
def extract_vocab_from_chunk(chunk_text: str):
    """
    Calls the model with a prompt to find up to 5 vocabulary words in a chunk,
    each with 'word' and 'quote'.

    Returns a list of dicts:
      [
        { "word": "...", "quote": "..." },
        ...
      ]
    or an empty list on parse error.
    """

    # We strongly instruct the model to only return valid JSON, checking
    # that the extracted word actually appears in the chunk to avoid hallucinations.
    prompt = f"""
You are an English workbook creator. Return ONLY valid JSON.

Instructions:
1. Look at the CHUNKED TEXT below.
2. Extract up to 5 unique, challenging vocabulary words that actually appear in this chunk (exact match).
3. For each word, return:
   - "word": the exact text of the word
   - "quote": up to three sentences, combining:
       (a) the sentence before, 
       (b) the sentence containing the word (with the word in **bold**),
       (c) and the sentence after, if available.
4. Output must be valid JSON. No extra text or disclaimers.

CHUNKED TEXT:
{chunk_text}
"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload)
        resp.raise_for_status()
        raw_response = resp.json().get("response", "")

        # Optional debug for each chunk
        with st.expander("Debug: Chunk Raw Model Response", expanded=False):
            st.write(raw_response)

        data = robust_parse_json(raw_response)
        if not data:
            # Could not parse as a list/dict
            return []

        # Filter out hallucinated words that do not actually appear in chunk_text
        chunk_lower = chunk_text.lower()
        cleaned = []
        for item in data:
            if not isinstance(item, dict):
                continue
            if "word" in item and "quote" in item:
                w = item["word"].strip()
                # Ensure the word is found in the chunk
                if w.lower() in chunk_lower:
                    cleaned.append(item)

        return cleaned

    except requests.exceptions.RequestException as e:
        st.error(f"Error contacting the model API: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

    return []

###############################################################################
# 4. Aggregate Vocab from All Chunks
###############################################################################
def generate_vocabulary_workbook(full_text: str):
    """
    1. Break into chunks.
    2. For each chunk, attempt to extract up to 5 vocab words.
    3. Keep track of unique words until we hit MAX_VOCAB_WORDS across all chunks.
    4. Return the aggregated list of unique dicts.
    """
    chunks = chunk_text(full_text)
    all_words = {}

    # Single progress bar for chunk processing
    progress_bar = st.progress(0)
    total_chunks = len(chunks)

    for i, chunk in enumerate(chunks, start=1):
        progress_bar.progress(i / total_chunks)

        chunk_vocab = extract_vocab_from_chunk(chunk)
        for entry in chunk_vocab:
            # Use the lowercased version as a key to avoid duplicates
            w_lower = entry["word"].strip().lower()
            if w_lower not in all_words:
                all_words[w_lower] = entry
                if len(all_words) >= MAX_VOCAB_WORDS:
                    break
        if len(all_words) >= MAX_VOCAB_WORDS:
            break

    progress_bar.progress(1.0)  # done
    return list(all_words.values())

###############################################################################
# 5. Save Results to Data Folder (JSON + TXT)
###############################################################################
def save_results(vocab_list, uploaded_filename):
    """
    Saves the vocabulary list in two formats:
      - JSON: VocabWords-<uploaded filename>.json
      - TXT:  VocabWords-<uploaded filename>.txt
    in the data folder.
    """
    base_name = os.path.splitext(uploaded_filename)[0]
    # Prefix with "VocabWords-"
    json_path = os.path.join(SAVE_DIR, f"VocabWords-{base_name}.json")
    txt_path = os.path.join(SAVE_DIR, f"VocabWords-{base_name}.txt")

    try:
        # JSON file
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(vocab_list, jf, indent=2)
        st.success(f"Saved JSON file: {json_path}")

        # TXT summary
        with open(txt_path, "w", encoding="utf-8") as tf:
            tf.write(f"Vocabulary words extracted from {uploaded_filename}\n")
            tf.write("---------------------------------------------------\n")
            for i, entry in enumerate(vocab_list, start=1):
                word = entry["word"]
                quote = entry["quote"]
                tf.write(f"{i}. WORD: {word}\n")
                tf.write(f"   QUOTE: {quote}\n")
                tf.write("---------------------------------------------------\n")

        st.success(f"Saved TXT summary: {txt_path}")

    except Exception as e:
        st.error(f"Error saving results: {e}")

###############################################################################
# 6. Streamlit Application
###############################################################################
def main():
    st.title("Vocabulary Workbook Generator")
    st.write(f"Upload a text file to extract up to {MAX_VOCAB_WORDS} challenging words with quotes.")

    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

    if uploaded_file is not None:
        text_content = uploaded_file.read().decode("utf-8")
        st.success("File uploaded successfully!")

        with st.spinner("Processing text and generating vocabulary..."):
            vocab_list = generate_vocabulary_workbook(text_content)

        if vocab_list:
            st.success(f"Generated {len(vocab_list)} vocabulary words!")
            # Show them on screen
            for i, item in enumerate(vocab_list, start=1):
                st.markdown(f"**{i}. {item['word']}**")
                st.markdown(f"> {item['quote']}")
                st.write("---")

            # Save results
            save_results(vocab_list, uploaded_file.name)
        else:
            st.error("No valid vocabulary entries found. The model may not have followed JSON format.")

if __name__ == "__main__":
    main()
