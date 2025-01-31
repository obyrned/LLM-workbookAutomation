import os
import json
import re
import requests
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path

###############################################################################
# Configuration
###############################################################################

# === Load API Key from .env ===
dotenv_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=dotenv_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY! Ensure it's set in your .env file.")

# OpenAI Model Configuration
OPENAI_MODEL = "gpt-4o"

# Directory to save output files
SAVE_DIR = os.path.abspath("data")
os.makedirs(SAVE_DIR, exist_ok=True)

# Number of words to collect overall
MAX_VOCAB_WORDS = 5

###############################################################################
# 1. Robust JSON Parsing
###############################################################################
def robust_parse_json(raw_response: str):
    """
    Tries to parse the model's raw response into a Python list of dicts.
    Returns a list of dictionaries, or an empty list if parsing fails.
    """
    try:
        parsed = json.loads(raw_response)
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"(\[\s*\{.*\}\s*\])", raw_response, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(1).strip())
            return parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            pass

    return []

###############################################################################
# 2. Text Chunking
###############################################################################
def chunk_text(full_text: str, max_chars=6000):
    """Splits 'full_text' into chunks of ~max_chars."""
    full_text = full_text.strip()
    return [full_text] if len(full_text) <= max_chars else [
        full_text[i:i+max_chars] for i in range(0, len(full_text), max_chars)
    ]

###############################################################################
# 3. Single-Chunk Processing
###############################################################################
def extract_vocab_from_chunk(chunk_text: str):
    """
    Calls OpenAI's GPT-4o to extract vocabulary words from a chunk.
    Returns a list of dicts with "word" and "quote".
    """
    prompt = f"""
You are an expert English workbook creator. Your task is to extract challenging **vocabulary words** from the given text while strictly adhering to JSON formatting.

### **Instructions**
1. **Extract exactly 5 unique and challenging vocabulary words** that appear **exactly as written** in the CHUNKED TEXT.
2. **Only select** words that are:  
   - **Nouns, verbs, adjectives, adverbs, phrasal verbs, or idioms.**  
   - **Not too easy** (e.g., avoid common words like "door," "big," "walk").  
   - **Full words only** (no partial words or prefixes).  
3. For each vocabulary word, return:
   - `"word"`: The **exact** word as it appears in the CHUNKED TEXT.
   - `"quote"`: A paragraph including:
     - **The sentence before** the word appears.
     - **The sentence containing the word**, with the word formatted in **bold**.
     - **The sentence after** the word, if available.
4. **Strict Formatting Rules**
   - Use **curly quotes** (“ ”) and **curly apostrophes** (’).  
   - **Do not duplicate words** (each word must be unique).  
   - **Ensure the correct word is highlighted in bold** (not a different word or full sentence).  
   - **Do not alter** the original text except to bold the word.  
5. **Output must be JSON only.**  
   - **Do not include any explanations, disclaimers, or extra formatting.**  
   - If fewer than 5 valid words are found, return only the ones available.  
   - If no suitable words are found, return an empty JSON array (`[]`).  

### **CHUNKED TEXT**  
{chunk_text}
"""

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    payload = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 1000
    }

    print("DEBUG: Sending API Request Payload:")
    print(json.dumps(payload, indent=2))  # Debugging output

    try:
        resp = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
        resp.raise_for_status()
        raw_response = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")

        return robust_parse_json(raw_response)

    except requests.exceptions.RequestException as e:
        st.error(f"Error contacting OpenAI API: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

    return []

###############################################################################
# 4. Aggregate Vocab from All Chunks
###############################################################################
def generate_vocabulary_workbook(full_text: str):
    """
    Processes text, extracts vocabulary, and ensures uniqueness.
    """
    chunks = chunk_text(full_text)
    all_words = {}

    progress_bar = st.progress(0)
    for i, chunk in enumerate(chunks, start=1):
        progress_bar.progress(i / len(chunks))

        chunk_vocab = extract_vocab_from_chunk(chunk)
        for entry in chunk_vocab:
            w_lower = entry["word"].strip().lower()
            if w_lower not in all_words:
                all_words[w_lower] = entry
                if len(all_words) >= MAX_VOCAB_WORDS:
                    break
        if len(all_words) >= MAX_VOCAB_WORDS:
            break

    progress_bar.progress(1.0)
    return list(all_words.values())

###############################################################################
# 5. Save Results to Data Folder (JSON + TXT)
###############################################################################
def save_results(vocab_list, uploaded_filename):
    """Saves vocabulary words in JSON and TXT format."""
    base_name = os.path.splitext(uploaded_filename)[0]
    json_path = os.path.join(SAVE_DIR, f"vocab10_{base_name}.json")
    txt_path = os.path.join(SAVE_DIR, f"vocab10_{base_name}.txt")

    try:
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(vocab_list, jf, indent=2, ensure_ascii=False)
        st.success(f"Saved JSON file: {json_path}")

        with open(txt_path, "w", encoding="utf-8") as tf:
            tf.write(f"Vocabulary words extracted from {uploaded_filename}\n")
            tf.write("---------------------------------------------------\n")
            for i, entry in enumerate(vocab_list, start=1):
                tf.write(f"{i}. WORD: {entry['word']}\n")
                tf.write(f"   QUOTE: {entry['quote']}\n")
                tf.write("---------------------------------------------------\n")

        st.success(f"Saved TXT summary: {txt_path}")

    except Exception as e:
        st.error(f"Error saving results: {e}")

###############################################################################
# 6. Streamlit Application
###############################################################################
def main():
    st.title("Vocabulary Workbook Generator")
    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

    if uploaded_file:
        text_content = uploaded_file.read().decode("utf-8")
        st.success("File uploaded successfully!")

        with st.spinner("Processing text..."):
            vocab_list = generate_vocabulary_workbook(text_content)

        if vocab_list:
            st.success(f"Generated {len(vocab_list)} vocabulary words!")
            save_results(vocab_list, uploaded_file.name)
        else:
            st.error("No valid vocabulary words found.")

if __name__ == "__main__":
    main()