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

# Load API Key
dotenv_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=dotenv_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY! Ensure it's set in your .env file.")

OPENAI_MODEL = "gpt-4o"
SAVE_DIR = os.path.abspath("data")
os.makedirs(SAVE_DIR, exist_ok=True)

MAX_VOCAB_WORDS = 5  # Number of words to extract

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
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, dict):
            return [parsed]  # Wrap single object in list
    except json.JSONDecodeError:
        pass

    # Attempt to extract JSON from within response
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
# 3. OpenAI Vocabulary Extraction
###############################################################################
def extract_vocab_from_chunk(chunk_text: str):
    """
    Calls OpenAI's GPT-4o to extract vocabulary words from a chunk.
    Returns a list of dicts with "word" and "quote".
    """
    prompt = f"""
    You are an expert in English education. Extract exactly **5** advanced vocabulary words from the given text.

    ### **Instructions**
    1. Extract **5 unique and challenging vocabulary words** appearing **exactly as written** in the text.
    2. Only select **nouns, verbs, adjectives, adverbs, phrasal verbs, or idioms**.
    3. Each vocabulary word must include:
       - `"word"`: The **exact** word as it appears in the text.
       - `"quote"`: A paragraph including:
         - **The sentence before** the word appears.
         - **The sentence containing the word**, with the word formatted in **bold**.
         - **The sentence after**, if available.
    4. **Return only valid JSON.**  
       - If fewer than 5 words are found, return only the ones available.
       - If no suitable words are found, return `[]`.

    ### TEXT:
    {chunk_text}
    """

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    payload = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 1000
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
        response.raise_for_status()
        raw_response = response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()

        print("DEBUG: Raw OpenAI Response:")
        print(raw_response)

        vocab_list = robust_parse_json(raw_response)
        return vocab_list

    except requests.exceptions.RequestException as e:
        print(f"Error contacting OpenAI API: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []

###############################################################################
# 4. Aggregate Vocabulary Extraction
###############################################################################
def generate_vocabulary_workbook(full_text: str, filename):
    """
    Processes text, extracts vocabulary, and ensures uniqueness.
    Saves output as vocab10_<filename>.json.
    """
    chunks = chunk_text(full_text)
    all_words = {}

    for chunk in chunks:
        chunk_vocab = extract_vocab_from_chunk(chunk)
        for entry in chunk_vocab:
            w_lower = entry["word"].strip().lower()
            if w_lower not in all_words:
                all_words[w_lower] = entry
                if len(all_words) >= MAX_VOCAB_WORDS:
                    break
        if len(all_words) >= MAX_VOCAB_WORDS:
            break

    vocab_list = list(all_words.values())

    # Save JSON file
    base_name, _ = os.path.splitext(filename)
    json_path = os.path.join(SAVE_DIR, f"vocab10_{base_name}.json")

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(vocab_list, file, indent=2, ensure_ascii=False)

    print(f"✅ Saved: {json_path}")
    return vocab_list

###############################################################################
# 5. Save Results to TXT
###############################################################################
def save_results(vocab_list, uploaded_filename):
    """
    Saves vocabulary words in TXT format.
    """
    base_name, _ = os.path.splitext(uploaded_filename)
    vocab_txt_path = os.path.join(SAVE_DIR, f"vocab10_{base_name}.txt")

    try:
        with open(vocab_txt_path, "w", encoding="utf-8") as tf:
            tf.write(f"Vocabulary words extracted from {uploaded_filename}\n")
            tf.write("---------------------------------------------------\n")
            for i, entry in enumerate(vocab_list, start=1):
                tf.write(f"{i}. WORD: {entry['word']}\n")
                tf.write(f"   QUOTE: {entry['quote']}\n")
                tf.write("---------------------------------------------------\n")

        print(f"✅ Saved: {vocab_txt_path}")

    except Exception as e:
        print(f"Error saving results: {e}")

###############################################################################
# 6. Streamlit UI
###############################################################################
def main():
    st.title("Extract Vocabulary Words")

    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

    if uploaded_file:
        text_content = uploaded_file.read().decode("utf-8")
        st.success("File uploaded successfully!")

        with st.spinner("Extracting vocabulary words..."):
            vocab_list = generate_vocabulary_workbook(text_content, uploaded_file.name)

        if vocab_list:
            st.success(f"Generated {len(vocab_list)} vocabulary words!")

            # Display results
            for i, entry in enumerate(vocab_list, start=1):
                st.markdown(f"**{i}. {entry['word']}**")
                st.markdown(f"> {entry['quote']}")
                st.write("---")

            # Save results
            save_results(vocab_list, uploaded_file.name)
            st.success("Results saved!")

        else:
            st.error("No valid vocabulary words found.")

    # "Next" button is always visible
    if st.button("Next →"):
        os.system("streamlit run vocab_synonym.py")  # Move to synonym extraction
        st.stop()

if __name__ == "__main__":
    main()

