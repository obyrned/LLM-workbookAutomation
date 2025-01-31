import os
import json
import requests
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
import glob

###############################################################################
# Configuration
###############################################################################

# Load API Key
dotenv_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=dotenv_path)

OLLAMA_URL = "http://localhost:11434/api/generate"  # Ollama server
LLAMA_MODEL = "llama3"

# Directory to save output files
SAVE_DIR = os.path.abspath("data")
os.makedirs(SAVE_DIR, exist_ok=True)

###############################################################################
# 1. Find the Latest vocab10 JSON File
###############################################################################
def get_latest_vocab10():
    """
    Finds the most recently created vocab10_*.json file.
    """
    json_files = sorted(glob.glob(os.path.join(SAVE_DIR, "vocab10_*.json")), key=os.path.getmtime, reverse=True)
    return json_files[0] if json_files else None

###############################################################################
# 2. Fetch Synonyms Using Llama
###############################################################################
def fetch_synonyms(word):
    """
    Calls Llama via Ollama to get 4 synonyms for the given word.
    If no synonyms are found, returns "_____".
    """
    prompt = f"""
Provide exactly **four** synonyms for the word '{word}'.  
- Each synonym must be **one word** or a **short phrase** (no explanations).  
- If no synonyms are found, return: **_____, _____, _____, _____**  
- Output format:  
  **synonym1, synonym2, synonym3, synonym4**
    """

    payload = {"model": LLAMA_MODEL, "prompt": prompt, "stream": False}

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        synonyms = response.json().get("response", "").strip()

        # Ensure we return exactly 4 synonyms, clean and split the output
        synonym_list = [syn.strip() for syn in synonyms.split(",") if syn.strip()]
        return synonym_list[:4] if len(synonym_list) == 4 else ["_____", "_____", "_____", "_____"]

    except requests.exceptions.RequestException as e:
        print(f"Error contacting Llama API: {e}")
        return ["_____", "_____", "_____", "_____"]

###############################################################################
# 3. Generate vocab20 JSON with Synonyms
###############################################################################
def generate_vocab20(vocab10_path):
    """
    Reads vocab10_<filename>.json, fetches synonyms for each word using Llama,
    and saves the new JSON file as vocab20_<filename>.json.
    """
    base_name = os.path.basename(vocab10_path).replace("vocab10_", "").replace(".json", "")
    vocab20_json_path = os.path.join(SAVE_DIR, f"vocab20_{base_name}.json")
    vocab20_txt_path = os.path.join(SAVE_DIR, f"vocab20_{base_name}.txt")

    if not os.path.exists(vocab10_path):
        print(f"Error: {vocab10_path} not found!")
        return None

    with open(vocab10_path, "r", encoding="utf-8") as file:
        vocab_list = json.load(file)

    updated_vocab_list = []
    
    for entry in vocab_list:
        word = entry["word"]
        quote = entry["quote"]
        synonyms = fetch_synonyms(word)

        updated_vocab_list.append({
            "word": word,
            "quote": quote,
            "synonyms": synonyms
        })

    # Save vocab20 JSON file
    with open(vocab20_json_path, "w", encoding="utf-8") as file:
        json.dump(updated_vocab_list, file, indent=2, ensure_ascii=False)

    # Save vocab20 TXT summary
    try:
        with open(vocab20_txt_path, "w", encoding="utf-8") as tf:
            tf.write(f"Vocabulary words and synonyms from {base_name}.txt\n")
            tf.write("---------------------------------------------------\n")
            for i, entry in enumerate(updated_vocab_list, start=1):
                tf.write(f"{i}. WORD: {entry['word']}\n")
                tf.write(f"   QUOTE: {entry['quote']}\n")
                tf.write(f"   SYNONYMS: {', '.join(entry['synonyms'])}\n")
                tf.write("---------------------------------------------------\n")

        print(f"✅ Saved: {vocab20_json_path}")
        print(f"✅ Saved: {vocab20_txt_path}")

    except Exception as e:
        print(f"Error saving vocab20 TXT: {e}")

    return updated_vocab_list

###############################################################################
# 4. Streamlit UI
###############################################################################
def main():
    st.title("Generate Synonyms for Vocabulary Words")

    # Find the latest vocab10 file
    latest_vocab10 = get_latest_vocab10()
    if latest_vocab10:
        st.success(f"Using latest file: {os.path.basename(latest_vocab10)}")
    else:
        st.warning("No existing vocabulary file found. Upload a new one below.")

    uploaded_file = st.file_uploader("Upload a .txt file to fetch synonyms", type=["txt"])

    vocab_list = None

    # If file is uploaded, process it
    if uploaded_file:
        text_content = uploaded_file.read().decode("utf-8")
        st.success("File uploaded successfully!")

        # Generate vocab10 if user uploads a new file
        from vocab_find import generate_vocabulary_workbook  # Import from vocab_find.py
        vocab_list = generate_vocabulary_workbook(text_content, uploaded_file.name)
        latest_vocab10 = os.path.join(SAVE_DIR, f"vocab10_{uploaded_file.name}.json")

    # Display table if vocab10 exists
    if latest_vocab10 and os.path.exists(latest_vocab10):
        with open(latest_vocab10, "r", encoding="utf-8") as file:
            vocab_list = json.load(file)

        if vocab_list:
            st.write("### Vocabulary Words & Quotes")
            # Show words in a table
            vocab_data = [{"Word": entry["word"], "Quote": entry["quote"]} for entry in vocab_list]
            st.table(vocab_data)

            # "Find Synonyms" Button
            if st.button("Find Synonyms"):
                with st.spinner("Fetching synonyms..."):
                    updated_vocab_list = generate_vocab20(latest_vocab10)

                if updated_vocab_list:
                    st.success("Synonyms added and saved!")
                    
                    # Display synonyms below words
                    st.write("### Vocabulary Words with Synonyms")
                    for entry in updated_vocab_list:
                        st.markdown(f"**{entry['word']}**")
                        st.markdown(f"> {entry['quote']}")
                        st.write(f"**Synonyms:** {', '.join(entry['synonyms'])}")
                        st.write("---")

if __name__ == "__main__":
    main()
