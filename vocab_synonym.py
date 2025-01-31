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

# Load environment variables from .env
dotenv_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=dotenv_path)

# Ollama server URLs and model names
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")  # Your local Ollama endpoint
LLAMA_MODEL = os.getenv("LLAMA_MODEL", "llama3.2:8b")                        # Model for synonyms
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-r1:14b")              # Model for definitions

# Directory to save output files
SAVE_DIR = os.path.abspath("data")
os.makedirs(SAVE_DIR, exist_ok=True)

###############################################################################
# 1. Find the Latest vocab10 JSON File
###############################################################################
def get_latest_vocab10():
    """
    Finds the most recently created vocab10_*.json file but does NOT auto-execute.
    """
    json_files = sorted(
        glob.glob(os.path.join(SAVE_DIR, "vocab10_*.json")),
        key=os.path.getmtime,
        reverse=True
    )
    return json_files[0] if json_files else None

###############################################################################
# 2. Fetch Synonyms (Up to 4)
###############################################################################
def fetch_synonyms(word):
    """
    Calls llama3.2:8b via Ollama to get exactly 4 synonyms for the given word.
    Returns a list of up to 4 synonyms.
    """
    prompt = f"""
Provide exactly four high-quality synonyms for the word '{word}'. 
Ensure all synonyms match the part of speech of the original word. 
Return only the synonyms as a comma-separated list.
Example Output: synonym01, synonym02, synonym03, synonym04
    """

    payload = {"model": LLAMA_MODEL, "prompt": prompt, "stream": False}
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        synonyms_raw = response.json().get("response", "").strip()
        # Clean up and split by commas
        synonym_list = [syn.strip() for syn in synonyms_raw.split(",") if syn.strip()]
        return synonym_list[:4]
    except requests.exceptions.RequestException as e:
        st.error(f"Error contacting Llama API for synonyms: {e}")
        return []

###############################################################################
# 3. Fetch Definitions (Contextual)
###############################################################################
def fetch_definition(word, quote):
    """
    Calls llama3.2:8b via Ollama to define the word in the context of the quote.
    Returns a single definition string or an empty string.
    """
    prompt = f"""
Define the word '{word}' in the specific context of this quote:
\"{quote}\"

- Provide only one concise, plain-text definition.
- Do not include explanations, disclaimers, or headings.
- Just the context-appropriate definition.
    """

    payload = {"model": LLAMA_MODEL, "prompt": prompt, "stream": False}
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        definition = response.json().get("response", "").strip()
        return definition
    except requests.exceptions.RequestException as e:
        st.error(f"Error contacting DeepSeek API for definitions: {e}")
        return ""

###############################################################################
# 4. Generate Master Vocab Files
###############################################################################
def generate_master_vocab(selected_vocab, base_name):
    """
    Generates vocab30_<base_name>.json and vocab30_<base_name>.txt files.
    """
    json_path = os.path.join(SAVE_DIR, f"vocab30_{base_name}.json")
    txt_path  = os.path.join(SAVE_DIR, f"vocab30_{base_name}.txt")

    # Save JSON
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(selected_vocab, f, indent=2, ensure_ascii=False)
        st.success(f"Master JSON file created: `{json_path}`")
    except Exception as e:
        st.error(f"Error saving JSON file: {e}")

    # Save TXT summary
    try:
        with open(txt_path, "w", encoding="utf-8") as tf:
            tf.write("Vocabulary Words - Contextual Definitions - Synonyms\n")
            tf.write("===================================================\n\n")
            for i, entry in enumerate(selected_vocab, start=1):
                tf.write(f"{i}. WORD: {entry['word']}\n")
                if entry.get("quote"):
                    tf.write(f"   QUOTE: \"{entry['quote']}\"\n")
                tf.write(f"   DEFINITION: {entry['definition']}\n")
                synonyms_str = ", ".join(entry['synonyms'])
                tf.write(f"   SYNONYMS: {synonyms_str}\n")
                tf.write("---------------------------------------------------\n\n")
        st.success(f"Master TXT file created: `{txt_path}`")
    except Exception as e:
        st.error(f"Error saving TXT file: {e}")

    return json_path, txt_path

###############################################################################
# 5. Main Streamlit Page
###############################################################################
def main():
    st.title("Generate Synonyms and Definitions for Vocabulary Words")

    # Initialize session state if not already
    if "vocab_list" not in st.session_state:
        st.session_state["vocab_list"] = []
    if "generation_complete" not in st.session_state:
        st.session_state["generation_complete"] = False

    # 1) Let user pick or upload the latest vocab10 file
    latest_vocab10 = get_latest_vocab10()
    selected_file = None

    st.write("### 1) Select a Vocabulary JSON File")
    if latest_vocab10:
        st.info(f"Latest file detected: `{os.path.basename(latest_vocab10)}`")
        selected_file = st.selectbox(
            "Choose a JSON file to process:",
            [latest_vocab10] + sorted(glob.glob(os.path.join(SAVE_DIR, "vocab10_*.json")))
        )

    uploaded_file = st.file_uploader("Or upload a `.json` file", type=["json"])
    if uploaded_file is not None:
        selected_file = os.path.join(SAVE_DIR, f"uploaded_{uploaded_file.name}")
        with open(selected_file, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Uploaded file: `{uploaded_file.name}` is now selected.")

    # 2) If a file is chosen, load its data
    if selected_file and os.path.exists(selected_file):
        with open(selected_file, "r", encoding="utf-8") as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                st.error("The selected JSON file is invalid.")
                data = []

        # Store the data in session_state if not already done or if user changed the file
        if data and (not st.session_state["vocab_list"] or st.session_state.get("current_file") != selected_file):
            # Reset vocab_list if a new file is selected
            st.session_state["vocab_list"] = [
                {
                    "word": item["word"],
                    "definition": item.get("definition", ""),
                    "quote": item.get("quote", ""),
                    "synonyms": []
                }
                for item in data
            ]
            st.session_state["current_file"] = selected_file
            st.session_state["generation_complete"] = False

        # 3) Button: Generate Synonyms and Definitions
        if not st.session_state["generation_complete"]:
            generate_button = st.button("Generate Synonyms and Definitions for All Words")
            if generate_button:
                with st.spinner("Generating synonyms and definitions for each word..."):
                    for vocab_item in st.session_state["vocab_list"]:
                        word = vocab_item["word"]
                        quote = vocab_item.get("quote", "")
                        # Fetch synonyms
                        vocab_item["synonyms"] = fetch_synonyms(word)
                        # Fetch definition
                        vocab_item["definition"] = fetch_definition(word, quote)
                st.session_state["generation_complete"] = True
                st.success("Synonyms and definitions generated successfully!")

        # 4) Display the enriched vocabulary list
        if st.session_state["generation_complete"]:
            st.write("### 2) Enriched Vocabulary List")
            for idx, entry in enumerate(st.session_state["vocab_list"], start=1):
                st.markdown(f"**{idx}. {entry['word']}**")
                if entry.get("quote"):
                    st.markdown(f"> *\"{entry['quote']}\"*")
                st.markdown(f"**Definition:** {entry['definition'] if entry['definition'] else 'N/A'}")
                st.markdown(f"**Synonyms:** {', '.join(entry['synonyms']) if entry['synonyms'] else 'N/A'}")
                st.markdown("---")

            # 5) Input for Base Filename
            st.write("### 3) Specify Base Filename for Master Vocab Files")
            base_name = st.text_input("Enter a base name (e.g., 'mybook'):", value="mybook")

            # 6) Button to Generate Master Vocab Files
            generate_master_button = st.button("Generate Master Vocab Files")
            if generate_master_button:
                if not base_name.strip():
                    st.error("Please enter a valid base name.")
                else:
                    json_path, txt_path = generate_master_vocab(st.session_state["vocab_list"], base_name)
                    st.session_state["master_files"] = {
                        "json": json_path,
                        "txt": txt_path
                    }

            # 7) Button to Download Master Vocab Files
            if "master_files" in st.session_state:
                st.write("### 4) Download Master Vocab Files")
                json_path = st.session_state["master_files"]["json"]
                txt_path  = st.session_state["master_files"]["txt"]

                with open(json_path, "r", encoding="utf-8") as f:
                    json_content = f.read()
                with open(txt_path, "r", encoding="utf-8") as f:
                    txt_content = f.read()

                st.download_button(
                    label="Download JSON",
                    data=json_content,
                    file_name=os.path.basename(json_path),
                    mime="application/json"
                )
                st.download_button(
                    label="Download TXT",
                    data=txt_content,
                    file_name=os.path.basename(txt_path),
                    mime="text/plain"
                )

            # 8) Button to Proceed Further (if needed)
            # Example: If you have another step, you can add a button or link here.
            # For now, we'll just notify the user that the process is complete.
            if "master_files" in st.session_state:
                st.success("Master vocab files generated and ready for download!")

    else:
        st.warning("No valid JSON file selected or found. Please upload or pick a file.")

###############################################################################
# Run the app
###############################################################################
if __name__ == "__main__":
    main()
