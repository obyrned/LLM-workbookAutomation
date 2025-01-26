import streamlit as st
import requests
import json
import re
import os

###############################################################################
# Configuration
###############################################################################
OLLAMA_URL = "http://localhost:11434/api/generate"  # Your Ollama endpoint
MODEL_NAME = "deepseek-r1:8b"

# Directory to save output files
SAVE_DIR = os.path.abspath("data")
os.makedirs(SAVE_DIR, exist_ok=True)

# Desired number of questions total
NUM_MC_QUESTIONS = 5
NUM_TF_QUESTIONS = 5

# Maximum chunk size (characters)
CHUNK_SIZE = 6000

# Maximum final re-prompts if we still haven't reached 5/5
MAX_FINAL_ATTEMPTS = 3

###############################################################################
# 1. Robust JSON Parsing
###############################################################################
def robust_parse_json(raw_response: str):
    """
    Tries to parse the model's raw response into a Python structure.
    - If it's a single dict, wraps it in a list.
    - If it's a list, returns it.
    - Otherwise, tries a regex approach.
    - If still invalid, returns [].
    """
    # Direct parse
    try:
        parsed = json.loads(raw_response)
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    # Regex approach for bracketed array
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
def chunk_text(full_text: str, max_chars=CHUNK_SIZE):
    """
    Splits 'full_text' into ~max_chars sized chunks for processing.
    """
    text = full_text.strip()
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start_idx = 0
    while start_idx < len(text):
        end_idx = min(start_idx + max_chars, len(text))
        chunks.append(text[start_idx:end_idx])
        start_idx = end_idx
    return chunks

###############################################################################
# 3. Single-Chunk Question Extraction
###############################################################################
def extract_questions_from_chunk(chunk_text: str, needed_mc: int, needed_tf: int):
    """
    Prompt the model for exactly `needed_mc` multiple-choice + `needed_tf` TF questions
    from 'chunk_text'.

    Returns dict: {
      "mc_questions": [...],
      "tf_questions": [...]
    } or {} on parse failure.

    We also filter out incomplete questions (missing required keys).
    """
    # Force the model to produce exactly needed_mc and needed_tf
    prompt = f"""
You are an English workbook creator. Return ONLY valid JSON.

Instructions:
1. Read ONLY the CHUNKED TEXT below.
2. Generate exactly {needed_mc} multiple-choice questions (4 options each: a-d).
3. Generate exactly {needed_tf} true/false questions.
4. All questions must be in literary present tense and based ONLY on the text.
5. For each multiple-choice question:
   - "question": The question text
   - "options": {{ "a": "...", "b": "...", "c": "...", "d": "..." }}
   - "correct": one of "a","b","c","d"
6. For each true/false question:
   - "question": The question text
   - "correct": "True" or "False"
7. Output a JSON dict with "mc_questions" and "tf_questions" keys.

Example valid JSON:
{{
  "mc_questions": [
    {{
      "question": "What does the character do?",
      "options": {{
        "a": "Runs away",
        "b": "Hides",
        "c": "Fights bravely",
        "d": "Asks for help"
      }},
      "correct": "c"
    }}
  ],
  "tf_questions": [
    {{
      "question": "The protagonist travels alone.",
      "correct": "False"
    }}
  ]
}}

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

        with st.expander("Debug: Chunk Raw Model Response", expanded=False):
            st.write(raw_response)

        data = robust_parse_json(raw_response)
        if not data:
            return {}

        # If data is a list, pick the first dict
        if isinstance(data, list):
            data = data[0]
        if not isinstance(data, dict):
            return {}

        # Must have mc_questions/tf_questions keys
        if "mc_questions" not in data or "tf_questions" not in data:
            return {}

        # Filter out incomplete MC
        good_mc = []
        for q in data["mc_questions"]:
            if (
                isinstance(q, dict)
                and "question" in q
                and "options" in q
                and "correct" in q
            ):
                good_mc.append(q)

        # Filter out incomplete TF
        good_tf = []
        for q in data["tf_questions"]:
            if (
                isinstance(q, dict)
                and "question" in q
                and "correct" in q
            ):
                good_tf.append(q)

        return {
            "mc_questions": good_mc,
            "tf_questions": good_tf
        }

    except requests.exceptions.RequestException as e:
        st.error(f"Error contacting the model API: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

    return {}

###############################################################################
# 4. Aggregator Logic
###############################################################################
def generate_questions_workbook(full_text: str):
    """
    1) Chunk pass: go chunk by chunk, each time requesting the needed # of MC/TF.
    2) If we still have < 5 MC or < 5 TF, do up to MAX_FINAL_ATTEMPTS with the FULL text.
    3) Return exactly 5 MC and 5 TF if possible (or as many as we got).
    """
    chunks = chunk_text(full_text)
    mc_questions = []
    tf_questions = []
    mc_seen = set()
    tf_seen = set()

    needed_mc = NUM_MC_QUESTIONS
    needed_tf = NUM_TF_QUESTIONS

    # Single progress bar
    progress_bar = st.progress(0)
    total_chunks = len(chunks)

    # Pass 1: Chunk loop
    for i, chunk_str in enumerate(chunks, start=1):
        progress_bar.progress(i / total_chunks)
        if needed_mc <= 0 and needed_tf <= 0:
            break

        data = extract_questions_from_chunk(chunk_str, needed_mc, needed_tf)
        if not data:
            continue

        for q in data["mc_questions"]:
            if needed_mc <= 0:
                break
            q_text = q["question"].strip().lower()
            if q_text not in mc_seen:
                mc_questions.append(q)
                mc_seen.add(q_text)
                needed_mc -= 1

        for q in data["tf_questions"]:
            if needed_tf <= 0:
                break
            q_text = q["question"].strip().lower()
            if q_text not in tf_seen:
                tf_questions.append(q)
                tf_seen.add(q_text)
                needed_tf -= 1

    progress_bar.progress(1.0)

    # Pass 2: Final attempts with FULL text if needed
    attempts_left = MAX_FINAL_ATTEMPTS
    while attempts_left > 0 and (needed_mc > 0 or needed_tf > 0):
        st.info(f"Re-prompting with FULL text (missing {needed_mc} MC, {needed_tf} TF), attempts left: {attempts_left}...")
        data = extract_questions_from_chunk(full_text, needed_mc, needed_tf)
        if not data:
            attempts_left -= 1
            continue

        # Attempt to fill MC
        for q in data["mc_questions"]:
            if needed_mc <= 0:
                break
            q_text = q["question"].strip().lower()
            if q_text not in mc_seen:
                mc_questions.append(q)
                mc_seen.add(q_text)
                needed_mc -= 1

        # Attempt to fill TF
        for q in data["tf_questions"]:
            if needed_tf <= 0:
                break
            q_text = q["question"].strip().lower()
            if q_text not in tf_seen:
                tf_questions.append(q)
                tf_seen.add(q_text)
                needed_tf -= 1

        attempts_left -= 1

    # Truncate if above 5
    mc_questions = mc_questions[:NUM_MC_QUESTIONS]
    tf_questions = tf_questions[:NUM_TF_QUESTIONS]

    return {
        "mc_questions": mc_questions,
        "tf_questions": tf_questions
    }

###############################################################################
# 5. Save Results to Data Folder
###############################################################################
def save_results(questions, uploaded_filename):
    """
    Overwrites any existing file with tfmc-<filename>.json and .txt
    """
    base_name = os.path.splitext(uploaded_filename)[0]
    json_path = os.path.join(SAVE_DIR, f"tfmc-{base_name}.json")
    txt_path = os.path.join(SAVE_DIR, f"tfmc-{base_name}.txt")

    try:
        # Overwrite JSON
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(questions, jf, indent=2)

        st.success(f"Saved JSON file (overwritten if existed): {json_path}")

        # Overwrite TXT
        with open(txt_path, "w", encoding="utf-8") as tf:
            tf.write(f"Questions extracted from {uploaded_filename}\n")
            tf.write("---------------------------------------------------\n")
            tf.write("Multiple-Choice Questions:\n\n")
            for i, q in enumerate(questions["mc_questions"], start=1):
                tf.write(f"{i}. {q['question']}\n")
                options = q.get("options", {})
                for key, val in options.items():
                    tf.write(f"   {key.upper()}: {val}\n")
                tf.write(f"   Correct Answer: {q['correct']}\n")
                tf.write("---------------------------------------------------\n\n")

            tf.write("True/False Questions:\n\n")
            for i, q in enumerate(questions["tf_questions"], start=1):
                tf.write(f"{i}. {q['question']}\n")
                tf.write(f"   Correct Answer: {q['correct']}\n")
                tf.write("---------------------------------------------------\n\n")

        st.success(f"Saved TXT file (overwritten if existed): {txt_path}")

    except Exception as e:
        st.error(f"Error saving results: {e}")

###############################################################################
# 6. Streamlit App
###############################################################################
def main():
    st.title("Comprehension Workbook Generator")
    st.write(f"Upload a text file to generate {NUM_MC_QUESTIONS} multiple-choice and {NUM_TF_QUESTIONS} true/false questions.")

    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

    if uploaded_file is not None:
        text_content = uploaded_file.read().decode("utf-8")
        st.success("File uploaded successfully!")

        with st.spinner("Generating MC/TF questions..."):
            questions = generate_questions_workbook(text_content)

        mc_list = questions.get("mc_questions", [])
        tf_list = questions.get("tf_questions", [])

        if not mc_list and not tf_list:
            st.error("No valid questions found. The model may not have followed JSON format or there's insufficient text.")
            return

        # Show results
        st.success("Questions generated successfully!")
        st.write("### Multiple-Choice Questions")
        for i, q in enumerate(mc_list, start=1):
            st.markdown(f"**{i}. {q['question']}**")
            options = q.get("options", {})
            for opt_key, opt_text in options.items():
                st.write(f"{opt_key.upper()}: {opt_text}")
            st.write(f"**Correct Answer:** {q['correct']}")
            st.write("---")

        st.write("### True/False Questions")
        for i, q in enumerate(tf_list, start=1):
            st.markdown(f"**{i}. {q['question']}**")
            st.write(f"**Correct Answer:** {q['correct']}")
            st.write("---")

        # Save results
        save_results(questions, uploaded_file.name)

if __name__ == "__main__":
    main()
