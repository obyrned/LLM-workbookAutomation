import streamlit as st
import requests
import json
import re
import os

# Configuration
###############################################################################
OLLAMA_URL = "http://localhost:11434/api/generate"  # Your Ollama endpoint
MODEL_NAME = "deepseek-r1:8b"

# Directory to save output files
SAVE_DIR = os.path.abspath("data")
os.makedirs(SAVE_DIR, exist_ok=True)

# Number of questions to generate
NUM_MC_QUESTIONS = 5
NUM_TF_QUESTIONS = 5

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
def extract_questions_from_chunk(chunk_text: str):
    """
    Calls the model with a prompt to generate:
      - 5 multiple-choice questions
      - 5 true/false questions
    based on the chunk text.

    Returns a dict with two keys:
      - "mc_questions": List of multiple-choice questions
      - "tf_questions": List of true/false questions
    or an empty dict on error.
    """

    prompt = f"""
You are an English workbook creator. Return ONLY valid JSON.

Instructions:
1. Look at the CHUNKED TEXT below.
2. Generate:
   - 5 multiple-choice questions (with 4 options each, labeled A-D).
   - 5 true/false questions.
3. All questions must be in literary present tense and based on the text.
4. For multiple-choice questions:
   - Include the correct answer (key: "correct").
   - Ensure the options are plausible but only one is correct.
5. For true/false questions:
   - Include the correct answer (key: "correct").
6. Output must be valid JSON. No extra text or disclaimers.

Example valid JSON:
{{
  "mc_questions": [
    {{
      "question": "What does the protagonist do when faced with danger?",
      "options": {{
        "A": "Run away",
        "B": "Stand and fight",
        "C": "Call for help",
        "D": "Freeze in fear"
      }},
      "correct": "B"
    }}
  ],
  "tf_questions": [
    {{
      "question": "The protagonist is always brave.",
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

        # Optional debug for each chunk
        with st.expander("Debug: Chunk Raw Model Response", expanded=False):
            st.write(raw_response)

        data = robust_parse_json(raw_response)
        if not data:
            # Could not parse as a list/dict
            return {}

        # Ensure the response has the expected structure
        if isinstance(data, list):
            data = data[0]  # Use the first item if it's a list

        if not isinstance(data, dict):
            return {}

        # Validate the structure
        if "mc_questions" not in data or "tf_questions" not in data:
            return {}

        return data

    except requests.exceptions.RequestException as e:
        st.error(f"Error contacting the model API: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

    return {}

###############################################################################
# 4. Aggregate Questions from All Chunks
###############################################################################
def generate_questions_workbook(full_text: str):
    """
    1. Break into chunks.
    2. For each chunk, attempt to generate questions.
    3. Accumulate questions until we hit NUM_MC_QUESTIONS and NUM_TF_QUESTIONS.
    4. Return the aggregated questions.
    """
    chunks = chunk_text(full_text)
    mc_questions = []
    tf_questions = []

    # Single progress bar for chunk processing
    progress_bar = st.progress(0)
    total_chunks = len(chunks)

    for i, chunk in enumerate(chunks, start=1):
        progress_bar.progress(i / total_chunks)

        chunk_questions = extract_questions_from_chunk(chunk)
        if not chunk_questions:
            continue

        # Add MC questions
        for q in chunk_questions.get("mc_questions", []):
            if len(mc_questions) < NUM_MC_QUESTIONS:
                mc_questions.append(q)

        # Add TF questions
        for q in chunk_questions.get("tf_questions", []):
            if len(tf_questions) < NUM_TF_QUESTIONS:
                tf_questions.append(q)

        # Stop if we have enough questions
        if len(mc_questions) >= NUM_MC_QUESTIONS and len(tf_questions) >= NUM_TF_QUESTIONS:
            break

    progress_bar.progress(1.0)  # done
    return {
        "mc_questions": mc_questions[:NUM_MC_QUESTIONS],
        "tf_questions": tf_questions[:NUM_TF_QUESTIONS]
    }

###############################################################################
# 5. Save Results to Data Folder (JSON + TXT)
###############################################################################
def save_results(questions, uploaded_filename):
    """
    Saves the questions in two formats:
      - JSON: tfmc-<uploaded filename>.json
      - TXT:  tfmc-<uploaded filename>.txt
    in the data folder.
    """
    base_name = os.path.splitext(uploaded_filename)[0]
    # Prefix with "tfmc-"
    json_path = os.path.join(SAVE_DIR, f"tfmc-{base_name}.json")
    txt_path = os.path.join(SAVE_DIR, f"tfmc-{base_name}.txt")

    try:
        # JSON file
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(questions, jf, indent=2)
        st.success(f"Saved JSON file: {json_path}")

        # TXT summary
        with open(txt_path, "w", encoding="utf-8") as tf:
            tf.write(f"Questions extracted from {uploaded_filename}\n")
            tf.write("---------------------------------------------------\n")
            tf.write("Multiple-Choice Questions:\n")
            for i, q in enumerate(questions["mc_questions"], start=1):
                tf.write(f"{i}. {q['question']}\n")
                for opt, text in q["options"].items():
                    tf.write(f"   {opt}: {text}\n")
                tf.write(f"   Correct Answer: {q['correct']}\n")
                tf.write("---------------------------------------------------\n")

            tf.write("\nTrue/False Questions:\n")
            for i, q in enumerate(questions["tf_questions"], start=1):
                tf.write(f"{i}. {q['question']}\n")
                tf.write(f"   Correct Answer: {q['correct']}\n")
                tf.write("---------------------------------------------------\n")

        st.success(f"Saved TXT summary: {txt_path}")

    except Exception as e:
        st.error(f"Error saving results: {e}")

###############################################################################
# 6. Streamlit Application
###############################################################################
def main():
    st.title("Comprehension Workbook Generator")
    st.write(f"Upload a text file to generate {NUM_MC_QUESTIONS} multiple-choice and {NUM_TF_QUESTIONS} true/false questions.")

    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

    if uploaded_file is not None:
        text_content = uploaded_file.read().decode("utf-8")
        st.success("File uploaded successfully!")

        with st.spinner("Processing text and generating questions..."):
            questions = generate_questions_workbook(text_content)

        if questions["mc_questions"] or questions["tf_questions"]:
            st.success("Questions generated successfully!")
            # Show MC questions
            st.write("### Multiple-Choice Questions")
            for i, q in enumerate(questions["mc_questions"], start=1):
                st.markdown(f"**{i}. {q['question']}**")
                for opt, text in q["options"].items():
                    st.write(f"{opt}: {text}")
                st.write(f"**Correct Answer:** {q['correct']}")
                st.write("---")

            # Show TF questions
            st.write("### True/False Questions")
            for i, q in enumerate(questions["tf_questions"], start=1):
                st.markdown(f"**{i}. {q['question']}**")
                st.write(f"**Correct Answer:** {q['correct']}")
                st.write("---")

            # Save results
            save_results(questions, uploaded_file.name)
        else:
            st.error("No valid questions found. The model may not have followed JSON format.")

if __name__ == "__main__":
    main()