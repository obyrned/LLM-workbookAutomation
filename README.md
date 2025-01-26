
Workbook Generator

The Comprehension Workbook Generator is a web application designed to transform novel text into educational materials. Using the DeepSeek-R1:8B model, it generates comprehension questions and vocabulary exercises, making it easy to create workbooks for lessons.

Requirements

- Ollama: Ensure Ollama is installed and running on port 11434.
- DeepSeek-R1:8B: The model is required for generating questions and vocabulary suggestions.
- Streamlit: Install Streamlit to run the application.

Features

1. Comprehension Questions (tfmc_workbook.py)
- Automatic Question Generation:
  - Creates exactly 5 multiple-choice questions (with 4 answer options each).
  - Creates exactly 5 true/false questions.
- Text Chunking: Processes lesson-sized chunks of novel text (~1300 lines) for meaningful question generation.
- Robust Parsing: Ensures valid JSON output from the model.
- Output Formats:
  - Saves questions as JSON and text files.
  - Provides a preview within the app.

2. Vocabulary Workbook (vocab_workbook.py)
- Vocabulary Extraction:
  - Identifies and highlights useful vocabulary based on uploaded text.
  - Suggests key terms for students to focus on.
- Customizable: Easily adjust parameters to suit different text types or lesson goals.

Installation

1. Clone the repository:
   git clone https://github.com/obyrned/comprehension-workbook-generator.git
   cd comprehension-workbook-generator

2. Install dependencies:
   pip install -r requirements.txt

3. Run the application:
   streamlit run app.py

4. Open your browser and navigate to:
   http://localhost:8501

Usage

Generating Questions
1. Upload a .txt file containing a lesson-sized chunk of text (~1300 lines).
2. The app will generate:
   - 5 multiple-choice questions.
   - 5 true/false questions.
3. Preview the questions in the app.
4. Download the results as JSON and text files (saved to the data directory).

Extracting Vocabulary
1. Use vocab_workbook.py to identify useful vocabulary terms.
2. Run the script on your text file to generate vocabulary lists for student practice.

Notes

- Ensure Ollama is running on port 11434 with the DeepSeek-R1:8B model loaded.
- For best results, use text with rich content to generate meaningful questions and vocabulary.

Example Output

Comprehension Questions
Multiple-Choice
1. What does the protagonist do?
   - A: Runs away
   - B: Hides
   - C: Fights bravely
   - D: Asks for help
   Correct Answer: C

True/False
1. The protagonist travels alone.
   Correct Answer: False

Vocabulary Example
- Words: protagonist, bravery, adventure
- Definitions: Derived from the text for clear context-based explanations.
