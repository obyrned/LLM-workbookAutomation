# Workbook Generator

The Comprehension Workbook Generator is a web application designed to transform novel text into educational materials. Using OpenAI's GPT-4o and the DeepSeek-R1:8B model, it generates comprehension questions and vocabulary exercises, making it easy to create workbooks for lessons.

## 📌 Requirements

### **1️⃣ Install Required Software**
- **Python 3.8+**: Ensure you have Python installed.
- **Streamlit**: Used for the interactive web-based interface.
- **python-dotenv**: Loads API keys from an environment file.
- **OpenAI API Key**: Required for vocabulary extraction.

### **2️⃣ Install Dependencies**
Run the following command to install all required dependencies:
```bash
pip install -r requirements.txt
```

### **3️⃣ OpenAI API Key (Required)**
This script requires an **OpenAI API key** for processing vocabulary.  
If you don’t have one, create an API key at **[OpenAI](https://platform.openai.com/signup/)**.

#### **➤ How to Set Up Your API Key**
1. **Create a `.env` file** in the project directory.
2. **Add your OpenAI API key** inside `.env` (replace `sk-xxxx` with your key):
```bash
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### **4️⃣ (Optional) Ollama + DeepSeek Setup**
The script **previously** supported **DeepSeek via Ollama**, but this is currently **disabled**.  
If you want to use it, ensure **Ollama is installed and running** on port **11434**.

## 🚀 **How to Run**
1. **Start the Streamlit app**:
```bash
streamlit run vocab_workbook.py
```
2. **Upload a `.txt` file**, and the app will extract vocabulary words.

## 📂 **Output Files**
- The extracted words will be saved in **`data/`** as:
  - **JSON**: `vocab10_<filename>.json`
  - **TXT**: `vocab10_<filename>.txt`

## 🔗 **Additional Notes**
- OpenAI API logs can be checked at [OpenAI API Logs](https://platform.openai.com/account/api-logs).
- If you encounter API errors, ensure your **API key is set correctly** in `.env`.

## **Features**

### **1. Comprehension Questions (`tfmc_workbook.py`)**
- **Automatic Question Generation**:
  - Creates exactly 5 multiple-choice questions (with 4 answer options each).
  - Creates exactly 5 true/false questions.
- **Text Chunking**: Processes lesson-sized chunks of novel text (~1300 lines) for meaningful question generation.
- **Robust Parsing**: Ensures valid JSON output from the model.
- **Output Formats**:
  - Saves questions as JSON and text files.
  - Provides a preview within the app.

### **2. Vocabulary Workbook (`vocab_workbook.py`)**
- **Vocabulary Extraction**:
  - Identifies and highlights useful vocabulary based on uploaded text.
  - Suggests key terms for students to focus on.
- **Customizable**: Easily adjust parameters to suit different text types or lesson goals.

## **Example Output**

### **Comprehension Questions**
#### Multiple-Choice
1. What does the protagonist do?
   - A: Runs away
   - B: Hides
   - C: Fights bravely
   - D: Asks for help  
   **Correct Answer**: C

#### True/False
1. The protagonist travels alone.  
   **Correct Answer**: False

### **Vocabulary Example**
- **Words**: protagonist, bravery, adventure
- **Definitions**: Derived from the text for clear context-based explanations.

---
