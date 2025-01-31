import streamlit as st
import os

def main():
    st.title("Vocabulary Workbook Generator")

    st.write("Welcome! Click the button below to start extracting vocabulary words.")

    # Button to move to the next step
    if st.button("Next →"):
        os.system("streamlit run vocab_find.py")  # Launch vocab_find.py in a new Streamlit session
        st.stop()  # Stops current execution

if __name__ == "__main__":
    main()
