
import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
import docx
import pytesseract
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURATION ---
# Replace this with your actual path to Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- EXTRACTION BRAIN ---
def extract_text_from_file(uploaded_file):
    filename = uploaded_file.name
    if filename.endswith('.pdf'):
        reader = PdfReader(uploaded_file)
        return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    
    elif filename.endswith('.docx'):
        doc = docx.Document(uploaded_file)
        return " ".join([para.text for para in doc.paragraphs])
    
    elif filename.endswith(('.png', '.jpg', '.jpeg')):
        return pytesseract.image_to_string(Image.open(uploaded_file))
    
    return None

# --- UI INTERFACE ---
st.set_page_config(page_title="AI Resume Matcher", layout="wide")
st.title("🎯 AI Resume Matcher")
st.markdown("Upload **PDF, DOCX, or Images** to match against a Job Description.")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Step 1: Job Description")
    jd_text = st.text_area("Paste the Job Description here...", height=300)

with col2:
    st.header("Step 2: Upload Resumes")
    uploaded_files = st.file_uploader("Drop files here", type=['pdf', 'docx', 'png', 'jpg', 'jpeg'], accept_multiple_files=True)

# --- THE MATCHING ENGINE ---
if st.button("🚀 Match Resumes"):
    if not jd_text or not uploaded_files:
        st.warning("Please provide both a Job Description and Resumes.")
    else:
        with st.spinner('Extracting text and calculating scores...'):
            resume_data = []
            for file in uploaded_files:
                text = extract_text_from_file(file)
                if text:
                    resume_data.append({"filename": file.name, "text": text})

            if resume_data:
                # ML Logic: TF-IDF and Cosine Similarity
                texts = [jd_text] + [r['text'] for r in resume_data]
                vectorizer = TfidfVectorizer(stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(texts)
                
                # Compare JD (index 0) with all resumes (index 1+)
                scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]

                # Results Table
                results_df = pd.DataFrame({
                    "Resume Name": [r['filename'] for r in resume_data],
                    "Match Score": [f"{round(s * 100, 2)}%" for s in scores],
                    "Raw Score": scores
                }).sort_values(by="Raw Score", ascending=False)

                st.success("Analysis Complete!")
                st.table(results_df[["Resume Name", "Match Score"]])
            else:
                st.error("Could not extract text from the uploaded files.")
