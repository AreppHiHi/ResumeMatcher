import streamlit as st
import pandas as pd
import platform
import pytesseract
from PIL import Image
from PyPDF2 import PdfReader
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. TESSERACT CONFIGURATION ---
# This part detects if you are on Windows or the Cloud
if platform.system() == "Windows":
    # Update this path if your tesseract is installed elsewhere
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else:
    # On Streamlit Cloud/Linux, it is usually in the PATH automatically
    pass

# --- 2. EXTRACTION ENGINE ---
def extract_text(uploaded_file):
    """Extracts text from PDF, DOCX, or Image files."""
    file_extension = uploaded_file.name.split('.')[-1].lower()
    text = ""
    
    try:
        if file_extension == 'pdf':
            reader = PdfReader(uploaded_file)
            for page in reader.pages:
                text += page.extract_text() or ""
                
        elif file_extension == 'docx':
            doc = docx.Document(uploaded_file)
            text = " ".join([para.text for para in doc.paragraphs])
            
        elif file_extension in ['jpg', 'jpeg', 'png']:
            img = Image.open(uploaded_file)
            text = pytesseract.image_to_string(img)
            
        return text.strip()
    except Exception as e:
        st.error(f"Error processing {uploaded_file.name}: {e}")
        return ""

# --- 3. STREAMLIT UI ---
st.set_page_config(page_title="AI Resume Matcher", page_icon="🎯")

st.title("🎯 AI Resume Matcher")
st.markdown("""
    This tool uses **Natural Language Processing (NLP)** to compare resumes against a job description. 
    It supports **PDFs, Word Docs, and Images (OCR)**.
""")

# Sidebar for instructions
with st.sidebar:
    st.header("How to use")
    st.info("1. Paste the Job Description.\n2. Upload one or more resumes.\n3. Click Match.")

# Layout Columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Job Description")
    jd_input = st.text_area("Paste the job requirements here...", height=300)

with col2:
    st.subheader("📁 Upload Resumes")
    resumes = st.file_uploader("Upload PDF, DOCX, or Images", 
                               type=['pdf', 'docx', 'png', 'jpg', 'jpeg'], 
                               accept_multiple_files=True)

# --- 4. MATCHING LOGIC ---
if st.button("Run AI Matching"):
    if not jd_input:
        st.warning("Please provide a Job Description.")
    elif not resumes:
        st.warning("Please upload at least one resume.")
    else:
        with st.spinner('AI is analyzing the documents...'):
            resume_texts = []
            filenames = []
            
            # Process each file
            for res in resumes:
                content = extract_text(res)
                if content:
                    resume_texts.append(content)
                    filenames.append(res.name)
            
            if resume_texts:
                # Vectorization using TF-IDF
                all_documents = [jd_input] + resume_texts
                vectorizer = TfidfVectorizer(stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(all_documents)
                
                # Calculate Cosine Similarity
                # Index 0 is the Job Description, Index 1+ are the resumes
                scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
                
                # Prepare Results Dataframe
                results = pd.DataFrame({
                    "Candidate Name": filenames,
                    "Match Score (%)": (scores * 100).round(2)
                }).sort_values(by="Match Score (%)", ascending=False)
                
                # Display Results
                st.success("Matching Complete!")
                
                # Highlight the top candidate
                st.balloons()
                st.write("### Ranking Table")
                st.dataframe(results, use_container_width=True)
                
                # Simple Data Visualization
                st.bar_chart(results.set_index("Candidate Name"))
            else:
                st.error("No text could be extracted from the uploaded files. Check if they are empty or corrupted.")

st.bar_chart(...)
