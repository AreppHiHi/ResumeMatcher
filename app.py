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
# Handles pathing for both your local Windows machine and Streamlit Cloud (Linux)
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

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
st.set_page_config(page_title="AI Resume Matcher", page_icon="🎯", layout="wide")

st.title("🎯 AI Resume Matcher")
st.markdown("Compare multiple resumes (PDF, DOCX, JPG) against a job description using NLP.")

# Layout Columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Job Description")
    jd_input = st.text_area("Paste the job requirements here...", height=300)

with col2:
    st.subheader("📁 Upload Resumes")
    resumes = st.file_uploader("Upload files", 
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
            
            # Process files
            for res in resumes:
                content = extract_text(res)
                if content:
                    resume_texts.append(content)
                    filenames.append(res.name)
            
            if resume_texts:
                # ML Logic: TF-IDF Vectorization
                all_documents = [jd_input] + resume_texts
                vectorizer = TfidfVectorizer(stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(all_documents)
                
                # Cosine Similarity calculation
                scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
                
                # Create DataFrame for processing
                results = pd.DataFrame({
                    "Candidate Name": filenames,
                    "Match Score": (scores * 100).round(2)
                }).sort_values(by="Match Score", ascending=False)
                
                # --- UI RESULTS ---
                st.success("Matching Complete!")
                st.balloons()

                # Display Table
                st.write("### Ranking Table")
                # We format the score for the table view
                display_df = results.copy()
                display_df["Match Score (%)"] = display_df["Match Score"].astype(str) + "%"
                st.dataframe(display_df[["Candidate Name", "Match Score (%)"]], use_container_width=True)
                
                # Display Chart
                st.write("### Visual Comparison")
                # Set index to name so chart labels are correct
                chart_data = results.set_index("Candidate Name")
                st.bar_chart(chart_data)

            else:
                st.error("Could not extract any text. Check if files are scanned images without OCR or empty.")
