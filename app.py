import os
import spacy
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2

# Load SpaCy model for text processing
nlp = spacy.load("en_core_web_sm")

# List of job-related keywords
job_keywords = ["python", "data analysis", "machine learning", "statistics", "sql", "data visualization", "deep learning", "NLP"]

# Function to extract text from PDFs
def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return text
    except:
        return ""

# Preprocess text (convert to lowercase, remove non-alphabetic characters)
def preprocess_text(text):
    doc = nlp(text.lower())
    return " ".join([token.text for token in doc if token.is_alpha])

# Calculate similarity score based on keywords
def calculate_similarity(resume_text, job_keywords):
    vectorizer = CountVectorizer(stop_words='english')
    corpus = [resume_text, " ".join(job_keywords)]
    vectors = vectorizer.fit_transform(corpus)
    similarity_matrix = cosine_similarity(vectors)
    return similarity_matrix[0, 1]

# Streamlit UI for uploading resumes and processing
st.title("Resume Screening Tool")
st.write("Upload your resumes and see the similarity scores for job keywords.")

# File uploader
uploaded_files = st.file_uploader("Choose PDF resumes", type="pdf", accept_multiple_files=True)

if uploaded_files:
    results = []

    for uploaded_file in uploaded_files:
        st.write(f"Processing: {uploaded_file.name}")
        resume_text = preprocess_text(extract_text_from_pdf(uploaded_file))

        if not resume_text.strip():
            st.warning(f"Skipping empty resume: {uploaded_file.name}")
            continue

        similarity_score = calculate_similarity(resume_text, job_keywords)

        results.append((uploaded_file.name, similarity_score if similarity_score > 0.3 else "No significant match"))

    # Sort by similarity score
    results.sort(key=lambda x: x[1] if isinstance(x[1], float) else 0, reverse=True)

    # Display results in Streamlit
    st.write("\n### Resume Screening Results")
    st.write(f"{'Resume Name':<40} {'Similarity Score'}")
    for resume, score in results:
        st.write(f"{resume:<40} {score if score == 'No significant match' else f'{score:.4f}'}")
