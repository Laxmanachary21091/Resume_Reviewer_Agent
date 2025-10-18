"""
Resume Reviewer Agent - Streamlit app
Tools: OpenAI + LangChain + Streamlit

This fixed version ensures compatibility with the latest LangChain API and adds better error handling.

Features:
- Upload PDF / DOCX / TXT resumes
- Extract text (PyPDF2, python-docx)
- Use OpenAI via LangChain to provide feedback and improvement suggestions

Run:
    streamlit run resume_reviewer_streamlit.py
"""

import os
import re
import tempfile
from typing import Optional

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import docx
except ImportError:
    docx = None

# -------------------- Utility Functions --------------------

def extract_text_from_pdf(file) -> str:
    if not PyPDF2:
        return "PyPDF2 not installed. Install it using: pip install PyPDF2"
    try:
        reader = PyPDF2.PdfReader(file)
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
        return text.strip()
    except Exception as e:
        return f"Error reading PDF: {e}"


def extract_text_from_docx(path) -> str:
    if not docx:
        return "python-docx not installed. Install it using: pip install python-docx"
    try:
        document = docx.Document(path)
        return "\n".join(p.text for p in document.paragraphs)
    except Exception as e:
        return f"Error reading DOCX: {e}"


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

# -------------------- LLM Setup --------------------

def get_llm(temp: float = 0.2):
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY not found. Please set it in your environment.")
    return ChatOpenAI(model="gpt-4o", temperature=temp)

PROMPT_TEMPLATE = """
You are a professional resume reviewer.

Review the following resume and provide structured feedback:
- Summary of strengths (2-3 lines)
- Score in 6 areas (Format, Clarity, Experience, Impact, Skills/ATS Fit, Overall)
- Actionable improvements with examples
- Suggested rewritten bullet points with impact metrics
- Formatting and layout suggestions
- Optional role-based keyword suggestions (if a role is provided)

Resume Text:
{resume_text}

Target Role: {target_role}
"""

# -------------------- Streamlit App --------------------

st.set_page_config(page_title="Resume Reviewer Agent", layout="centered")
st.title("🧠 Resume Reviewer Agent")
st.markdown("Upload your resume (PDF, DOCX, or TXT) to get detailed AI feedback.")

uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])
target_role = st.text_input("Target Role (optional)", placeholder="e.g., Data Scientist, ML Engineer")
temp = st.slider("Model Creativity (Temperature)", 0.0, 1.0, 0.2, 0.05)

if uploaded_file:
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    extracted_text = ""
    if suffix.lower() == ".pdf":
        with open(tmp_path, "rb") as f:
            extracted_text = extract_text_from_pdf(f)
    elif suffix.lower() == ".docx":
        extracted_text = extract_text_from_docx(tmp_path)
    else:
        try:
            extracted_text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        except Exception:
            extracted_text = ""

    extracted_text = clean_text(extracted_text)

    if not extracted_text:
        st.error("No text could be extracted. Make sure it's not a scanned image.")
    else:
        st.subheader("Extracted Text (first 800 chars)")
        st.code(extracted_text[:800] + ("..." if len(extracted_text) > 800 else ""))

        if st.button("Run Resume Review"):
            with st.spinner("Analyzing resume using OpenAI..."):
                try:
                    llm = get_llm(temp)
                    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["resume_text", "target_role"])
                    chain = LLMChain(llm=llm, prompt=prompt)
                    result = chain.run(resume_text=extracted_text, target_role=target_role)

                    st.success("✅ Review Completed!")
                    st.markdown(result)
                except Exception as e:
                    st.error(f"Error: {e}")
else:
    st.info("Please upload a resume file to get started.")

