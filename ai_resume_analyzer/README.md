# AI Resume Analyzer

A beginner-friendly Python project that compares a PDF resume with a job description.

## Features
- Upload a text-based PDF resume
- Paste a job description
- TF-IDF + cosine similarity score
- Matching and potentially missing skills
- Basic suggestions
- Downloadable text report

## Setup

1. Install Python 3.10 or newer.
2. Extract this folder and open a terminal in it.
3. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   ```

4. Activate it:

   Windows:
   ```bash
   .venv\Scripts\activate
   ```

   macOS/Linux:
   ```bash
   source .venv/bin/activate
   ```

5. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

6. Run the app:

   ```bash
   streamlit run app.py
   ```

The app opens in your browser, usually at `http://localhost:8501`.

## How it works
- `pdfplumber` extracts text from the PDF.
- `TfidfVectorizer` converts text into numerical vectors.
- Cosine similarity estimates text similarity.
- A built-in keyword list identifies skills.

## Limitations
- Scanned/image-only PDFs may not work without OCR.
- Skill detection depends on the built-in skill list and exact wording.
- The similarity score is not a probability of getting hired and should not be treated as an automated hiring decision.
- Only include skills in your resume that you actually know.
