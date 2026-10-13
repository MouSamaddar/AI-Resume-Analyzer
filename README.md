# ResumeIQ: AI Resume Screener

ResumeIQ is a Streamlit web app that screens and ranks multiple PDF resumes against a job description. It combines TF-IDF text similarity with skill-coverage matching, then shows the results in a responsive, themed dashboard.

> **Note:** ResumeIQ is a screening aid built with traditional NLP. It is **not** a real ATS and does not predict hiring outcomes.

---

## Features

- **Bulk upload:** screen many text-based PDF resumes at once
- **Ranked results:** every resume gets an overall score and a Strong / Moderate / Low match label
- **Skill matching and gap detection:** see which job skills each resume has and which are missing
- **Detailed insights (optional):**
  - Per-resume deep dive (sections found, contact details, shared keywords, quantified results)
  - Improvement suggestions for each resume
  - Skills matrix across all resumes
  - Score comparison charts and a full screening table
- **Downloadable report:** export the ranking as CSV
- **Dark and light themes**
- **Responsive design:** works on phones, tablets, laptops and desktops
- **Polished UI:** custom sidebar, live status panel, gradient cards and Plotly charts

---

## How scoring works

For each resume:

```
Overall score = 50% × Text similarity + 50% × Skill coverage
```

| Component | Description |
|---|---|
| **Text similarity** | TF-IDF vectors (unigrams + bigrams, English stop words removed) compared with cosine similarity between the resume and job description |
| **Skill coverage** | Percentage of skills found in the job description that also appear in the resume, based on a predefined skill list |

If no known skills are found in the job description, the overall score falls back to the text similarity score alone.

| Overall score | Label |
|---|---|
| 60% and above | Strong match |
| 35% – 59% | Moderate match |
| Below 35% | Low match |

---

## Tech stack

- [Python](https://www.python.org/)
- [Streamlit](https://streamlit.io/): web UI
- [pdfplumber](https://github.com/jsvine/pdfplumber): PDF text extraction
- [scikit-learn](https://scikit-learn.org/): TF-IDF and cosine similarity
- [Pandas](https://pandas.pydata.org/): data handling
- [Plotly](https://plotly.com/python/): interactive charts

---

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## requirements.txt

```txt
streamlit
pdfplumber
plotly
pandas
scikit-learn
```

---

## Usage

1. Open the **Dashboard** from the sidebar.
2. Upload one or more **text-based PDF resumes**.
3. Paste the **job description** into the text box.
4. Click **✦ Screen resumes**.
5. Review the ranking, then toggle **View all detailed insights** for the deep dive, skills matrix and charts.
6. Click **⬇ Download ranking (CSV)** to export the results.

**Tip:** the more detailed the job description, the more accurate the ranking.

---

## Project structure

```
.
├── app.py              # Full Streamlit application (UI, CSS, NLP logic)
├── requirements.txt    # Python dependencies
└── README.md
```

---

## Customization

- **Add or remove skills:** edit the `SKILLS` list near the top of `app.py`.
- **Change score weighting:** edit the `0.5 * text_score + 0.5 * coverage` line in the screening logic.
- **Change match thresholds:** edit `score_color()` and `score_label()`.
- **Change colors and theme:** edit the color constants under `THEME COLORS`.

---

## Limitations

- Works only with **text-based PDFs**. Scanned or image-only resumes are not readable (OCR is not included).
- Skill detection uses a **predefined keyword list**, so it may miss synonyms and cannot verify real experience.
- TF-IDF measures **wording overlap**, not meaning or candidate quality.
- Scores are a screening aid and should never be the only basis for a hiring decision.

---

## Possible improvements

- OCR support for scanned resumes
- Semantic matching with embeddings
- Support for DOCX resumes
- Larger and customizable skill database
- Per-skill weighting from the job description

---

## Author

Made by **<MouSamaddar>**.
