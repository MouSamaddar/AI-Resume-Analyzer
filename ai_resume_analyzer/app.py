import io
import re
import html

import streamlit as st
import pdfplumber
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="ResumeIQ | AI Resume Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

if "analysis" not in st.session_state:
    st.session_state.analysis = None


# ---------------- THEME COLORS ----------------

DARK = st.session_state.theme == "Dark"

BG = "#0B1020" if DARK else "#F3F6FC"
PANEL = "#121A2D" if DARK else "#FFFFFF"
PANEL2 = "#1A2540" if DARK else "#EDF2F8"
TEXT = "#F1F5F9" if DARK else "#172033"
MUTED = "#9AA8BF" if DARK else "#64748B"
BORDER = "rgba(148,163,184,.25)"
PURPLE = "#8B5CF6"
CYAN = "#22D3EE"
GREEN = "#34D399"
RED = "#FB7185"

# Gradient colours for the sidebar brand name
BRAND_FROM = "#FFFFFF" if DARK else "#172033"
BRAND_MID = "#C4B5FD" if DARK else "#7C3AED"
BRAND_TO = "#67E8F9" if DARK else "#0891B2"

# Height (in px) of the job description box and minimum height of the resume box
INPUT_BOX_HEIGHT = 220


# ---------------- CUSTOM CSS ----------------

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;600;700;800&display=swap');

    .stApp {{
        background: {BG};
        color: {TEXT};
        font-family: 'DM Sans', sans-serif;
        -webkit-text-size-adjust: 100%;
    }}

    [data-testid="stHeader"] {{
        background: transparent;
    }}

    /* HIDE DEPLOY BUTTON AND 3-DOTS MENU */
    [data-testid="stToolbar"],
    [data-testid="stDeployButton"],
    .stDeployButton,
    [data-testid="stMainMenu"],
    #MainMenu {{
        display: none !important;
        visibility: hidden !important;
    }}

    /* KEEP THE SIDEBAR OPEN BUTTON VISIBLE (IMPORTANT ON MOBILE) */
    [data-testid="stExpandSidebarButton"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {{
        display: flex !important;
        visibility: visible !important;
        color: {TEXT} !important;
    }}

    [data-testid="stExpandSidebarButton"] *,
    [data-testid="stSidebarCollapsedControl"] *,
    [data-testid="collapsedControl"] * {{
        color: {TEXT} !important;
    }}

    /* HIDE "Made with Streamlit" FOOTER AND TOP COLOR BAR */
    footer,
    [data-testid="stDecoration"] {{
        display: none !important;
    }}

    [data-testid="stSidebar"] {{
        background:
            radial-gradient(circle at 15% 0%, rgba(139, 92, 246, .25), transparent 42%),
            radial-gradient(circle at 100% 100%, rgba(34, 211, 238, .14), transparent 40%),
            {PANEL};
        border-right: 1px solid {BORDER};
    }}

    [data-testid="stSidebar"] * {{
        color: {TEXT};
    }}

    .block-container {{
        max-width: 1500px;
        padding-top: 1.8rem;
        padding-bottom: 3rem;
    }}

    h1, h2, h3 {{
        color: {TEXT};
        font-family: 'Manrope', sans-serif;
        letter-spacing: -0.035em;
    }}

    p, label {{
        color: {TEXT};
    }}

    .brand {{
        font-family: 'Manrope', sans-serif;
        font-size: 1.35rem;
        font-weight: 800;
    }}

    .brand-icon {{
        display: inline-flex;
        width: 38px;
        height: 38px;
        align-items: center;
        justify-content: center;
        margin-right: 9px;
        border-radius: 12px;
        color: white;
        background: linear-gradient(135deg, #8B5CF6, #22D3EE);
    }}

    .hero {{
        padding: 30px;
        border: 1px solid {BORDER};
        border-radius: 24px;
        background: linear-gradient(120deg, {PANEL}, {PANEL2});
        margin-bottom: 25px;
    }}

    .eyebrow {{
        color: {CYAN};
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
    }}

    .hero-title {{
        font-family: 'Manrope', sans-serif;
        font-size: clamp(26px, 4vw, 43px);
        font-weight: 800;
        line-height: 1.2;
        letter-spacing: -1.8px;
        margin-top: 12px;
        color: {TEXT};
    }}

    .muted {{
        color: {MUTED};
        font-size: 13px;
    }}

    .card {{
        background: {PANEL};
        border: 1px solid {BORDER};
        border-radius: 18px;
        padding: 20px;
        height: 100%;
        box-sizing: border-box;
        word-break: break-word;
    }}

    .card-label {{
        color: {MUTED};
        font-size: 12px;
        font-weight: 700;
        letter-spacing: .7px;
    }}

    .card-value {{
        font-family: 'Manrope', sans-serif;
        font-size: 30px;
        font-weight: 800;
        margin: 8px 0 4px;
    }}

    .card-note {{
        color: {MUTED};
        font-size: 12px;
    }}

    .section-title {{
        color: {TEXT};
        font-family: 'Manrope', sans-serif;
        font-size: 19px;
        font-weight: 800;
        margin: 22px 0 14px;
    }}

    .skill {{
        display: inline-block;
        padding: 7px 12px;
        margin: 4px 5px 4px 0;
        border-radius: 20px;
        font-size: 12px;
        background: {PANEL2};
        border: 1px solid {BORDER};
        color: {TEXT};
        max-width: 100%;
        word-break: break-word;
    }}

    .matched {{
        background: rgba(52,211,153,.12);
        border-color: rgba(52,211,153,.3);
        color: {GREEN};
    }}

    .missing {{
        background: rgba(251,113,133,.10);
        border-color: rgba(251,113,133,.3);
        color: {RED};
    }}

    .tip {{
        background: {PANEL};
        border: 1px solid {BORDER};
        border-radius: 13px;
        padding: 15px;
        margin: 8px 0;
        color: {TEXT};
        font-size: 14px;
    }}

    .notice {{
        background: {PANEL2};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 13px;
        color: {MUTED};
        font-size: 13px;
    }}

    /* BOTH BUTTONS: SAME SIZE AND SAME PURPLE GRADIENT */

    div.stButton > button {{
        width: 100% !important;
        min-height: 44px !important;
        height: 44px !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        color: #FFFFFF !important;
        background: linear-gradient(
            100deg, #7C3AED, #6366F1
        ) !important;
        border: 1px solid transparent !important;
        box-shadow: none !important;
        transition: all 0.2s ease !important;
    }}

    /* SAME HOVER STYLE FOR BOTH BUTTONS */

    div.stButton > button:hover {{
        background: linear-gradient(
            100deg, #6D28D9, #4F46E5
        ) !important;
        color: #FFFFFF !important;
        border: 1px solid #A78BFA !important;
        transform: translateY(-1px);
    }}

    /* KEEP BUTTON TEXT WHITE */

    div.stButton > button p,
    div.stButton > button span,
    div.stButton > button div {{
        color: #FFFFFF !important;
    }}

    div.stButton > button:hover p,
    div.stButton > button:hover span,
    div.stButton > button:hover div {{
        color: #FFFFFF !important;
    }}

    /* KEYBOARD FOCUS */

    div.stButton > button:focus-visible {{
        outline: 2px solid #A78BFA !important;
        outline-offset: 3px !important;
        box-shadow: none !important;
    }}

    /* DOWNLOAD BUTTON */

    div.stDownloadButton > button {{
        width: 100% !important;
        min-height: 44px !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        color: {TEXT} !important;
        background: {PANEL2} !important;
        border: 1px solid {BORDER} !important;
    }}

    div.stDownloadButton > button p,
    div.stDownloadButton > button span {{
        color: {TEXT} !important;
    }}

    /* ---------- EQUAL-SIZE INPUT BOXES ---------- */
    /* Both the resume uploader and the job description box share
       the same width (equal columns), the same height, the same
       border, the same radius and the same background. */

    /* Resume uploader box */
    [data-testid="stFileUploader"] {{
        box-sizing: border-box !important;
        min-height: {INPUT_BOX_HEIGHT}px !important;
        height: auto !important;
        background: {PANEL2} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 14px !important;
        padding: 14px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        overflow: visible !important;
    }}

    /* UPLOADED FILE LIST: SHOW EVERY FILE NAME CLEARLY */
    [data-testid="stFileUploaderFile"],
    [data-testid="stFileUploaderFileName"] {{
        color: {TEXT} !important;
    }}

    [data-testid="stFileUploaderFileName"] {{
        overflow: visible !important;
        text-overflow: clip !important;
        white-space: normal !important;
        word-break: break-word !important;
    }}

    /* File size text (was too dark to read) */
    [data-testid="stFileUploaderFile"] small,
    [data-testid="stFileUploaderFileData"] small,
    [data-testid="stFileUploaderFile"] [data-testid="stCaptionContainer"] {{
        color: {MUTED} !important;
    }}

    [data-testid="stFileUploaderFile"] svg,
    [data-testid="stFileUploaderDeleteBtn"] svg {{
        color: {MUTED} !important;
    }}

    [data-testid="stFileUploaderDeleteBtn"] button {{
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        min-height: 0 !important;
        height: auto !important;
    }}

    /* ---------- FILE LIST PAGINATION ("Showing page 1 of 3" + ARROWS) ---------- */

    [data-testid="stFileUploaderPagination"] {{
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
        gap: 12px !important;
        margin-top: 10px !important;
        padding-top: 10px !important;
        border-top: 1px solid {BORDER} !important;
    }}

    /* Grey, readable "Showing page X of Y" text */
    [data-testid="stFileUploaderPagination"] small,
    [data-testid="stFileUploaderPagination"] p,
    [data-testid="stFileUploaderPagination"] span,
    [data-testid="stFileUploaderPagination"] div,
    [data-testid="stFileUploaderDropzone"] ~ div small {{
        color: {MUTED} !important;
        opacity: 1 !important;
        font-size: 13px !important;
        font-weight: 500 !important;
    }}

    /* Previous / next arrow buttons */
    [data-testid="stFileUploaderPagination"] button {{
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 34px !important;
        min-width: 34px !important;
        height: 34px !important;
        min-height: 34px !important;
        padding: 0 !important;
        border-radius: 10px !important;
        background: {PANEL} !important;
        border: 1px solid {BORDER} !important;
        color: {TEXT} !important;
        opacity: 1 !important;
        box-shadow: none !important;
        cursor: pointer !important;
        transition: all 0.15s ease !important;
    }}

    [data-testid="stFileUploaderPagination"] button svg,
    [data-testid="stFileUploaderPagination"] button span,
    [data-testid="stFileUploaderPagination"] button p {{
        color: {TEXT} !important;
        fill: {TEXT} !important;
        opacity: 1 !important;
    }}

    [data-testid="stFileUploaderPagination"] button:hover:not(:disabled) {{
        background: rgba(139, 92, 246, 0.25) !important;
        border-color: {PURPLE} !important;
        transform: none !important;
    }}

    /* Greyed-out arrow when there is no previous / next page */
    [data-testid="stFileUploaderPagination"] button:disabled {{
        opacity: 0.35 !important;
        cursor: not-allowed !important;
    }}

    /* Job description box (outer wrapper that draws the border) */
    .stTextArea [data-baseweb="textarea"],
    .stTextArea [data-baseweb="base-input"] {{
        box-sizing: border-box !important;
        background: {PANEL2} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 14px !important;
    }}

    .stTextArea [data-baseweb="textarea"]:focus-within,
    .stTextArea [data-baseweb="base-input"]:focus-within {{
        border: 1px solid {PURPLE} !important;
    }}

    /* Job description textarea itself */
    .stTextArea textarea {{
        box-sizing: border-box !important;
        height: {INPUT_BOX_HEIGHT}px !important;
        min-height: {INPUT_BOX_HEIGHT}px !important;
        background: {PANEL2} !important;
        color: {TEXT} !important;
        -webkit-text-fill-color: {TEXT} !important;
        caret-color: {CYAN} !important;
        cursor: text !important;
        border-radius: 14px !important;
        border: none !important;
        padding: 14px !important;
    }}

    /* CLEAR "READY TO TYPE" FEEDBACK WHEN THE BOX IS CLICKED */
    .stTextArea [data-baseweb="textarea"]:focus-within,
    .stTextArea [data-baseweb="base-input"]:focus-within {{
        border: 1px solid {CYAN} !important;
        box-shadow: 0 0 0 3px rgba(34, 211, 238, 0.22) !important;
    }}

    .stTextArea textarea:focus {{
        caret-color: {CYAN} !important;
        outline: none !important;
        box-shadow: none !important;
    }}

    /* PLACEHOLDER FADES SLIGHTLY WHEN FOCUSED SO IT FEELS EDITABLE */
    .stTextArea textarea:focus::placeholder {{
        opacity: 0.55 !important;
    }}

    /* GREY PLACEHOLDER TEXT */
    .stTextArea textarea::placeholder {{
        color: {MUTED} !important;
        -webkit-text-fill-color: {MUTED} !important;
        opacity: 1 !important;
    }}

    .stTextArea textarea::-webkit-input-placeholder {{
        color: {MUTED} !important;
        -webkit-text-fill-color: {MUTED} !important;
        opacity: 1 !important;
    }}

    .stTextArea textarea::-moz-placeholder {{
        color: {MUTED} !important;
        opacity: 1 !important;
    }}

    /* ---------- RESUME RANKING ---------- */

    .best-card {{
        background: linear-gradient(120deg, rgba(124,58,237,.25), rgba(34,211,238,.12));
        border: 1px solid rgba(139,92,246,.55);
        border-radius: 18px;
        padding: 20px;
    }}

    .best-name {{
        font-family: 'Manrope', sans-serif;
        font-size: 24px;
        font-weight: 800;
        margin: 8px 0 4px;
        color: {TEXT};
        word-break: break-word;
    }}

    .rank-card {{
        background: {PANEL};
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 16px 18px;
        margin: 10px 0 4px;
    }}

    .rank-card.top {{
        border-color: rgba(139,92,246,.7);
        box-shadow: 0 0 0 1px rgba(139,92,246,.25);
    }}

    .rank-head {{
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 10px;
    }}

    .rank-badge {{
        min-width: 38px;
        height: 38px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 12px;
        font-family: 'Manrope', sans-serif;
        font-weight: 800;
        font-size: 14px;
        color: #FFFFFF;
        background: linear-gradient(135deg, #8B5CF6, #22D3EE);
    }}

    .rank-name {{
        flex: 1;
        min-width: 0;
        font-weight: 700;
        font-size: 15px;
        color: {TEXT};
        word-break: break-word;
    }}

    .rank-score {{
        font-family: 'Manrope', sans-serif;
        font-size: 24px;
        font-weight: 800;
    }}

    .bar-track {{
        width: 100%;
        height: 8px;
        border-radius: 8px;
        background: {PANEL2};
        overflow: hidden;
    }}

    .bar-fill {{
        height: 100%;
        border-radius: 8px;
    }}

    /* ---------- SIDEBAR REDESIGN ---------- */

    [data-testid="stSidebarContent"] {{
        background: transparent !important;
    }}

    /* Brand header */
    .side-brand {{
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 6px 2px 2px;
    }}

    .side-logo {{
        width: 50px;
        height: 50px;
        min-width: 50px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 16px;
        font-size: 24px;
        color: #FFFFFF;
        background: linear-gradient(135deg, #8B5CF6, #22D3EE);
        box-shadow: 0 8px 24px rgba(139, 92, 246, .45);
        animation: logoPulse 3s ease-in-out infinite;
    }}

    @keyframes logoPulse {{
        0% {{ box-shadow: 0 0 0 0 rgba(139, 92, 246, .55); }}
        70% {{ box-shadow: 0 0 0 14px rgba(139, 92, 246, 0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(139, 92, 246, 0); }}
    }}

    .side-brand-name {{
        font-family: 'Manrope', sans-serif;
        font-size: 24px;
        font-weight: 800;
        letter-spacing: -0.5px;
        line-height: 1.1;
        background: linear-gradient(90deg, {BRAND_FROM}, {BRAND_MID}, {BRAND_TO});
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent !important;
        color: transparent !important;
    }}

    .side-tagline {{
        margin-top: 3px;
        font-size: 12px;
        color: {MUTED};
    }}

    .side-divider {{
        height: 1px;
        margin: 20px 0;
        background: linear-gradient(
            90deg,
            transparent,
            rgba(139, 92, 246, .6),
            rgba(34, 211, 238, .4),
            transparent
        );
    }}

    .side-title {{
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 2px;
        color: {CYAN};
        margin: 0 0 10px;
    }}

    /* Navigation + theme: radio options become pill buttons.
       Every pill is forced to the SAME width so Dashboard,
       How it works and About are equal-sized boxes. */
    [data-testid="stSidebar"] [role="radiogroup"] {{
        gap: 7px !important;
        width: 100% !important;
        align-items: stretch !important;
    }}

    [data-testid="stSidebar"] [role="radiogroup"] label {{
        flex: 1 1 auto;
        width: 100% !important;
        min-width: 0 !important;
        box-sizing: border-box !important;
        display: flex !important;
        align-items: center !important;
        margin: 0 !important;
        padding: 11px 14px !important;
        border-radius: 13px;
        border: 1px solid {BORDER};
        background: rgba(148, 163, 184, .07);
        cursor: pointer;
        transition: all .18s ease;
    }}

    [data-testid="stSidebar"] [role="radiogroup"] label > div:first-of-type:not([data-testid="stMarkdownContainer"]) {{
        display: none !important;
    }}

    [data-testid="stSidebar"] [role="radiogroup"] label:hover {{
        background: rgba(139, 92, 246, .18);
        border-color: rgba(139, 92, 246, .55);
        transform: translateY(-1px);
    }}

    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{
        background: linear-gradient(100deg, #7C3AED, #6366F1);
        border-color: #A78BFA;
        box-shadow: 0 8px 20px rgba(124, 58, 237, .38);
    }}

    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) *,
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {{
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }}

    [data-testid="stSidebar"] [role="radiogroup"] label p {{
        font-size: 14px !important;
    }}

    /* Cards inside the sidebar */
    .side-card {{
        background: {PANEL2};
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 14px;
    }}

    .side-card-head {{
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 2px;
        color: {CYAN};
        margin-bottom: 12px;
    }}

    .live-dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: {GREEN};
        box-shadow: 0 0 0 0 rgba(52, 211, 153, .7);
        animation: liveDot 1.8s infinite;
    }}

    .live-dot.idle {{
        background: {MUTED};
        animation: none;
    }}

    @keyframes liveDot {{
        0% {{ box-shadow: 0 0 0 0 rgba(52, 211, 153, .7); }}
        70% {{ box-shadow: 0 0 0 8px rgba(52, 211, 153, 0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(52, 211, 153, 0); }}
    }}

    .side-stats {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
    }}

    .side-stat {{
        background: {PANEL};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 10px 12px;
    }}

    .side-stat-label {{
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1px;
        color: {MUTED};
    }}

    .side-stat-value {{
        font-family: 'Manrope', sans-serif;
        font-size: 24px;
        font-weight: 800;
        margin-top: 2px;
    }}

    .side-top {{
        margin-top: 8px;
        padding: 10px 12px;
        border-radius: 12px;
        font-size: 12.5px;
        color: {TEXT};
        background: linear-gradient(120deg, rgba(124, 58, 237, .25), rgba(34, 211, 238, .12));
        border: 1px solid rgba(139, 92, 246, .45);
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }}

    .side-empty {{
        font-size: 12.5px;
        line-height: 1.55;
        color: {MUTED};
    }}

    .side-step {{
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 7px 0;
        font-size: 13px;
        color: {TEXT};
    }}

    .side-step-num {{
        width: 24px;
        height: 24px;
        min-width: 24px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        font-size: 11px;
        font-weight: 800;
        color: #FFFFFF;
        background: linear-gradient(135deg, #8B5CF6, #22D3EE);
    }}

    .side-tip {{
        padding: 14px;
        border-radius: 14px;
        font-size: 12.5px;
        line-height: 1.55;
        color: {TEXT};
        background: linear-gradient(135deg, rgba(139, 92, 246, .20), rgba(34, 211, 238, .10));
        border: 1px solid rgba(139, 92, 246, .38);
    }}

    .side-footer {{
        margin-top: 14px;
        text-align: center;
        font-size: 11.5px;
        color: {MUTED};
    }}

    /* ---------- TABS ---------- */

    [data-baseweb="tab-list"] button p {{
        color: {TEXT} !important;
        font-weight: 600 !important;
    }}

    hr {{
        border-color: {BORDER};
    }}

    /* =====================================================
       RESPONSIVE LAYOUT (phones, tablets, laptops, desktops)
       ===================================================== */

    /* Never allow sideways page scrolling */
    html, body,
    [data-testid="stAppViewContainer"] {{
        overflow-x: hidden !important;
        max-width: 100vw;
    }}

    /* Columns wrap onto new rows instead of being squeezed */
    [data-testid="stHorizontalBlock"] {{
        flex-wrap: wrap !important;
        row-gap: 1rem !important;
    }}

    /* Top-level columns: never narrower than this */
    [data-testid="stColumn"],
    [data-testid="column"] {{
        min-width: 190px !important;
    }}

    /* Columns nested inside other columns (metric cards) */
    [data-testid="stColumn"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
    [data-testid="column"] [data-testid="stHorizontalBlock"] > [data-testid="column"] {{
        min-width: 150px !important;
    }}

    /* ----- LAPTOP SMALL / TABLET (up to 1100px) ----- */
    @media (max-width: 1100px) {{

        .block-container {{
            padding-left: 1.5rem !important;
            padding-right: 1.5rem !important;
        }}

        [data-testid="stColumn"],
        [data-testid="column"] {{
            min-width: 260px !important;
        }}

        [data-testid="stColumn"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
        [data-testid="column"] [data-testid="stHorizontalBlock"] > [data-testid="column"] {{
            min-width: 150px !important;
        }}

        .hero {{
            padding: 24px;
        }}
    }}

    /* ----- PHONE (up to 640px) ----- */
    @media (max-width: 640px) {{

        .block-container {{
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            padding-top: 3.5rem !important;
            padding-bottom: 2rem !important;
        }}

        [data-testid="stColumn"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
        [data-testid="column"] [data-testid="stHorizontalBlock"] > [data-testid="column"] {{
            min-width: 140px !important;
        }}

        h2 {{
            font-size: 1.5rem !important;
        }}

        h3 {{
            font-size: 1.15rem !important;
        }}

        .hero {{
            padding: 20px;
            border-radius: 18px;
            margin-bottom: 18px;
        }}

        .hero-title {{
            font-size: 26px;
            letter-spacing: -1px;
        }}

        .eyebrow {{
            font-size: 11px;
        }}

        .section-title {{
            font-size: 17px;
            margin: 18px 0 12px;
        }}

        .card {{
            padding: 14px;
            border-radius: 14px;
        }}

        .card-value {{
            font-size: 24px;
        }}

        .best-card {{
            padding: 16px;
        }}

        .best-name {{
            font-size: 20px;
        }}

        .rank-card {{
            padding: 14px;
        }}

        .rank-head {{
            gap: 10px;
        }}

        .rank-badge {{
            min-width: 34px;
            height: 34px;
            font-size: 13px;
        }}

        .rank-name {{
            font-size: 14px;
        }}

        .rank-score {{
            font-size: 20px;
        }}

        .tip {{
            font-size: 13px;
            padding: 12px;
        }}

        .skill {{
            font-size: 11.5px;
            padding: 6px 10px;
        }}

        [data-testid="stFileUploader"] {{
            min-height: auto !important;
            padding: 10px !important;
        }}

        [data-baseweb="tab-list"] button p {{
            font-size: 13px !important;
        }}

        [data-testid="stFileUploaderPagination"] {{
            flex-wrap: wrap !important;
        }}
    }}

    /* ----- SMALL PHONE (up to 380px) ----- */
    @media (max-width: 380px) {{

        .hero-title {{
            font-size: 23px;
        }}

        .card-value {{
            font-size: 21px;
        }}

        .side-brand-name {{
            font-size: 21px;
        }}
    }}
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------- SKILL DATABASE ----------------

SKILLS = [
    "python", "java", "c++", "javascript", "typescript",
    "react", "node.js", "html", "css", "sql", "mysql",
    "postgresql", "mongodb", "nosql", "django", "flask",
    "fastapi", "rest api", "git", "github", "linux", "aws",
    "azure", "docker", "kubernetes", "machine learning",
    "deep learning", "artificial intelligence", "generative ai",
    "nlp", "natural language processing", "pandas", "numpy",
    "scikit-learn", "tensorflow", "pytorch", "data analysis",
    "excel", "power bi", "tableau", "communication",
    "problem solving", "dbms", "operating systems",
    "computer networks", "oops", "agile", "streamlit",
    "data structures", "algorithms", "spring boot",
    "express.js", "php", "testing", "ci/cd", "statistics"
]


# ---------------- HELPER FUNCTIONS ----------------

def extract_pdf_text(uploaded_file):
    text_parts = []

    with pdfplumber.open(
        io.BytesIO(uploaded_file.getvalue())
    ) as pdf:
        for page in pdf.pages:
            text_parts.append(page.extract_text() or "")

    return "\n".join(text_parts).strip()


def normalize(text):
    return re.sub(r"\s+", " ", text.lower())


def find_skills(text):
    clean_text = normalize(text)
    found = []

    for skill in SKILLS:
        pattern = (
            r"(?<![a-z0-9])"
            + re.escape(skill)
            + r"(?![a-z0-9])"
        )

        if re.search(pattern, clean_text):
            found.append(skill)

    return sorted(set(found))


def calculate_similarity(resume_text, job_text):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    try:
        matrix = vectorizer.fit_transform(
            [resume_text, job_text]
        )

        score = cosine_similarity(
            matrix[0:1],
            matrix[1:2]
        )[0][0]

        return round(float(score) * 100, 1)

    except ValueError:
        return 0.0


def make_card(label, value, note, color):
    return f"""
    <div class="card">
        <div class="card-label">{html.escape(label)}</div>
        <div class="card-value" style="color:{color}">
            {html.escape(str(value))}
        </div>
        <div class="card-note">{html.escape(note)}</div>
    </div>
    """


def short_label(rank, name, limit=22):
    """Short chart label so long file names never squash the charts."""
    if len(name) > limit:
        name = name[: limit - 1] + "…"
    return f"#{rank} {name}"


# ---------------- DETAILED INSIGHT HELPERS ----------------

SECTION_PATTERNS = {
    "Summary / Objective": r"\b(summary|objective|profile|about me)\b",
    "Education": r"\b(education|academic|qualifications?)\b",
    "Experience": r"\b(experience|internships?|employment|work history)\b",
    "Projects": r"\bprojects?\b",
    "Skills": r"\b(skills|technologies|technical skills)\b",
    "Certifications": r"\b(certifications?|certificates?|courses)\b",
    "Achievements": r"\b(achievements?|awards?|honou?rs?|accomplishments?)\b"
}


def read_pdf(uploaded_file):
    text_parts = []

    with pdfplumber.open(
        io.BytesIO(uploaded_file.getvalue())
    ) as pdf:
        pages = len(pdf.pages)

        for page in pdf.pages:
            text_parts.append(page.extract_text() or "")

    return "\n".join(text_parts).strip(), pages


def analyze_structure(text):
    clean_text = normalize(text)

    found_sections = []
    missing_sections = []

    for name, pattern in SECTION_PATTERNS.items():
        if re.search(pattern, clean_text):
            found_sections.append(name)
        else:
            missing_sections.append(name)

    phone_matches = re.findall(r"\+?\d[\d\s().-]{8,}\d", text)

    has_phone = any(
        len(re.sub(r"\D", "", match)) >= 10
        for match in phone_matches
    )

    contacts = {
        "Email": bool(re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)),
        "Phone": has_phone,
        "LinkedIn": "linkedin.com" in clean_text,
        "GitHub": "github.com" in clean_text
    }

    numbers = len(
        re.findall(r"\d+(?:\.\d+)?\s?%|\$\s?\d+|\b\d+\+", text)
    )

    return {
        "sections": found_sections,
        "missing_sections": missing_sections,
        "contacts": contacts,
        "numbers": numbers
    }


def shared_keywords(resume_text, job_text, top_n=10):
    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z+#]{2,}\b"
        )

        matrix = vectorizer.fit_transform(
            [resume_text, job_text]
        ).toarray()

        terms = vectorizer.get_feature_names_out()

    except ValueError:
        return []

    scored = [
        (terms[i], matrix[0][i] * matrix[1][i])
        for i in range(len(terms))
        if matrix[0][i] > 0 and matrix[1][i] > 0
    ]

    scored.sort(key=lambda item: item[1], reverse=True)

    return [term for term, _ in scored[:top_n]]


def chips(items, css_class="", empty="None"):
    if not items:
        return (
            '<span class="muted">'
            + html.escape(empty)
            + "</span>"
        )

    return "".join(
        f'<span class="skill {css_class}">'
        + html.escape(str(item))
        + "</span>"
        for item in items
    )


def build_suggestions(row):
    tips = []

    if row["missing"]:
        tips.append(
            "Job skills not detected: "
            + ", ".join(row["missing"][:8])
            + ". Add them only if the candidate genuinely has them."
        )

    if "Projects" in row["missing_sections"]:
        tips.append(
            "No Projects section detected. Add projects with the "
            "tools used and the results achieved."
        )

    if "Experience" in row["missing_sections"]:
        tips.append(
            "No Experience or Internship section detected. "
            "Add practical experience if available."
        )

    if "Skills" in row["missing_sections"]:
        tips.append(
            "No dedicated Skills section detected. A clear skills "
            "section makes screening easier."
        )

    if row["numbers"] < 2:
        tips.append(
            "Few quantified results found. Add numbers, such as "
            "percentages, users, or time saved."
        )

    if row["words"] < 200:
        tips.append(
            "The resume is quite short. Add more detail about "
            "projects and responsibilities."
        )
    elif row["words"] > 900:
        tips.append(
            "The resume is long. Consider shortening it to the "
            "most relevant content."
        )

    if row["pages"] > 2:
        tips.append(
            "The resume is over 2 pages. One or two pages is "
            "usually preferred."
        )

    missing_contacts = [
        name for name, present in row["contacts"].items()
        if not present
    ]

    if missing_contacts:
        tips.append(
            "Contact details not detected: "
            + ", ".join(missing_contacts)
            + "."
        )

    if row["text"] < 10:
        tips.append(
            "Low wording overlap with the job description. Use "
            "accurate terminology from the role where it applies."
        )

    if not tips:
        tips.append(
            "This resume looks well structured and well matched "
            "to the job description."
        )

    return tips


# ---------------- SIDEBAR HELPERS ----------------

NAV_LABELS = {
    "Dashboard": "🏠  Dashboard",
    "How it works": "💡  How it works",
    "About": "ℹ️  About"
}

THEME_LABELS = {
    "Dark": "🌙  Dark",
    "Light": "☀️  Light"
}


def side_divider():
    st.markdown(
        '<div class="side-divider"></div>',
        unsafe_allow_html=True
    )


def build_sidebar_status():
    analysis = st.session_state.analysis

    if not analysis:
        return (
            '<div class="side-card">'
            '<div class="side-card-head">'
            '<span class="live-dot idle"></span>LIVE STATUS</div>'
            '<div class="side-empty">No analysis yet. Upload resumes, '
            'paste a job description and press '
            '<b>Screen resumes</b>.</div>'
            '</div>'
        )

    rows = analysis["rows"]
    top = rows[0]
    average = sum(r["overall"] for r in rows) / len(rows)

    return (
        '<div class="side-card">'
        '<div class="side-card-head">'
        '<span class="live-dot"></span>LIVE STATUS</div>'
        '<div class="side-stats">'
        '<div class="side-stat">'
        '<div class="side-stat-label">SCREENED</div>'
        f'<div class="side-stat-value" style="color:{PURPLE}">'
        f'{len(rows)}</div>'
        '</div>'
        '<div class="side-stat">'
        '<div class="side-stat-label">TOP SCORE</div>'
        f'<div class="side-stat-value" style="color:{GREEN}">'
        f'{top["overall"]:.0f}%</div>'
        '</div>'
        '<div class="side-stat">'
        '<div class="side-stat-label">AVERAGE</div>'
        f'<div class="side-stat-value" style="color:{CYAN}">'
        f'{average:.0f}%</div>'
        '</div>'
        '<div class="side-stat">'
        '<div class="side-stat-label">JOB SKILLS</div>'
        f'<div class="side-stat-value" style="color:{TEXT}">'
        f'{len(analysis["job_skills"])}</div>'
        '</div>'
        '</div>'
        f'<div class="side-top">🏆 {html.escape(top["name"])}</div>'
        '</div>'
    )


# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.markdown(
        '<div class="side-brand">'
        '<div class="side-logo">✦</div>'
        '<div>'
        '<div class="side-brand-name">ResumeIQ</div>'
        '<div class="side-tagline">AI-powered career toolkit</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

    side_divider()

    st.markdown(
        '<div class="side-title">MENU</div>',
        unsafe_allow_html=True
    )

    page = st.radio(
        "Workspace",
        list(NAV_LABELS.keys()),
        format_func=lambda option: NAV_LABELS[option],
        label_visibility="collapsed"
    )

    side_divider()

    # Filled at the end of the script so it always shows fresh results
    status_slot = st.empty()

    side_divider()

    st.markdown(
        '<div class="side-title">QUICK GUIDE</div>'
        '<div class="side-step">'
        '<span class="side-step-num">1</span>'
        '<span>Upload your PDF resumes</span></div>'
        '<div class="side-step">'
        '<span class="side-step-num">2</span>'
        '<span>Paste the job description</span></div>'
        '<div class="side-step">'
        '<span class="side-step-num">3</span>'
        '<span>Screen and compare the ranking</span></div>',
        unsafe_allow_html=True
    )

    side_divider()

    st.markdown(
        '<div class="side-title">APPEARANCE</div>',
        unsafe_allow_html=True
    )

    theme_choice = st.radio(
        "Choose theme",
        ["Dark", "Light"],
        index=0 if st.session_state.theme == "Dark" else 1,
        horizontal=True,
        format_func=lambda option: THEME_LABELS[option],
        label_visibility="collapsed"
    )

    if theme_choice != st.session_state.theme:
        st.session_state.theme = theme_choice
        st.rerun()

    st.markdown(
        '<div class="side-title" style="margin-top:20px">'
        'PREFERENCES</div>',
        unsafe_allow_html=True
    )

    show_details = st.toggle(
        "Show detailed analysis",
        value=True
    )

    side_divider()

    st.markdown(
        '<div class="side-tip">'
        '💡 <b>Pro tip</b><br>'
        'Use text-based PDFs (not scanned images) and paste a detailed '
        'job description for the most accurate ranking.'
        '</div>'
        '<div class="side-footer">'
        'ResumeIQ · Career Intelligence'
        '</div>',
        unsafe_allow_html=True
    )


# ---------------- HERO ----------------

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">CAREER INTELLIGENCE</div>
        <div class="hero-title">
            Make your next application<br>
            your strongest one.
        </div>
        <p class="muted">
            Analyze your resume, discover skill gaps, and prepare
            for your target role with a smarter resume workflow.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------- INFORMATION PAGES ----------------

if page == "How it works":

    st.markdown("## How ResumeIQ works")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            make_card(
                "STEP 01", "Upload",
                "Upload many text-based PDF resumes.",
                PURPLE
            ),
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            make_card(
                "STEP 02", "Compare",
                "Paste your target job description.",
                CYAN
            ),
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            make_card(
                "STEP 03", "Improve",
                "See which resume ranks best.",
                GREEN
            ),
            unsafe_allow_html=True
        )

    st.markdown("### Technologies used")

    st.write(
        "Python, Streamlit, pdfplumber, Pandas, "
        "TF-IDF, cosine similarity, and Plotly."
    )

    st.info(
        "The score represents text similarity, not a real ATS "
        "score or probability of getting hired."
    )


elif page == "About":

    st.markdown("## About ResumeIQ")

    st.write(
        "ResumeIQ is a resume analysis project built with Python "
        "and traditional NLP techniques."
    )

    st.markdown("### Features")

    st.write(
        "- Multiple PDF resume upload\n"
        "- Job description comparison\n"
        "- Ranked resume screening\n"
        "- Skill matching and gap detection\n"
        "- Interactive visual dashboard\n"
        "- Dark and light themes\n"
        "- Downloadable ranking (CSV)"
    )

    st.warning(
        "Skill detection uses a predefined keyword list. "
        "It may miss synonyms and cannot verify a candidate's experience."
    )


# ---------------- MAIN DASHBOARD ----------------

else:

    st.markdown(
        '<div class="section-title">01 / Upload and configure</div>',
        unsafe_allow_html=True
    )

    # EQUAL WIDTH COLUMNS: 50% EACH (stack automatically on small screens)
    left, right = st.columns([1, 1], gap="large")

    with left:

        st.markdown("### Your resumes")

        resume_files = st.file_uploader(
            "Upload PDF resumes (select multiple files)",
            type=["pdf"],
            accept_multiple_files=True,
            help="Select as many text-based PDF resumes as you want."
        )

        if resume_files:
            st.success(f"Ready: {len(resume_files)} resume(s) uploaded")

    with right:

        st.markdown("### Target job description")

        job_description = st.text_area(
            "Paste job description",
            height=INPUT_BOX_HEIGHT,
            placeholder=(
                "Paste the job responsibilities, required skills, "
                "qualifications, and experience..."
            ),
            label_visibility="collapsed"
        )

    # EQUAL WIDTH COLUMNS: 50% EACH

    run_col, reset_col = st.columns(
        [1, 1],
        gap="small"
    )

    with run_col:

        analyze = st.button(
            "✦ Screen resumes",
            type="primary",
            use_container_width=True,
            key="analyze_resume"
        )

    with reset_col:

        reset = st.button(
            "↻ Reset results",
            use_container_width=True,
            key="reset_results"
        )

    if reset:
        st.session_state.analysis = None
        st.rerun()

    # ---------------- SCREENING LOGIC ----------------

    if analyze:

        if not resume_files:

            st.error("Please upload at least one PDF resume.")

        elif not job_description.strip():

            st.error("Please paste a job description.")

        else:

            job_skills_list = find_skills(job_description)
            rows = []
            failed = []
            total = len(resume_files)

            progress = st.progress(0, text="Screening resumes...")

            seen_names = {}

            for index, file in enumerate(resume_files, 1):

                seen_names[file.name] = seen_names.get(file.name, 0) + 1

                if seen_names[file.name] == 1:
                    display_name = file.name
                else:
                    display_name = f"{file.name} ({seen_names[file.name]})"

                try:

                    text, pages = read_pdf(file)

                    if len(text) < 30:

                        failed.append(file.name)

                    else:

                        text_score = calculate_similarity(
                            text,
                            job_description
                        )

                        resume_skills = find_skills(text)

                        matched = sorted(
                            set(resume_skills) & set(job_skills_list)
                        )

                        missing = sorted(
                            set(job_skills_list) - set(resume_skills)
                        )

                        if job_skills_list:

                            coverage = round(
                                len(matched) / len(job_skills_list) * 100,
                                1
                            )

                            overall = round(
                                0.5 * text_score + 0.5 * coverage,
                                1
                            )

                        else:

                            coverage = None
                            overall = text_score

                        structure = analyze_structure(text)

                        rows.append({
                            "name": display_name,
                            "overall": overall,
                            "text": text_score,
                            "coverage": coverage,
                            "matched": matched,
                            "missing": missing,
                            "extra": sorted(
                                set(resume_skills) - set(job_skills_list)
                            ),
                            "all_skills": resume_skills,
                            "words": len(text.split()),
                            "pages": pages,
                            "sections": structure["sections"],
                            "missing_sections": structure["missing_sections"],
                            "contacts": structure["contacts"],
                            "numbers": structure["numbers"],
                            "keywords": shared_keywords(
                                text,
                                job_description
                            )
                        })

                except Exception:

                    failed.append(file.name)

                progress.progress(
                    index / total,
                    text=f"Screened {index} of {total} resumes"
                )

            progress.empty()

            if rows:

                rows.sort(
                    key=lambda r: (r["overall"], r["text"]),
                    reverse=True
                )

                st.session_state.analysis = {
                    "rows": rows,
                    "failed": failed,
                    "job_skills": job_skills_list
                }

            else:

                st.session_state.analysis = None

                st.error(
                    "No readable text was found in the uploaded files. "
                    "Try text-based PDFs instead of scanned images."
                )

    # ---------------- RESULTS ----------------

    result = st.session_state.analysis

    if result:

        rows = result["rows"]
        failed = result["failed"]
        job_skills = result["job_skills"]
        best = rows[0]
        average = sum(r["overall"] for r in rows) / len(rows)

        def score_color(value):
            if value >= 60:
                return GREEN
            if value >= 35:
                return CYAN
            return RED

        def score_label(value):
            if value >= 60:
                return "Strong match"
            if value >= 35:
                return "Moderate match"
            return "Low match"

        st.divider()

        # -------- BEST MATCH --------

        st.markdown(
            '<div class="section-title">02 / Best resume for this job</div>',
            unsafe_allow_html=True
        )

        score_col, metrics_col = st.columns(
            [1, 2],
            gap="large"
        )

        with score_col:

            fig = go.Figure(
                data=[
                    go.Pie(
                        values=[best["overall"], 100 - best["overall"]],
                        hole=0.78,
                        sort=False,
                        direction="clockwise",
                        marker=dict(
                            colors=[PURPLE, PANEL2]
                        ),
                        textinfo="none",
                        hoverinfo="skip"
                    )
                ]
            )

            fig.add_annotation(
                text=(
                    f"<b>{best['overall']:.0f}%</b>"
                    "<br><span style='font-size:12px'>BEST SCORE</span>"
                ),
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=25, color=TEXT)
            )

            fig.update_layout(
                height=260,
                margin=dict(l=0, r=0, t=5, b=5),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False}
            )

        with metrics_col:

            best_name = html.escape(best["name"])

            st.markdown(
                '<div class="best-card">'
                '<div class="card-label">🏆 TOP CANDIDATE</div>'
                f'<div class="best-name">{best_name}</div>'
                f'<div class="card-note">{score_label(best["overall"])} · '
                f'{len(best["matched"])} matching skills · '
                f'{len(best["missing"])} skills to review</div>'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown("<br>", unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)

            with m1:
                st.markdown(
                    make_card(
                        "RESUMES SCREENED",
                        len(rows),
                        "Successfully analyzed",
                        PURPLE
                    ),
                    unsafe_allow_html=True
                )

            with m2:
                st.markdown(
                    make_card(
                        "AVERAGE SCORE",
                        f"{average:.0f}%",
                        "Across all resumes",
                        CYAN
                    ),
                    unsafe_allow_html=True
                )

            with m3:
                st.markdown(
                    make_card(
                        "JOB SKILLS FOUND",
                        len(job_skills),
                        "Detected in description",
                        GREEN
                    ),
                    unsafe_allow_html=True
                )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            '<div class="notice">'
            'Score = 50% text similarity (TF-IDF + cosine similarity) '
            '+ 50% skill coverage of the job description. '
            'This is a screening aid, not a hiring decision or a real ATS score.'
            '</div>',
            unsafe_allow_html=True
        )

        if failed:

            st.warning(
                "Could not read these files: " + ", ".join(failed)
            )

        # -------- RANKED LIST --------

        st.markdown(
            '<div class="section-title">03 / Resume screening ranking</div>',
            unsafe_allow_html=True
        )

        for rank, row in enumerate(rows, 1):

            color = score_color(row["overall"])
            width = min(max(row["overall"], 0), 100)
            top_class = " top" if rank == 1 else ""

            if row["coverage"] is None:
                coverage_text = "N/A"
            else:
                coverage_text = f"{row['coverage']:.0f}%"

            matched_chips = "".join(
                '<span class="skill matched">'
                + html.escape(skill)
                + "</span>"
                for skill in row["matched"]
            ) or '<span class="muted">None detected</span>'

            missing_chips = "".join(
                '<span class="skill missing">'
                + html.escape(skill)
                + "</span>"
                for skill in row["missing"]
            ) or '<span class="muted">None - all job skills found</span>'

            card_html = (
                f'<div class="rank-card{top_class}">'
                '<div class="rank-head">'
                f'<div class="rank-badge">#{rank}</div>'
                f'<div class="rank-name">{html.escape(row["name"])}</div>'
                f'<div class="rank-score" style="color:{color}">'
                f'{row["overall"]:.0f}%</div>'
                '</div>'
                '<div class="bar-track">'
                f'<div class="bar-fill" style="width:{width}%;'
                f'background:{color}"></div>'
                '</div>'
                '<div class="muted" style="margin-top:8px">'
                f'{score_label(row["overall"])} · '
                f'Text match {row["text"]:.0f}% · '
                f'Skill coverage {coverage_text} · '
                f'{len(row["matched"])} matched · '
                f'{len(row["missing"])} to review'
                '</div>'
                '</div>'
            )

            st.markdown(card_html, unsafe_allow_html=True)

            if show_details:

                with st.expander(f"Skill details for {row['name']}"):

                    st.markdown(
                        '<div class="card-label">MATCHING SKILLS</div>'
                        + matched_chips
                        + '<div class="card-label" style="margin-top:12px">'
                        'SKILLS TO REVIEW</div>'
                        + missing_chips,
                        unsafe_allow_html=True
                    )

        # -------- TABLE DATA (used by insights and download) --------

        # chart_data has a short "Label" column used only by the charts.
        chart_data = pd.DataFrame([
            {
                "Rank": rank,
                "Label": short_label(rank, row["name"]),
                "Resume": row["name"],
                "Overall %": row["overall"],
                "Text match %": row["text"],
                "Skill coverage %": (
                    row["coverage"]
                    if row["coverage"] is not None
                    else None
                ),
                "Matched skills": len(row["matched"]),
                "Skills to review": len(row["missing"]),
                "Words": row["words"],
                "Pages": row["pages"]
            }
            for rank, row in enumerate(rows, 1)
        ])

        # table_data is what the user sees in tables and downloads.
        table_data = chart_data.drop(columns=["Label"])

        # -------- ALL DETAILED INSIGHTS (OPTIONAL) --------

        st.markdown(
            '<div class="section-title">04 / Detailed insights</div>',
            unsafe_allow_html=True
        )

        show_insights = st.toggle(
            "🔎 View all detailed insights",
            value=False,
            help="Open an in-depth view with a per-resume deep dive, "
                 "a skills matrix, and score comparison charts."
        )

        if show_insights:

            tab_deep, tab_matrix, tab_scores = st.tabs([
                "🔍 Resume deep dive",
                "🧩 Skills matrix",
                "📊 Score comparison"
            ])

            # ----- TAB 1: DEEP DIVE -----

            with tab_deep:

                labels = [
                    f"#{rank} · {row['name']}"
                    for rank, row in enumerate(rows, 1)
                ]

                choice = st.selectbox(
                    "Select a resume to inspect",
                    labels
                )

                selected = rows[labels.index(choice)]
                gap = best["overall"] - selected["overall"]

                d1, d2, d3, d4 = st.columns(4)

                with d1:
                    st.markdown(
                        make_card(
                            "OVERALL SCORE",
                            f"{selected['overall']:.0f}%",
                            score_label(selected["overall"]),
                            score_color(selected["overall"])
                        ),
                        unsafe_allow_html=True
                    )

                with d2:
                    st.markdown(
                        make_card(
                            "TEXT MATCH",
                            f"{selected['text']:.0f}%",
                            "TF-IDF similarity",
                            CYAN
                        ),
                        unsafe_allow_html=True
                    )

                with d3:
                    coverage_value = (
                        "N/A"
                        if selected["coverage"] is None
                        else f"{selected['coverage']:.0f}%"
                    )

                    st.markdown(
                        make_card(
                            "SKILL COVERAGE",
                            coverage_value,
                            "Job skills found in resume",
                            PURPLE
                        ),
                        unsafe_allow_html=True
                    )

                with d4:
                    st.markdown(
                        make_card(
                            "LENGTH",
                            f"{selected['words']} words",
                            f"{selected['pages']} page(s)",
                            GREEN
                        ),
                        unsafe_allow_html=True
                    )

                st.markdown("<br>", unsafe_allow_html=True)

                if gap <= 0:
                    gap_text = (
                        "This is the top-ranked resume for the "
                        "given job description."
                    )
                else:
                    gap_text = (
                        f"This resume scores {gap:.0f} points "
                        f"below the top candidate ({best['name']})."
                    )

                st.markdown(
                    f'<div class="notice">{html.escape(gap_text)}</div>',
                    unsafe_allow_html=True
                )

                st.markdown("<br>", unsafe_allow_html=True)

                info_left, info_right = st.columns(2, gap="large")

                with info_left:

                    contact_found = [
                        name for name, present
                        in selected["contacts"].items() if present
                    ]

                    contact_missing = [
                        name for name, present
                        in selected["contacts"].items() if not present
                    ]

                    st.markdown(
                        '<div class="card">'
                        '<div class="card-label">RESUME SECTIONS FOUND</div>'
                        + chips(
                            selected["sections"],
                            "matched",
                            "No standard sections detected"
                        )
                        + '<div class="card-label" style="margin-top:14px">'
                        'SECTIONS NOT DETECTED</div>'
                        + chips(
                            selected["missing_sections"],
                            "missing",
                            "All standard sections present"
                        )
                        + '<div class="card-label" style="margin-top:14px">'
                        'CONTACT DETAILS FOUND</div>'
                        + chips(
                            contact_found,
                            "matched",
                            "None detected"
                        )
                        + '<div class="card-label" style="margin-top:14px">'
                        'CONTACT DETAILS NOT DETECTED</div>'
                        + chips(
                            contact_missing,
                            "missing",
                            "All detected"
                        )
                        + '</div>',
                        unsafe_allow_html=True
                    )

                with info_right:

                    st.markdown(
                        '<div class="card">'
                        '<div class="card-label">'
                        'TOP KEYWORDS SHARED WITH THE JOB DESCRIPTION</div>'
                        + chips(
                            selected["keywords"],
                            "",
                            "No shared keywords found"
                        )
                        + '<div class="card-label" style="margin-top:14px">'
                        'QUANTIFIED RESULTS DETECTED</div>'
                        f'<div class="card-value" style="color:{CYAN}">'
                        f'{selected["numbers"]}</div>'
                        '<div class="card-note">'
                        'Percentages, amounts and numbers with "+"'
                        '</div>'
                        '</div>',
                        unsafe_allow_html=True
                    )

                st.markdown("<br>", unsafe_allow_html=True)

                sk1, sk2, sk3 = st.columns(3, gap="large")

                with sk1:
                    st.markdown(
                        '<div class="card">'
                        '<div class="card-label">MATCHING SKILLS</div>'
                        + chips(
                            selected["matched"],
                            "matched",
                            "None detected"
                        )
                        + '</div>',
                        unsafe_allow_html=True
                    )

                with sk2:
                    st.markdown(
                        '<div class="card">'
                        '<div class="card-label">SKILLS TO REVIEW</div>'
                        + chips(
                            selected["missing"],
                            "missing",
                            "No gaps - all job skills found"
                        )
                        + '</div>',
                        unsafe_allow_html=True
                    )

                with sk3:
                    st.markdown(
                        '<div class="card">'
                        '<div class="card-label">ADDITIONAL SKILLS</div>'
                        + chips(
                            selected["extra"],
                            "",
                            "No other skills detected"
                        )
                        + '</div>',
                        unsafe_allow_html=True
                    )

                st.markdown(
                    '<div class="section-title">'
                    'Suggestions for this resume</div>',
                    unsafe_allow_html=True
                )

                for number, tip in enumerate(
                    build_suggestions(selected), 1
                ):
                    st.markdown(
                        f'<div class="tip">'
                        f'<b style="color:{PURPLE}">{number:02d}</b>'
                        f' &nbsp; {html.escape(tip)}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

            # ----- TAB 2: SKILLS MATRIX -----

            with tab_matrix:

                if job_skills:

                    st.markdown("##### Which resume has which job skill")

                    matrix_data = {"Skill": job_skills}

                    for row in rows:
                        matrix_data[row["name"]] = [
                            "✅" if skill in row["matched"] else "❌"
                            for skill in job_skills
                        ]

                    st.dataframe(
                        pd.DataFrame(matrix_data),
                        use_container_width=True,
                        hide_index=True
                    )

                    st.markdown(
                        "##### How common each job skill is "
                        "across all resumes"
                    )

                    counts = [
                        sum(1 for row in rows if skill in row["matched"])
                        for skill in job_skills
                    ]

                    demand = pd.DataFrame({
                        "Skill": job_skills,
                        "Resumes with skill": counts,
                        "Share of resumes %": [
                            round(count / len(rows) * 100)
                            for count in counts
                        ]
                    }).sort_values("Share of resumes %")

                    st.dataframe(
                        demand,
                        use_container_width=True,
                        hide_index=True
                    )

                    st.caption(
                        "Skills at the top are the hardest to find "
                        "among your candidates."
                    )

                else:

                    st.info(
                        "No known skills were detected in the job "
                        "description, so there is no skills matrix to "
                        "show. Try adding specific skills such as "
                        "Python, SQL, or Docker."
                    )

            # ----- TAB 3: SCORE COMPARISON -----

            with tab_scores:

                st.markdown("##### Overall score by resume")

                fig2 = px.bar(
                    chart_data,
                    x="Overall %",
                    y="Label",
                    orientation="h",
                    color_discrete_sequence=[PURPLE],
                    text="Overall %",
                    hover_name="Resume"
                )

                fig2.update_traces(
                    texttemplate="%{text:.0f}%",
                    textposition="outside",
                    cliponaxis=False
                )

                fig2.update_layout(
                    height=max(260, 55 * len(rows) + 60),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color=TEXT,
                    margin=dict(l=10, r=40, t=10, b=30),
                    xaxis=dict(
                        range=[0, 115],
                        title="",
                        showgrid=False
                    ),
                    yaxis=dict(
                        title="",
                        autorange="reversed",
                        automargin=True
                    ),
                    showlegend=False
                )

                st.plotly_chart(
                    fig2,
                    use_container_width=True,
                    config={"displayModeBar": False}
                )

                st.markdown("##### Text match vs skill coverage")

                chart_long = chart_data.melt(
                    id_vars=["Label", "Resume"],
                    value_vars=["Text match %", "Skill coverage %"],
                    var_name="Metric",
                    value_name="Score"
                )

                fig3 = px.bar(
                    chart_long,
                    x="Label",
                    y="Score",
                    color="Metric",
                    barmode="group",
                    color_discrete_map={
                        "Text match %": CYAN,
                        "Skill coverage %": PURPLE
                    },
                    hover_name="Resume"
                )

                fig3.update_layout(
                    height=400,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color=TEXT,
                    margin=dict(l=10, r=10, t=40, b=20),
                    xaxis=dict(
                        tickangle=-35,
                        title="",
                        automargin=True
                    ),
                    yaxis=dict(
                        title="",
                        range=[0, 105],
                        automargin=True
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="left",
                        x=0,
                        title_text=""
                    )
                )

                st.plotly_chart(
                    fig3,
                    use_container_width=True,
                    config={"displayModeBar": False}
                )

                st.markdown("##### Full screening table")

                st.dataframe(
                    table_data,
                    use_container_width=True,
                    hide_index=True
                )

        # -------- DOWNLOAD REPORT --------

        st.download_button(
            "⬇ Download ranking (CSV)",
            data=table_data.to_csv(index=False),
            file_name="resumeiq_ranking.csv",
            mime="text/csv",
            use_container_width=True
        )

    else:

        st.markdown(
            f"""
            <div class="card" style="text-align:center;padding:35px">
                <div style="font-size:36px;color:{PURPLE}">✦</div>
                <h3>Your resume ranking will appear here</h3>
                <p class="muted">
                    Upload multiple resumes and add a job description to
                    see which resume is the best match.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


# ---------------- SIDEBAR LIVE STATUS (fresh results) ----------------

status_slot.markdown(
    build_sidebar_status(),
    unsafe_allow_html=True
)


# ---------------- FOOTER ----------------

st.divider()

st.markdown(
    f"""
    <p class="muted" style="text-align:center">
        ResumeIQ · Built with Python, Streamlit, NLP and scikit-learn
        · {st.session_state.theme} mode
    </p>
    """,
    unsafe_allow_html=True
)