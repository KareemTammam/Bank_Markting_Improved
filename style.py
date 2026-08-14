"""Design system: colors, fonts, and shared CSS for the whole app."""

import streamlit as st

# ---------------------------------------------------------------------------
# Color system (used in Python for charts AND mirrored in the CSS below)
# ---------------------------------------------------------------------------
COLORS = {
    "bg": "#FFFFFF",
    "surface": "#FFFFFF",
    "border": "#E5E9F0",
    "text": "#0F172A",       # charcoal / dark navy
    "text_muted": "#64748B",
    "primary": "#2563EB",    # blue
    "primary_dark": "#1D4ED8",
    "primary_light": "#EFF4FF",
    "success": "#10B981",    # green
    "success_light": "#ECFDF5",
    "warning": "#F59E0B",    # orange
    "warning_light": "#FFFBEB",
    "danger": "#EF4444",     # red
    "danger_light": "#FEF2F2",
    "navy": "#0F172A",
}

METRIC_COLORS = {
    "Accuracy": COLORS["primary"],
    "Precision": COLORS["warning"],
    "Recall": COLORS["success"],
    "F1": COLORS["danger"],
}

MODEL_PALETTE = [
    "#2563EB", "#10B981", "#F59E0B", "#8B5CF6",
    "#EF4444", "#06B6D4", "#EC4899", "#84CC16",
]


def inject_css():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}

        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}

        .stApp {{
            background-color: {COLORS['bg']};
        }}

        section[data-testid="stSidebar"] {{
            background-color: {COLORS['surface']};
            border-right: 1px solid {COLORS['border']};
        }}

        section[data-testid="stSidebar"] .block-container {{
            padding-top: 1.5rem;
        }}

        .block-container {{
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }}

        h1, h2, h3, h4 {{
            color: {COLORS['text']} !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em;
        }}

        p, li, span, label {{
            color: {COLORS['text']};
        }}

        .muted {{
            color: {COLORS['text_muted']};
        }}

        /* ---- Sidebar nav radio styled as pills ---- */
        section[data-testid="stSidebar"] div[role="radiogroup"] {{
            gap: 4px;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] label {{
            background-color: transparent;
            border-radius: 10px;
            padding: 10px 14px;
            width: 100%;
            transition: background-color 0.15s ease;
            font-weight: 500;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
            background-color: {COLORS['primary_light']};
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] {{
            background-color: {COLORS['primary']};
        }}

        /* ---- Buttons ---- */
        .stButton > button {{
            background-color: {COLORS['primary']};
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.6rem 1.4rem;
            font-weight: 600;
            transition: all 0.15s ease;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
        }}
        .stButton > button:hover {{
            background-color: {COLORS['primary_dark']};
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
            transform: translateY(-1px);
        }}
        .stButton > button p {{ color: white !important; font-weight: 600; }}

        /* ---- Generic card ---- */
        .card {{
            background-color: {COLORS['bg']};
            border: 1px solid {COLORS['border']};
            border-radius: 16px;
            padding: 1.5rem 1.6rem;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
            transition: box-shadow 0.15s ease;
        }}
        .card:hover {{
            box-shadow: 0 6px 20px rgba(15, 23, 42, 0.08);
        }}

        /* ---- KPI card ---- */
        .kpi-card {{
            background-color: {COLORS['bg']};
            border: 1px solid {COLORS['border']};
            border-radius: 16px;
            padding: 1.3rem 1.4rem;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        }}
        .kpi-label {{
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            color: {COLORS['text_muted']};
            text-transform: uppercase;
            margin-bottom: 0.4rem;
        }}
        .kpi-value {{
            font-size: 1.9rem;
            font-weight: 800;
            color: {COLORS['text']};
            line-height: 1.1;
        }}
        .kpi-sub {{
            font-size: 0.8rem;
            color: {COLORS['text_muted']};
            margin-top: 0.3rem;
        }}

        /* ---- Badges ---- */
        .badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.02em;
        }}
        .badge-blue {{ background-color: {COLORS['primary_light']}; color: {COLORS['primary_dark']}; }}
        .badge-green {{ background-color: {COLORS['success_light']}; color: #047857; }}
        .badge-orange {{ background-color: {COLORS['warning_light']}; color: #B45309; }}

        /* ---- Hero ---- */
        .hero-title {{
            font-size: 2.6rem;
            font-weight: 800;
            color: {COLORS['text']};
            line-height: 1.15;
            letter-spacing: -0.03em;
            margin-bottom: 0.6rem;
        }}
        .hero-sub {{
            font-size: 1.1rem;
            color: {COLORS['text_muted']};
            max-width: 640px;
            line-height: 1.55;
        }}

        /* ---- Prediction result ---- */
        .result-card {{
            border-radius: 20px;
            padding: 2rem 2.2rem;
            text-align: center;
            border: 1px solid;
        }}
        .result-yes {{
            background-color: {COLORS['success_light']};
            border-color: #A7F3D0;
        }}
        .result-no {{
            background-color: {COLORS['warning_light']};
            border-color: #FDE68A;
        }}
        .result-label {{
            font-size: 0.85rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: {COLORS['text_muted']};
        }}
        .result-value {{
            font-size: 2.4rem;
            font-weight: 800;
            margin: 0.3rem 0 0.6rem 0;
        }}
        .result-yes .result-value {{ color: #047857; }}
        .result-no .result-value {{ color: #B45309; }}

        hr {{
            border-color: {COLORS['border']};
        }}

        div[data-testid="stMetricValue"] {{
            font-weight: 800;
            color: {COLORS['text']};
        }}

        .step-num {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 28px; height: 28px;
            border-radius: 50%;
            background-color: {COLORS['primary_light']};
            color: {COLORS['primary_dark']};
            font-weight: 700;
            font-size: 0.85rem;
            margin-right: 0.5rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
