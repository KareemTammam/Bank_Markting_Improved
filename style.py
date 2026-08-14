"""Design system: colors, fonts, and shared CSS for the whole app."""

import streamlit as st

# ---------------------------------------------------------------------------
# Color system (used in Python for charts AND mirrored in the CSS below)
# ---------------------------------------------------------------------------
COLORS = {
    "bg": "#FFFFFF",
    "surface": "#F8FAFC",
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
    # ---- Sidebar (dark navy, matches the framed hero accent below) ----
    "sidebar_bg": "#0F1B2B",
    "sidebar_bg_soft": "#172033",
    "sidebar_text": "#E8EEF4",
    "sidebar_text_muted": "#94A3B8",
    "sidebar_border": "#22304A",
    # ---- Hero / framed-title gradient ----
    "hero_from": "#0F4C81",
    "hero_to": "#1A7A8A",
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


# ---------------------------------------------------------------------------
# Dark-mode overrides for the MAIN content area only. The sidebar always
# stays on the dark navy palette above (the person explicitly wants it
# stable/consistent), so only these keys change when the toggle is on.
# ---------------------------------------------------------------------------
DARK_OVERRIDES = {
    "bg": "#0B1220",
    "surface": "#111A2E",
    "border": "#243146",
    "text": "#E8EEF4",
    "text_muted": "#94A3B8",
    "primary_light": "rgba(37, 99, 235, 0.18)",
    "success_light": "rgba(16, 185, 129, 0.16)",
    "warning_light": "rgba(245, 158, 11, 0.16)",
    "danger_light": "rgba(239, 68, 68, 0.16)",
}


def inject_css(dark: bool = False):
    c = dict(COLORS)
    if dark:
        c.update(DARK_OVERRIDES)
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Poppins:wght@700;800;900&display=swap');

        /* Bridges our palette into Streamlit's own CSS variables, so native
           widgets (inputs, dataframes, sliders) also follow dark/light mode
           without us having to hand-style every internal component. */
        :root, .stApp {{
            --primary-color: {c['primary']};
            --background-color: {c['bg']};
            --secondary-background-color: {c['surface']};
            --text-color: {c['text']};
        }}

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}

        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{background: transparent !important; box-shadow: none !important;}}
        [data-testid="stMainMenu"] {{display: none;}}
        [data-testid="stAppDeployButton"] {{display: none;}}
        [data-testid="stStatusWidget"] {{display: none;}}
        [data-testid="stToolbarActions"] {{display: none;}}
        /* Keep ONLY the "expand sidebar" arrow (the button that reopens a
           collapsed sidebar) — it lives inside the same toolbar as the
           icons above, under data-testid="stExpandSidebarButton". Hiding
           the whole toolbar, or the whole header, hides this button too
           and traps the person with no way to bring the sidebar back. */
        [data-testid="stExpandSidebarButton"] {{
            visibility: visible !important;
            display: flex !important;
            opacity: 1 !important;
        }}

        .stApp {{
            background-color: {c['bg']};
        }}

        section[data-testid="stSidebar"] {{
            background-color: {c['sidebar_bg']};
            border-right: 1px solid {c['sidebar_border']};
        }}

        section[data-testid="stSidebar"] > div {{
            padding-top: 0.6rem;
        }}

        section[data-testid="stSidebar"] .block-container {{
            padding-top: 0.3rem;
            padding-bottom: 1rem;
        }}

        /* Make every default Streamlit text element inside the sidebar
           legible against the dark navy background. */
        section[data-testid="stSidebar"] * {{
            color: {c['sidebar_text']} !important;
        }}
        section[data-testid="stSidebar"] hr {{
            border-color: {c['sidebar_border']} !important;
        }}
        /* Muted variant for secondary sidebar text (wins the tie against
           the wildcard rule above because it's declared after it). */
        .sidebar-muted, .sidebar-muted * {{
            color: {c['sidebar_text_muted']} !important;
        }}

        /* ---- Sidebar "Dataset info" mini stat row ---- */
        .sidebar-stat-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: {c['sidebar_bg_soft']};
            border: 1px solid {c['sidebar_border']};
            border-radius: 10px;
            padding: 0.4rem 0.75rem;
            margin-bottom: 0.35rem;
        }}
        .sidebar-stat-label {{
            font-size: 0.78rem;
            color: {c['sidebar_text_muted']} !important;
        }}
        .sidebar-stat-value {{
            font-size: 0.88rem;
            font-weight: 700;
            color: #FFFFFF !important;
        }}

        /* ---- Compact single-line divider (replaces spacer-div + hr pairs,
           which were the main reason sidebar content needed scrolling) ---- */
        .sidebar-divider {{
            border: none;
            border-top: 1px dashed {c['sidebar_border']};
            margin: 0.6rem 0 !important;
        }}

        /* ---- Signature, bottom of sidebar ---- */
        .dev-signature-wrap {{
            text-align: center;
            margin-top: 0.6rem;
            padding: 0.5rem 0.6rem 0.3rem 0.6rem;
            border-top: 1px dashed {c['sidebar_border']};
        }}
        .dev-signature {{
            display: inline-block;
            font-family: 'Poppins', 'Inter', sans-serif;
            font-weight: 800;
            font-size: 1.3rem;
            line-height: 1.2;
            margin: 0;
            letter-spacing: 0.02em;
            background: linear-gradient(90deg, #60A5FA 0%, #A78BFA 40%, #F472B6 75%, #FBBF24 100%);
            background-size: 200% auto;
            -webkit-background-clip: text !important;
            background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            color: transparent !important;
            filter: drop-shadow(0 2px 6px rgba(96, 165, 250, 0.35));
            cursor: pointer;
            transition: transform 0.25s ease, filter 0.25s ease, background-position 0.5s ease;
        }}
        .dev-signature:hover {{
            transform: scale(1.08) translateY(-1px);
            filter: drop-shadow(0 4px 14px rgba(167, 139, 250, 0.55));
            background-position: right center;
        }}
        .dev-signature-sub {{
            font-size: 0.62rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: rgba(232, 238, 244, 0.45) !important;
            margin-top: 0.15rem;
        }}

        .block-container {{
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }}

        h1, h2, h3, h4 {{
            color: {c['text']} !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em;
        }}

        p, li, span, label {{
            color: {c['text']};
        }}

        .muted {{
            color: {c['text_muted']};
        }}

        /* ---- Sidebar nav radio styled as pills ---- */
        section[data-testid="stSidebar"] div[role="radiogroup"] {{
            gap: 2px;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] label {{
            background-color: transparent;
            border-radius: 10px;
            padding: 7px 12px;
            width: 100%;
            transition: background-color 0.15s ease;
            font-weight: 500;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
            background-color: rgba(255, 255, 255, 0.08);
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] {{
            background-color: {c['primary']};
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] * {{
            color: #FFFFFF !important;
        }}

        /* ---- Sidebar file-uploader / widgets, dark-matched ---- */
        section[data-testid="stSidebar"] [data-testid="stFileUploader"] {{
            background: {c['sidebar_bg_soft']} !important;
            border: 1px dashed {c['primary']} !important;
            border-radius: 14px;
        }}
        section[data-testid="stSidebar"] [data-testid="stFileUploader"] section {{
            background: {c['sidebar_bg_soft']} !important;
        }}

        /* ---- Native input widgets follow the palette in both modes ---- */
        [data-testid="stForm"] {{
            background-color: {c['surface']};
            border: 1px solid {c['border']};
            border-radius: 18px;
            padding: 1.6rem 1.8rem;
        }}
        .stNumberInput input, .stTextInput input, .stTextArea textarea {{
            background-color: {c['bg']} !important;
            color: {c['text']} !important;
            border-color: {c['border']} !important;
        }}
        .stSelectbox div[data-baseweb="select"] > div {{
            background-color: {c['bg']} !important;
            color: {c['text']} !important;
            border-color: {c['border']} !important;
        }}
        [data-testid="stDataFrame"], [data-testid="stTable"] {{
            background-color: {c['bg']};
            border-radius: 12px;
        }}

        /* ---- Buttons ---- */
        .stButton > button {{
            background-color: {c['primary']};
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.6rem 1.4rem;
            font-weight: 600;
            transition: all 0.15s ease;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
        }}
        .stButton > button:hover {{
            background-color: {c['primary_dark']};
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
            transform: translateY(-1px);
        }}
        .stButton > button p {{ color: white !important; font-weight: 600; }}

        /* ---- Generic card ---- */
        .card {{
            background-color: {c['bg']};
            border: 1px solid {c['border']};
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
            background-color: {c['bg']};
            border: 1px solid {c['border']};
            border-top: 3px solid {c['primary']};
            border-radius: 16px;
            padding: 1.3rem 1.4rem;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        }}
        .kpi-label {{
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            color: {c['text_muted']};
            text-transform: uppercase;
            margin-bottom: 0.4rem;
        }}
        .kpi-value {{
            font-size: 1.9rem;
            font-weight: 800;
            color: {c['text']};
            line-height: 1.1;
        }}
        .kpi-sub {{
            font-size: 0.8rem;
            color: {c['text_muted']};
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
        .badge-blue {{ background-color: {c['primary_light']}; color: {c['primary_dark']}; }}
        .badge-green {{ background-color: {c['success_light']}; color: #047857; }}
        .badge-orange {{ background-color: {c['warning_light']}; color: #B45309; }}

        /* ---- Hero ---- */
        .hero-title {{
            font-size: 2.6rem;
            font-weight: 800;
            color: {c['text']};
            line-height: 1.15;
            letter-spacing: -0.03em;
            margin-bottom: 0.6rem;
        }}
        .hero-sub {{
            font-size: 1.1rem;
            color: {c['text_muted']};
            max-width: 640px;
            line-height: 1.55;
        }}

        /* ---- Framed hero (project title, home page) ---- */
        .hero-frame {{
            background: linear-gradient(135deg, {c['hero_from']} 0%, {c['hero_to']} 100%);
            border-radius: 20px;
            padding: 2rem 2.2rem;
            color: white;
            box-shadow: 0 8px 24px rgba(15, 76, 129, 0.18);
        }}
        .hero-frame .hero-title {{
            color: #FFFFFF;
        }}
        .hero-frame .hero-sub {{
            color: rgba(255,255,255,0.88);
        }}
        .hero-frame .badge {{
            background-color: rgba(255,255,255,0.18) !important;
            color: #FFFFFF !important;
        }}
        .hero-frame .stButton > button {{
            background-color: rgba(255,255,255,0.14);
            border: 1px solid rgba(255,255,255,0.35);
        }}
        .hero-frame .stButton > button:hover {{
            background-color: rgba(255,255,255,0.26);
            box-shadow: none;
        }}
        .hero-frame .stButton > button p {{
            color: #FFFFFF !important;
        }}

        /* ---- Small framed page header (used at the top of every page) ---- */
        .page-header {{
            background: linear-gradient(135deg, {c['hero_from']} 0%, {c['hero_to']} 100%);
            border-radius: 16px;
            padding: 1.3rem 1.6rem;
            color: white;
            margin-bottom: 1.3rem;
            box-shadow: 0 4px 14px rgba(15, 76, 129, 0.15);
        }}
        .page-header-title {{
            font-size: 1.5rem;
            font-weight: 800;
            color: #FFFFFF;
            letter-spacing: -0.01em;
            margin: 0;
        }}
        .page-header-sub {{
            font-size: 0.95rem;
            color: rgba(255,255,255,0.85);
            margin-top: 0.35rem;
            max-width: 720px;
            line-height: 1.5;
        }}

        /* ---- Sidebar brand frame ---- */
        .brand-frame {{
            display: flex;
            align-items: center;
            gap: 10px;
            background: linear-gradient(135deg, {c['hero_from']} 0%, {c['hero_to']} 100%);
            border-radius: 14px;
            padding: 0.65rem 0.9rem;
            margin-bottom: 0.7rem;
            box-shadow: 0 4px 12px rgba(15, 76, 129, 0.25);
        }}
        .brand-frame .brand-title {{
            font-weight: 800;
            font-size: 1.1rem;
            color: #FFFFFF !important;
            line-height: 1.1;
        }}
        .brand-frame .brand-sub {{
            font-size: 0.72rem;
            color: rgba(255,255,255,0.8) !important;
        }}

        /* ---- Prediction result ---- */
        .result-card {{
            border-radius: 20px;
            padding: 2rem 2.2rem;
            text-align: center;
            border: 1px solid;
        }}
        .result-yes {{
            background-color: {c['success_light']};
            border-color: #A7F3D0;
        }}
        .result-no {{
            background-color: {c['warning_light']};
            border-color: #FDE68A;
        }}
        .result-label {{
            font-size: 0.85rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: {c['text_muted']};
        }}
        .result-value {{
            font-size: 2.4rem;
            font-weight: 800;
            margin: 0.3rem 0 0.6rem 0;
        }}
        .result-yes .result-value {{ color: #047857; }}
        .result-no .result-value {{ color: #B45309; }}

        hr {{
            border-color: {c['border']};
        }}

        div[data-testid="stMetricValue"] {{
            font-weight: 800;
            color: {c['text']};
        }}

        .step-num {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 28px; height: 28px;
            border-radius: 50%;
            background-color: {c['primary_light']};
            color: {c['primary_dark']};
            font-weight: 700;
            font-size: 0.85rem;
            margin-right: 0.5rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str = "", icon: str = ""):
    """Renders a small framed gradient header, matching the sidebar/hero
    accent, for use at the top of every page."""
    label = f"{icon}  {title}" if icon else title
    sub_html = f'<div class="page-header-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="page-header">
            <div class="page-header-title">{label}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
