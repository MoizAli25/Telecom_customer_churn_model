import streamlit as st

st.set_page_config(
    page_title="ChurnIQ — Telecom Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load CSS ──
def load_css():
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ── Custom Component Helpers ──
def card(content: str, padding="1.5rem"):
    st.markdown(f"""
    <div style="
        background:#1C2333;
        border:1px solid #2A3350;
        border-radius:12px;
        padding:{padding};
        margin-bottom:1rem;
    ">{content}</div>
    """, unsafe_allow_html=True)

def metric_card(label, value, delta=None, color="#4F7FFF"):
    delta_html = ""
    if delta:
        arrow = "▲" if delta > 0 else "▼"
        delta_color = "#06D6A0" if delta > 0 else "#FF4B6E"
        delta_html = f'<div style="color:{delta_color};font-size:0.8rem;margin-top:4px">{arrow} {abs(delta):.1f}%</div>'
    st.markdown(f"""
    <div style="
        background:#1C2333;
        border:1px solid #2A3350;
        border-left:3px solid {color};
        border-radius:12px;
        padding:1.2rem 1.5rem;
        height:100%;
    ">
        <div style="color:#8B97B8;font-size:0.75rem;
                    text-transform:uppercase;letter-spacing:0.08em;
                    font-weight:600;margin-bottom:6px">{label}</div>
        <div style="color:#4F7FFF;font-size:1.8rem;
                    font-weight:700;line-height:1">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def risk_badge(risk_tier):
    cfg = {
        "Very High": ("#FF4B6E", "rgba(255,75,110,0.12)"),
        "High"     : ("#FF8C42", "rgba(255,140,66,0.12)"),
        "Medium"   : ("#FFD166", "rgba(255,209,102,0.12)"),
        "Low"      : ("#06D6A0", "rgba(6,214,160,0.12)"),
    }
    color, bg = cfg.get(risk_tier, ("#8B97B8", "rgba(139,151,184,0.12)"))
    return f"""<span style="
        background:{bg};color:{color};
        border:1px solid {color};
        border-radius:20px;padding:3px 12px;
        font-size:0.78rem;font-weight:600;
        letter-spacing:0.04em;
    ">{risk_tier} Risk</span>"""

def section_header(title, subtitle=None):
    sub = f'<div style="color:#8B97B8;font-size:0.9rem;margin-top:4px">{subtitle}</div>' if subtitle else ""
    st.markdown(f"""
    <div style="margin-bottom:1.5rem">
        <h2 style="color:#4F7FFF;font-size:1.4rem;
                   font-weight:700;margin:0;letter-spacing:-0.01em">{title}</h2>
        {sub}
    </div>
    """, unsafe_allow_html=True)

# ── Sidebar ──
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 1.5rem">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px">
            <span style="font-size:1.5rem">⚡</span>
            <span style="font-size:1.1rem;font-weight:700;
                         color:#4F7FFF;letter-spacing:-0.01em">ChurnIQ</span>
        </div>
        <div style="color:#556080;font-size:0.75rem;
                    padding-left:2px;letter-spacing:0.02em">
            TELECOM INTELLIGENCE PLATFORM
        </div>
    </div>
    <hr style="border-color:#2A3350;margin:0 0 1rem">
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="color:#8B97B8;font-size:0.72rem;
                text-transform:uppercase;letter-spacing:0.08em;
                font-weight:600;margin-bottom:8px">Navigation</div>
    """, unsafe_allow_html=True)

    st.page_link("app.py",             label="🏠  Overview",          )
    st.page_link("pages/01_predict.py", label="🔍  Customer Predict",  )
    st.page_link("pages/02_batch.py",   label="📊  Batch Analysis",    )
    st.page_link("pages/03_insights.py",label="💡  Model Insights",    )

    st.markdown("<hr style='border-color:#2A3350;margin:1rem 0'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="color:#556080;font-size:0.72rem;line-height:1.6">
        Model: Random Forest<br>
        AUC: 0.8327 &nbsp;|&nbsp; Recall: 0.78<br>
        Dataset: IBM Telco (7,032 customers)
    </div>
    """, unsafe_allow_html=True)

# ── HOME PAGE ──
st.markdown("""
<div style="margin-bottom:2.5rem">
    <div style="color:#4F7FFF;font-size:0.8rem;font-weight:600;
                letter-spacing:0.1em;text-transform:uppercase;
                margin-bottom:10px">Telecom Customer Intelligence</div>
    <h1 style="font-size:2.4rem;font-weight:700;color:#4F7FFF;
               margin:0;letter-spacing:-0.02em;line-height:1.2">
        Predict churn.<br>
        <span style="color:#4F7FFF">Act before they leave.</span>
    </h1>
    <p style="color:#8B97B8;font-size:1rem;margin-top:12px;
              max-width:560px;line-height:1.7">
        End-to-end churn prediction with explainable AI and
        personalized retention recommendations for every customer segment.
    </p>
</div>
""", unsafe_allow_html=True)

# ── KPI Row ──
c1, c2, c3, c4 = st.columns(4)
with c1: metric_card("Model AUC",        "0.8327", color="#4F7FFF")
with c2: metric_card("Churn Recall",     "78%",    color="#06D6A0")
with c3: metric_card("Dataset Size",     "7,032",  color="#FF8C42")
with c4: metric_card("Baseline Churn",   "26.5%",  color="#FF4B6E")

st.markdown("<div style='margin:2rem 0'></div>", unsafe_allow_html=True)

# ── Feature Cards ──
section_header("Platform Capabilities", "Three integrated modules for end-to-end churn management")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style="background:#1C2333;border:1px solid #2A3350;
                border-top:3px solid #4F7FFF;border-radius:12px;
                padding:1.5rem;height:200px">
        <div style="font-size:1.5rem;margin-bottom:10px">🔍</div>
        <div style="font-weight:600;font-size:1rem;
                    color:#4F7FFF;margin-bottom:8px">Customer Predict</div>
        <div style="color:#8B97B8;font-size:0.85rem;line-height:1.6">
            Enter a customer profile and get instant churn probability,
            risk tier, segment classification, and targeted retention actions.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background:#1C2333;border:1px solid #2A3350;
                border-top:3px solid #06D6A0;border-radius:12px;
                padding:1.5rem;height:200px">
        <div style="font-size:1.5rem;margin-bottom:10px">📊</div>
        <div style="font-weight:600;font-size:1rem;
                    color:#4F7FFF;margin-bottom:8px">Batch Analysis</div>
        <div style="color:#8B97B8;font-size:0.85rem;line-height:1.6">
            Upload a CSV of customers and get churn predictions,
            segment assignments, and priority-ranked retention recommendations in bulk.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="background:#1C2333;border:1px solid #2A3350;
                border-top:3px solid #FF8C42;border-radius:12px;
                padding:1.5rem;height:200px">
        <div style="font-size:1.5rem;margin-bottom:10px">💡</div>
        <div style="font-weight:600;font-size:1rem;
                    color:#4F7FFF;margin-bottom:8px">Model Insights</div>
        <div style="color:#8B97B8;font-size:0.85rem;line-height:1.6">
            Explore SHAP-based feature importance, segment profiles,
            and key churn drivers with interactive visualizations.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin:2rem 0'></div>", unsafe_allow_html=True)

# ── Segment Overview ──
section_header("Customer Segment Overview", "Risk distribution across 4 behavioral segments")

seg_data = [
    ("Critical Churn-Risk", "2,126", "54.7%", "#FF4B6E", "rgba(255,75,110,0.08)"),
    ("Vulnerable High-Risk","1,744", "28.5%", "#FF8C42", "rgba(255,140,66,0.08)"),
    ("Stable Mid-Risk",     "2,029",  "9.2%", "#FFD166", "rgba(255,209,102,0.08)"),
    ("Loyal Low-Risk",      "1,133",  "2.0%", "#06D6A0", "rgba(6,214,160,0.08)"),
]

cols = st.columns(4)
for col, (seg, count, churn, color, bg) in zip(cols, seg_data):
    with col:
        st.markdown(f"""
        <div style="background:{bg};border:1px solid {color};
                    border-radius:12px;padding:1.2rem;text-align:center">
            <div style="color:{color};font-size:1.6rem;
                        font-weight:700;line-height:1">{churn}</div>
            <div style="color:#4F7FFF;font-size:0.78rem;
                        font-weight:600;margin:6px 0 4px">{seg}</div>
            <div style="color:#556080;font-size:0.75rem">{count} customers</div>
        </div>
        """, unsafe_allow_html=True)
        
