
import re
import os
import requests
import streamlit as st
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageClassification
try:
    from groq import Groq
    GROQ_SDK_OK = True
except Exception:
    GROQ_SDK_OK = False

# ============================================================
# FARMIQ v4
# SIH26131 — Crop Health Intelligence
#
# Navigation:
# IMAGE RECOGNITION
#   • Plant Image
#   • Diagnosis
#   • Confidence / Severity
#   • Expert Validation
#
# WEATHER INTELLIGENCE
#   • Current Weather
#   • Disease Risk
#   • 3-day Forecast
#
# SOIL & NUTRITION
#   • Soil Assessment
#   • Nutrient Requirement
#
# DECISION ENGINE
#   • Multimodal Assessment
#   • Treatment Recommendation
#
# MONITORING
#   • Follow-up
#   • Regional Intelligence
#
# English + Hindi interface
# Visual data: confidence bars, disease probabilities,
# weather risk meter, forecast chart, soil nutrient chart.
# ============================================================

st.set_page_config(
    page_title="FARMIQ | Crop Health Intelligence",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_ID = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"

IMAGE_TFM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

CITIES = {
    "Pune, Maharashtra": (18.5204, 73.8567),
    "Bhopal, Madhya Pradesh": (23.2599, 77.4126),
    "Nagpur, Maharashtra": (21.1458, 79.0882),
    "Nashik, Maharashtra": (19.9975, 73.7898),
    "Mumbai, Maharashtra": (19.0760, 72.8777),
    "Indore, Madhya Pradesh": (22.7196, 75.8577),
}

# ------------------------------------------------------------
# Treatment knowledge base — demonstration decision support.
# Verify local registration, label, dose, PHI and agronomy
# before any real-world recommendation.
# ------------------------------------------------------------
RULES = {
    "Tomato___Early_blight": {
        "name": "Tomato Early Blight",
        "hi": "टमाटर अर्ली ब्लाइट",
        "type": "Fungal disease",
        "type_hi": "फफूंद रोग",
        "options": ["Mancozeb", "Azoxystrobin"],
        "actions": [
            "Remove severely affected leaves where practical.",
            "Improve canopy airflow and reduce prolonged leaf wetness.",
            "For confirmed disease requiring chemical control, consider a locally registered option and follow its label.",
        ],
        "actions_hi": [
            "जहाँ संभव हो, बहुत अधिक प्रभावित पत्तियों को हटाएँ।",
            "फसल में हवा का संचार बढ़ाएँ और पत्तियों पर लंबे समय तक नमी रहने से बचाएँ।",
            "रोग की पुष्टि होने पर ही स्थानीय रूप से पंजीकृत विकल्प पर विचार करें और लेबल का पालन करें।",
        ],
    },
    "Tomato___Late_blight": {
        "name": "Tomato Late Blight",
        "hi": "टमाटर लेट ब्लाइट",
        "type": "Fungal/oomycete disease",
        "type_hi": "फफूंद/ऊमाइसीट रोग",
        "options": ["Mancozeb", "Metalaxyl + Mancozeb"],
        "actions": [
            "Reduce prolonged leaf wetness and remove severely affected tissue where practical.",
            "For confirmed high-risk disease, consider a locally registered option according to its label.",
        ],
        "actions_hi": [
            "पत्तियों पर लंबे समय तक नमी कम करें और जहाँ संभव हो बहुत अधिक प्रभावित भाग हटाएँ।",
            "पुष्ट उच्च-जोखिम रोग में स्थानीय रूप से पंजीकृत विकल्प पर विचार करें और लेबल का पालन करें।",
        ],
    },
    "Potato___Early_blight": {
        "name": "Potato Early Blight",
        "hi": "आलू अर्ली ब्लाइट",
        "type": "Fungal disease",
        "type_hi": "फफूंद रोग",
        "options": ["Mancozeb", "Azoxystrobin"],
        "actions": [
            "Maintain field sanitation and avoid unnecessary leaf wetness.",
            "Consider a locally registered fungicide option after diagnosis is confirmed.",
        ],
        "actions_hi": [
            "खेत की स्वच्छता बनाए रखें और अनावश्यक पत्ती-नमी से बचें।",
            "रोग की पुष्टि के बाद स्थानीय रूप से पंजीकृत फफूंदनाशी विकल्प पर विचार करें।",
        ],
    },
    "Potato___Late_blight": {
        "name": "Potato Late Blight",
        "hi": "आलू लेट ब्लाइट",
        "type": "Fungal/oomycete disease",
        "type_hi": "फफूंद/ऊमाइसीट रोग",
        "options": ["Mancozeb", "Metalaxyl + Mancozeb"],
        "actions": [
            "Reduce prolonged leaf wetness and remove severely affected material where practical.",
            "For confirmed high-risk disease, consider a locally registered option according to its label.",
        ],
        "actions_hi": [
            "पत्तियों पर लंबे समय तक नमी कम करें और बहुत अधिक प्रभावित भाग हटाएँ।",
            "पुष्ट उच्च-जोखिम रोग में स्थानीय रूप से पंजीकृत विकल्प पर विचार करें।",
        ],
    },
    "Pepper,_bell___Bacterial_spot": {
        "name": "Pepper Bacterial Spot",
        "hi": "शिमला मिर्च बैक्टीरियल स्पॉट",
        "type": "Bacterial disease",
        "type_hi": "बैक्टीरियल रोग",
        "options": ["Copper-based registered option where locally approved"],
        "actions": [
            "Maintain sanitation and remove severely affected tissue where practical.",
            "Avoid prolonged leaf wetness and overhead irrigation where appropriate.",
        ],
        "actions_hi": [
            "खेत की स्वच्छता रखें और बहुत अधिक प्रभावित भाग हटाएँ।",
            "जहाँ उपयुक्त हो, लंबे समय तक पत्ती-नमी और ऊपर से सिंचाई से बचें।",
        ],
    },
}

NUTRIENTS = {
    "N": {
        "en": "Nitrogen",
        "hi": "नाइट्रोजन",
        "source": "appropriate nitrogen source",
        "source_hi": "उपयुक्त नाइट्रोजन स्रोत",
        "note": "Low N can be associated with generalized older-leaf yellowing and weak vegetative growth.",
        "note_hi": "कम N से पुरानी पत्तियों का सामान्य पीलापन और कमजोर वनस्पतिक वृद्धि जुड़ी हो सकती है।",
    },
    "P": {
        "en": "Phosphorus",
        "hi": "फॉस्फोरस",
        "source": "appropriate phosphorus source",
        "source_hi": "उपयुक्त फॉस्फोरस स्रोत",
        "note": "Low P can affect growth and root development.",
        "note_hi": "कम P वृद्धि और जड़ विकास को प्रभावित कर सकता है।",
    },
    "K": {
        "en": "Potassium",
        "hi": "पोटैशियम",
        "source": "appropriate potassium source",
        "source_hi": "उपयुक्त पोटैशियम स्रोत",
        "note": "Low K can increase stress susceptibility in some crops.",
        "note_hi": "कुछ फसलों में कम K तनाव के प्रति संवेदनशीलता बढ़ा सकता है।",
    },
}

TEXT = {
    "en": {
        "profile": "Field Profile",
        "farmer": "Farmer name",
        "crop": "Crop",
        "variety": "Variety",
        "stage": "Growth stage",
        "location": "Location",
        "soil": "Soil test",
        "use_soil": "Use recent soil-test values",
        "image": "IMAGE RECOGNITION",
        "plant_image": "Plant Image",
        "diagnosis": "Diagnosis",
        "confidence": "Confidence & Severity",
        "expert": "Expert Validation",
        "weather": "WEATHER INTELLIGENCE",
        "current_weather": "Current Weather",
        "risk": "Disease Risk",
        "forecast": "3-day Forecast",
        "nutrition": "SOIL & NUTRITION",
        "soil_assess": "Soil Assessment",
        "nutrient_need": "Nutrient Requirement",
        "decision": "DECISION ENGINE",
        "health": "Multimodal Health Assessment",
        "treatment": "Treatment Recommendation",
        "monitor": "MONITORING",
        "follow": "Follow-up",
        "regional": "Regional Intelligence",
        "demo": "Guided Demo",
        "run_demo": "Run Guided Demo",
        "analyze": "Analyze Image",
        "continue": "Continue",
        "back": "Back",
        "high": "HIGH",
        "moderate": "MODERATE",
        "low": "LOW",
        "intervention": "INTERVENTION RECOMMENDED",
        "targeted": "TARGETED MANAGEMENT",
        "validation": "EXPERT VALIDATION",
        "monitor_decision": "MONITOR",
    },
    "hi": {
        "profile": "खेत की जानकारी",
        "farmer": "किसान का नाम",
        "crop": "फसल",
        "variety": "किस्म",
        "stage": "फसल की अवस्था",
        "location": "स्थान",
        "soil": "मृदा परीक्षण",
        "use_soil": "हाल का मृदा परीक्षण उपयोग करें",
        "image": "इमेज पहचान",
        "plant_image": "पौधे की तस्वीर",
        "diagnosis": "रोग पहचान",
        "confidence": "विश्वास स्तर और गंभीरता",
        "expert": "विशेषज्ञ सत्यापन",
        "weather": "मौसम इंटेलिजेंस",
        "current_weather": "वर्तमान मौसम",
        "risk": "रोग जोखिम",
        "forecast": "3-दिन का पूर्वानुमान",
        "nutrition": "मृदा और पोषण",
        "soil_assess": "मृदा आकलन",
        "nutrient_need": "पोषक तत्व आवश्यकता",
        "decision": "निर्णय इंजन",
        "health": "मल्टीमॉडल फसल स्वास्थ्य आकलन",
        "treatment": "उपचार सुझाव",
        "monitor": "निगरानी",
        "follow": "फॉलो-अप",
        "regional": "क्षेत्रीय इंटेलिजेंस",
        "demo": "गाइडेड डेमो",
        "run_demo": "गाइडेड डेमो चलाएँ",
        "analyze": "इमेज का विश्लेषण करें",
        "continue": "आगे बढ़ें",
        "back": "वापस",
        "high": "उच्च",
        "moderate": "मध्यम",
        "low": "कम",
        "intervention": "हस्तक्षेप की सलाह",
        "targeted": "लक्षित प्रबंधन",
        "validation": "विशेषज्ञ सत्यापन",
        "monitor_decision": "निगरानी",
    },
}

# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>
:root {
  --far-bg: #f4f8f6;
  --far-surface: #ffffff;
  --far-surface2: #eef5f1;
  --far-text: #10231a;
  --far-muted: #64736b;
  --far-border: #d8e4dd;
  --far-input: #ffffff;
  --far-accent: #176b4d;
  --far-accent2: #2d8a65;
  --far-shadow: 0 8px 28px rgba(16,35,26,.07);
}
.stApp { background:var(--far-bg) !important; color:var(--far-text) !important; }
.block-container { max-width:1380px; padding-top:1rem; color:var(--far-text) !important; }
[data-testid="stSidebar"] { background:var(--far-surface) !important; border-right:1px solid var(--far-border) !important; }
[data-testid="stSidebar"] * { color:var(--far-text) !important; }
.stMarkdown, .stMarkdown p, .stMarkdown li, label, .stCaption { color:var(--far-text) !important; }
[data-testid="stMetric"] {
  background:var(--far-surface) !important;
  border:1px solid var(--far-border) !important;
  border-radius:16px !important;
  box-shadow:var(--far-shadow);
}
[data-testid="stMetricLabel"], [data-testid="stMetricValue"] { color:var(--far-text) !important; }
div[data-testid="stExpander"] { background:var(--far-surface) !important; border:1px solid var(--far-border) !important; border-radius:16px; }
.stTextInput input, .stTextArea textarea, .stNumberInput input,
[data-baseweb="select"] > div, [data-baseweb="input"] > div {
  background:var(--far-input) !important; color:var(--far-text) !important;
  border-color:var(--far-border) !important;
}
[data-baseweb="popover"], [data-baseweb="menu"] { background:var(--far-surface) !important; }
[data-baseweb="menu"] * { color:var(--far-text) !important; }
.stButton > button {
  border-color:var(--far-border) !important;
  color:var(--far-text) !important;
  background:var(--far-surface) !important;
  border-radius:11px !important;
}
.stButton > button[kind="primary"] {
  background:var(--far-accent) !important; color:#fff !important; border-color:var(--far-accent) !important;
}
.stChatMessage {
  background:var(--far-surface) !important;
  border:1px solid var(--far-border) !important;
  border-radius:16px !important;
}
.stChatInputContainer {
  background:var(--far-surface) !important;
}
.hero {
 background:linear-gradient(135deg,var(--far-accent),var(--far-accent2));
 color:white; border-radius:24px; padding:1.45rem 1.8rem;
 margin-bottom:.8rem; box-shadow:0 12px 34px rgba(0,0,0,.12);
}
.hero h1 { margin:0; font-size:2.55rem; }
.hero p { margin:.35rem 0 0; font-size:1.05rem; }
.section {
 background:var(--far-surface); color:var(--far-text);
 border:1px solid var(--far-border); border-radius:20px;
 padding:1.25rem; margin:.85rem 0; box-shadow:var(--far-shadow);
}
.section-title { color:var(--far-text); font-size:1.35rem; font-weight:800; margin-bottom:.75rem; }
.kpi { background:var(--far-surface2); border:1px solid var(--far-border); border-radius:15px; padding:.8rem; }
.kpi-label { color:var(--far-muted); font-size:.8rem; }
.kpi-value { color:var(--far-text); font-size:1.12rem; font-weight:800; margin-top:.2rem; }
.green { background:rgba(46,160,91,.12); border:1px solid rgba(46,160,91,.35); border-radius:16px; padding:1rem; color:var(--far-text); }
.yellow { background:rgba(235,175,45,.12); border:1px solid rgba(235,175,45,.4); border-radius:16px; padding:1rem; color:var(--far-text); }
.red { background:rgba(220,75,65,.12); border:1px solid rgba(220,75,65,.38); border-radius:16px; padding:1rem; color:var(--far-text); }
.blue { background:rgba(60,130,220,.12); border:1px solid rgba(60,130,220,.3); border-radius:16px; padding:1rem; color:var(--far-text); }
.small { color:var(--far-muted) !important; font-size:.86rem; }
.navhint { padding:.55rem .7rem; background:var(--far-surface2); border:1px solid var(--far-border); border-radius:10px; color:var(--far-text) !important; font-size:.82rem; margin-bottom:.5rem; }
.meter { height:14px; background:var(--far-surface2); border:1px solid var(--far-border); border-radius:999px; overflow:hidden; margin:.5rem 0 1rem; }
.meter-fill { height:100%; border-radius:999px; background:linear-gradient(90deg,#67bd8c,#1a8c62); }
.progress-label { display:flex; justify-content:space-between; font-size:.78rem; margin:.35rem 0; }
.progress-track { height:8px; border-radius:99px; background:var(--far-surface2); overflow:hidden; margin-bottom:.7rem; }
.progress-fill { height:100%; border-radius:99px; background:linear-gradient(90deg,#2d8a65,#62c995); }
.chat-header {
  background:linear-gradient(135deg,var(--far-surface2),var(--far-surface));
  border:1px solid var(--far-border); border-radius:20px; padding:1rem 1.15rem;
  box-shadow:var(--far-shadow); margin-top:1rem;
}
.chat-header h3 { margin:0; color:var(--far-text); }
.signal-card {
  background:var(--far-surface); border:1px solid var(--far-border);
  border-radius:18px; padding:1rem; box-shadow:var(--far-shadow);
}
</style>
""", unsafe_allow_html=True)


# Initialize UI preferences BEFORE any theme-dependent rendering.
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "language" not in st.session_state:
    st.session_state.language = "English"

# ============================================================
# FARMIQ LIGHT / DARK THEME
# ============================================================
_THEME = {
    "light": {
        "bg": "#f4f8f6", "surface": "#ffffff", "surface2": "#eef5f1",
        "text": "#10231a", "muted": "#64736b", "border": "#d8e4dd",
        "input": "#ffffff", "accent": "#176b4d", "accent2": "#2d8a65",
        "shadow": "0 8px 28px rgba(16,35,26,.07)"
    },
    "dark": {
        "bg": "#07100c", "surface": "#101b16", "surface2": "#17251e",
        "text": "#f2faf5", "muted": "#aabdb2", "border": "#30463b",
        "input": "#0c1511", "accent": "#55c88d", "accent2": "#79dda8",
        "shadow": "0 10px 30px rgba(0,0,0,.30)"
    },
}
_tc = _THEME[st.session_state.theme]
st.markdown(f"""
<style>
:root {{
 --far-bg:{_tc['bg']}; --far-surface:{_tc['surface']}; --far-surface2:{_tc['surface2']};
 --far-text:{_tc['text']}; --far-muted:{_tc['muted']}; --far-border:{_tc['border']};
 --far-input:{_tc['input']}; --far-accent:{_tc['accent']}; --far-accent2:{_tc['accent2']};
 --far-shadow:{_tc['shadow']};
}}
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
  background:var(--far-bg) !important;
  color:var(--far-text) !important;
  color-scheme: { "dark" if st.session_state.theme == "dark" else "light" };
}}
[data-testid="stHeader"] {{ background:var(--far-bg) !important; }}
[data-testid="stToolbar"] {{ background:transparent !important; }}
[data-testid="stSidebar"] {{
  background:var(--far-surface) !important;
  border-right:1px solid var(--far-border) !important;
}}
[data-testid="stSidebar"] * {{ color:var(--far-text) !important; }}
.stApp, .stAppViewContainer, .main, .block-container {{
  color:var(--far-text) !important;
}}
[data-baseweb="select"] *,
[data-baseweb="input"] *,
[data-baseweb="textarea"] *,
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {{
  color:var(--far-text) !important;
  -webkit-text-fill-color:var(--far-text) !important;
}}
[data-baseweb="select"] > div,
[data-baseweb="input"] > div {{
  background:var(--far-input) !important;
  border-color:var(--far-border) !important;
}}
[data-baseweb="popover"],
[data-baseweb="menu"],
[data-baseweb="menu"] > div {{
  background:var(--far-surface) !important;
}}
[data-baseweb="menu"] * {{ color:var(--far-text) !important; }}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HELPERS
# ============================================================

def t(key):
    lang = st.session_state.get("lang", "en")
    return TEXT[lang].get(key, key)

def h_or_e(en, hi):
    return hi if st.session_state.get("lang") == "hi" else en

def pretty(label):
    return re.sub(r"\s+", " ", label.replace("___", " — ").replace("_", " ")).strip()

def render_meter(value, label=""):
    pct = max(0, min(100, value))
    st.markdown(
        f"""
        <div class="small"><b>{label}</b> {pct:.0f}%</div>
        <div class="meter"><div class="meter-fill" style="width:{pct}%"></div></div>
        """,
        unsafe_allow_html=True,
    )

try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False


def visual_gauge(value, title, subtitle="", color=None):
    value = max(0, min(100, float(value)))
    if not PLOTLY_OK:
        render_meter(value, title)
        return
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix":"%", "font":{"size":28}},
        title={"text":title, "font":{"size":15}},
        gauge={
            "axis":{"range":[0,100], "tickwidth":1},
            "bar":{"color":color or "#2d8a65"},
            "bgcolor":"rgba(0,0,0,0)",
            "borderwidth":0,
            "steps":[
                {"range":[0,40], "color":"rgba(46,160,91,.16)"},
                {"range":[40,70], "color":"rgba(235,175,45,.16)"},
                {"range":[70,100], "color":"rgba(220,75,65,.16)"},
            ],
        },
    ))
    fig.update_layout(
        height=245, margin=dict(l=18,r=18,t=45,b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color":"#f2faf5" if st.session_state.theme=="dark" else "#10231a"},
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
    if subtitle:
        st.caption(subtitle)

def render_signal_dashboard(conf, risk_score, soil_score, completeness):
    vals = [conf*100, risk_score, soil_score, completeness]
    labels = [
        h_or_e("AI confidence","AI विश्वास"),
        h_or_e("Weather risk","मौसम जोखिम"),
        h_or_e("Soil context","मृदा संदर्भ"),
        h_or_e("Data completeness","डेटा पूर्णता"),
    ]
    if not PLOTLY_OK:
        for lab,val in zip(labels,vals):
            render_meter(val, lab)
        return
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals + [vals[0]], theta=labels + [labels[0]],
        fill="toself", name=h_or_e("Crop health signals","फसल स्वास्थ्य संकेत"),
        line={"width":3}, opacity=.78
    ))
    fig.update_layout(
        polar={
            "radialaxis":{"visible":True,"range":[0,100],"gridcolor":"#71847a"},
            "angularaxis":{"gridcolor":"#71847a"},
            "bgcolor":"rgba(0,0,0,0)",
        },
        showlegend=False, height=380,
        margin=dict(l=45,r=45,t=30,b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color":"#f2faf5" if st.session_state.theme=="dark" else "#10231a"},
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

def visual_bar(labels, values, title, y_title, suffix=""):
    if not PLOTLY_OK:
        st.bar_chart(pd.DataFrame({"Value": values}, index=labels))
        return
    fig = go.Figure(go.Bar(
        x=labels, y=values,
        text=[f"{v:.0f}{suffix}" for v in values],
        textposition="outside",
        hovertemplate="%{x}<br>%{y:.1f}" + suffix + "<extra></extra>"
    ))
    fig.update_layout(
        title=title, height=340, margin=dict(l=20,r=20,t=60,b=35),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis_title=y_title, xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="#DCE8E1"),
        font=dict(size=13)
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

def visual_pie(labels, values, title, suffix="%"):
    """Presentation-friendly pie/donut chart for part-to-whole data."""
    if not PLOTLY_OK:
        # Streamlit fallback: show a compact table rather than a misleading bar.
        total = sum(values) or 1
        st.dataframe(
            pd.DataFrame({
                "Category": labels,
                "Share": [round(v / total * 100, 1) for v in values]
            }),
            hide_index=True,
            use_container_width=True
        )
        return

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.48,
        textinfo="label+percent",
        hovertemplate="%{label}<br>%{value:.1f}" + suffix + "<br>%{percent}<extra></extra>",
        sort=False
    ))
    fig.update_layout(
        title=title,
        height=360,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
        legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center")
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

def visual_line(x, series, title, y_title):
    if not PLOTLY_OK:
        st.line_chart(pd.DataFrame(series, index=x))
        return
    fig = go.Figure()
    for name, values in series.items():
        fig.add_trace(go.Scatter(
            x=x, y=values, mode="lines+markers", name=name,
            line=dict(width=3), marker=dict(size=8)
        ))
    fig.update_layout(
        title=title, height=360, margin=dict(l=20,r=20,t=60,b=35),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis_title=y_title, xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="#DCE8E1"), hovermode="x unified",
        font=dict(size=13)
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ============================================================
# MODEL
# ============================================================

@st.cache_resource(show_spinner="Loading crop-disease AI model...")
def load_model():
    # Direct torchvision preprocessing avoids the previous
    # AutoImageProcessor metadata incompatibility.
    m = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    m.eval()
    return m

def predict_image(img):
    m = load_model()
    x = IMAGE_TFM(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(m(pixel_values=x).logits, dim=-1)[0]
    vals, idx = torch.topk(probs, min(5, len(probs)))
    return [
        {"label": m.config.id2label.get(int(i), str(i)),
         "score": float(v)}
        for v, i in zip(vals, idx)
    ]

# ============================================================
# WEATHER
# ============================================================

def get_weather(lat, lon):
    r = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat, "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,precipitation,rain,wind_speed_10m",
            "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min",
            "forecast_days": 3,
            "timezone": "auto",
        },
        timeout=8,
    )
    r.raise_for_status()
    return r.json()

def weather_risk(w, label):
    c = w["current"]
    temp = float(c.get("temperature_2m", 25) or 25)
    hum = float(c.get("relative_humidity_2m", 60) or 60)
    rain = float(c.get("rain", 0) or 0) + float(c.get("precipitation", 0) or 0)

    score = 0
    why = []

    if hum >= 80:
        score += 40
        why.append(h_or_e(f"High humidity ({hum:.0f}%)",
                           f"उच्च आर्द्रता ({hum:.0f}%)"))
    elif hum >= 70:
        score += 20
        why.append(h_or_e(f"Elevated humidity ({hum:.0f}%)",
                           f"बढ़ी हुई आर्द्रता ({hum:.0f}%)"))

    if rain > 0:
        score += 30
        why.append(h_or_e("Rainfall / precipitation present",
                           "वर्षा / वर्षण मौजूद है"))

    favorable = (10 <= temp <= 25) if "late_blight" in label.lower() else (18 <= temp <= 32)
    if favorable:
        score += 25
        why.append(h_or_e(
            f"Temperature ({temp:.1f}°C) is favorable for this disease profile",
            f"तापमान ({temp:.1f}°C) इस रोग प्रोफाइल के लिए अनुकूल है"
        ))

    score = min(score, 100)
    level = "HIGH" if score >= 65 else "MODERATE" if score >= 35 else "LOW"
    return level, score, why

# ============================================================
# SESSION STATE
# ============================================================

if "lang" not in st.session_state:
    st.session_state.lang = "en"
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "nav" not in st.session_state:
    st.session_state.nav = "plant_image"
if "nav_index" not in st.session_state:
    st.session_state.nav_index = 0
if "_ui_language_changed" not in st.session_state:
    st.session_state._ui_language_changed = False


if "demo" not in st.session_state:
    st.session_state.demo = True

if "predictions" not in st.session_state:
    st.session_state.predictions = None

if "weather" not in st.session_state:
    st.session_state.weather = None

if "risk" not in st.session_state:
    st.session_state.risk = None

if "soil_flags" not in st.session_state:
    st.session_state.soil_flags = []


# ============================================================
# AI CROP ASSISTANT — GROQ
# ============================================================
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "groq_api_key" not in st.session_state:
    _stored_groq_key = os.getenv("GROQ_API_KEY", "")
    try:
        _stored_groq_key = _stored_groq_key or st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        pass
    st.session_state.groq_api_key = _stored_groq_key

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

def get_groq_key():
    """Read Groq key from session, environment, or Streamlit secrets."""
    key = st.session_state.get("groq_api_key", "") or os.getenv("GROQ_API_KEY", "")
    if key:
        return key
    try:
        return st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        return ""

def ensure_demo_assessment():
    """Populate a clearly-labelled demo assessment so the assistant is never empty in Guided Demo."""
    if not st.session_state.get("demo", False):
        return
    if not st.session_state.get("predictions"):
        st.session_state.predictions = [
            {"label": "Tomato___Early_blight", "score": 0.87},
            {"label": "Tomato___Late_blight", "score": 0.06},
            {"label": "Tomato___healthy", "score": 0.04},
        ]
    if not st.session_state.get("weather"):
        st.session_state.weather = {
            "current": {
                "temperature_2m": 28.0,
                "relative_humidity_2m": 82.0,
                "precipitation": 2.4,
                "wind_speed_10m": 9.0,
            },
            "daily": {},
        }
        st.session_state.weather_live = False
    if not st.session_state.get("risk"):
        st.session_state.risk = {
            "level": "HIGH",
            "score": 78,
            "why": "High humidity and recent moisture create favorable conditions for disease development in this demo scenario.",
        }
    if not st.session_state.get("soil_flags"):
        st.session_state.soil_flags = ["Moderate nitrogen context", "Potassium should be monitored"]


def current_farmer_context():
    pred = st.session_state.get("predictions") or []
    weather = st.session_state.get("weather") or {}
    risk = st.session_state.get("risk") or {}
    flags = st.session_state.get("soil_flags") or []
    top = pred[0] if pred else None
    current = weather.get("current", {}) if isinstance(weather, dict) else {}

    diagnosis = pretty(top["label"]) if top else None
    confidence = round(top["score"] * 100, 1) if top else None
    decision = None
    if top:
        conf = top["score"]
        risk_level = risk.get("level") if risk else None
        if conf < 0.65:
            decision = "EXPERT VALIDATION"
        elif risk_level == "HIGH" or conf >= 0.85:
            decision = "INTERVENTION RECOMMENDED"
        elif risk_level == "LOW" and conf < 0.75:
            decision = "MONITOR"
        else:
            decision = "TARGETED MANAGEMENT"

    treatment_options = []
    if top and top.get("label") in RULES:
        rule = RULES[top["label"]]
        treatment_options = rule.get("active_ingredients", []) or rule.get("options", []) or []

    return {
        "crop": crop,
        "variety": variety,
        "growth_stage": stage,
        "location": location,
        "diagnosis": diagnosis,
        "diagnosis_confidence": confidence,
        "weather_risk": risk.get("level"),
        "weather_risk_score": risk.get("score"),
        "risk_reason": risk.get("why"),
        "temperature_c": current.get("temperature_2m"),
        "humidity_pct": current.get("relative_humidity_2m"),
        "rain_mm": current.get("precipitation"),
        "wind_kmh": current.get("wind_speed_10m"),
        "soil_N": n if soil_available else None,
        "soil_P": p if soil_available else None,
        "soil_K": k if soil_available else None,
        "soil_pH": ph if soil_available else None,
        "soil_flags": flags,
        "decision": decision,
        "candidate_treatment_categories": treatment_options,
        "data_mode": "Guided demo scenario" if st.session_state.get("demo") else "Live / user-provided assessment",
    }


def groq_answer(user_text):
    """Call Groq directly over HTTPS so the assistant works even if groq-sdk is missing."""
    api_key = get_groq_key().strip()
    if not api_key:
        return None, h_or_e(
            "Add your Groq API key in the sidebar to activate the AI assistant.",
            "AI सहायक शुरू करने के लिए साइडबार में Groq API key जोड़ें।"
        )

    ctx = current_farmer_context()
    language_instruction = (
        "Respond in simple farmer-friendly Hindi. Keep crop and disease names in English in parentheses when useful."
        if st.session_state.lang == "hi"
        else "Respond in clear, concise farmer-friendly English."
    )
    system = f"""
You are FARMIQ AI Crop Assistant, the conversational layer of a crop-health decision-support system.
{language_instruction}

IMPORTANT OPERATING RULES:
1. FARMIQ's vision model, weather-risk logic, soil inputs and decision engine are the source of truth for the CURRENT FIELD.
2. Explain those signals clearly; do not pretend the LLM itself diagnosed the plant.
3. Never invent current-field measurements, diagnosis, weather, soil values, pesticide doses, registrations, or guarantees.
4. If confidence is below 65%, recommend expert/lab validation before treatment.
5. For treatment questions, only discuss the candidate treatment categories supplied by FARMIQ. Tell the farmer to verify local registration, product label, dose, PHI and local agronomist guidance. Never invent a dose.
6. For nutrient questions, explain that N/P/K recommendations are crop- and soil-context dependent and should be verified before application.
7. If the user asks something unrelated to agriculture, answer briefly and offer to return to the crop assessment.
8. Prefer an actionable structure: what we see -> why it matters -> what to do next -> what to monitor.
9. Clearly identify when the app is using Guided Demo data.
10. Do not claim certainty beyond the supplied confidence/risk signals.

CURRENT FARMIQ ASSESSMENT:
{ctx}
"""

    messages = [{"role": "system", "content": system}]
    messages.extend(st.session_state.chat_messages[-10:])
    messages.append({"role": "user", "content": user_text})

    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": 0.25,
        "max_completion_tokens": 900,
        "stream": False,
        "include_reasoning": False,
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=45,
        )
        if response.status_code != 200:
            try:
                detail = response.json().get("error", {}).get("message", response.text)
            except Exception:
                detail = response.text
            return None, f"Groq API error ({response.status_code}): {detail}"
        data = response.json()
        answer = data["choices"][0]["message"]["content"].strip()
        return answer, None
    except requests.exceptions.Timeout:
        return None, h_or_e(
            "Groq took too long to respond. Check your internet connection and try again.",
            "Groq ने जवाब देने में बहुत समय लिया। इंटरनेट कनेक्शन जाँचकर फिर प्रयास करें।"
        )
    except requests.exceptions.RequestException as exc:
        return None, f"Could not reach Groq: {exc}"
    except Exception as exc:
        return None, f"Assistant error: {exc}"


def test_groq_connection():
    key = get_groq_key().strip()
    if not key:
        return False, h_or_e("No Groq API key entered.", "Groq API key दर्ज नहीं की गई है।")
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": "Reply with exactly: FARMIQ ONLINE"}],
                "temperature": 0,
                "max_completion_tokens": 20,
                "stream": False,
                "include_reasoning": False,
            },
            timeout=20,
        )
        if r.status_code == 200:
            return True, "FARMIQ ONLINE"
        try:
            msg = r.json().get("error", {}).get("message", r.text)
        except Exception:
            msg = r.text
        return False, f"HTTP {r.status_code}: {msg}"
    except Exception as exc:
        return False, str(exc)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🌱 FARMIQ")
    st.caption(h_or_e("Crop Health Intelligence", "फसल स्वास्थ्य इंटेलिजेंस"))

    # Language and theme are preferences only. Navigation keeps a stable index
    # so changing language never jumps the user back to the first module.
    language = st.radio(
        "Language / भाषा",
        ["English", "हिन्दी"],
        horizontal=True,
        index=0 if st.session_state.lang == "en" else 1,
        key="language_selector",
    )
    new_lang = "hi" if language == "हिन्दी" else "en"
    if new_lang != st.session_state.lang:
        st.session_state.lang = new_lang
        st.session_state.language = "English" if new_lang == "en" else "हिन्दी"
        st.session_state._ui_language_changed = True
        # nav_index is the stable position of the module across translations.
        st.rerun()

    theme_choice = st.radio(
        "Theme / थीम",
        ["☀️ Light", "🌙 Dark"],
        horizontal=True,
        index=0 if st.session_state.theme == "light" else 1,
        key="theme_selector",
    )
    new_theme = "dark" if theme_choice == "🌙 Dark" else "light"
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.rerun()

    st.divider()
    with st.expander("🤖 " + h_or_e("AI Crop Assistant", "AI फसल सहायक"), expanded=True):
        st.caption(h_or_e(
            "Powered by Groq. Key is kept in this session only.",
            "Groq द्वारा संचालित। API key केवल इस session में रखी जाती है।"
        ))
        _configured_key = get_groq_key().strip()
        if _configured_key:
            st.success("🔐 Groq API key configured", icon="✓")
            st.caption(h_or_e(
                "Using your saved environment/Streamlit secret. No need to enter it again.",
                "सहेजी गई environment/Streamlit secret का उपयोग हो रहा है। दोबारा key डालने की जरूरत नहीं है।"
            ))
        else:
            st.session_state.groq_api_key = st.text_input(
                "Groq API Key",
                value="",
                type="password",
                placeholder="gsk_...",
                help="For local demos, paste your Groq key here. For deployment, use Streamlit Secrets."
            )
        if st.button("🔌 Test FARMIQ AI connection", use_container_width=True):
            ok, msg = test_groq_connection()
            if ok:
                st.success("✓ " + msg)
            else:
                st.error(msg)
        st.caption("Model: openai/gpt-oss-20b • Free-tier limits apply")
        st.caption(("🌙 " if st.session_state.theme == "dark" else "☀️ ") + h_or_e("Current theme: " + st.session_state.theme.title(), "वर्तमान थीम: " + ("डार्क" if st.session_state.theme == "dark" else "लाइट")))

    st.divider()

    st.markdown("### 🌾 " + t("profile"))

    farmer = st.text_input(t("farmer"), "Demo Farmer")
    crop = st.selectbox(t("crop"), ["Tomato", "Potato", "Pepper", "Other"])
    variety = st.text_input(t("variety"), "Local / not specified")
    stage = st.selectbox(
        t("stage"),
        ["Seedling", "Vegetative", "Flowering", "Fruiting", "Maturity"],
        index=2,
    )
    location = st.selectbox(t("location"), list(CITIES))
    lat, lon = CITIES[location]

    st.divider()

    st.markdown("### 🧪 " + t("soil"))
    soil_available = st.checkbox(t("use_soil"), True)
    st.session_state.soil_available = soil_available

    n = st.number_input("N", 0., 200., 70., 1., disabled=not soil_available)
    p = st.number_input("P", 0., 100., 30., 1., disabled=not soil_available)
    k = st.number_input("K", 0., 400., 160., 1., disabled=not soil_available)
    ph = st.number_input("pH", 3., 10., 6.8, .1, disabled=not soil_available)

    st.divider()

    st.markdown("### 🎬 " + t("demo"))
    st.session_state.demo = st.toggle(
        h_or_e("Guided demo scenario", "गाइडेड डेमो"),
        value=st.session_state.demo,
    )

    # --------------------------------------------------------
    # Progressive workflow navigation
    # --------------------------------------------------------
    st.divider()
    st.markdown("### 🧭 " + h_or_e("Crop Health Journey", "फसल स्वास्थ्य यात्रा"))

    STEPS = [
        ("plant_image", "📷", "plant_image", "IMAGE RECOGNITION"),
        ("diagnosis", "🦠", "diagnosis", "IMAGE RECOGNITION"),
        ("confidence", "📊", "confidence", "IMAGE RECOGNITION"),
        ("expert", "👨‍🌾", "expert", "IMAGE RECOGNITION"),
        ("current_weather", "🌡️", "current_weather", "WEATHER INTELLIGENCE"),
        ("risk", "⚠️", "risk", "WEATHER INTELLIGENCE"),
        ("forecast", "🔮", "forecast", "WEATHER INTELLIGENCE"),
        ("soil_assess", "🧪", "soil_assess", "SOIL & NUTRITION"),
        ("nutrient_need", "🌿", "nutrient_need", "SOIL & NUTRITION"),
        ("health", "🧠", "health", "DECISION ENGINE"),
        ("treatment", "💊", "treatment", "DECISION ENGINE"),
        ("follow", "🔄", "follow", "MONITORING"),
        ("regional", "🗺️", "regional", "MONITORING"),
        ("assistant", "🤖", "AI Crop Assistant", "MONITORING"),
    ]
    GROUP_NAMES = {
        "IMAGE RECOGNITION": ("🖼️", "IMAGE RECOGNITION", "इमेज पहचान"),
        "WEATHER INTELLIGENCE": ("🌦️", "WEATHER INTELLIGENCE", "मौसम इंटेलिजेंस"),
        "SOIL & NUTRITION": ("🧪", "SOIL & NUTRITION", "मृदा और पोषण"),
        "DECISION ENGINE": ("🧠", "DECISION ENGINE", "निर्णय इंजन"),
        "MONITORING": ("🔄", "MONITORING", "निगरानी"),
    }

    step_keys = [x[0] for x in STEPS]
    if st.session_state.nav not in step_keys:
        st.session_state.nav = "plant_image"
    current_index = step_keys.index(st.session_state.nav)
    st.session_state.nav_index = current_index

    progress = current_index / (len(STEPS) - 1)
    st.markdown(
        "<div class='progress-label'><span>" +
        h_or_e("Journey progress", "यात्रा प्रगति") +
        f"</span><b>{current_index + 1}/{len(STEPS)}</b></div>" +
        f"<div class='progress-track'><div class='progress-fill' style='width:{progress*100:.1f}%'></div></div>",
        unsafe_allow_html=True,
    )

    last_group = None
    for idx, (key, icon, text_key, group) in enumerate(STEPS):
        if group != last_group:
            gi, ge, gh = GROUP_NAMES[group]
            st.caption(f"{gi} {h_or_e(ge, gh)}")
            last_group = group

        label = f"{icon} {t(text_key)}"
        if idx < current_index:
            label = "✓ " + label
        elif idx == current_index:
            label = "● " + label

        if st.button(
            label,
            key=f"stage_{key}",
            use_container_width=True,
            type="primary" if idx == current_index else "secondary",
        ):
            st.session_state.nav = key
            st.session_state.nav_index = idx
            st.rerun()

    st.caption(h_or_e(
        "Jump to any stage anytime. Use Continue → for the natural workflow.",
        "आप किसी भी चरण पर कभी भी जा सकते हैं। सामान्य प्रक्रिया के लिए आगे बढ़ें → का उपयोग करें।"
    ))

# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🌱 FARMIQ</h1>
        <p>AI-Powered Crop Health Intelligence & Early-Warning Decision Support</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="navhint">
    <b>{h_or_e("Two independent intelligence paths:",
               "दो स्वतंत्र इंटेलिजेंस पाथ:")}</b>
    🖼️ {h_or_e("Image Recognition", "इमेज पहचान")}
    &nbsp; + &nbsp;
    🌦️ {h_or_e("Weather Intelligence", "मौसम इंटेलिजेंस")}
    &nbsp; → &nbsp;
    🧠 {h_or_e("Decision Engine", "निर्णय इंजन")}
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 1. PLANT IMAGE
# ============================================================


# ============================================================
# FARMIQ COMMAND CENTER
# ============================================================
if st.session_state.nav in ("health", "treatment", "regional"):
    _ctx_pred = st.session_state.get("predictions") or []
    _ctx_top = _ctx_pred[0] if _ctx_pred else None
    _ctx_disease = pretty(_ctx_top["label"]) if _ctx_top else h_or_e("Awaiting diagnosis", "निदान की प्रतीक्षा")
    _ctx_conf = (_ctx_top["score"] * 100) if _ctx_top else 0
    _ctx_risk = st.session_state.get("risk") or {}
    _ctx_risk_score = float(_ctx_risk.get("score", 0) or 0)
    _ctx_soil = 72 if st.session_state.get("soil_available", False) else 35
    _ctx_complete = 92 if (_ctx_top and _ctx_risk and st.session_state.get("soil_available", False)) else 65

    st.markdown(
        '<div class="hero">'
        '<h1>🌱 FARMIQ Crop Health Command Center</h1>'
        '<p>' + h_or_e(
            "See the evidence behind every recommendation.",
            "हर सिफारिश के पीछे मौजूद साक्ष्य देखें।"
        ) + '</p></div>',
        unsafe_allow_html=True
    )

    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.markdown(f'<div class="kpi"><div class="kpi-label">{h_or_e("Diagnosis","निदान")}</div><div class="kpi-value">🦠 {_ctx_disease}</div></div>', unsafe_allow_html=True)
    with d2:
        st.markdown(f'<div class="kpi"><div class="kpi-label">{h_or_e("AI confidence","AI विश्वास")}</div><div class="kpi-value">{_ctx_conf:.0f}%</div></div>', unsafe_allow_html=True)
    with d3:
        _risk_label = _ctx_risk.get("level") or h_or_e("Pending","लंबित")
        st.markdown(f'<div class="kpi"><div class="kpi-label">{h_or_e("Weather risk","मौसम जोखिम")}</div><div class="kpi-value">⚠️ {_risk_label} {_ctx_risk_score:.0f}%</div></div>', unsafe_allow_html=True)
    with d4:
        st.markdown(f'<div class="kpi"><div class="kpi-label">{h_or_e("Data completeness","डेटा पूर्णता")}</div><div class="kpi-value">✓ {_ctx_complete:.0f}%</div></div>', unsafe_allow_html=True)

    st.markdown("### " + h_or_e("Why FARMIQ recommends an action", "FARMIQ यह कार्रवाई क्यों सुझाता है"))
    render_signal_dashboard(_ctx_conf/100, _ctx_risk_score, _ctx_soil, _ctx_complete)

    q1, q2 = st.columns(2)
    with q1:
        st.markdown(
            '<div class="signal-card"><b>🔎 ' + h_or_e("Vision signal", "विज़न सिग्नल") +
            '</b><br><span class="small">' +
            h_or_e("Leaf-image evidence and confidence from the disease model.",
                   "पत्ती की इमेज से मिले साक्ष्य और रोग मॉडल का विश्वास स्तर।") +
            '</span></div>', unsafe_allow_html=True
        )
    with q2:
        st.markdown(
            '<div class="signal-card"><b>🌦️ ' + h_or_e("Environmental signal", "पर्यावरणीय सिग्नल") +
            '</b><br><span class="small">' +
            h_or_e("Weather conditions are evaluated as a separate risk signal.",
                   "मौसम की स्थिति को अलग जोखिम सिग्नल के रूप में आंका जाता है।") +
            '</span></div>', unsafe_allow_html=True
        )

    st.markdown("### " + h_or_e("Decision pipeline", "निर्णय प्रक्रिया"))
    st.markdown(
        '<div class="navhint">📷 <b>' + h_or_e("See", "देखें") +
        '</b> → 🦠 <b>' + h_or_e("Detect", "पहचानें") +
        '</b> → 🌦️ <b>' + h_or_e("Predict", "पूर्वानुमान") +
        '</b> → 🧪 <b>' + h_or_e("Contextualize", "संदर्भ जोड़ें") +
        '</b> → 🧠 <b>' + h_or_e("Decide", "निर्णय लें") +
        '</b> → 💊 <b>' + h_or_e("Act", "कार्रवाई") +
        '</b> → 🔄 <b>' + h_or_e("Learn", "सीखें") +
        '</b></div>',
        unsafe_allow_html=True
    )


if st.session_state.nav == "plant_image":

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("## 📷 " + t("plant_image"))
    st.write(h_or_e(
        "Upload a clear crop/leaf image. The image model is one signal; FARMIQ later combines it with weather, soil and crop context.",
        "फसल/पत्ती की साफ तस्वीर अपलोड करें। इमेज मॉडल एक संकेत है; FARMIQ बाद में इसे मौसम, मृदा और फसल की जानकारी के साथ जोड़ता है।"
    ))

    image_file = st.file_uploader(
        h_or_e("Upload image", "तस्वीर अपलोड करें"),
        ["jpg", "jpeg", "png"],
    )

    if image_file:
        st.session_state.image = Image.open(image_file).convert("RGB")

    if st.session_state.get("image") is not None:
        a, b = st.columns([1, 1.25])
        with a:
            st.image(st.session_state.image, caption=h_or_e("Field image", "खेत की तस्वीर"),
                     use_container_width=True)
        with b:
            st.markdown(
                f"""
                <div class="blue">
                <b>{h_or_e("FARMIQ will check:", "FARMIQ जांचेगा:")}</b><br><br>
                📷 {h_or_e("Visible symptoms", "दिखने वाले लक्षण")}<br>
                🌱 {h_or_e("Crop & growth stage", "फसल और अवस्था")}<br>
                🌦️ {h_or_e("Environmental context", "पर्यावरणीय जानकारी")}<br>
                🧪 {h_or_e("Soil context", "मृदा जानकारी")}<br>
                🎯 {h_or_e("Prediction confidence", "पूर्वानुमान विश्वास स्तर")}
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("🔍 " + t("analyze"), type="primary", use_container_width=True):
                if st.session_state.demo:
                    st.session_state.predictions = [
                        {"label": "Tomato___Early_blight", "score": .87},
                        {"label": "Tomato___Late_blight", "score": .06},
                        {"label": "Tomato___healthy", "score": .04},
                    ]
                    st.success(h_or_e("Guided demonstration result generated.",
                                      "गाइडेड डेमो परिणाम तैयार है।"))
                else:
                    try:
                        st.session_state.predictions = predict_image(st.session_state.image)
                        st.success(h_or_e("Live AI prediction completed.",
                                          "लाइव AI पूर्वानुमान पूरा हुआ।"))
                    except Exception as e:
                        st.error(h_or_e("Live AI model could not be loaded.",
                                        "लाइव AI मॉडल लोड नहीं हो सका।"))
                        st.code(str(e))

    elif st.session_state.demo:
        st.markdown(
            '<div class="blue"><b>🎬 Guided demo:</b> you can use the complete '
            'workflow without uploading an image.</div>',
            unsafe_allow_html=True,
        )
        if st.button("▶ " + t("run_demo"), type="primary"):
            st.session_state.predictions = [
                {"label": "Tomato___Early_blight", "score": .87},
                {"label": "Tomato___Late_blight", "score": .06},
                {"label": "Tomato___healthy", "score": .04},
            ]

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# 2. DIAGNOSIS
# ============================================================

elif st.session_state.nav == "diagnosis":

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("## 🦠 " + t("diagnosis"))

    pred = st.session_state.predictions

    if not pred:
        st.info(h_or_e(
            "No diagnosis yet. Open Plant Image and analyze an image, or run Guided Demo.",
            "अभी कोई रोग पहचान उपलब्ध नहीं है। Plant Image खोलकर तस्वीर का विश्लेषण करें या Guided Demo चलाएँ।"
        ))
    else:
        label = pred[0]["label"]
        conf = pred[0]["score"]
        name = RULES.get(label, {}).get("name", pretty(label))

        if st.session_state.demo:
            st.warning(h_or_e(
                "GUIDED DEMO — sample diagnosis, not a live prediction.",
                "गाइडेड डेमो — यह नमूना रोग पहचान है, लाइव पूर्वानुमान नहीं।"
            ))

        a,b,c = st.columns(3)
        a.metric(h_or_e("Detected issue","पहचानी गई समस्या"), name)
        b.metric(t("confidence"), f"{conf*100:.0f}%")
        c.metric(h_or_e("Crop","फसल"), crop)

        render_meter(conf*100, h_or_e("AI confidence", "AI विश्वास स्तर"))

        st.markdown("### " + h_or_e("Top model predictions", "मुख्य मॉडल पूर्वानुमान"))

        chart = pd.DataFrame({
            h_or_e("Prediction","पूर्वानुमान"):
                [pretty(x["label"]) for x in pred],
            h_or_e("Confidence","विश्वास %"):
                [x["score"]*100 for x in pred],
        }).set_index(h_or_e("Prediction","पूर्वानुमान"))

        visual_pie(
        chart.index.tolist(),
        chart[h_or_e("Confidence","विश्वास %")].tolist(),
        h_or_e("AI diagnosis probability mix", "AI रोग पहचान संभावनाएँ"),
        "%"
    )

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# 3. CONFIDENCE / SEVERITY
# ============================================================

elif st.session_state.nav == "confidence":

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("## 📊 " + t("confidence"))

    pred = st.session_state.predictions
    risk = st.session_state.risk

    if not pred:
        st.info(h_or_e("Run image diagnosis first.",
                       "पहले इमेज रोग पहचान चलाएँ।"))
    else:
        conf = pred[0]["score"]
        risk_level = risk["level"] if risk else "NOT ASSESSED"

        if conf < .65:
            severity = "Uncertain"
        elif conf >= .85 and risk_level == "HIGH":
            severity = "High"
        elif conf >= .75:
            severity = "Moderate"
        else:
            severity = "Low"

        a,b,c = st.columns(3)
        a.metric(h_or_e("AI confidence","AI विश्वास स्तर"), f"{conf*100:.0f}%")
        b.metric(h_or_e("Environmental risk","पर्यावरणीय जोखिम"), risk_level)
        c.metric(h_or_e("Prototype severity","प्रोटोटाइप गंभीरता"), severity)

        render_meter(conf*100, h_or_e("Model confidence", "मॉडल विश्वास स्तर"))

        if conf < .65:
            st.markdown(
                '<div class="red"><h3>⚠️ Expert validation required</h3>'
                'The system should not trigger a pesticide recommendation from a low-confidence result.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="green"><h3>✓ Confidence sufficient for contextual assessment</h3>'
                'The result still needs environmental and field context before action.</div>',
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# 4. EXPERT VALIDATION
# ============================================================

elif st.session_state.nav == "expert":

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("## 👨‍🌾 " + t("expert"))

    pred = st.session_state.predictions
    conf = pred[0]["score"] if pred else 0

    st.markdown(
        """
        <div class="blue">
        <b>Human-in-the-loop design</b><br><br>
        FARMIQ does not assume the AI is always correct. Low-confidence,
        severe or atypical cases can be escalated to an agricultural expert
        or laboratory. Confirmed outcomes can later become feedback data.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.metric(h_or_e("Current AI confidence","वर्तमान AI विश्वास स्तर"), f"{conf*100:.0f}%")

    if conf < .65:
        st.error(h_or_e("ESCALATE FOR VALIDATION", "सत्यापन के लिए भेजें"))
    elif conf < .80:
        st.warning(h_or_e("Expert review recommended", "विशेषज्ञ समीक्षा की सलाह"))
    else:
        st.success(h_or_e("High-confidence screening", "उच्च-विश्वास स्क्रीनिंग"))

    if st.button("📨 " + h_or_e("Send simulated case for expert validation",
                                "विशेषज्ञ सत्यापन के लिए केस भेजें")):
        st.success(h_or_e(
            "Case queued for expert validation — prototype workflow.",
            "केस विशेषज्ञ सत्यापन के लिए भेजा गया — प्रोटोटाइप वर्कफ़्लो।"
        ))

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# 5. CURRENT WEATHER
# ============================================================

elif st.session_state.nav == "current_weather":

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("## 🌡️ " + t("current_weather"))

    try:
        w = get_weather(lat, lon)
        st.session_state.weather = w
        st.session_state.weather_live = True
    except Exception:
        w = {
            "current": {
                "temperature_2m": 27,
                "relative_humidity_2m": 82,
                "rain": 2.5,
                "precipitation": 2.5,
                "wind_speed_10m": 8,
            },
            "daily": {
                "time": ["Today", "Tomorrow", "Day 3"],
                "precipitation_sum": [2.5, 6, 1.2],
                "temperature_2m_max": [28,27,29],
                "temperature_2m_min": [20,20,21],
            }
        }
        st.session_state.weather = w
        st.session_state.weather_live = False

    if st.session_state.weather_live:
        st.success(h_or_e("Live weather loaded.", "लाइव मौसम डेटा लोड हुआ।"))
    else:
        st.warning(h_or_e("Live weather unavailable — fallback demo data.",
                           "लाइव मौसम उपलब्ध नहीं — डेमो डेटा दिखाया जा रहा है।"))

    c = w["current"]
    a,b,c1,d = st.columns(4)
    a.metric(h_or_e("Temperature","तापमान"), f"{c['temperature_2m']} °C")
    b.metric(h_or_e("Humidity","आर्द्रता"), f"{c['relative_humidity_2m']} %")
    c1.metric(h_or_e("Rainfall","वर्षा"), f"{c['rain']} mm")
    d.metric(h_or_e("Wind","हवा"), f"{c['wind_speed_10m']} km/h")

    st.markdown(
        '<div class="blue"><b>Why weather matters:</b> weather does not diagnose '
        'the disease. It provides the environmental context used by the separate '
        'risk-assessment path.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# 6. DISEASE RISK
# ============================================================

elif st.session_state.nav == "risk":

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("## ⚠️ " + t("risk"))

    pred = st.session_state.predictions
    if not pred:
        st.info(h_or_e(
            "Run image diagnosis first so the weather path can assess risk for that disease profile.",
            "पहले इमेज रोग पहचान चलाएँ ताकि मौसम पाथ उस रोग प्रोफाइल के लिए जोखिम का आकलन कर सके।"
        ))
    else:
        if st.session_state.weather is None:
            try:
                st.session_state.weather = get_weather(lat, lon)
                st.session_state.weather_live = True
            except Exception:
                st.session_state.weather = {
                    "current": {"temperature_2m":27,"relative_humidity_2m":82,
                                "rain":2.5,"precipitation":2.5,"wind_speed_10m":8}
                }
                st.session_state.weather_live = False

        level, score, why = weather_risk(
            st.session_state.weather,
            pred[0]["label"]
        )
        st.session_state.risk = {"level":level, "score":score, "why":why}

        st.markdown(
            f"""
            <div class="{'red' if level=='HIGH' else 'yellow' if level=='MODERATE' else 'green'}">
                <h1 style="margin:0;">{'🔴' if level=='HIGH' else '🟡' if level=='MODERATE' else '🟢'} {level}</h1>
                <p>{h_or_e("Environmental disease risk",
                            "पर्यावरणीय रोग जोखिम")} — {score}/100</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        render_meter(score, h_or_e("Environmental risk score",
                                   "पर्यावरणीय जोखिम स्कोर"))

        st.markdown("### " + h_or_e("Why?", "क्यों?"))
        for x in why:
            st.write("• " + x)

        st.markdown(
            '<div class="blue"><b>Separate model path:</b> image recognition '
            'answers “what may be happening”; weather assessment answers '
            '“are current conditions favorable for worsening/development?”</div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# 7. FORECAST
# ============================================================

elif st.session_state.nav == "forecast":

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("## 🔮 " + t("forecast"))

    if st.session_state.weather is None:
        try:
            st.session_state.weather = get_weather(lat, lon)
            st.session_state.weather_live = True
        except Exception:
            st.session_state.weather = {
                "daily": {
                    "time":["Today","Tomorrow","Day 3"],
                    "precipitation_sum":[2.5,6,1.2],
                    "temperature_2m_max":[28,27,29],
                    "temperature_2m_min":[20,20,21]
                }
            }
            st.session_state.weather_live = False

    daily = st.session_state.weather.get("daily", {})
    if daily:
        df = pd.DataFrame({
            "Day": daily.get("time", [])[:3],
            "Rainfall": daily.get("precipitation_sum", [])[:3],
            "Max Temp": daily.get("temperature_2m_max", [])[:3],
            "Min Temp": daily.get("temperature_2m_min", [])[:3],
        })

        st.dataframe(df, hide_index=True, use_container_width=True)

        visual_line(
            df["Day"].tolist(),
            {
                h_or_e("Max temperature", "अधिकतम तापमान"): df["Max Temp"].tolist(),
                h_or_e("Min temperature", "न्यूनतम तापमान"): df["Min Temp"].tolist(),
            },
            h_or_e("3-day weather trend", "3-दिन का मौसम रुझान"),
            "°C"
        )
        visual_bar(
            df["Day"].tolist(),
            df["Rainfall"].tolist(),
            h_or_e("Rainfall outlook", "वर्षा पूर्वानुमान"),
            h_or_e("Rainfall", "वर्षा"),
            " mm"
        )

    st.caption(h_or_e(
        "Prototype forecast visualization. Production disease forecasting requires crop/disease-specific field validation.",
        "प्रोटोटाइप पूर्वानुमान विज़ुअलाइज़ेशन। वास्तविक रोग पूर्वानुमान के लिए फसल/रोग-विशिष्ट फील्ड वैलिडेशन आवश्यक है।"
    ))

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# 8. SOIL ASSESSMENT
# ============================================================

elif st.session_state.nav == "soil_assess":

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("## 🧪 " + t("soil_assess"))

    if not soil_available:
        st.markdown(
            '<div class="blue"><b>No soil-test data.</b> FARMIQ withholds '
            'nutrient-specific recommendations instead of guessing.</div>',
            unsafe_allow_html=True,
        )
    else:
        values = pd.DataFrame({
            "N": [n],
            "P": [p],
            "K": [k],
        })

        a,b,c,d = st.columns(4)
        a.metric("N", f"{n:.0f}")
        b.metric("P", f"{p:.0f}")
        c.metric("K", f"{k:.0f}")
        d.metric("pH", f"{ph:.1f}")

        st.markdown("### " + h_or_e("Visual soil profile", "मृदा पोषण प्रोफाइल"))
        visual_bar(
            ["N", "P", "K", "pH"],
            [n, p, k, ph],
            h_or_e("Soil nutrient profile", "मृदा पोषण प्रोफाइल"),
            h_or_e("Measured value", "मापा गया मान")
        )

        flags = []
        if n < 40: flags.append("N")
        if p < 20: flags.append("P")
        if k < 120: flags.append("K")

        st.session_state.soil_flags = flags

        if flags:
            names = [NUTRIENTS[x][h_or_e("en","hi")] for x in flags]
            st.markdown(
                f'<div class="yellow"><h3>🟡 {h_or_e("Potential deficiency flags","संभावित कमी के संकेत")}: {", ".join(names)}</h3></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="green"><h3>🟢 No obvious N/P/K deficiency flag</h3>'
                'Do not add external nutrients without evidence of need.</div>',
                unsafe_allow_html=True,
            )

        if ph < 5.5 or ph > 8:
            st.warning(h_or_e(
                f"pH {ph:.1f} needs crop-specific interpretation.",
                f"pH {ph:.1f} की फसल-विशिष्ट व्याख्या आवश्यक है।"
            ))

    st.caption(h_or_e(
        "Prototype screening only — not universal fertilizer thresholds.",
        "यह केवल प्रोटोटाइप स्क्रीनिंग है — सार्वभौमिक उर्वरक सीमा नहीं।"
    ))

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# 9. NUTRIENT REQUIREMENT
# ============================================================

elif st.session_state.nav == "nutrient_need":

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("## 🌿 " + t("nutrient_need"))

    flags = st.session_state.get("soil_flags", [])

    if not soil_available:
        st.info(h_or_e(
            "Provide recent soil-test values to unlock nutrient screening.",
            "पोषक तत्व स्क्रीनिंग के लिए हाल का मृदा परीक्षण डेटा दें।"
        ))
    elif not flags:
        st.success(h_or_e(
            "No external N/P/K intervention is triggered by the current screening.",
            "वर्तमान स्क्रीनिंग के अनुसार बाहरी N/P/K हस्तक्षेप की आवश्यकता नहीं दिखती।"
        ))
    else:
        st.markdown(
            '<div class="yellow"><b>External nutrient support may be required.</b>'
            '<br>Confirm with crop-specific soil interpretation before application.</div>',
            unsafe_allow_html=True,
        )
        for x in flags:
            info = NUTRIENTS[x]
            st.markdown(
                f"### {info['en'] if st.session_state.lang=='en' else info['hi']}"
            )
            st.write(
                info["source"] if st.session_state.lang=="en" else info["source_hi"]
            )
            st.caption(
                info["note"] if st.session_state.lang=="en" else info["note_hi"]
            )

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# 10. MULTIMODAL HEALTH ASSESSMENT
# ============================================================

elif st.session_state.nav == "health":

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("## 🧠 " + t("health"))

    pred = st.session_state.predictions
    risk = st.session_state.risk

    if not pred:
        st.info(h_or_e(
            "Image recognition has not produced a diagnosis yet.",
            "इमेज पहचान ने अभी कोई रोग परिणाम नहीं दिया है।"
        ))
    else:
        conf = pred[0]["score"]
        risk_level = risk["level"] if risk else "NOT ASSESSED"
        risk_score = risk["score"] if risk else 0

        if conf < .65:
            decision = "EXPERT VALIDATION"
        elif risk_level == "HIGH" or conf >= .85:
            decision = "INTERVENTION RECOMMENDED"
        elif risk_level == "LOW" and conf < .75:
            decision = "MONITOR"
        else:
            decision = "TARGETED MANAGEMENT"

        labels = {
            "EXPERT VALIDATION": ("⚠️", t("validation")),
            "INTERVENTION RECOMMENDED": ("🔴", t("intervention")),
            "TARGETED MANAGEMENT": ("🟡", t("targeted")),
            "MONITOR": ("🟢", t("monitor_decision")),
        }

        icon, decision_text = labels[decision]

        st.markdown(
            f"""
            <div class="{'red' if decision in ['EXPERT VALIDATION','INTERVENTION RECOMMENDED']
                         else 'yellow' if decision=='TARGETED MANAGEMENT'
                         else 'green'}">
                <h1 style="margin:0;">{icon} {decision_text}</h1>
                <p>{h_or_e("The decision engine combines independent signals instead of relying on the image alone.",
                            "निर्णय इंजन केवल इमेज पर निर्भर नहीं करता; यह अलग-अलग संकेतों को जोड़ता है।")}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### " + h_or_e("Crop Health Signal Matrix", "फसल स्वास्थ्य सिग्नल मैट्रिक्स"))
        soil_score = 72 if soil_available else 35
        completeness = 92 if soil_available and risk else (72 if risk else 52)
        render_signal_dashboard(conf, risk_score, soil_score, completeness)

        gc1, gc2, gc3 = st.columns(3)
        with gc1:
            visual_gauge(conf*100, h_or_e("AI confidence","AI विश्वास"),
                         h_or_e("Image model certainty","इमेज मॉडल की निश्चितता"))
        with gc2:
            visual_gauge(risk_score, h_or_e("Disease risk","रोग जोखिम"),
                         h_or_e("Weather-driven risk signal","मौसम आधारित जोखिम संकेत"))
        with gc3:
            visual_gauge(soil_score, h_or_e("Soil context","मृदा संदर्भ"),
                         h_or_e("Supporting context, not diagnosis","सहायक संदर्भ, निदान नहीं"))

        st.markdown("### " + h_or_e("Signal dashboard", "सिग्नल डैशबोर्ड"))
        a,b,c = st.columns(3)
        with a:
            st.metric("📷 " + h_or_e("Image confidence","इमेज विश्वास"), f"{conf*100:.0f}%")
            render_meter(conf*100)
        with b:
            st.metric("🌦️ " + h_or_e("Weather risk","मौसम जोखिम"), risk_level)
            render_meter(risk_score)
        with c:
            soil_status = "Available" if soil_available else "Missing"
            st.metric("🧪 " + h_or_e("Soil context","मृदा संदर्भ"), soil_status)

        st.markdown(
            """
            <div class="blue">
            <b>Architecture:</b><br><br>
            📷 Image model → visible disease signal<br>
            🌦️ Weather path → environmental risk signal<br>
            🧪 Soil path → nutrient/stress context<br>
            🌱 Crop context → crop + growth stage + location<br><br>
            ↓<br><br>
            🧠 Decision Engine → action pathway
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# 11. TREATMENT
# ============================================================

elif st.session_state.nav == "treatment":

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("## 💊 " + t("treatment"))

    pred = st.session_state.predictions

    if not pred:
        st.info(h_or_e("Run diagnosis first.",
                       "पहले रोग पहचान चलाएँ।"))
    else:
        label = pred[0]["label"]
        conf = pred[0]["score"]
        risk = st.session_state.risk["level"] if st.session_state.risk else None

        if conf < .65:
            st.markdown(
                '<div class="red"><h2>⚠️ Do not recommend pesticide yet</h2>'
                'Low-confidence cases require expert/lab validation.</div>',
                unsafe_allow_html=True,
            )
        elif "healthy" in label.lower():
            st.markdown(
                '<div class="green"><h2>🟢 No disease intervention triggered</h2>'
                'Avoid unnecessary pesticide application.</div>',
                unsafe_allow_html=True,
            )
        elif label in RULES:
            rule = RULES[label]
            use_hi = st.session_state.lang == "hi"

            name = rule["hi"] if use_hi else rule["name"]
            typ = rule["type_hi"] if use_hi else rule["type"]
            actions = rule["actions_hi"] if use_hi else rule["actions"]

            st.markdown(
                f'<div class="red"><h2>🔴 {name}</h2>'
                f'<b>{h_or_e("Type","प्रकार")}:</b> {typ}<br>'
                f'<b>{h_or_e("Environmental risk","पर्यावरणीय जोखिम")}:</b> {risk or "Not assessed"}</div>',
                unsafe_allow_html=True,
            )

            a,b = st.columns(2)

            with a:
                st.markdown("### 🦠 " + h_or_e("Disease management","रोग प्रबंधन"))
                st.write("**" + h_or_e("Candidate active ingredients / options:",
                                       "संभावित सक्रिय तत्व / विकल्प:") + "**")
                for option in rule["options"]:
                    st.write("• " + option)

            with b:
                st.markdown("### 🌱 " + h_or_e("Integrated actions","समेकित प्रबंधन"))
                for action in actions:
                    st.write("✓ " + action)

            st.divider()

            flags = st.session_state.get("soil_flags", [])
            st.markdown("### 🧪 " + h_or_e("External nutrient decision",
                                           "बाहरी पोषक तत्व निर्णय"))

            if flags:
                for x in flags:
                    info = NUTRIENTS[x]
                    st.write(
                        f"**{info['hi'] if use_hi else info['en']}:** "
                        f"{info['source_hi'] if use_hi else info['source']} "
                        f"{h_or_e('after confirming crop-specific requirement.',
                                 'फसल-विशिष्ट आवश्यकता की पुष्टि के बाद।')}"
                    )
            else:
                st.success(h_or_e(
                    "No external N/P/K intervention triggered by current soil screening.",
                    "वर्तमान मृदा स्क्रीनिंग के अनुसार बाहरी N/P/K हस्तक्षेप ट्रिगर नहीं हुआ।"
                ))

            st.markdown(
                '<div class="yellow"><b>⚠️ Safety gate:</b> FARMIQ is decision support, '
                'not an unsupervised pesticide prescription. Verify current local '
                'registration, crop/disease label, dose, pre-harvest interval and '
                'resistance-management guidance before real-world use.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info(h_or_e(
                "No verified crop-specific intervention rule is stored for this prediction; FARMIQ will not invent one.",
                "इस पूर्वानुमान के लिए सत्यापित फसल-विशिष्ट उपचार नियम उपलब्ध नहीं है; FARMIQ अनुमान से उपचार नहीं बनाएगा।"
            ))

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# 12. FOLLOW-UP
# ============================================================

elif st.session_state.nav == "follow":

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("## 🔄 " + t("follow"))

    st.write(h_or_e(
        "The decision does not end at treatment. Field outcome becomes feedback.",
        "निर्णय उपचार पर समाप्त नहीं होता। खेत का परिणाम आगे के फीडबैक में बदलता है।"
    ))

    status = st.radio(
        h_or_e("Field outcome", "खेत का परिणाम"),
        [
            h_or_e("Not checked yet", "अभी जांच नहीं हुई"),
            h_or_e("Improving", "सुधार हो रहा है"),
            h_or_e("Unchanged", "कोई बदलाव नहीं"),
            h_or_e("Worsening", "स्थिति बिगड़ रही है"),
        ],
        horizontal=True,
    )

    if "Improving" in status or "सुधार" in status:
        st.success(h_or_e(
            "Improvement recorded. A follow-up image can validate recovery.",
            "सुधार दर्ज किया गया। फॉलो-अप तस्वीर से रिकवरी सत्यापित की जा सकती है।"
        ))
    elif "Worsening" in status or "बिगड़" in status:
        st.error(h_or_e(
            "Worsening recorded → prioritize expert validation.",
            "स्थिति बिगड़ने का रिकॉर्ड → विशेषज्ञ सत्यापन को प्राथमिकता दें।"
        ))
    elif "Unchanged" in status or "बदलाव" in status:
        st.warning(h_or_e(
            "No improvement → re-check diagnosis, weather risk and management.",
            "सुधार नहीं → रोग पहचान, मौसम जोखिम और प्रबंधन फिर जांचें।"
        ))
    else:
        st.info(h_or_e(
            "The next field observation closes the decision loop.",
            "अगला खेत निरीक्षण निर्णय चक्र को पूरा करेगा।"
        ))

    st.markdown(
        '<div class="blue"><b>Learning loop:</b> validated outcomes can later '
        'become feedback data for improving crop/disease models.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# 13. REGIONAL INTELLIGENCE
# ============================================================

elif st.session_state.nav == "regional":

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("## 🗺️ " + t("regional"))

    st.write(h_or_e(
        "Validated farmer observations can be aggregated into regional early-warning intelligence.",
        "सत्यापित किसान अवलोकनों को क्षेत्रीय प्रारंभिक चेतावनी इंटेलिजेंस में बदला जा सकता है।"
    ))

    sample = pd.DataFrame({
        "latitude": [23.26,23.29,23.24,23.31,23.22],
        "longitude": [77.41,77.39,77.45,77.36,77.43],
        "Crop": ["Tomato"]*5,
        "Issue": ["Early blight","Early blight","Late blight","Early blight","Healthy"],
        "Status": ["Validated sample","Validated sample","Review","Validated sample","Healthy"],
    })

    st.map(sample[["latitude","longitude"]])

    a,b,c = st.columns(3)
    a.metric(h_or_e("Observations","अवलोकन"), "5")
    b.metric(h_or_e("Priority cases","प्राथमिक केस"), "3")
    c.metric(h_or_e("Hotspot status","हॉटस्पॉट स्थिति"), "Emerging")

    st.dataframe(sample, hide_index=True, use_container_width=True)

    st.markdown(
        '<div class="blue"><b>System-level flow:</b> farmer observations → '
        'expert validation → regional hotspot intelligence → extension/official response.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# PROGRESSIVE WORKFLOW ORDER
# Defined before quick-access code uses it.
# ============================================================
WORKFLOW = [
    "plant_image", "diagnosis", "confidence", "expert",
    "current_weather", "risk", "forecast",
    "soil_assess", "nutrient_need", "health", "treatment",
    "follow", "regional", "assistant"
]

# ============================================================
# AI CROP ASSISTANT — QUICK ACCESS
# ============================================================
if st.session_state.nav != "assistant":
    with st.expander("🤖 " + h_or_e("Ask FARMIQ AI", "FARMIQ AI से पूछें"), expanded=False):
        st.caption(h_or_e(
            "Open the AI Assistant stage for the full conversation.",
            "पूरी बातचीत के लिए AI Assistant चरण खोलें।"
        ))
        if st.button(h_or_e("Open AI Assistant →", "AI सहायक खोलें →"), key="open_ai_assistant", type="primary"):
            st.session_state.nav = "assistant"
            st.session_state.nav_index = WORKFLOW.index("assistant") if "assistant" in WORKFLOW else len(WORKFLOW)-1
            st.rerun()

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    f"""
    <div class="small" style="text-align:center;padding:1.2rem;">
    🌱 FARMIQ • SIH26131 • {h_or_e("MVP prototype","MVP प्रोटोटाइप")} •
    🖼️ {h_or_e("Image Recognition","इमेज पहचान")} +
    🌦️ {h_or_e("Weather Intelligence","मौसम इंटेलिजेंस")} →
    🧠 {h_or_e("Decision Engine","निर्णय इंजन")}<br>
    {h_or_e("Prototype decision support — verify agricultural recommendations before real-world use.",
             "प्रोटोटाइप निर्णय सहायता — वास्तविक उपयोग से पहले कृषि सुझावों का सत्यापन आवश्यक है।")}
    </div>
    """,
    unsafe_allow_html=True,
)



# ============================================================
# DEDICATED AI CROP ASSISTANT
# ============================================================
if st.session_state.nav == "assistant":
    # In Guided Demo, guarantee that the assistant has a coherent assessment to explain.
    ensure_demo_assessment()

    st.markdown(
        '<div class="hero"><h1>🤖 ' +
        h_or_e("FARMIQ AI Crop Assistant", "FARMIQ AI फसल सहायक") +
        '</h1><p>' +
        h_or_e(
            "Ask questions about your current crop-health assessment. FARMIQ supplies the evidence; the assistant explains it.",
            "अपने वर्तमान फसल-स्वास्थ्य आकलन के बारे में पूछें। FARMIQ प्रमाण देता है; AI सहायक उसे समझाता है।"
        ) + '</p></div>',
        unsafe_allow_html=True
    )

    api_ready = bool(get_groq_key().strip())
    if api_ready:
        st.success(h_or_e(
            f"🟢 AI Assistant connected • {GROQ_MODEL}",
            f"🟢 AI सहायक कनेक्टेड • {GROQ_MODEL}"
        ))
    else:
        st.warning(h_or_e(
            "🔴 AI Assistant is not connected. Enter your Groq API key in the sidebar, then test the connection.",
            "🔴 AI सहायक कनेक्टेड नहीं है। साइडबार में Groq API key डालें और कनेक्शन टेस्ट करें।"
        ))

    ctx = current_farmer_context()
    cc1, cc2, cc3, cc4 = st.columns(4)
    with cc1: st.metric(h_or_e("Crop", "फसल"), ctx["crop"])
    with cc2: st.metric(h_or_e("Diagnosis", "निदान"), ctx["diagnosis"] or "—")
    with cc3: st.metric(h_or_e("Confidence", "विश्वास"), f'{ctx["diagnosis_confidence"] or 0:.0f}%')
    with cc4: st.metric(h_or_e("Risk", "जोखिम"), str(ctx["weather_risk"] or "—"))

    # Context panel makes it obvious what the LLM is grounded in.
    with st.expander("🔎 " + h_or_e("FARMIQ evidence supplied to the assistant", "सहायक को दिया गया FARMIQ प्रमाण"), expanded=True):
        e1,e2,e3,e4 = st.columns(4)
        e1.write(f"🌱 **{h_or_e('Crop','फसल')}:** {ctx['crop']}")
        e2.write(f"🦠 **{h_or_e('Diagnosis','निदान')}:** {ctx['diagnosis'] or 'Not assessed'}")
        e3.write(f"🌦️ **{h_or_e('Weather risk','मौसम जोखिम')}:** {ctx['weather_risk'] or 'Not assessed'}")
        e4.write(f"🧠 **{h_or_e('Decision','निर्णय')}:** {ctx['decision'] or 'Not assessed'}")
        if ctx.get("temperature_c") is not None:
            st.caption(
                f"Weather: {ctx['temperature_c']}°C • {ctx.get('humidity_pct','—')}% RH • "
                f"Rain {ctx.get('rain_mm','—')} mm • Wind {ctx.get('wind_kmh','—')} km/h"
            )
        if soil_available:
            st.caption(f"Soil: N {ctx['soil_N']} • P {ctx['soil_P']} • K {ctx['soil_K']} • pH {ctx['soil_pH']}")
        if ctx.get("soil_flags"):
            st.caption("Soil flags: " + ", ".join(ctx["soil_flags"]))
        st.caption("Mode: " + ctx.get("data_mode", "Assessment"))

    st.markdown("### " + h_or_e("Ask FARMIQ", "FARMIQ से पूछें"))

    # One-click farmer questions.
    q1,q2,q3,q4 = st.columns(4)
    quick_question = None
    with q1:
        if st.button("❓ Why is my crop at risk?", use_container_width=True):
            quick_question = "Why is my crop at risk? Explain the diagnosis, weather risk, and the main reason in simple terms."
    with q2:
        if st.button("🧭 What should I do now?", use_container_width=True):
            quick_question = "What should I do now? Give me the safest practical next steps based only on my FARMIQ assessment."
    with q3:
        if st.button("🧪 Explain my soil", use_container_width=True):
            quick_question = "Explain my soil and nutrient context and tell me what I should monitor. Do not invent fertilizer doses."
    with q4:
        if st.button("💊 Do I need treatment?", use_container_width=True):
            quick_question = "Do I need treatment? Explain whether FARMIQ recommends intervention, monitoring, or expert validation, and why."

    cclear, cstatus = st.columns([1,4])
    with cclear:
        if st.button("🗑️ " + h_or_e("Clear chat", "चैट साफ करें"), use_container_width=True):
            st.session_state.chat_messages = []
            st.rerun()
    with cstatus:
        st.caption(h_or_e(
            "The assistant remembers the recent conversation and receives the latest FARMIQ assessment on every question.",
            "सहायक हाल की बातचीत याद रखता है और हर प्रश्न पर नवीनतम FARMIQ आकलन प्राप्त करता है।"
        ))

    # Render history.
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input(
        h_or_e("Ask FARMIQ anything about this crop...", "इस फसल के बारे में FARMIQ से कुछ भी पूछें...")
    )
    if quick_question and not prompt:
        prompt = quick_question

    if prompt:
        # IMPORTANT: call Groq before appending the user message so it is not duplicated.
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner(h_or_e("FARMIQ is thinking...", "FARMIQ सोच रहा है...")):
                answer, error = groq_answer(prompt)
            if error:
                st.error(error)
                answer_to_store = error
            else:
                st.markdown(answer)
                answer_to_store = answer
        st.session_state.chat_messages.append({"role":"user","content":prompt})
        st.session_state.chat_messages.append({"role":"assistant","content":answer_to_store})
        st.rerun()

# ============================================================
# PROGRESSIVE WORKFLOW CONTROLLER
# ============================================================
if st.session_state.nav in WORKFLOW:
    wi = WORKFLOW.index(st.session_state.nav)
    st.markdown("---")
    bc1, bc2, bc3 = st.columns([1, 1.6, 1])
    with bc1:
        if wi > 0 and st.button("← " + t("back"), key="workflow_back", use_container_width=True):
            st.session_state.nav = WORKFLOW[wi-1]
            st.session_state.nav_index = wi-1
            st.rerun()
    with bc2:
        st.caption(h_or_e(
            f"Stage {wi+1} of {len(WORKFLOW)} • Sidebar remains fully jumpable",
            f"चरण {wi+1} / {len(WORKFLOW)} • साइडबार से किसी भी चरण पर जा सकते हैं"
        ))
    with bc3:
        if wi < len(WORKFLOW)-1 and st.button(t("continue") + " →", key="workflow_continue", type="primary", use_container_width=True):
            st.session_state.nav = WORKFLOW[wi+1]
            st.session_state.nav_index = wi+1
            st.rerun()

