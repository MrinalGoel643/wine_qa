import streamlit as st
import requests

API_URL = "https://wine-quality-ml.azurewebsites.net/api/predict"

st.set_page_config(page_title="Wine Quality Predictor", layout="centered", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    [data-testid="stAppViewContainer"] {
        background: #0a0a0a;
    }
    [data-testid="stHeader"] {
        background-color: transparent;
    }
    [data-testid="stToolbar"] {
        display: none;
    }
    [data-testid="stDecoration"] {
        display: none;
    }
    [data-testid="stStatusWidget"] {
        display: none;
    }
    .main > div:first-child:empty {
        display: none;
    }
    .element-container:has(.main-title) ~ .element-container:first-of-type {
        display: none;
    }
    section[data-testid="stSidebar"] {
        display: none;
    }
    .block-container {
        padding-top: 2rem;
    }
    div[data-testid="stVerticalBlock"] > div:empty {
        display: none !important;
    }
    .main-title {
        text-align: center;
        padding: 3rem 0 2rem 0;
    }
    .main-title h1 {
        color: #ffffff;
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 0.8rem;
        letter-spacing: -1px;
    }
    .main-title p {
        color: #888;
        font-size: 1.2rem;
        margin-top: 0;
        font-weight: 400;
    }
    .input-section {
        background: #161616;
        padding: 2.5rem;
        border-radius: 16px;
        border: 1px solid #222;
        margin: 2rem 0;
    }
    .section-header {
        color: #ffffff;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid #2a2a2a;
    }
    .stButton>button {
        background: linear-gradient(135deg, #c41e3a 0%, #8b1528 100%);
        color: white;
        border-radius: 12px;
        padding: 1rem 2.5rem;
        font-size: 1.1rem;
        border: none;
        font-weight: 700;
        width: 100%;
        margin-top: 2rem;
        transition: all 0.3s;
        letter-spacing: 0.5px;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(196,30,58,0.4);
    }
    [data-testid="stNumberInput"] label {
        color: #ccc !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
    }
    [data-testid="stNumberInput"] input {
        background-color: #0f0f0f !important;
        border: 1px solid #2a2a2a !important;
        color: #fff !important;
        font-size: 1.05rem !important;
        border-radius: 8px !important;
    }
    [data-testid="stSlider"] label {
        color: #ccc !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
    }
    [data-testid="stSlider"] div[data-baseweb="slider"] {
        margin-top: 0.5rem;
    }
    .stRadio > label {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
    }
    .stRadio [role="radiogroup"] {
        gap: 2rem;
        padding: 0.8rem 0;
    }
    .stRadio label[data-baseweb="radio"] {
        background-color: transparent !important;
    }
    .stRadio label[data-baseweb="radio"] span {
        color: #aaa !important;
        font-size: 1.05rem !important;
    }
    .result-card {
        background: linear-gradient(135deg, #c41e3a 0%, #8b1528 100%);
        padding: 3rem;
        border-radius: 16px;
        text-align: center;
        margin: 2.5rem 0;
        box-shadow: 0 10px 40px rgba(196,30,58,0.3);
        border: 1px solid rgba(255,255,255,0.1);
    }
    .score-display {
        font-size: 5rem;
        font-weight: 900;
        color: white;
        margin: 1rem 0;
        text-shadow: 0 4px 8px rgba(0,0,0,0.3);
        letter-spacing: -2px;
    }
    .score-label {
        color: rgba(255,255,255,0.85);
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 3px;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    .category-badge {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        padding: 0.7rem 2rem;
        border-radius: 25px;
        color: white;
        font-weight: 700;
        margin-top: 1.2rem;
        font-size: 1.15rem;
        letter-spacing: 1px;
    }
    .group-label {
        color: #999;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 1.2rem;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="main-title">
        <h1>🍷 Wine Quality Predictor</h1>
        <p>analyze chemical properties to predict wine quality</p>
    </div>
""", unsafe_allow_html=True)

st.markdown('<div class="input-section">', unsafe_allow_html=True)

st.markdown('<div class="section-header">Wine Type</div>', unsafe_allow_html=True)
wine_type = st.radio("", ["red", "white"], horizontal=True, label_visibility="collapsed")

st.markdown('<div class="section-header" style="margin-top: 2.5rem;">Chemical Properties</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<p class="group-label">Acidity</p>', unsafe_allow_html=True)
    fixed_acidity = st.slider("Fixed Acidity", min_value=0.0, max_value=20.0, value=7.4, step=0.1)
    volatile_acidity = st.slider("Volatile Acidity", min_value=0.0, max_value=2.0, value=0.7, step=0.01)
    citric_acid = st.slider("Citric Acid", min_value=0.0, max_value=2.0, value=0.0, step=0.01)
    ph = st.slider("pH Level", min_value=2.5, max_value=4.5, value=3.51, step=0.01)

with col2:
    st.markdown('<p class="group-label">Sulfur & Salts</p>', unsafe_allow_html=True)
    free_sulfur = st.slider("Free SO₂", min_value=0.0, max_value=100.0, value=11.0, step=1.0)
    total_sulfur = st.slider("Total SO₂", min_value=0.0, max_value=300.0, value=34.0, step=1.0)
    sulphates = st.slider("Sulphates", min_value=0.0, max_value=2.0, value=0.56, step=0.01)
    chlorides = st.slider("Chlorides", min_value=0.0, max_value=1.0, value=0.076, step=0.001, format="%.3f")

with col3:
    st.markdown('<p class="group-label">Composition</p>', unsafe_allow_html=True)
    residual_sugar = st.slider("Residual Sugar", min_value=0.0, max_value=50.0, value=1.9, step=0.1)
    density = st.slider("Density", min_value=0.98, max_value=1.01, value=0.9978, step=0.0001, format="%.4f")
    alcohol = st.slider("Alcohol %", min_value=8.0, max_value=15.0, value=9.4, step=0.1)

st.markdown('</div>', unsafe_allow_html=True)

if st.button("Analyze Wine Quality"):
    payload = {
        "fixed_acidity": fixed_acidity,
        "volatile_acidity": volatile_acidity,
        "citric_acid": citric_acid,
        "residual_sugar": residual_sugar,
        "chlorides": chlorides,
        "free_sulfur_dioxide": free_sulfur,
        "total_sulfur_dioxide": total_sulfur,
        "density": density,
        "pH": ph,
        "sulphates": sulphates,
        "alcohol": alcohol,
        "wine_type": wine_type
    }
    
    try:
        with st.spinner("analyzing chemical composition..."):
            response = requests.post(API_URL, json=payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                st.session_state.quality_score = result.get("predicted_quality", 0)
                st.session_state.category = result.get("quality_category", "unknown")
            else:
                st.error(f"api error: {response.status_code}")
                
    except requests.exceptions.Timeout:
        st.error("request timed out")
    except Exception as e:
        st.error(f"something went wrong: {str(e)}")

if "quality_score" in st.session_state:
    st.markdown(f"""
        <div class="result-card">
            <div class="score-label">Quality Score</div>
            <div class="score-display">{st.session_state.quality_score:.1f}</div>
            <div class="category-badge">{st.session_state.category.upper()}</div>
        </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.quality_score >= 6.5:
        st.success("exceptional quality characteristics")
    elif st.session_state.quality_score >= 5.5:
        st.info("solid quality profile")
    else:
        st.warning("room for improvement")