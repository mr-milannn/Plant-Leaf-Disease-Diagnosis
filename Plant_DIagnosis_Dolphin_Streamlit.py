import os
import re
import tempfile
from io import BytesIO

import numpy as np
import requests
from PIL import Image, ImageOps

import streamlit as st
import plotly.express as px

import tensorflow as tf
from tensorflow.keras.models import load_model
from fpdf import FPDF


# ---------------------------
# Page + Globals
# ---------------------------
st.set_page_config(page_title="Plant Disease Classifier", layout="wide")

# Read OpenRouter key safely
API_KEY = st.secrets["api"]["key"] if "api" in st.secrets and "key" in st.secrets["api"] else None
if not API_KEY:
    st.error("❌ OpenRouter API key missing. Add it in `.streamlit/secrets.toml` under [api].")
    st.stop()

MODEL_PATH = "plant_disease_model.h5"

# Class names
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___healthy',
    'Strawberry___Leaf_scorch', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
]


# ---------------------------
# Caching model load (faster)
# ---------------------------
@st.cache_resource(show_spinner=True)
def load_tf_model(path: str):
    return load_model(path)

model = load_tf_model(MODEL_PATH)


# ---------------------------
# Helpers
# ---------------------------
def normalize_disease_name(name: str) -> str:
    return name.replace("___", " ").replace("_", " ").strip()


def local_fallback(disease: str, language: str):
    if language == "Hindi":
        return {
            "overview": f"{disease} के लिए संक्षिप्त जानकारी।",
            "prevention": "• साफ उपकरण रखें\n• पत्तियों पर पानी देर तक न टिकने दें\n• संतुलित उर्वरक का उपयोग\n• संक्रमित पत्तियाँ हटाएँ",
            "treatment": "• उपयुक्त फफूँदनाशक/कीटनाशक (सिफारिश अनुसार)\n• प्रभावित भाग काटें और नष्ट करें\n• सिंचाई व पोषण प्रबंधन सुधारें"
        }
    else:
        return {
            "overview": f"Brief info for {disease}.",
            "prevention": "• Sanitize tools\n• Avoid prolonged leaf wetness\n• Balanced fertilization\n• Remove infected leaves",
            "treatment": "• Use appropriate fungicide/insecticide (as recommended)\n• Prune and destroy affected parts\n• Improve irrigation and nutrition management"
        }


def parse_sections(text: str, language: str):
    sections = {"overview": "", "prevention": "", "treatment": ""}
    t = text.strip()

    heads = {
        "overview": r"(overview|summary|अवलोकन|सारांश)",
        "prevention": r"(prevention|रोकथाम)",
        "treatment": r"(treatment|उपचार)"
    }

    pattern = re.compile(
        rf"(?P<ov>{heads['overview']}\s*:?\s*(?P<ovtxt>.*?))(?=(?:{heads['prevention']}|{heads['treatment']}|$))"
        rf"|(?P<pr>{heads['prevention']}\s*:?\s*(?P<prtxt>.*?))(?=(?:{heads['treatment']}|$))"
        rf"|(?P<tr>{heads['treatment']}\s*:?\s*(?P<trtxt>.*))",
        flags=re.IGNORECASE | re.DOTALL
    )

    matches = pattern.finditer(t)
    for m in matches:
        if m.group("ovtxt"):
            sections["overview"] = m.group("ovtxt").strip()
        if m.group("prtxt"):
            sections["prevention"] = m.group("prtxt").strip()
        if m.group("trtxt"):
            sections["treatment"] = m.group("trtxt").strip()

    if not any(sections.values()):
        sections["overview"] = t
        if language == "Hindi":
            sections["prevention"] = "• सामान्य रोकथाम: साफ-सफाई, संतुलित पोषण, समय पर सिंचाई"
            sections["treatment"] = "• उपचार: अनुशंसित औषधि/कृषि सलाह के अनुसार"
        else:
            sections["prevention"] = "• General prevention: sanitation, balanced nutrition, timely irrigation"
            sections["treatment"] = "• Treatment: as per recommended agri/plant protection guidelines"

    for k, v in sections.items():
        if not v.strip():
            sections[k] = "No information available." if k == "overview" else "• No information available."
    return sections


def get_ai_suggestion(disease_name: str, language: str = "English"):
    if "healthy" in disease_name.lower():
        return local_fallback("Healthy Plant", language)

    clean = normalize_disease_name(disease_name)
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://plant-leaf-disease-diagnosis.streamlit.app",
        "X-Title": "Plant Disease Classifier"
    }

    prompt = (
        f"You are a plant disease expert. For the disease '{clean}', "
        f"write three clear sections in {language}.\n\n"
        f"Overview:\nPrevention:\nTreatment:\n"
    )

    payload = {
        "model": "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
        "messages": [
            {"role": "system", "content": "Respond in three sections: Overview, Prevention, Treatment."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 400,
        "temperature": 0.2
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=25)
        st.write("🧪 AI status:", resp.status_code)

        if resp.status_code != 200:
            st.write("🧪 AI body:", resp.text[:200])
            return local_fallback(clean, language)

        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return local_fallback(clean, language)

        content = choices[0]["message"]["content"].strip()
        return parse_sections(content, language)

    except Exception as e:
        st.write("🧪 AI exception:", str(e))
        return local_fallback(clean, language)


def create_pdf(image: Image.Image, prediction: str, suggestion: dict, language="English") -> BytesIO:
    pdf = FPDF()
    pdf.add_page()

    font_path = "NotoSansDevanagari-Regular.ttf"
    if os.path.exists(font_path):
        pdf.add_font("Noto", "", font_path, uni=True)
        pdf.set_font("Noto", "", 16)
    else:
        pdf.set_font("Arial", "", 16)

    pdf.cell(0, 10, "Plant Disease Report", ln=True, align="C")
    pdf.ln(5)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            image.save(tmp_path)
        pdf.image(tmp_path, x=60, w=90)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    disp_pred = normalize_disease_name(prediction)
    pdf.set_font("Arial", "", 14)
    pdf.ln(8)
    pdf.cell(0, 10, f"Prediction: {disp_pred}", ln=True)
    pdf.set_font("Arial", "", 12)

    def as_text(label, body):
        return f"{label}:\n{body}"

    pdf.ln(2)
    pdf.multi_cell(0, 7, as_text("Overview", suggestion.get("overview", "-")))
    pdf.ln(2)
    pdf.multi_cell(0, 7, as_text("Prevention", suggestion.get("prevention", "-")))
    pdf.ln(2)
    pdf.multi_cell(0, 7, as_text("Treatment", suggestion.get("treatment", "-")))

    out = BytesIO(pdf.output(dest="S").encode("latin-1", "ignore"))
    out.seek(0)
    return out


# ---------------------------
# Streamlit UI
# ---------------------------
st.sidebar.header("User Guidance")
st.sidebar.markdown(
    "- Ensure leaf is clean and not blurred\n"
    "- Use good lighting\n"
    "- Leaf should be flat and fully visible\n"
    "- Avoid background clutter"
)

language = st.sidebar.selectbox("Select Language for AI Suggestions", ["English", "Hindi"])

st.title("🌿 Plant Disease Classifier")

uploaded_file = st.file_uploader("Upload Plant Leaf Image", type=["jpg", "jpeg", "png"])
capture = st.camera_input("Or capture from Webcam")

if "history" not in st.session_state:
    st.session_state.history = []
if "disease_counts" not in st.session_state:
    st.session_state.disease_counts = {}

image = None
if uploaded_file is not None:
    uploaded_file.seek(0)
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image).convert("RGB")
elif capture is not None:
    capture.seek(0)
    image = Image.open(capture)
    image = ImageOps.exif_transpose(image).convert("RGB")

if image is not None:
    st.image(image, caption="Uploaded / Captured Image", use_column_width=True)

    arr = np.array(image.resize((150, 150)), dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    pred_probs = model.predict(arr)
    top3_idx = pred_probs[0].argsort()[-3:][::-1]
    top3 = [(CLASS_NAMES[i], float(pred_probs[0][i])) for i in top3_idx]
    prediction = top3[0][0].strip()

    st.write(f"🧪 DEBUG | prediction → '{prediction}'")

    suggestion = get_ai_suggestion(prediction, language=language)

    st.subheader("🔍 Top 3 Predictions")
    for cls, prob in top3:
        st.write(f"{normalize_disease_name(cls)} : {prob*100:.2f}%")

    st.subheader("🤖 AI Suggestions")
    with st.expander("Overview", expanded=True):
        st.write(suggestion.get("overview", "-"))
    with st.expander("Prevention", expanded=True):
        st.write(suggestion.get("prevention", "-").replace("-", "•"))
    with st.expander("Treatment", expanded=True):
        st.write(suggestion.get("treatment", "-").replace("-", "•"))

    pdf_bytes = create_pdf(image, prediction, suggestion, language=language)
    st.download_button(
        "📄 Download PDF Report",
        data=pdf_bytes.getvalue(),
        file_name="Plant_Report.pdf",
        mime="application/pdf"
    )

    st.session_state.history.append({"prediction": prediction})
    st.session_state.disease_counts[prediction] = st.session_state.disease_counts.get(prediction, 0) + 1

if st.session_state.history:
    st.subheader("📊 Session Disease Statistics")
    counts = st.session_state.disease_counts
    df_counts = {"Disease": [normalize_disease_name(k) for k in counts.keys()],
                 "Count": [v for v in counts.values()]}
    fig = px.bar(df_counts, x="Disease", y="Count", color="Count", text="Count")
    st.plotly_chart(fig, use_container_width=True)
