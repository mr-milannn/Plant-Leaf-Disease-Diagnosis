import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import requests
import json
import re
from io import BytesIO
import tempfile
from fpdf import FPDF
import plotly.express as px
import os

# --- Load Model ---
MODEL_PATH = "plant_disease_model.h5"
model = load_model(MODEL_PATH)


# --- Class Names ---
class_names = [ 'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
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

# --- OpenRouter API Key ---
API_KEY = st.secrets["api"]["key"]

# --- Session state ---
if 'history' not in st.session_state:
    st.session_state.history = []

if 'disease_counts' not in st.session_state:
    st.session_state.disease_counts = {}

# --- Extract Sections (robust for English + Hindi) ---
def extract_sections(content, language='English'):
    sections = {"overview":"", "prevention":"", "treatment":""}

    headers = {
        "English": {"overview": ["Overview", "Summary"], "prevention": ["Prevention"], "treatment": ["Treatment"]},
        "Hindi": {"overview": ["अवलोकन","सारांश"], "prevention": ["रोकथाम"], "treatment": ["उपचार"]}
    }

    for key, labels in headers[language].items():
        # Look ahead for other headers to stop
        other_labels = sum([v for k,v in headers[language].items() if k!=key], [])
        pattern = re.compile(rf'({"|".join(labels)})\s*[:\-]?\s*(.*?)(?=(?:{"|".join(other_labels)}|$))', re.DOTALL)
        match = pattern.search(content)
        if match:
            sections[key] = match.group(2).strip()
        else:
            sections[key] = "No information available." if key=="overview" else "• No information available."
    return sections

# --- AI Suggestion Function ---
def get_ai_suggestion(disease_name, language='English'):
    try:
        if "healthy" in disease_name.lower():
            if language=="English":
                return {"overview":"This plant appears healthy!",
                        "prevention":"• Water appropriately\n• Ensure proper sunlight\n• Fertilize as needed\n• Monitor regularly",
                        "treatment":"No treatment needed."}
            else:
                return {"overview":"यह पौधा स्वस्थ प्रतीत होता है!",
                        "prevention":"• उचित मात्रा में पानी दें\n• पर्याप्त धूप सुनिश्चित करें\n• आवश्यकतानुसार उर्वरक डालें\n• नियमित निरीक्षण करें",
                        "treatment":"उपचार की आवश्यकता नहीं है।"}

        headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
        prompt = f"Provide concise Overview, Prevention, and Treatment for plant disease: {disease_name} in {language}."
        data = {
            "model": "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
            "messages": [
                {"role": "system", "content": "You are a plant disease expert. Respond clearly in three sections: Overview, Prevention, Treatment."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 300  # free Venice model limit
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(data)
        )
        result = response.json()
        print("Raw API Response:", json.dumps(result, indent=2))

        if "choices" in result and result["choices"]:
            content = result["choices"][0]["message"]["content"].strip()
        else:
            content = f"No AI suggestions available. Reason: {result.get('error', {}).get('message', '')}"
        print("Extracted content:", content)

        # --- Robust section parsing ---
        sections = {"overview":"", "prevention":"", "treatment":""}
        lines = content.splitlines()
        current_section = None
        for line in lines:
            line_lower = line.lower()
            if any(h in line_lower for h in ["overview","अवलोकन","सारांश"]):
                current_section = "overview"
            elif any(h in line_lower for h in ["prevention","रोकथाम"]):
                current_section = "prevention"
            elif any(h in line_lower for h in ["treatment","उपचार"]):
                current_section = "treatment"
            elif current_section:
                if sections[current_section]:
                    sections[current_section] += "\n" + line.strip()
                else:
                    sections[current_section] = line.strip()

        for key in sections:
            if not sections[key]:
                sections[key] = "No information available." if key=="overview" else "• No information available."

        return sections

    except Exception as e:
        return {"overview":f"Error: {e}",
                "prevention":"• No prevention info available.",
                "treatment":"• No treatment info available."}

# --- PDF Function ---
def create_pdf(image, prediction, suggestion, language='English'):
    pdf = FPDF()
    pdf.add_page()

    # Use proper TTF font
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    font_path = "NotoSans-Regular.ttf"   # since it is in repo root
    pdfmetrics.registerFont(TTFont("NotoSans", font_path))

    if not os.path.exists(font_path):
        raise RuntimeError(f"Font file not found: {font_path}")
    pdf.add_font("Noto", "", font_path, uni=True)

    pdf.set_font("Noto", '', 16)
    pdf.cell(0, 10, "Plant Disease Report", ln=True, align="C")
    pdf.ln(10)

    # Save image temporarily
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp.name)
        pdf.image(tmp.name, x=60, w=90)

    pdf.ln(10)
    pdf.set_font("Noto", '', 14)
    pdf.cell(0, 10, f"Prediction: {prediction.replace('___',' - ')}", ln=True)
    pdf.ln(5)

    pdf.set_font("Noto", '', 12)
    pdf.multi_cell(0, 8, f"Overview:\n{suggestion['overview']}")
    pdf.ln(2)
    pdf.multi_cell(0, 8, f"Prevention:\n{suggestion['prevention']}")
    pdf.ln(2)
    pdf.multi_cell(0, 8, f"Treatment:\n{suggestion['treatment']}")

    pdf_bytes = BytesIO(pdf.output(dest='S').encode('utf-8'))
    pdf_bytes.seek(0)
    return pdf_bytes

# --- Sidebar ---
st.sidebar.header("User Guidance")
st.sidebar.markdown("""
- Ensure leaf is clean and not blurred  
- Use good lighting  
- Leaf should be flat and fully visible  
- Avoid background clutter
""")
st.sidebar.header("Symptom Checklist")
spots = st.sidebar.checkbox("Spots on leaves")
yellowing = st.sidebar.checkbox("Yellowing")
wilting = st.sidebar.checkbox("Wilting")
holes = st.sidebar.checkbox("Holes or eaten parts")

language = st.sidebar.selectbox("Select Language for AI Suggestions", ["English", "Hindi"])

# --- Main App ---
st.title("🌿 Plant Disease Classifier - Feature Rich")

uploaded_file = st.file_uploader("Upload Plant Leaf Image", type=["jpg","jpeg","png"])
capture = st.camera_input("Or capture from Webcam")

image = None
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
elif capture:
    image = Image.open(capture).convert("RGB")

if image:
    st.image(image, caption="Uploaded / Captured Image", use_column_width=True)
    img_array = np.array(image.resize((150,150)))/255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred_probs = model.predict(img_array)
    top3_idx = pred_probs[0].argsort()[-3:][::-1]
    top3 = [(class_names[i], pred_probs[0][i]) for i in top3_idx]
    prediction = top3[0][0]

    # AI Suggestions
    suggestion = get_ai_suggestion(prediction, language=language)

    # Severe disease alert
    severe_diseases = ["Late_blight", "Bacterial_spot", "Tomato_Yellow_Leaf_Curl_Virus"]
    if any(disease in prediction for disease in severe_diseases):
        st.warning("⚠️ Severe disease detected! Take immediate action.")

    # Top-3 Predictions
    st.subheader("🔍 Top 3 Predictions")
    for cls, prob in top3:
        st.write(f"{cls.replace('___',' - ')} : {prob*100:.2f}%")

    # AI Suggestions Cards
    st.subheader("🤖 AI Suggestions")
    with st.expander("Overview"):
        st.write(suggestion.get('overview', '-'))
    with st.expander("Prevention"):
        st.write(suggestion.get('prevention', '-').replace("-","•"))
    with st.expander("Treatment"):
        st.write(suggestion.get('treatment', '-').replace("-","•"))

    # PDF Download
    pdf_bytes = create_pdf(image, prediction, suggestion, language=language)
    st.download_button("📄 Download PDF Report", data=pdf_bytes.getvalue(),
                       file_name="Plant_Report.pdf", mime="application/pdf")

    # Update session history
    st.session_state.history.append({"image": image, "prediction": prediction})
    st.session_state.disease_counts[prediction] = st.session_state.disease_counts.get(prediction,0)+1

# --- Dashboard ---
if st.session_state.history:
    st.subheader("📊 Session Disease Statistics")
    counts = st.session_state.disease_counts
    df_counts = {"Disease":[k.replace('___',' - ') for k in counts.keys()],
                 "Count": [v for v in counts.values()]}
    fig = px.bar(df_counts, x="Disease", y="Count", color="Count", text="Count")
    st.plotly_chart(fig, use_container_width=True)


