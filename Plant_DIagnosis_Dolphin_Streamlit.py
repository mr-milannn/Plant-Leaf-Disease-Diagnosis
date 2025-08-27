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
try:
    model = load_model(MODEL_PATH)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

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
try:
    API_KEY = st.secrets["api"]["key"]
except KeyError:
    st.error("API key not found in secrets. Please add your API key to Streamlit secrets.")
    st.stop()

# --- Session state ---
if 'history' not in st.session_state:
    st.session_state.history = []

if 'disease_counts' not in st.session_state:
    st.session_state.disease_counts = {}

# --- AI Suggestion Function ---
def get_ai_suggestion(disease_name, language='English'):
    """Get AI suggestions for plant disease with proper error handling"""
    try:
        # Handle healthy plants
        if "healthy" in disease_name.lower():
            if language == "English":
                return {
                    "overview": "This plant appears healthy! No disease detected.",
                    "prevention": "• Water appropriately\n• Ensure proper sunlight\n• Fertilize as needed\n• Monitor regularly for early disease signs",
                    "treatment": "No treatment needed. Continue regular care."
                }
            else:
                return {
                    "overview": "यह पौधा स्वस्थ प्रतीत होता है! कोई बीमारी नहीं मिली।",
                    "prevention": "• उचित मात्रा में पानी दें\n• पर्याप्त धूप सुनिश्चित करें\n• आवश्यकतानुसार उर्वरक डालें\n• नियमित निरीक्षण करें",
                    "treatment": "उपचार की आवश्यकता नहीं है। नियमित देखभाल जारी रखें।"
                }

        # Clean disease name for better API results
        clean_disease = disease_name.replace("___", " ").replace("_", " ")
        
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        
        prompt = f"""Provide detailed information about the plant disease: {clean_disease}
        
        Please structure your response in exactly these three sections:
        
        Overview: Brief description of the disease and its characteristics
        
        Prevention: Specific prevention methods (use bullet points)
        
        Treatment: Recommended treatment options (use bullet points)
        
        Language: {language}"""
        
        data = {
            "model": "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
            "messages": [
                {
                    "role": "system", 
                    "content": f"You are a plant disease expert. Always respond in {language}. Structure your response with clear Overview, Prevention, and Treatment sections."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "max_tokens": 500,
            "temperature": 0.7
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(data),
            timeout=30
        )
        
        if response.status_code != 200:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return get_fallback_suggestion(disease_name, language)
        
        result = response.json()
        
        if "choices" in result and result["choices"]:
            content = result["choices"][0]["message"]["content"].strip()
            return parse_ai_response(content, language)
        else:
            error_msg = result.get('error', {}).get('message', 'Unknown error')
            st.warning(f"API returned no content: {error_msg}")
            return get_fallback_suggestion(disease_name, language)

    except requests.exceptions.Timeout:
        st.warning("API request timed out. Using fallback suggestions.")
        return get_fallback_suggestion(disease_name, language)
    except requests.exceptions.RequestException as e:
        st.error(f"Network error: {e}")
        return get_fallback_suggestion(disease_name, language)
    except Exception as e:
        st.error(f"Unexpected error getting AI suggestions: {e}")
        return get_fallback_suggestion(disease_name, language)

def parse_ai_response(content, language='English'):
    """Parse AI response into structured sections with clean formatting"""
    sections = {"overview": "", "prevention": "", "treatment": ""}
    
    def clean_text(text):
        """Clean and format text content"""
        if not text:
            return text
            
        # Remove excessive asterisks and formatting
        text = re.sub(r'\*{2,}', '', text)  # Remove ** formatting
        text = re.sub(r'\*+', '', text)     # Remove remaining asterisks
        
        # Remove multiple spaces and normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Clean up bullet points - remove existing ones first
        text = re.sub(r'^[\s]*[•\-\*]+[\s]*', '', text, flags=re.MULTILINE)
        
        # Split into sentences/points and clean each one
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        cleaned_lines = []
        
        for line in lines:
            # Remove leading/trailing punctuation and spaces
            line = line.strip(' •-*')
            if line:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def format_bullet_points(text):
        """Format text as clean bullet points"""
        if not text:
            return text
            
        # Split by newlines and filter out empty lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # If it's just one line, return as is (like overview)
        if len(lines) <= 1:
            return text
            
        # Format as bullet points
        formatted_lines = []
        for line in lines:
            if line and not line.startswith('•'):
                formatted_lines.append(f"• {line}")
            elif line:
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    # Try to extract sections using headers
    if language == "English":
        overview_pattern = r'(?:Overview|OVERVIEW)[:\-\s]*(.*?)(?=(?:Prevention|PREVENTION|Treatment|TREATMENT)|$)'
        prevention_pattern = r'(?:Prevention|PREVENTION)[:\-\s]*(.*?)(?=(?:Treatment|TREATMENT)|$)'
        treatment_pattern = r'(?:Treatment|TREATMENT)[:\-\s]*(.*?)'

def get_fallback_suggestion(disease_name, language='English'):
    """Provide fallback suggestions when AI is not available"""
    clean_name = disease_name.replace("___", " ").replace("_", " ")
    
    if language == "English":
        return {
            "overview": f"Disease detected: {clean_name}. This is a common plant disease that requires attention.",
            "prevention": "• Ensure proper plant spacing for air circulation\n• Water at soil level, avoid wetting leaves\n• Remove infected plant debris\n• Apply preventive fungicides if recommended\n• Monitor plants regularly",
            "treatment": "• Remove affected leaves/parts immediately\n• Apply appropriate fungicide or treatment\n• Improve growing conditions\n• Isolate infected plants if necessary\n• Consult local agricultural extension for specific advice"
        }
    else:
        return {
            "overview": f"बीमारी का पता चला: {clean_name}। यह एक आम पौधे की बीमारी है जिस पर ध्यान देने की आवश्यकता है।",
            "prevention": "• हवा की आवाजाही के लिए उचित पौधे की दूरी सुनिश्चित करें\n• मिट्टी के स्तर पर पानी दें, पत्तियों को भिगोने से बचें\n• संक्रमित पौधे के मलबे को हटा दें\n• यदि अनुशंसित हो तो निवारक कवकनाशी का प्रयोग करें\n• नियमित रूप से पौधों की निगरानी करें",
            "treatment": "• प्रभावित पत्तियों/भागों को तुरंत हटा दें\n• उपयुक्त कवकनाशी या उपचार लागू करें\n• बढ़ती परिस्थितियों में सुधार करें\n• यदि आवश्यक हो तो संक्रमित पौधों को अलग करें\n• विशिष्ट सलाह के लिए स्थानीय कृषि विस्तार से सलाह लें"
        }

# --- PDF Function ---
def create_pdf(image, prediction, suggestion, language='English'):
    """Create PDF report with proper error handling"""
    try:
        pdf = FPDF()
        pdf.add_page()
        
        # Try to use custom font, fall back to built-in if not available
        try:
            font_path = "NotoSans-Regular.ttf"
            if os.path.exists(font_path):
                pdf.add_font("Noto", "", font_path, uni=True)
                pdf.set_font("Noto", '', 16)
            else:
                pdf.set_font("Arial", 'B', 16)
        except:
            pdf.set_font("Arial", 'B', 16)
        
        pdf.cell(0, 10, "Plant Disease Report", ln=True, align="C")
        pdf.ln(10)

        # Save and add image
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                image.save(tmp.name)
                pdf.image(tmp.name, x=60, w=90)
                os.unlink(tmp.name)  # Clean up temp file
        except Exception as e:
            st.warning(f"Could not add image to PDF: {e}")

        pdf.ln(10)
        pdf.set_font("Arial", 'B', 14)
        
        # Handle text encoding issues
        try:
            clean_prediction = prediction.replace('___', ' - ')
            pdf.cell(0, 10, f"Prediction: {clean_prediction}", ln=True)
        except:
            pdf.cell(0, 10, "Prediction: [Unable to display]", ln=True)
        
        pdf.ln(5)
        pdf.set_font("Arial", '', 12)
        
        # Add sections with error handling
        for section_name, content in [("Overview", suggestion['overview']), 
                                     ("Prevention", suggestion['prevention']), 
                                     ("Treatment", suggestion['treatment'])]:
            try:
                pdf.multi_cell(0, 8, f"{section_name}:\n{content}")
                pdf.ln(2)
            except Exception as e:
                pdf.multi_cell(0, 8, f"{section_name}: [Content could not be displayed]")
                pdf.ln(2)

        return BytesIO(pdf.output(dest='S').encode('latin-1'))
    
    except Exception as e:
        st.error(f"Error creating PDF: {e}")
        return None

# --- Sidebar ---
st.sidebar.header("🔍 User Guidance")
st.sidebar.markdown("""
**For best results:**
- Ensure leaf is clean and not blurred  
- Use good lighting  
- Leaf should be flat and fully visible  
- Avoid background clutter
- Take close-up shots showing disease symptoms clearly
""")

st.sidebar.header("📋 Symptom Checklist")
spots = st.sidebar.checkbox("Spots on leaves")
yellowing = st.sidebar.checkbox("Yellowing")
wilting = st.sidebar.checkbox("Wilting")
holes = st.sidebar.checkbox("Holes or eaten parts")

language = st.sidebar.selectbox("Select Language for AI Suggestions", ["English", "Hindi"])

# Display selected symptoms
if any([spots, yellowing, wilting, holes]):
    st.sidebar.write("**Observed symptoms:**")
    if spots: st.sidebar.write("• Spots detected")
    if yellowing: st.sidebar.write("• Yellowing detected")
    if wilting: st.sidebar.write("• Wilting detected")
    if holes: st.sidebar.write("• Holes/damage detected")

# --- Main App ---
st.title("🌿 Plant Disease Classifier - Advanced")
st.markdown("Upload an image or use your camera to identify plant diseases and get AI-powered suggestions.")

# Create columns for better layout
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("📁 Upload Plant Leaf Image", type=["jpg", "jpeg", "png"])

with col2:
    capture = st.camera_input("📷 Or capture from Camera")

# Process image
image = None
image_source = None

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        image_source = "uploaded"
        st.success("Image uploaded successfully!")
    except Exception as e:
        st.error(f"Error loading uploaded image: {e}")

elif capture is not None:
    try:
        image = Image.open(capture).convert("RGB")
        image_source = "camera"
        st.success("Image captured successfully!")
    except Exception as e:
        st.error(f"Error loading captured image: {e}")

if image is not None:
    # Display image
    st.image(image, caption=f"Image from {image_source}", use_column_width=True)
    
    # Preprocess image for prediction
    try:
        img_array = np.array(image.resize((150, 150))) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        with st.spinner("Analyzing image..."):
            pred_probs = model.predict(img_array)
            
        top3_idx = pred_probs[0].argsort()[-3:][::-1]
        top3 = [(class_names[i], pred_probs[0][i]) for i in top3_idx]
        prediction = top3[0][0]
        confidence = top3[0][1]

        # Display results
        st.subheader("🔍 Prediction Results")
        
        # Main prediction with confidence
        st.metric("Primary Prediction", 
                 prediction.replace('___', ' - '), 
                 f"{confidence*100:.1f}% confidence")

        # Top-3 predictions in expandable section
        with st.expander("View all top predictions"):
            for i, (cls, prob) in enumerate(top3):
                st.write(f"{i+1}. {cls.replace('___', ' - ')}: {prob*100:.2f}%")

        # Severe disease alert
        severe_diseases = ["Late_blight", "Bacterial_spot", "Tomato_Yellow_Leaf_Curl_Virus", "Black_rot"]
        if any(disease in prediction for disease in severe_diseases):
            st.error("⚠️ **ALERT:** Severe disease detected! Immediate action recommended.")

        # Get AI suggestions
        st.subheader("🤖 AI-Powered Recommendations")
        
        with st.spinner("Getting AI recommendations..."):
            suggestion = get_ai_suggestion(prediction, language=language)

        # Display suggestions in cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            with st.container():
                st.markdown("### 📖 Overview")
                st.write(suggestion.get('overview', 'No information available.'))
        
        with col2:
            with st.container():
                st.markdown("### 🛡️ Prevention")
                st.write(suggestion.get('prevention', '• No prevention info available.'))
        
        with col3:
            with st.container():
                st.markdown("### 💊 Treatment")
                st.write(suggestion.get('treatment', '• No treatment info available.'))

        # PDF Download
        st.subheader("📄 Generate Report")
        if st.button("🔄 Generate PDF Report"):
            with st.spinner("Creating PDF report..."):
                pdf_bytes = create_pdf(image, prediction, suggestion, language=language)
                if pdf_bytes:
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_bytes.getvalue(),
                        file_name=f"Plant_Disease_Report_{prediction.split('___')[0]}.pdf",
                        mime="application/pdf"
                    )
                    st.success("PDF report generated successfully!")

        # Update session history
        st.session_state.history.append({
            "image": image, 
            "prediction": prediction,
            "confidence": confidence,
            "source": image_source
        })
        st.session_state.disease_counts[prediction] = st.session_state.disease_counts.get(prediction, 0) + 1

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.error("Please try uploading a different image or check your model file.")

# --- Session Statistics Dashboard ---
if st.session_state.history:
    st.subheader("📊 Session Analysis Dashboard")
    
    # Statistics
    total_predictions = len(st.session_state.history)
    unique_diseases = len(st.session_state.disease_counts)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Predictions", total_predictions)
    with col2:
        st.metric("Unique Diseases Found", unique_diseases)
    with col3:
        healthy_count = sum(1 for pred in st.session_state.disease_counts.keys() if 'healthy' in pred.lower())
        st.metric("Healthy Plants", healthy_count)
    
    # Disease distribution chart
    if st.session_state.disease_counts:
        counts_data = {
            "Disease": [k.replace('___', ' - ') for k in st.session_state.disease_counts.keys()],
            "Count": list(st.session_state.disease_counts.values())
        }
        
        fig = px.bar(
            x=counts_data["Disease"], 
            y=counts_data["Count"],
            title="Disease Detection Frequency",
            labels={"x": "Disease Type", "y": "Detection Count"}
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent predictions
    with st.expander("Recent Predictions History"):
        for i, item in enumerate(reversed(st.session_state.history[-5:])):  # Show last 5
            st.write(f"{i+1}. {item['prediction'].replace('___', ' - ')} "
                    f"({item['confidence']*100:.1f}% confidence) - "
                    f"Source: {item['source']}")

# --- Footer ---
st.markdown("---")
st.markdown("**Note:** This tool provides AI-assisted suggestions for educational purposes. "
           "Always consult with agricultural experts or extension services for professional advice.")

# Clear session button
if st.button("🗑️ Clear Session History"):
    st.session_state.history = []
    st.session_state.disease_counts = {}
    st.success("Session history cleared!")
    st.experimental_rerun()
else:
    overview_pattern = r'(?:अवलोकन|सारांश)[:\-\s]*(.*?)(?=(?:रोकथाम|उपचार)|$)'
    prevention_pattern = r'(?:रोकथाम)[:\-\s]*(.*?)(?=(?:उपचार)|$)'
    treatment_pattern = r'(?:उपचार)[:\-\s]*(.*?)'

def get_fallback_suggestion(disease_name, language='English'):
    """Provide fallback suggestions when AI is not available"""
    clean_name = disease_name.replace("___", " ").replace("_", " ")
    
    if language == "English":
        return {
            "overview": f"Disease detected: {clean_name}. This is a common plant disease that requires attention.",
            "prevention": "• Ensure proper plant spacing for air circulation\n• Water at soil level, avoid wetting leaves\n• Remove infected plant debris\n• Apply preventive fungicides if recommended\n• Monitor plants regularly",
            "treatment": "• Remove affected leaves/parts immediately\n• Apply appropriate fungicide or treatment\n• Improve growing conditions\n• Isolate infected plants if necessary\n• Consult local agricultural extension for specific advice"
        }
    else:
        return {
            "overview": f"बीमारी का पता चला: {clean_name}। यह एक आम पौधे की बीमारी है जिस पर ध्यान देने की आवश्यकता है।",
            "prevention": "• हवा की आवाजाही के लिए उचित पौधे की दूरी सुनिश्चित करें\n• मिट्टी के स्तर पर पानी दें, पत्तियों को भिगोने से बचें\n• संक्रमित पौधे के मलबे को हटा दें\n• यदि अनुशंसित हो तो निवारक कवकनाशी का प्रयोग करें\n• नियमित रूप से पौधों की निगरानी करें",
            "treatment": "• प्रभावित पत्तियों/भागों को तुरंत हटा दें\n• उपयुक्त कवकनाशी या उपचार लागू करें\n• बढ़ती परिस्थितियों में सुधार करें\n• यदि आवश्यक हो तो संक्रमित पौधों को अलग करें\n• विशिष्ट सलाह के लिए स्थानीय कृषि विस्तार से सलाह लें"
        }

# --- PDF Function ---
def create_pdf(image, prediction, suggestion, language='English'):
    """Create PDF report with proper error handling"""
    try:
        pdf = FPDF()
        pdf.add_page()
        
        # Try to use custom font, fall back to built-in if not available
        try:
            font_path = "NotoSans-Regular.ttf"
            if os.path.exists(font_path):
                pdf.add_font("Noto", "", font_path, uni=True)
                pdf.set_font("Noto", '', 16)
            else:
                pdf.set_font("Arial", 'B', 16)
        except:
            pdf.set_font("Arial", 'B', 16)
        
        pdf.cell(0, 10, "Plant Disease Report", ln=True, align="C")
        pdf.ln(10)

        # Save and add image
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                image.save(tmp.name)
                pdf.image(tmp.name, x=60, w=90)
                os.unlink(tmp.name)  # Clean up temp file
        except Exception as e:
            st.warning(f"Could not add image to PDF: {e}")

        pdf.ln(10)
        pdf.set_font("Arial", 'B', 14)
        
        # Handle text encoding issues
        try:
            clean_prediction = prediction.replace('___', ' - ')
            pdf.cell(0, 10, f"Prediction: {clean_prediction}", ln=True)
        except:
            pdf.cell(0, 10, "Prediction: [Unable to display]", ln=True)
        
        pdf.ln(5)
        pdf.set_font("Arial", '', 12)
        
        # Add sections with error handling
        for section_name, content in [("Overview", suggestion['overview']), 
                                     ("Prevention", suggestion['prevention']), 
                                     ("Treatment", suggestion['treatment'])]:
            try:
                pdf.multi_cell(0, 8, f"{section_name}:\n{content}")
                pdf.ln(2)
            except Exception as e:
                pdf.multi_cell(0, 8, f"{section_name}: [Content could not be displayed]")
                pdf.ln(2)

        return BytesIO(pdf.output(dest='S').encode('latin-1'))
    
    except Exception as e:
        st.error(f"Error creating PDF: {e}")
        return None

# --- Sidebar ---
st.sidebar.header("🔍 User Guidance")
st.sidebar.markdown("""
**For best results:**
- Ensure leaf is clean and not blurred  
- Use good lighting  
- Leaf should be flat and fully visible  
- Avoid background clutter
- Take close-up shots showing disease symptoms clearly
""")

st.sidebar.header("📋 Symptom Checklist")
spots = st.sidebar.checkbox("Spots on leaves")
yellowing = st.sidebar.checkbox("Yellowing")
wilting = st.sidebar.checkbox("Wilting")
holes = st.sidebar.checkbox("Holes or eaten parts")

language = st.sidebar.selectbox("Select Language for AI Suggestions", ["English", "Hindi"])

# Display selected symptoms
if any([spots, yellowing, wilting, holes]):
    st.sidebar.write("**Observed symptoms:**")
    if spots: st.sidebar.write("• Spots detected")
    if yellowing: st.sidebar.write("• Yellowing detected")
    if wilting: st.sidebar.write("• Wilting detected")
    if holes: st.sidebar.write("• Holes/damage detected")

# --- Main App ---
st.title("🌿 Plant Disease Classifier - Advanced")
st.markdown("Upload an image or use your camera to identify plant diseases and get AI-powered suggestions.")

# Create columns for better layout
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("📁 Upload Plant Leaf Image", type=["jpg", "jpeg", "png"])

with col2:
    capture = st.camera_input("📷 Or capture from Camera")

# Process image
image = None
image_source = None

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        image_source = "uploaded"
        st.success("Image uploaded successfully!")
    except Exception as e:
        st.error(f"Error loading uploaded image: {e}")

elif capture is not None:
    try:
        image = Image.open(capture).convert("RGB")
        image_source = "camera"
        st.success("Image captured successfully!")
    except Exception as e:
        st.error(f"Error loading captured image: {e}")

if image is not None:
    # Display image
    st.image(image, caption=f"Image from {image_source}", use_column_width=True)
    
    # Preprocess image for prediction
    try:
        img_array = np.array(image.resize((150, 150))) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        with st.spinner("Analyzing image..."):
            pred_probs = model.predict(img_array)
            
        top3_idx = pred_probs[0].argsort()[-3:][::-1]
        top3 = [(class_names[i], pred_probs[0][i]) for i in top3_idx]
        prediction = top3[0][0]
        confidence = top3[0][1]

        # Display results
        st.subheader("🔍 Prediction Results")
        
        # Main prediction with confidence
        st.metric("Primary Prediction", 
                 prediction.replace('___', ' - '), 
                 f"{confidence*100:.1f}% confidence")

        # Top-3 predictions in expandable section
        with st.expander("View all top predictions"):
            for i, (cls, prob) in enumerate(top3):
                st.write(f"{i+1}. {cls.replace('___', ' - ')}: {prob*100:.2f}%")

        # Severe disease alert
        severe_diseases = ["Late_blight", "Bacterial_spot", "Tomato_Yellow_Leaf_Curl_Virus", "Black_rot"]
        if any(disease in prediction for disease in severe_diseases):
            st.error("⚠️ **ALERT:** Severe disease detected! Immediate action recommended.")

        # Get AI suggestions
        st.subheader("🤖 AI-Powered Recommendations")
        
        with st.spinner("Getting AI recommendations..."):
            suggestion = get_ai_suggestion(prediction, language=language)

        # Display suggestions in cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            with st.container():
                st.markdown("### 📖 Overview")
                st.write(suggestion.get('overview', 'No information available.'))
        
        with col2:
            with st.container():
                st.markdown("### 🛡️ Prevention")
                prevention_text = suggestion.get('prevention', '• No prevention info available.')
                # Ensure bullet points are formatted correctly
                if not prevention_text.strip().startswith('•'):
                    prevention_text = '• ' + prevention_text.replace('\n', '\n• ')
                st.write(prevention_text)
        
        with col3:
            with st.container():
                st.markdown("### 💊 Treatment")
                treatment_text = suggestion.get('treatment', '• No treatment info available.')
                # Ensure bullet points are formatted correctly
                if not treatment_text.strip().startswith('•'):
                    treatment_text = '• ' + treatment_text.replace('\n', '\n• ')
                st.write(treatment_text)

        # PDF Download
        st.subheader("📄 Generate Report")
        if st.button("🔄 Generate PDF Report"):
            with st.spinner("Creating PDF report..."):
                pdf_bytes = create_pdf(image, prediction, suggestion, language=language)
                if pdf_bytes:
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_bytes.getvalue(),
                        file_name=f"Plant_Disease_Report_{prediction.split('___')[0]}.pdf",
                        mime="application/pdf"
                    )
                    st.success("PDF report generated successfully!")

        # Update session history
        st.session_state.history.append({
            "image": image, 
            "prediction": prediction,
            "confidence": confidence,
            "source": image_source
        })
        st.session_state.disease_counts[prediction] = st.session_state.disease_counts.get(prediction, 0) + 1

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.error("Please try uploading a different image or check your model file.")

# --- Session Statistics Dashboard ---
if st.session_state.history:
    st.subheader("📊 Session Analysis Dashboard")
    
    # Statistics
    total_predictions = len(st.session_state.history)
    unique_diseases = len(st.session_state.disease_counts)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Predictions", total_predictions)
    with col2:
        st.metric("Unique Diseases Found", unique_diseases)
    with col3:
        healthy_count = sum(1 for pred in st.session_state.disease_counts.keys() if 'healthy' in pred.lower())
        st.metric("Healthy Plants", healthy_count)
    
    # Disease distribution chart
    if st.session_state.disease_counts:
        counts_data = {
            "Disease": [k.replace('___', ' - ') for k in st.session_state.disease_counts.keys()],
            "Count": list(st.session_state.disease_counts.values())
        }
        
        fig = px.bar(
            x=counts_data["Disease"], 
            y=counts_data["Count"],
            title="Disease Detection Frequency",
            labels={"x": "Disease Type", "y": "Detection Count"}
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent predictions
    with st.expander("Recent Predictions History"):
        for i, item in enumerate(reversed(st.session_state.history[-5:])):  # Show last 5
            st.write(f"{i+1}. {item['prediction'].replace('___', ' - ')} "
                    f"({item['confidence']*100:.1f}% confidence) - "
                    f"Source: {item['source']}")

# --- Footer ---
st.markdown("---")
st.markdown("**Note:** This tool provides AI-assisted suggestions for educational purposes. "
           "Always consult with agricultural experts or extension services for professional advice.")

# Clear session button
if st.button("🗑️ Clear Session History"):
    st.session_state.history = []
    st.session_state.disease_counts = {}
    st.success("Session history cleared!")
    st.experimental_rerun()
    
    overview_match = re.search(overview_pattern, content, re.DOTALL | re.IGNORECASE)
    prevention_match = re.search(prevention_pattern, content, re.DOTALL | re.IGNORECASE)
    treatment_match = re.search(treatment_pattern, content, re.DOTALL | re.IGNORECASE)
    
    if overview_match:
        sections["overview"] = clean_text(overview_match.group(1).strip())
    if prevention_match:
        sections["prevention"] = clean_text(prevention_match.group(1).strip())
    if treatment_match:
        sections["treatment"] = clean_text(treatment_match.group(1).strip())
    
    # If no sections found, try line-by-line parsing
    if not any(sections.values()):
        lines = content.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in ["overview", "अवलोकन", "सारांश"]):
                current_section = "overview"
                # Extract content after the header if it's on the same line
                content_part = re.sub(r'^.*?(?:overview|अवलोकन|सारांश)[:\-\s]*', '', line, flags=re.IGNORECASE)
                if content_part.strip():
                    sections[current_section] = clean_text(content_part.strip())
            elif any(keyword in line_lower for keyword in ["prevention", "रोकथाम"]):
                current_section = "prevention"
                content_part = re.sub(r'^.*?(?:prevention|रोकथाम)[:\-\s]*', '', line, flags=re.IGNORECASE)
                if content_part.strip():
                    sections[current_section] = clean_text(content_part.strip())
            elif any(keyword in line_lower for keyword in ["treatment", "उपचार"]):
                current_section = "treatment"
                content_part = re.sub(r'^.*?(?:treatment|उपचार)[:\-\s]*', '', line, flags=re.IGNORECASE)
                if content_part.strip():
                    sections[current_section] = clean_text(content_part.strip())
            elif current_section and line:
                clean_line = clean_text(line)
                if sections[current_section]:
                    sections[current_section] += "\n" + clean_line
                else:
                    sections[current_section] = clean_line
    
    # Format prevention and treatment as bullet points, keep overview as paragraph
    sections["prevention"] = format_bullet_points(sections["prevention"])
    sections["treatment"] = format_bullet_points(sections["treatment"])
    
    # Ensure all sections have content
    for key in sections:
        if not sections[key].strip():
            if language == "English":
                sections[key] = "Information not available." if key == "overview" else "• No specific recommendations available."
            else:
                sections[key] = "जानकारी उपलब्ध नहीं है।" if key == "overview" else "• कोई विशिष्ट सिफारिश उपलब्ध नहीं है।"
    
    return sections

def get_fallback_suggestion(disease_name, language='English'):
    """Provide fallback suggestions when AI is not available"""
    clean_name = disease_name.replace("___", " ").replace("_", " ")
    
    if language == "English":
        return {
            "overview": f"Disease detected: {clean_name}. This is a common plant disease that requires attention.",
            "prevention": "• Ensure proper plant spacing for air circulation\n• Water at soil level, avoid wetting leaves\n• Remove infected plant debris\n• Apply preventive fungicides if recommended\n• Monitor plants regularly",
            "treatment": "• Remove affected leaves/parts immediately\n• Apply appropriate fungicide or treatment\n• Improve growing conditions\n• Isolate infected plants if necessary\n• Consult local agricultural extension for specific advice"
        }
    else:
        return {
            "overview": f"बीमारी का पता चला: {clean_name}। यह एक आम पौधे की बीमारी है जिस पर ध्यान देने की आवश्यकता है।",
            "prevention": "• हवा की आवाजाही के लिए उचित पौधे की दूरी सुनिश्चित करें\n• मिट्टी के स्तर पर पानी दें, पत्तियों को भिगोने से बचें\n• संक्रमित पौधे के मलबे को हटा दें\n• यदि अनुशंसित हो तो निवारक कवकनाशी का प्रयोग करें\n• नियमित रूप से पौधों की निगरानी करें",
            "treatment": "• प्रभावित पत्तियों/भागों को तुरंत हटा दें\n• उपयुक्त कवकनाशी या उपचार लागू करें\n• बढ़ती परिस्थितियों में सुधार करें\n• यदि आवश्यक हो तो संक्रमित पौधों को अलग करें\n• विशिष्ट सलाह के लिए स्थानीय कृषि विस्तार से सलाह लें"
        }

# --- PDF Function ---
def create_pdf(image, prediction, suggestion, language='English'):
    """Create PDF report with proper error handling"""
    try:
        pdf = FPDF()
        pdf.add_page()
        
        # Try to use custom font, fall back to built-in if not available
        try:
            font_path = "NotoSans-Regular.ttf"
            if os.path.exists(font_path):
                pdf.add_font("Noto", "", font_path, uni=True)
                pdf.set_font("Noto", '', 16)
            else:
                pdf.set_font("Arial", 'B', 16)
        except:
            pdf.set_font("Arial", 'B', 16)
        
        pdf.cell(0, 10, "Plant Disease Report", ln=True, align="C")
        pdf.ln(10)

        # Save and add image
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                image.save(tmp.name)
                pdf.image(tmp.name, x=60, w=90)
                os.unlink(tmp.name)  # Clean up temp file
        except Exception as e:
            st.warning(f"Could not add image to PDF: {e}")

        pdf.ln(10)
        pdf.set_font("Arial", 'B', 14)
        
        # Handle text encoding issues
        try:
            clean_prediction = prediction.replace('___', ' - ')
            pdf.cell(0, 10, f"Prediction: {clean_prediction}", ln=True)
        except:
            pdf.cell(0, 10, "Prediction: [Unable to display]", ln=True)
        
        pdf.ln(5)
        pdf.set_font("Arial", '', 12)
        
        # Add sections with error handling
        for section_name, content in [("Overview", suggestion['overview']), 
                                     ("Prevention", suggestion['prevention']), 
                                     ("Treatment", suggestion['treatment'])]:
            try:
                pdf.multi_cell(0, 8, f"{section_name}:\n{content}")
                pdf.ln(2)
            except Exception as e:
                pdf.multi_cell(0, 8, f"{section_name}: [Content could not be displayed]")
                pdf.ln(2)

        return BytesIO(pdf.output(dest='S').encode('latin-1'))
    
    except Exception as e:
        st.error(f"Error creating PDF: {e}")
        return None

# --- Sidebar ---
st.sidebar.header("🔍 User Guidance")
st.sidebar.markdown("""
**For best results:**
- Ensure leaf is clean and not blurred  
- Use good lighting  
- Leaf should be flat and fully visible  
- Avoid background clutter
- Take close-up shots showing disease symptoms clearly
""")

st.sidebar.header("📋 Symptom Checklist")
spots = st.sidebar.checkbox("Spots on leaves")
yellowing = st.sidebar.checkbox("Yellowing")
wilting = st.sidebar.checkbox("Wilting")
holes = st.sidebar.checkbox("Holes or eaten parts")

language = st.sidebar.selectbox("Select Language for AI Suggestions", ["English", "Hindi"])

# Display selected symptoms
if any([spots, yellowing, wilting, holes]):
    st.sidebar.write("**Observed symptoms:**")
    if spots: st.sidebar.write("• Spots detected")
    if yellowing: st.sidebar.write("• Yellowing detected")
    if wilting: st.sidebar.write("• Wilting detected")
    if holes: st.sidebar.write("• Holes/damage detected")

# --- Main App ---
st.title("🌿 Plant Disease Classifier - Advanced")
st.markdown("Upload an image or use your camera to identify plant diseases and get AI-powered suggestions.")

# Create columns for better layout
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("📁 Upload Plant Leaf Image", type=["jpg", "jpeg", "png"])

with col2:
    capture = st.camera_input("📷 Or capture from Camera")

# Process image
image = None
image_source = None

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        image_source = "uploaded"
        st.success("Image uploaded successfully!")
    except Exception as e:
        st.error(f"Error loading uploaded image: {e}")

elif capture is not None:
    try:
        image = Image.open(capture).convert("RGB")
        image_source = "camera"
        st.success("Image captured successfully!")
    except Exception as e:
        st.error(f"Error loading captured image: {e}")

if image is not None:
    # Display image
    st.image(image, caption=f"Image from {image_source}", use_column_width=True)
    
    # Preprocess image for prediction
    try:
        img_array = np.array(image.resize((150, 150))) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        with st.spinner("Analyzing image..."):
            pred_probs = model.predict(img_array)
            
        top3_idx = pred_probs[0].argsort()[-3:][::-1]
        top3 = [(class_names[i], pred_probs[0][i]) for i in top3_idx]
        prediction = top3[0][0]
        confidence = top3[0][1]

        # Display results
        st.subheader("🔍 Prediction Results")
        
        # Main prediction with confidence
        st.metric("Primary Prediction", 
                 prediction.replace('___', ' - '), 
                 f"{confidence*100:.1f}% confidence")

        # Top-3 predictions in expandable section
        with st.expander("View all top predictions"):
            for i, (cls, prob) in enumerate(top3):
                st.write(f"{i+1}. {cls.replace('___', ' - ')}: {prob*100:.2f}%")

        # Severe disease alert
        severe_diseases = ["Late_blight", "Bacterial_spot", "Tomato_Yellow_Leaf_Curl_Virus", "Black_rot"]
        if any(disease in prediction for disease in severe_diseases):
            st.error("⚠️ **ALERT:** Severe disease detected! Immediate action recommended.")

        # Get AI suggestions
        st.subheader("🤖 AI-Powered Recommendations")
        
        with st.spinner("Getting AI recommendations..."):
            suggestion = get_ai_suggestion(prediction, language=language)

        # Display suggestions in cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            with st.container():
                st.markdown("### 📖 Overview")
                st.write(suggestion.get('overview', 'No information available.'))
        
        with col2:
            with st.container():
                st.markdown("### 🛡️ Prevention")
                prevention_text = suggestion.get('prevention', '• No prevention info available.')
                # Ensure bullet points are formatted correctly
                if not prevention_text.strip().startswith('•'):
                    prevention_text = '• ' + prevention_text.replace('\n', '\n• ')
                st.write(prevention_text)
        
        with col3:
            with st.container():
                st.markdown("### 💊 Treatment")
                treatment_text = suggestion.get('treatment', '• No treatment info available.')
                # Ensure bullet points are formatted correctly
                if not treatment_text.strip().startswith('•'):
                    treatment_text = '• ' + treatment_text.replace('\n', '\n• ')
                st.write(treatment_text)

        # PDF Download
        st.subheader("📄 Generate Report")
        if st.button("🔄 Generate PDF Report"):
            with st.spinner("Creating PDF report..."):
                pdf_bytes = create_pdf(image, prediction, suggestion, language=language)
                if pdf_bytes:
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_bytes.getvalue(),
                        file_name=f"Plant_Disease_Report_{prediction.split('___')[0]}.pdf",
                        mime="application/pdf"
                    )
                    st.success("PDF report generated successfully!")

        # Update session history
        st.session_state.history.append({
            "image": image, 
            "prediction": prediction,
            "confidence": confidence,
            "source": image_source
        })
        st.session_state.disease_counts[prediction] = st.session_state.disease_counts.get(prediction, 0) + 1

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.error("Please try uploading a different image or check your model file.")

# --- Session Statistics Dashboard ---
if st.session_state.history:
    st.subheader("📊 Session Analysis Dashboard")
    
    # Statistics
    total_predictions = len(st.session_state.history)
    unique_diseases = len(st.session_state.disease_counts)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Predictions", total_predictions)
    with col2:
        st.metric("Unique Diseases Found", unique_diseases)
    with col3:
        healthy_count = sum(1 for pred in st.session_state.disease_counts.keys() if 'healthy' in pred.lower())
        st.metric("Healthy Plants", healthy_count)
    
    # Disease distribution chart
    if st.session_state.disease_counts:
        counts_data = {
            "Disease": [k.replace('___', ' - ') for k in st.session_state.disease_counts.keys()],
            "Count": list(st.session_state.disease_counts.values())
        }
        
        fig = px.bar(
            x=counts_data["Disease"], 
            y=counts_data["Count"],
            title="Disease Detection Frequency",
            labels={"x": "Disease Type", "y": "Detection Count"}
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent predictions
    with st.expander("Recent Predictions History"):
        for i, item in enumerate(reversed(st.session_state.history[-5:])):  # Show last 5
            st.write(f"{i+1}. {item['prediction'].replace('___', ' - ')} "
                    f"({item['confidence']*100:.1f}% confidence) - "
                    f"Source: {item['source']}")

# --- Footer ---
st.markdown("---")
st.markdown("**Note:** This tool provides AI-assisted suggestions for educational purposes. "
           "Always consult with agricultural experts or extension services for professional advice.")

# Clear session button
if st.button("🗑️ Clear Session History"):
    st.session_state.history = []
    st.session_state.disease_counts = {}
    st.success("Session history cleared!")
    st.experimental_rerun()



