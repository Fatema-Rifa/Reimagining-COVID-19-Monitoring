import streamlit as st
import pandas as pd
import numpy as np
import joblib
import google.generativeai as genai
from datetime import datetime

# Page config
st.set_page_config(page_title="AI Health Status Predictor", layout="centered")

# Title with better visual hierarchy
st.title("COVID-19 Health Status Predictor")
st.markdown("""
<style>
.big-font {
    font-size:18px !important;
}
</style>
""", unsafe_allow_html=True)

# Load the trained model and scaler
@st.cache_resource
def load_model():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

GEMINI_API_KEY = "YOUR_API_KEY_HERE"

# Initialize Gemini
def init_gemini():
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        return genai.GenerativeModel('gemini-pro')
    except Exception as e:
        st.error(f"Failed to initialize Gemini AI: {str(e)}")
        return None

gemini_model = init_gemini()

# Feature input using sliders with better organization
st.header("Patient Vital Signs")
col1, col2, col3 = st.columns(3)
with col1:
    oxygen = st.slider("Oxygen Level (SpO2 %)", min_value=80.0, max_value=100.0, value=98.0, step=0.1,
                      help="Normal range is typically 95-100%")
with col2:
    pulse = st.slider("Pulse Rate (bpm)", min_value=40.0, max_value=150.0, value=70.0, step=1.0,
                     help="Normal resting heart rate is 60-100 bpm for adults")
with col3:
    temp = st.slider("Body Temperature (°F)", min_value=90.0, max_value=105.0, value=98.6, step=0.1,
                    help="Normal body temperature is around 98.6°F")

# Additional health information
with st.expander("Additional Health Information (Optional)"):
    symptoms = st.text_area("Describe any symptoms (optional)", 
                           placeholder="e.g., dizziness, shortness of breath, fatigue...")
    medical_history = st.text_area("Relevant medical history (optional)", 
                                 placeholder="e.g., asthma, heart conditions...")

# Create a DataFrame for the input
input_data = pd.DataFrame([[oxygen, pulse, temp]], columns=['Oxygen', 'PulseRate', 'Temperature'])

# Scale the input
input_scaled = scaler.transform(input_data)

# Enhanced prediction function with Gemini integration
def generate_health_advice(prediction, vitals, symptoms_text="", history_text=""):
    if not gemini_model:
        return "AI explanation not available at this time."
    
    health_status = "potentially abnormal" if prediction == 1 else "likely normal"
    
    prompt = f"""
    You are a medical assistant providing clear, helpful information to patients about their health status.
    Based on these vital signs:
    - Oxygen: {vitals['Oxygen']}%
    - Pulse: {vitals['PulseRate']} bpm
    - Temperature: {vitals['Temperature']}°F
    
    The prediction model indicates a {health_status} health status.
    
    Additional symptoms: {symptoms_text if symptoms_text else "None provided"}
    Medical history: {history_text if history_text else "None provided"}
    
    Please provide:
    1. A simple explanation of what these vitals mean
    2. Context about the prediction (what might cause abnormal values)
    3. Recommended next steps (when to seek medical attention)
    4. General wellness tips based on these readings
    
    Keep the response under 300 words, professional but empathetic, and emphasize when immediate medical care is needed.
    """
    
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error generating health advice: {str(e)}")
        return "Could not generate health advice at this time."

# Prediction button with enhanced output
if st.button("🔍 Analyze Health Status", type="primary"):
    with st.spinner("Analyzing health status..."):
        # Basic prediction
        prediction = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1] if hasattr(model, "predict_proba") else None
        
        # Display prediction with better visuals
        st.subheader("Health Assessment")
        if prediction == 1:
            st.error("🚨 **Prediction: Potentially Abnormal**")
            st.markdown('<p class="big-font">These readings suggest possible health concerns.</p>', unsafe_allow_html=True)
        else:
            st.success("✅ **Prediction: Likely Normal**")
            st.markdown('<p class="big-font">These readings appear within normal ranges.</p>', unsafe_allow_html=True)
        
        if prob is not None:
            st.metric("Prediction Confidence", f"{prob*100:.2f}%")
        
        # Generate and display Gemini AI insights
        st.subheader("AI Health Advisor Insights")
        vitals_dict = {'Oxygen': oxygen, 'PulseRate': pulse, 'Temperature': temp}
        advice = generate_health_advice(prediction, vitals_dict, symptoms, medical_history)
        st.markdown(f"<div style='background-color:#f0f2f6; padding:15px; border-radius:10px;'>{advice}</div>", 
                   unsafe_allow_html=True)
        
        # Add disclaimer
        st.caption("ℹ️ This tool is not a substitute for professional medical advice. Always consult with a healthcare provider for medical concerns.")

# Replace the footer section with this version:
st.markdown("---")
current_date = datetime.now().strftime("%Y-%m-%d")
footer_html = f"""
<style>
.footer {{
    font-size: 12px;
    color: #666;
    text-align: center;
}}
</style>
<div class="footer">
    Last updated: {current_date} | Medical AI Assistant v1.1
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)