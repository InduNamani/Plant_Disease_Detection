import streamlit as st

# CRITICAL: set_page_config MUST be the first Streamlit command
st.set_page_config(
    page_title="🌿 Plant Disease Detector",
    page_icon="🌿",
    layout="wide"
)

import joblib
import numpy as np
from PIL import Image
import os
from datetime import datetime
from supabase import create_client, Client
from dotenv import load_dotenv
from utils import extract_features, preprocess_image_for_display
import tempfile

# Load environment variables
load_dotenv()

# Initialize Supabase
@st.cache_resource
def init_supabase():
    """Connect to Supabase database"""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    
    if not url or not key:
        st.warning("⚠️ Supabase not configured. Running without database.")
        return None
    
    return create_client(url, key)

supabase = init_supabase()


# Load trained model
@st.cache_resource
def load_model():
    """Load the trained ML model"""
    try:
        model = joblib.load('models/plant_model.pkl')
        class_names = joblib.load('models/class_names.pkl')
        return model, class_names
    except:
        st.error("❌ Model not found! Please run train_model.py first.")
        st.stop()

model, class_names = load_model()


def predict_disease(image_file):
    """
    Predict plant disease from uploaded image
    
    Returns: prediction, confidence
    """
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        tmp_file.write(image_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Extract features
        features = extract_features(tmp_path)
        features = features.reshape(1, -1)  # Reshape for single prediction
        
        # Predict
        prediction = model.predict(features)[0]
        
        # Get prediction probabilities (confidence)
        probabilities = model.predict_proba(features)[0]
        confidence = max(probabilities) * 100
        
        return prediction, confidence, probabilities
        
    finally:
        # Clean up temp file
        os.unlink(tmp_path)


def save_to_database(disease, confidence, user_name):
    """Save prediction to Supabase"""
    
    if supabase is None:
        return False
    
    try:
        data = {
            "disease_name": disease,
            "confidence": float(confidence),
            "user_name": user_name,
            "timestamp": datetime.now().isoformat()
        }
        
        supabase.table("predictions").insert(data).execute()
        return True
        
    except Exception as e:
        st.error(f"Database error: {e}")
        return False


def get_disease_info(disease_name):
    """Get information about the disease"""
    
    disease_info = {
        "Tomato___Bacterial_spot": {
            "description": "Bacterial disease causing dark spots on leaves and fruit.",
            "symptoms": "Small, dark, water-soaked spots on leaves",
            "treatment": "Use copper-based fungicides, remove infected plants"
        },
        "Tomato___Early_blight": {
            "description": "Fungal disease with dark spots with concentric rings.",
            "symptoms": "Brown spots with target-like rings on older leaves",
            "treatment": "Apply fungicide, improve air circulation, crop rotation"
        },
        "Tomato___Late_blight": {
            "description": "Serious fungal disease that can destroy entire crops.",
            "symptoms": "Large, irregular brown/black spots on leaves and stems",
            "treatment": "Remove infected plants immediately, use fungicide preventively"
        },
        "Tomato___Leaf_Mold": {
            "description": "Fungal disease thriving in humid conditions.",
            "symptoms": "Pale green/yellow spots on upper leaf, olive-green mold below",
            "treatment": "Reduce humidity, improve ventilation, apply fungicide"
        },
        "Tomato___healthy": {
            "description": "Plant is healthy! No disease detected.",
            "symptoms": "Green, vibrant leaves with no spots or discoloration",
            "treatment": "Continue regular care and monitoring"
        }
    }
    
    return disease_info.get(disease_name, {
        "description": "Disease information not available",
        "symptoms": "N/A",
        "treatment": "Consult agricultural expert"
    })


# ==================== STREAMLIT UI ====================

# Title
st.title("🌿 Plant Disease Detection System")
st.markdown("Upload a tomato leaf image to detect diseases using Machine Learning")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.info("""
    This app uses **Machine Learning** to detect plant diseases from leaf images.
    
    **How it works:**
    1. Upload leaf image
    2. AI extracts features (color, texture, shape)
    3. Random Forest algorithm predicts disease
    4. Results saved to database
    
    **Supported:** Tomato plants (5 classes)
    """)
    
    st.header("📊 Model Info")
    st.metric("Algorithm", "Random Forest")
    st.metric("Classes", len(class_names))
    
    # Show recent predictions
    if supabase:
        st.header("📜 Recent Predictions")
        try:
            response = supabase.table("predictions").select("*").order("timestamp", desc=True).limit(5).execute()
            if response.data:
                for pred in response.data:
                    st.text(f"• {pred['disease_name'][:20]}...")
        except:
            st.text("No predictions yet")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload Image")
    
    # User name input
    user_name = st.text_input("Your Name (optional)", placeholder="e.g., John Farmer")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a leaf image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of a tomato leaf"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = preprocess_image_for_display(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Predict button
        if st.button("🔍 Detect Disease", type="primary", use_container_width=True):
            with st.spinner("Analyzing image..."):
                
                # Predict
                prediction, confidence, probabilities = predict_disease(uploaded_file)
                
                # Display results in col2
                with col2:
                    st.header("📊 Results")
                    
                    # Disease name
                    disease_display = prediction.replace("___", " - ").replace("_", " ")
                    
                    if "healthy" in prediction.lower():
                        st.success(f"### ✅ {disease_display}")
                    else:
                        st.error(f"### ⚠️ {disease_display}")
                    
                    # Confidence meter
                    st.metric("Confidence", f"{confidence:.1f}%")
                    st.progress(confidence / 100)
                    
                    # Disease information
                    info = get_disease_info(prediction)
                    
                    st.subheader("📖 Disease Information")
                    st.write(f"**Description:** {info['description']}")
                    st.write(f"**Symptoms:** {info['symptoms']}")
                    st.write(f"**Treatment:** {info['treatment']}")
                    
                    # Save to database
                    if user_name:
                        if save_to_database(prediction, confidence, user_name):
                            st.success("✅ Prediction saved to database!")
                    
                    # Show all probabilities
                    with st.expander("🔬 View All Predictions"):
                        for idx, class_name in enumerate(class_names):
                            prob = probabilities[idx] * 100
                            st.write(f"{class_name}: {prob:.2f}%")

with col2:
    if uploaded_file is None:
        st.header("👈 Start Here")
        st.info("Upload a tomato leaf image to begin disease detection")
        
        # Example images section
        st.subheader("📷 Example Symptoms")
        st.markdown("""
        **Healthy:** Bright green, no spots  
        **Bacterial Spot:** Small dark spots  
        **Early Blight:** Target-like rings  
        **Late Blight:** Large irregular patches  
        **Leaf Mold:** Yellow spots with mold underneath
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with Python, Scikit-learn, Streamlit & Supabase | 
    <a href='https://github.com' target='_blank'>GitHub</a></p>
</div>
""", unsafe_allow_html=True)
