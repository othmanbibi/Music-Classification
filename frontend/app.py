import streamlit as st
import requests
import pandas as pd
import os
import time

# Page configuration
st.set_page_config(
    page_title="🎵 Music Genre Prediction",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .prediction-result {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
    }
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 5px;
    }
    .status-connected {
        background-color: #28a745;
    }
    .status-disconnected {
        background-color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Title with styling
st.markdown('<div class="main-header"><h1>🎵 Music Genre Prediction</h1><p>Upload your audio file and discover its genre!</p></div>', unsafe_allow_html=True)

# Use environment variable for backend URL
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:5000")

def test_backend_connection():
    """Test if backend is accessible"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except requests.exceptions.RequestException:
        return False, None

def make_prediction(file):
    """Make prediction API call"""
    try:
        # Reset file pointer
        file.seek(0)
        
        with st.spinner('🎵 Analyzing your music... This may take a moment!'):
            response = requests.post(
                f"{BACKEND_URL}/predict", 
                files={"file": file},
                timeout=60  # Increased timeout for ML processing
            )
            
        if response.status_code == 200:
            return True, response.json()
        else:
            error_msg = "Unknown error"
            try:
                error_data = response.json()
                error_msg = error_data.get('error', f"HTTP {response.status_code}")
            except:
                error_msg = f"HTTP {response.status_code}"
            return False, error_msg
            
    except requests.exceptions.Timeout:
        return False, "Request timed out. The file might be too large or the server is busy."
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to the backend service. Please try again later."
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"

# Backend status check
col1, col2 = st.columns([3, 1])
with col1:
    st.subheader("📤 Upload Audio File")
with col2:
    is_connected, backend_info = test_backend_connection()
    if is_connected:
        st.markdown('<span class="status-indicator status-connected"></span>Backend Connected', unsafe_allow_html=True)
        if backend_info and 'model_status' in backend_info:
            if backend_info['model_status'] == 'loaded':
                st.success("✅ Model Ready")
            else:
                st.error("❌ Model Not Loaded")
    else:
        st.markdown('<span class="status-indicator status-disconnected"></span>Backend Disconnected', unsafe_allow_html=True)
        st.error("❌ Cannot connect to backend")
        st.info(f"Trying to connect to: {BACKEND_URL}")

# File uploader
uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=["wav", "mp3"],
    help="Supported formats: WAV, MP3. Max file size: 200MB"
)

if uploaded_file is not None:
    # Display file info
    file_details = {
        "Filename": uploaded_file.name,
        "File size": f"{uploaded_file.size / (1024*1024):.2f} MB",
        "File type": uploaded_file.type
    }
    
    st.write("📁 **File Details:**")
    st.json(file_details)
    
    # Audio player
    st.write("🎧 **Preview:**")
    st.audio(uploaded_file, format='audio/wav')
    
    # Prediction button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        predict_button = st.button(
            "🎵 Predict Genre", 
            type="primary",
            disabled=not is_connected,
            use_container_width=True
        )
    
    if predict_button:
        if not is_connected:
            st.error("❌ Cannot make prediction: Backend service is not available")
        else:
            success, result = make_prediction(uploaded_file)
            
            if success:
                st.markdown('<div class="prediction-result">', unsafe_allow_html=True)
                
                # Main prediction result
                confidence = result.get('confidence', 0) * 100
                st.success(f"🎵 **Predicted Genre: {result['genre'].upper()}**")
                st.info(f"🎯 **Confidence: {confidence:.1f}%**")
                
                # Probability distribution
                st.write("### 📊 Genre Probability Distribution")
                
                # Use the probabilities dict if available, otherwise fall back to the list
                if 'probabilities' in result and isinstance(result['probabilities'], dict):
                    genres = list(result['probabilities'].keys())
                    probs = [result['probabilities'][genre] * 100 for genre in genres]
                else:
                    # Fallback to original format
                    genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
                             'jazz', 'metal', 'pop', 'reggae', 'rock']
                    probs = [round(p * 100, 2) for p in result.get("all_probabilities", result.get("probabilities", []))]
                
                # Create DataFrame for visualization
                df = pd.DataFrame({
                    "Genre": genres,
                    "Probability (%)": probs
                }).sort_values("Probability (%)", ascending=False)
                
                # Display results in columns
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.write("**Detailed Results:**")
                    st.dataframe(
                        df.style.highlight_max(axis=0, subset=['Probability (%)']),
                        use_container_width=True
                    )
                
                with col2:
                    st.write("**Probability Chart:**")
                    st.bar_chart(df.set_index("Genre")["Probability (%)"])
                
                # Top 3 predictions
                st.write("### 🏆 Top 3 Predictions")
                top_3 = df.head(3)
                for idx, row in top_3.iterrows():
                    if idx == 0:
                        st.write(f"🥇 **{row['Genre']}**: {row['Probability (%)']:.2f}%")
                    elif idx == 1:
                        st.write(f"🥈 **{row['Genre']}**: {row['Probability (%)']:.2f}%")
                    elif idx == 2:
                        st.write(f"🥉 **{row['Genre']}**: {row['Probability (%)']:.2f}%")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
            else:
                st.error(f"❌ Prediction failed: {result}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🎵 Music Genre Classification using Deep Learning</p>
    <p>Supported formats: WAV, MP3 • Processing time: ~10-30 seconds</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This application uses a deep learning model trained on audio features 
    to classify music into different genres.
    
    **Supported Genres:**
    - Blues
    - Classical  
    - Country
    - Disco
    - Hip Hop
    - Jazz
    - Metal
    - Pop
    - Reggae
    - Rock
    
    **How it works:**
    1. Upload your audio file
    2. The model analyzes mel-spectrograms
    3. Predictions are made using overlapping audio chunks
    4. Final genre is determined by majority vote
    """)
    
    st.header("🔧 Technical Info")
    st.write(f"""
    **Backend URL:** {BACKEND_URL}
    **Status:** {"✅ Connected" if is_connected else "❌ Disconnected"}
    """)
    
    if st.button("🔄 Refresh Connection"):
        st.rerun()