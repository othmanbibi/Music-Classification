import streamlit as st
import requests
import pandas as pd
import os

# Page configuration
st.set_page_config(
    page_title="🎵 Music Genre Prediction",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS Styling
st.markdown("""
<style>
    .main-header {text-align:center;padding:2rem 0;background:linear-gradient(90deg,#667eea 0%,#764ba2 100%);color:white;border-radius:10px;margin-bottom:2rem;}
    .prediction-result {background-color:#f0f2f6;padding:1rem;border-radius:10px;border-left:5px solid #667eea;}
    .status-indicator {display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;}
    .status-connected {background-color:#28a745;}
    .status-disconnected {background-color:#dc3545;}
</style>
""", unsafe_allow_html=True)



st.markdown('<div class="main-header"><h1>🎵 Music Genre Prediction</h1><p>Upload your audio file and discover its genre!</p></div>', unsafe_allow_html=True)

# Backend URL
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:5000")

# Genres
GENRES = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']

# Backend connection test
def test_backend_connection():
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except requests.exceptions.RequestException:
        return False, None

# Prediction function
def make_prediction(file):
    try:
        file.seek(0)
        with st.spinner('🎵 Analyzing your music...'):
            response = requests.post(f"{BACKEND_URL}/predict", files={"file": file}, timeout=60)
        if response.status_code == 200:
            return True, response.json()
        else:
            try:
                return False, response.json().get('error', f"HTTP {response.status_code}")
            except:
                return False, f"HTTP {response.status_code}"
    except requests.exceptions.Timeout:
        return False, "Request timed out."
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to backend."
    except Exception as e:
        return False, str(e)

# Connection indicator
col1, col2 = st.columns([3,1])
with col1:
    st.subheader("📤 Upload Audio File")
with col2:
    is_connected, backend_info = test_backend_connection()
    if is_connected:
        st.markdown('<span class="status-indicator status-connected"></span>Backend Connected', unsafe_allow_html=True)
        if backend_info and backend_info.get('model_status') == 'loaded':
            st.success("✅ Model Ready")
        else:
            st.error("❌ Model Not Loaded")
    else:
        st.markdown('<span class="status-indicator status-disconnected"></span>Backend Disconnected', unsafe_allow_html=True)
        st.error("❌ Cannot connect to backend")

# File uploader
uploaded_file = st.file_uploader("Choose an audio file", type=["wav","mp3"])

if uploaded_file and is_connected:
    # File info and preview
    st.json({"Filename": uploaded_file.name, "Size (MB)": uploaded_file.size/(1024*1024)})
    st.audio(uploaded_file, format='audio/wav')

    if st.button("🎵 Predict Genre"):


        success, result = make_prediction(uploaded_file)
        if success:
            
            predicted_genre = result.get("genre","Unknown")
            all_probs = result.get("probabilities",[])
            confidence = 0
            if predicted_genre in GENRES and len(all_probs)==len(GENRES):
                confidence = all_probs[GENRES.index(predicted_genre)] * 100

            st.success(f"🎵 Predicted Genre: {predicted_genre.upper()}")
            st.info(f"🎯 Dominence: {confidence:.1f}%")

            # Probability table & chart
            if len(all_probs)==len(GENRES):
                df = pd.DataFrame({"Genre":GENRES,"Probability (%)":[p*100 for p in all_probs]}).sort_values("Probability (%)",ascending=False)
                col1, col2 = st.columns([1,2])
                with col1:
                    st.dataframe(df.style.highlight_max(axis=0, subset=['Probability (%)']), use_container_width=True)
                with col2:
                    st.bar_chart(df.set_index("Genre")["Probability (%)"])
                
                #Top 3
                top_3 = df.head(3)
                st.write("### 🏆 Top 3 Predictions")
                for rank, row in enumerate(top_3.itertuples()):
                    medal = "🥇" if rank==0 else "🥈" if rank==1 else "🥉"
                    st.write(f"{medal} **{row.Genre}**: {row._2:.2f}%")

            
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
