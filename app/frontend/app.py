import streamlit as st
import requests
import pandas as pd


st.title("Music Genre Prediction")

uploaded_file = st.file_uploader("Upload an audio file (wav or mp3)", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    
    if st.button("Predict Genre"):
        try:
            response = requests.post(
                "http://localhost:5000/predict",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Convert to percentages
                probs = [round(p * 100, 2) for p in data["probabilities"]]
                
                st.success(f"🎵 Predicted Genre: **{data['genre']}**")
                
                # Show probabilities as table + bar chart
                df = pd.DataFrame({
                    "Genre": ['blues','classical','country','disco','hiphop',
                            'jazz','metal','pop','reggae','rock'],
                    "Probability (%)": probs
                })
                
                st.write("### 🎚️ Prediction Confidence")
                st.dataframe(df.set_index("Genre"))
                st.bar_chart(df.set_index("Genre"))
            else:
                st.error(f"Error {response.status_code}: Could not get prediction")
        
        except Exception as e:
            st.error(f"Request failed: {e}")
