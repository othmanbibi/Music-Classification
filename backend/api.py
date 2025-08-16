print("Starting Flask API...")

from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import librosa
import tempfile
from pydub import AudioSegment
from tensorflow.image import resize
import os

app = Flask(__name__)

MODEL_PATH = r'C:\Projects\Music_ML_Pr\music-ml-app\models\Trained_model_Music_Genre_Class.h5'
model = load_model(MODEL_PATH)

# Define the classes (music genres)
classes = ['blues', 'classical', 'country', 'disco', 'hiphop',
           'jazz', 'metal', 'pop', 'reggae', 'rock']


# Load and process the audio file
def load_and_process_file(file_path, target_shape=(150, 150)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None, mono=True)

    chunk_duration = 4
    overlap_duration = 2
    chunk_sample = chunk_duration * sample_rate
    overlap_sample = overlap_duration * sample_rate

    num_chunks = int(np.ceil((len(audio_data) - chunk_sample) /
                                    (chunk_sample - overlap_sample)))

    for i in range(num_chunks):
        start = i * (chunk_sample - overlap_sample)
        end = start + chunk_sample
        chunk = audio_data[start:end]

        if len(chunk) == 0:
            continue

        mel_spectogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
        mel_spectogram = resize(np.expand_dims(mel_spectogram, axis=-1), target_shape)

        data.append(mel_spectogram)

    return np.array(data)


def model_predict_genre(test_audio_data):
    Y_pred = model.predict(test_audio_data)
    predicted_genres = np.argmax(Y_pred, axis=1)
    unique_elements, count = np.unique(predicted_genres, return_counts=True)
    max_count = np.max(count)
    dominant_genre = unique_elements[count == max_count]

    genre_probabilities = np.mean(Y_pred, axis=0)

    return genre_probabilities, dominant_genre



@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    # Create a temp file path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp_path = tmp.name  # save path first

    try:
        # Convert mp3 to wav if needed
        if file.filename.endswith(".mp3"):
            sound = AudioSegment.from_mp3(file)
            sound.export(tmp_path, format="wav")
        else:
            file.save(tmp_path)

        # Now process the closed file safely
        test_audio_data = load_and_process_file(tmp_path)
        proba, prediction = model_predict_genre(test_audio_data)
        predicted_label = classes[int(prediction[0])]   # map index to genre name
        return jsonify({"genre": predicted_label, "probabilities": proba.tolist()})



    finally:
        # Clean up safely
        if os.path.exists(tmp_path):
            os.remove(tmp_path)



if __name__ == "__main__":
    # Get port from environment variable (Render sets this)
    port = int(os.environ.get('PORT', 5000))
    
    print(f"Starting Flask app on port {port}")
    
    # Run with host 0.0.0.0 to accept external connections
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,  # Disable debug in production
        threaded=True  # Enable threading for better performance
    )