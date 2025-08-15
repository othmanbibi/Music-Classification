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


#Define the classes (music genres)
classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

#Load and Process the audio file
def load_and_process_file(file_path, target_shape=(150, 150)):
        data = []
        audio_data, sample_rate = librosa.load(file_path, sr = None) #Keep original sampling rate
        
        #Performing Audio Preprocessing
        
        #Define the duration of Chunk and overlap
        chunk_duration = 4
        overlap_duration = 2
        
        #Convert duration to sample
        chunk_sample = chunk_duration * sample_rate
        overlap_sample = overlap_duration * sample_rate
        
        #Calculate the number of chunks
        num_chunks = int(np.ceil((len(audio_data) - chunk_sample) / (chunk_sample - overlap_sample))) +1
        
        #iterate over each chunk
        for i in range(num_chunks):
            #calculate start and end indices of the chunk
            start = i*(chunk_sample - overlap_sample)
            end = start + chunk_sample
            #extract chunk audio
            chunk = audio_data[start:end]
            
            #Compute melspectogram
            mel_spectogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
            #Resize matrix based on target_shape
            mel_spectogram = resize(np.expand_dims(mel_spectogram, axis=-1), target_shape)
            
            data.append(mel_spectogram)
            
        return  np.array(data)


def model_predict_genre(test_audio_data):
    Y_pred = model.predict(test_audio_data)
    predicted_genres = np.argmax(Y_pred, axis=1)
    unique_elements, count = np.unique(predicted_genres, return_counts=True)
    max_count = np.max(count)
    dominent_genre = unique_elements[count == max_count]
    #Calculate the average probablities of each class in the audio file
    genre_probabilites = np.mean(Y_pred, axis=0) #Average by rows
    return genre_probabilites, dominent_genre




@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        # Convert mp3 to wav if needed
        if file.filename.endswith(".mp3"):
            sound = AudioSegment.from_mp3(file)
            sound.export(tmp.name, format="wav")
        else:
            file.save(tmp.name)

        test_audio_data = load_and_process_file(tmp.name)
        proba, prediction = model_predict_genre(test_audio_data)

    return jsonify({"genre": prediction.tolist(), "probabilities": proba.tolist()})



    
if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)

