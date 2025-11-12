from flask import Flask, request, render_template
import librosa
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model & scaler
qda = joblib.load("models/qda_model.pkl")
scaler = joblib.load("models/scaler.pkl")


def extract_features(file_path):
    y, sr = librosa.load(file_path, mono=True)
    
    # Feature 1: length (matching CSV)
    length = len(y)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zero_cross = librosa.feature.zero_crossing_rate(y)
    harmony = librosa.effects.harmonic(y)
    percussive = librosa.effects.percussive(y)
    tempo = librosa.beat.tempo(y=y, sr=sr)[0]

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    features = [
        length,  # ✅ Added missing feature!

        np.mean(chroma), np.var(chroma),
        np.mean(rms), np.var(rms),
        np.mean(spectral_centroid), np.var(spectral_centroid),
        np.mean(spectral_bandwidth), np.var(spectral_bandwidth),
        np.mean(rolloff), np.var(rolloff),
        np.mean(zero_cross), np.var(zero_cross),
        np.mean(harmony), np.var(harmony),
        np.mean(percussive), np.var(percussive),
        tempo
    ]

    for i in range(20):
        features.append(np.mean(mfcc[i]))
        features.append(np.var(mfcc[i]))

    return np.array(features).reshape(1, -1)



@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "audio" not in request.files:
            return render_template("index.html", result="No file uploaded")

        file = request.files["audio"]

        if file.filename == "":
            return render_template("index.html", result="No file selected")

        # Save temporary
        path = "uploaded.wav"
        file.save(path)

        # Extract features & scale
        features = extract_features(path)
        scaled = scaler.transform(features)

        # Predict genre
        prediction = qda.predict(scaled)[0]

        return render_template("index.html", result=prediction)

    return render_template("index.html", result=None)


if __name__ == "__main__":
    app.run(debug=True)





