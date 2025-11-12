import streamlit as st
import librosa
import numpy as np
import joblib
import tempfile

# Load trained model & scaler
qda = joblib.load("models/qda_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Feature extractor (same as Flask)
def extract_features(file_path):
    y, sr = librosa.load(file_path, mono=True)
    
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
        length,
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

# Streamlit UI
st.title("🎵 Music Genre Classification")
st.write("Upload an audio file to predict its genre (GTZAN + Bollywood support)")

uploaded_file = st.file_uploader("Upload Audio File", type=["wav", "mp3"])

if uploaded_file is not None:
    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    try:
        # Extract & scale features
        features = extract_features(file_path)
        scaled = scaler.transform(features)

        # Predict
        prediction = qda.predict(scaled)[0]

        st.success(f"🎧 **Predicted Genre: {prediction}**")

    except Exception as e:
        st.error(f"Error processing file: {e}")
