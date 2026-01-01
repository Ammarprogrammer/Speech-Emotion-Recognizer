
import streamlit as st
import os
import random
import json
import numpy as np
import librosa
import soundfile as sf
from tensorflow.keras.models import load_model

# Constants
SR = 66150
DURATION = 3.0
TARGET_SAMPLES = int(SR * DURATION)
N_MFCC = 40
MAX_LEN = 180
TRIM_SILENCE = True

# Load the trained model
MODEL_PATH = 'emotion_model.h5'
if not os.path.exists(MODEL_PATH):
    st.error("Model not found. Please train the model first by running `speech_emotion_recognition.py`.")
    st.stop()
model = load_model(MODEL_PATH)

# Load duas
with open("DeepLearning/dua_suggession.json", "r", encoding="utf-8") as file:
    duas = json.load(file)

def load_and_fix_length(path, sr=SR, target_samples=TARGET_SAMPLES, trim_silence=TRIM_SILENCE):
    y, file_sr = librosa.load(path, sr=sr, mono=True)
    if trim_silence:
        y, _ = librosa.effects.trim(y, top_db=20)
    if len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y)), 'constant')
    elif len(y) > target_samples:
        start = max(0, (len(y) - target_samples) // 2)
        y = y[start:start + target_samples]
    return y

def extract_mfcc_features(y, sr=SR, n_mfcc=N_MFCC):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features = np.vstack([mfcc, chroma, mel, contrast])

    if features.shape[0] < MAX_LEN:
        features = np.pad(features, ((0, MAX_LEN - features.shape[0]), (0, 0)), mode='constant')
    else:
        features = features[:MAX_LEN, :]
    return features

def predict_emotion(audio_path):
    try:
        y = load_and_fix_length(audio_path)
        feat = extract_mfcc_features(y)
        feat = np.expand_dims(feat, axis=0)
        pred = model.predict(feat)
        pred_class = np.argmax(pred, axis=1)[0]
        confidence = float(pred[0][pred_class])
        class_labels = [
            'neutral', 'calm', 'happy', 'sad',
            'angry', 'fearful', 'disgust', 'surprised'
        ]
        return class_labels[pred_class].upper(), confidence
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None

def get_random_dua(emotion):
    if emotion not in duas:
        return {"error": "Emotion not found in dua list"}
    selected = random.choice(duas[emotion])
    selected_video = random.choice(selected["video_links"])
    return {
        "dua_arabic": selected["dua_arabic"],
        "dua_english": selected["dua_english"],
        "caption": selected["caption"],
        "video_link": selected_video
    }

# Streamlit UI
st.set_page_config(page_title="Emotion Checker", page_icon="😊", layout="wide")

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(to right, #ff7e5f, #feb47b);
        color: white;
    }
    .st-emotion-cache-1kyxreq {
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .st-emotion-cache-1kyxreq>div {
        width: 100%;
        max-width: 800px;
        background: rgba(0, 0, 0, 0.5);
        padding: 2rem;
        border-radius: 10px;
    }
    h1, h2, h3, p {
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.title("Check Your Emotion")

uploaded_file = st.file_uploader("Upload an audio file (wav, mp3, ogg, flac)", type=["wav", "mp3", "ogg", "flac"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with open(os.path.join("temp_audio.wav"), "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(uploaded_file, format='audio/wav')

    emotion, confidence = predict_emotion("temp_audio.wav")

    if emotion:
        st.header(f"Predicted Emotion: {emotion} ({confidence*100:.2f}%)")

        dua = get_random_dua(emotion)
        if "error" not in dua:
            st.subheader(dua["caption"])
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"### {dua['dua_arabic']}")
            with col2:
                st.markdown(f"### {dua['dua_english']}")

            # Basic validation for video link
            if "yourvideo.com" not in dua["video_link"]:
                 st.video(dua["video_link"])
            else:
                 st.info("Placeholder video link. Please update in emotion.json")

    # Clean up the temporary file
    os.remove("temp_audio.wav")
