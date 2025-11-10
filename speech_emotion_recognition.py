import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import soundfile as sf   # pip install soundfile (for robust file reading)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import random

# These three lines make your entire ML experiment predictable and repeatable, ensuring that every run of your code gives the same training results.
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

import kagglehub
# Download RAVDESS dataset
path = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")
print("Path to dataset files:", path)

# Emotion mapping based on RAVDESS naming convention
emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Audio parameters
SR = 66150
DURATION = 3.0
TARGET_SAMPLES = int(SR * DURATION)
N_MFCC = 40
MAX_LEN = 180
TRIM_SILENCE = True

X, Y = [], []

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

# Walk through files and extract emotion from filename
for root, _, files in os.walk(path):
    for fname in files:
        if not fname.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
            continue

        # Extract emotion code (3rd number in filename: e.g. "03-01-05-01-02-02-12.wav" → "05")
        try:
            parts = fname.split('-')
            if len(parts) < 3:
                continue
            emotion_code = parts[2]
            emotion = emotion_map.get(emotion_code)
            if emotion is None:
                continue

            file_path = os.path.join(root, fname)
            y = load_and_fix_length(file_path)
            feat = extract_mfcc_features(y)
            X.append(feat)
            Y.append(emotion)
        except Exception as e:
            print(f"Error processing {fname}: {e}")

X = np.array(X)
Y = np.array(Y)

print(f"✅ Extracted samples: {X.shape}")
print(f"✅ Labels shape: {Y.shape}")
print(f"Unique emotions found: {set(Y)}")


le = LabelEncoder()
y_encoded = le.fit_transform(Y)
print("Label classes:", le.classes_)

# Save for training
np.save("X_mfcc_meanstd.npy", X)
np.save("y_labels.npy", y_encoded)

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

model = Sequential([
    LSTM(256, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.4),
    LSTM(128),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(len(np.unique(y_encoded)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=70, batch_size=16)
model.evaluate(X_test, Y_test, verbose=0)

# Plot accuracy
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Test Acc')
plt.legend()
plt.show()

# Evaluate
loss, acc = model.evaluate(np.array(X_test), np.array(Y_test))
print(f"Test Accuracy: {acc*100:.2f}%")


def predict_emotion(model, voice_path):
    try:
        # Load and preprocess
        y = load_and_fix_length(voice_path)
        feat = extract_mfcc_features(y)
        feat = np.expand_dims(feat, axis=0)  # Add batch dimension

        # Predict
        pred = model.predict(feat)
        pred_class = np.argmax(pred, axis=1)[0]
        confidence = float(pred[0][pred_class])

        # Class label names
        class_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

        print(f"\n🎤 File: {os.path.basename(voice_path)}")
        print(f"Predicted Emotion: {class_labels[pred_class].upper()}")
        print(f"Confidence: {confidence*100:.2f}%\n")

    except Exception as e:
        print(f"❌ Error during prediction: {e}")

# 🔁 Loop for multiple predictions
while True:
    voice_file = input("🎧 Enter path to your voice file (.wav): ").strip()
    if not os.path.isfile(voice_file):
        print("⚠️ File not found! Try again.")
        continue

    predict_emotion(model, voice_file)

    again = input("Do you want to test another voice? (yes/no): ").strip().lower()
    if again in ['no', 'exit', 'quit']:
        print("👋 Exiting emotion prediction.")
        break





