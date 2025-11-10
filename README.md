🗣️ Speech Emotion Recognition using LSTM
This project detects human emotions from voice recordings using deep learning (LSTM) based on extracted MFCC (Mel-Frequency Cepstral Coefficients) features.
The goal is to make machines understand emotional tone (happy, sad, angry, neutral, etc.) from speech.

🚀 Overview
This project uses the RAVDESS Emotional Speech Audio Dataset and trains a Long Short-Term Memory (LSTM) network to classify emotions from voice data.
It can also work with custom recordings, allowing users to predict emotions from their own voice files.

🎯 Features
✅ Emotion classification from voice recordings
✅ MFCC feature extraction using Librosa
✅ LSTM-based neural network for sequential feature learning
✅ Custom voice prediction support (user input)
✅ Configurable preprocessing (sampling rate, duration, trimming, padding)

🧠 Model Architecture
Input: MFCC features (MAX_LEN x N_MFCC)
↓
LSTM Layer (128 units)
↓
Dropout (0.3)
↓
Dense (64, activation='relu')
↓
Dense (number_of_emotions, activation='softmax')

📂 Dataset
🎵 RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
Contains 24 actors expressing 8 emotions in speech:

Code	Emotion
01	Neutral
02	Calm
03	Happy
04	Sad
05	Angry
06	Fearful
07	Disgust
08	Surprised

Each audio file follows a naming convention like:
03-01-05-01-02-02-12.wav → Emotion code 05 = Angry

⚙️ Installation
1️⃣ Clone the repository
git clone https://github.com/<your-username>/speech_emotion_recognition.git
cd speech_emotion_recognition

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Download dataset
Using KaggleHub:
import kagglehub
path = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")
Or download manually from Kaggle
.

🧩 Training the Model
python train_model.py
This script will:
Load the dataset
Extract MFCC features
Train an LSTM model
Save the trained model as emotion_model.h5

🎧 Predict Emotion from a Voice File
After training, you can test your own voice:
python predict_emotion.py
Then enter your .wav file path when prompted.

Example Output:
🎤 File: myvoice.wav
Predicted Emotion: HAPPY
Confidence: 94.32%
