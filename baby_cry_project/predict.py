import sounddevice as sd
import numpy as np
import librosa
import joblib
from scipy.io.wavfile import write
from keras.models import load_model

# -------- COMMON FEATURE FUNCTION --------
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=5)

    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    energy = np.mean(librosa.feature.rms(y=y))

    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0

    volume = np.mean(np.abs(y))

    features = np.hstack([mfcc, zcr, energy*5, pitch*5, volume*10])
    return features

# -------- LOAD MODEL --------
model = load_model("cry_model.h5", compile=False)
scaler = joblib.load("scaler.pkl")

labels = ["hunger", "pain", "sleep", "discomfort"]

print("🎤 Recording... Play baby cry near mic")

# -------- RECORD AUDIO --------
fs = 44100
duration = 5   # ✅ SAME everywhere

audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
sd.wait()

write("test.wav", fs, audio)

# -------- SILENCE CHECK --------
y_audio, sr = librosa.load("test.wav", duration=5)
volume = np.mean(np.abs(y_audio))

if volume < 0.01:
    print("❌ No baby cry detected")
    exit()

# -------- FEATURE EXTRACTION --------
features = extract_features("test.wav")

# -------- SCALING --------
features = scaler.transform([features])

# -------- PREDICTION --------
pred = model.predict(features)

print("Prediction probabilities:", pred)

predicted_index = np.argmax(pred)
confidence = pred[0][predicted_index] * 100

print("✅ Predicted Cry Type:", labels[predicted_index])
print("🔥 Confidence:", round(confidence, 2), "%")