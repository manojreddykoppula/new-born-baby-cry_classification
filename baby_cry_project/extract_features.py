import librosa
import numpy as np
import os

# -------- COMMON FEATURE FUNCTION (SAME FOR ALL FILES) --------
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=5)

    audios = [
        y,  # original
        y + 0.005 * np.random.randn(len(y)),  # noise
        librosa.effects.pitch_shift(y, sr=sr, n_steps=2),  # pitch
        librosa.effects.time_stretch(y, rate=0.8)  # speed
    ]

    features_list = []

    for audio in audios:
        mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
        energy = np.mean(librosa.feature.rms(y=audio))

        pitches, _ = librosa.piptrack(y=audio, sr=sr)
        pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0

        volume = np.mean(np.abs(audio))

        features = np.hstack([mfcc, zcr, energy*5, pitch*5, volume*10])
        features_list.append(features)

    return features_list

# -------- DATASET PROCESSING --------
X = []
y = []

labels = {
    "hunger": 0,
    "pain": 1,
    "sleep": 2,
    "discomfort": 3
}

for label in labels:
    folder = "dataset/" + label
    
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            path = os.path.join(folder, file)

            features_list = extract_features(path)

            if features_list is not None:
                for f in features_list:
                    X.append(f)
                    y.append(labels[label])

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Save files
np.save("X.npy", X)
np.save("y.npy", y)

print("✅ Features extracted successfully")
print("Shape of X:", X.shape)
print("Labels count:", np.bincount(y))