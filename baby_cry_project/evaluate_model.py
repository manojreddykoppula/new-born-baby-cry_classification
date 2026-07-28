import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
from sklearn.model_selection import train_test_split

# Load features
X = np.load("X.npy")
y = np.load("y.npy")

# Load scaler and normalize
scaler = joblib.load("scaler.pkl")
X = scaler.transform(X)

# Split into train/test (same as training)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, stratify=y
)

# One-hot encode labels
y_test_cat = tf.keras.utils.to_categorical(y_test, 4)

# Load trained model
model = load_model("cry_model.h5")

# Evaluate on test set
loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"Test Accuracy: {accuracy*100:.2f}%")