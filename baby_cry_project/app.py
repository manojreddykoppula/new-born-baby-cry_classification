import tkinter as tk
from tkinter import messagebox
import numpy as np
import librosa
import tensorflow as tf
import joblib
import sounddevice as sd
from scipy.io.wavfile import write
import threading
import pyttsx3
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- CONSTANTS & STYLE ---
BG_COLOR = "#F0F7FF"  # Soft sky blue
HEADER_COLOR = "#7EB6FF"  # Bright friendly blue
CARD_COLOR = "#FFFFFF"
ACCENT_PINK = "#FF9AA2"
ACCENT_MINT = "#B2E2D2"
ACCENT_YELLOW = "#FFF5BA"
TEXT_COLOR = "#4A4A4A"

# Global variables for model, scaler, and audio file
audio_file = "test.wav"
engine = pyttsx3.init()
model = None # Will be loaded once
scaler = None # Will be loaded once

# ---------------- FEATURE EXTRACTION ----------------
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=10)

        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        energy = np.mean(librosa.feature.rms(y=y))

        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0

        volume = np.mean(np.abs(y))

        features = np.hstack([mfcc, zcr, energy*5, pitch*5, volume*10])
        return features

    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None

# ---------------- SUGGESTIONS ----------------
def get_suggestions(label):
    data = {
        "Hunger 🍼": ("✅ Feed immediately\n✅ Maintain routine", "❌ Don't delay feeding\nAvoid overfeeding", "💡 Observe feeding patterns"),
        "Pain 😢": ("✅ Check for fever\n✅ Comfort baby", "❌ Don't ignore long cries", "💡 Consult doctor if persistent"),
        "Sleepy 😴": ("✅ Dim the lights\n✅ Rock gently", "❌ Avoid loud noises", "💡 Create a calm environment"),
        "Discomfort 😣": ("✅ Check diaper\n✅ Adjust clothes", "❌ Don't ignore irritation", "💡 Ensure room temp is okay")
    }
    return data.get(label, ("...", "...", "..."))

# ---------------- LOGIC ----------------
def speak(text):
    engine.say(text)
    engine.runAndWait()

def record_audio():
    def _record():
        global audio_file
        fs = 44100
        duration = 10
        
        # Update GUI only from the main thread using after()
        root.after(0, lambda: record_btn.config(text="🔴 Recording...", state="disabled"))
        root.after(0, lambda: loading_label.config(text="Recording audio..."))

        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32') # Specify dtype for clarity
        sd.wait()
        write(audio_file, fs, recording) # Use global audio_file

        # Update GUI from the main thread after recording is done
        root.after(0, lambda: record_btn.config(text="🎤 Record", state="normal"))
        root.after(0, lambda: loading_label.config(text="Recording complete!"))
        
        # Now that recording is done, plot the waveform
        root.after(0, plot_waveform)
        
    threading.Thread(target=_record).start() # Start recording in a new thread

def plot_waveform():
    for widget in graph_frame.winfo_children():
        widget.destroy()
    try:
        y, sr = librosa.load(audio_file)
        fig = plt.Figure(figsize=(6, 2), dpi=80, facecolor=BG_COLOR)
        ax = fig.add_subplot(111)
        ax.plot(y, color=HEADER_COLOR)
        ax.axis('off') # Clean look
        canvas = FigureCanvasTkAgg(fig, master=graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()
    except Exception as e:
        print(f"Error plotting waveform: {e}")
        loading_label.config(text="Error loading audio for waveform.")


def predict_audio():
    if model is None or scaler is None:
        messagebox.showerror("Error", "AI model or scaler not loaded. Please restart the application.")
        return

    def _predict():
        try:
            root.after(0, lambda: loading_label.config(text="✨ Thinking... ✨"))
            
            features = extract_features(audio_file)
            if features is None:
                raise ValueError("Failed to extract features from audio.")

            # 🔴 ADD THIS BLOCK HERE
            volume = np.mean(np.abs(features))
            if volume < 0.001:
                root.after(0, lambda: messagebox.showwarning("Warning", "No baby cry detected"))
                root.after(0, lambda: loading_label.config(text="No significant sound detected"))
                return
            
            
            # Reshape for scalar if necessary, then scale
            features_reshaped = features.reshape(1, -1) if features.ndim == 1 else features
            features_scaled = scaler.transform(features_reshaped)
            
            prediction = model.predict(features_scaled)
            label_index = np.argmax(prediction)
            confidence = round(np.max(prediction) * 100, 1)
            classes = ["Hunger 🍼", "Pain 😢", "Sleepy 😴", "Discomfort 😣"]
            result = classes[label_index]

            root.after(0, lambda: result_label.config(text=result))
            root.after(0, lambda: confidence_label.config(text=f"{confidence}% Match"))
            
            do, dont, tip = get_suggestions(result)
            root.after(0, lambda: do_label.config(text=f"DO:\n{do}"))
            root.after(0, lambda: dont_label.config(text=f"DON'T:\n{dont}"))
            root.after(0, lambda: tip_label.config(text=f"TOP TIP: {tip}"))

            threading.Thread(target=speak, args=(f"Baby is {result}",)).start()
            root.after(0, lambda: loading_label.config(text="Analysis Complete! ✅"))
        except Exception as e:
            root.after(0, lambda: messagebox.showerror("Error", str(e)))
            root.after(0, lambda: loading_label.config(text=""))

    threading.Thread(target=_predict).start()

# ---------------- GUI ----------------
root = tk.Tk()
root.title("Baby Care AI")
root.geometry("900x850")
root.configure(bg=BG_COLOR)

# Load model and scaler once at startup
try:
    model = tf.keras.models.load_model("cry_model.h5")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    messagebox.showerror("Loading Error", f"Failed to load AI model or scaler: {e}\nPlease ensure 'cry_model.h5' and 'scaler.pkl' are in the same directory.")
    root.destroy() # Close application if essential files are missing

# Custom Title Bar / Header
# ... (rest of your GUI code remains largely the same)
header = tk.Frame(root, bg=HEADER_COLOR, pady=20)
header.pack(fill="x")

tk.Label(header, text="🌈 New Born Baby Cry Classification And Analysis   👶", 
         font=("Comic Sans MS", 32, "bold"), 
         bg=HEADER_COLOR, fg="white").pack()

# Main Container
container = tk.Frame(root, bg=BG_COLOR)
container.pack(pady=20, padx=40, fill="both", expand=True)

# Buttons Section
btn_frame = tk.Frame(container, bg=BG_COLOR)
btn_frame.pack(pady=10)

btn_style = {"font": ("Segoe UI", 12, "bold"), "fg": "white", "width": 15, "bd": 0, "cursor": "hand2", "pady": 10}

record_btn = tk.Button(btn_frame, text="🎤 Record", command=record_audio, bg=ACCENT_PINK, **btn_style)
record_btn.grid(row=0, column=0, padx=15)

predict_btn = tk.Button(btn_frame, text="🔍 Analyze", command=predict_audio, bg=HEADER_COLOR, **btn_style)
predict_btn.grid(row=0, column=1, padx=15)

loading_label = tk.Label(container, text="", font=("Segoe UI", 11, "italic"), bg=BG_COLOR, fg=HEADER_COLOR)
loading_label.pack()

# Visual Waveform Area
graph_frame = tk.Frame(container, bg=BG_COLOR, height=150)
graph_frame.pack(fill="x", pady=10)

# Result Card
result_card = tk.Frame(container, bg=CARD_COLOR, padx=20, pady=20, highlightthickness=2, highlightbackground="#E0E0E0")
result_card.pack(fill="x", pady=10)

result_label = tk.Label(result_card, text="Waiting for baby...", font=("Comic Sans MS", 28, "bold"), bg=CARD_COLOR, fg=TEXT_COLOR)
result_label.pack()

confidence_label = tk.Label(result_card, text="Ready to help", font=("Segoe UI", 12), bg=CARD_COLOR, fg="gray")
confidence_label.pack()

# Advice Cards
advice_frame = tk.Frame(container, bg=BG_COLOR)
advice_frame.pack(fill="x", pady=20)

card_font = ("Segoe UI", 11, "bold")

do_label = tk.Label(advice_frame, text="DO:\n---", bg=ACCENT_MINT, font=card_font, width=25, height=5, wraplength=200, justify="center")
do_label.grid(row=0, column=0, padx=10, sticky="nsew")

dont_label = tk.Label(advice_frame, text="DON'T:\n---", bg="#FFD1D1", font=card_font, width=25, height=5, wraplength=200, justify="center")
dont_label.grid(row=0, column=1, padx=10, sticky="nsew")

advice_frame.columnconfigure(0, weight=1)
advice_frame.columnconfigure(1, weight=1)

tip_label = tk.Label(container, text="💡 Tip: Record at least 5 seconds for best results!", 
                     bg=ACCENT_YELLOW, font=("Segoe UI", 12, "italic"), pady=15, relief="flat")
tip_label.pack(fill="x", pady=10)

root.mainloop()

