# calculated information
# population mean = 3.70 
# STD = 0.34

# STEP 0 PREPARE
# import libraries
import os
import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import tensorflow as tf



# STEP 1 LOAD SAMPLES
# load a sample
audio = "data/ravdess/Actor_02/03-01-04-01-02-02-02.wav"
y, sr = librosa.load(audio, sr = None)
print("Audio Loaded")
print(f"Sampling Rate: {sr}, Audio Duration: {len(y)/sr} seconds")
y = librosa.util.normalize(y)

# load a wave length
plt.figure(figsize = (10,4))
librosa.display.waveshow(y, sr=sr, color = "pink")
plt.title("Sample 03-01-04-01-02-02-02 from Actor 2")
plt.xlabel("Time (secs)")
plt.ylabel("Amplitude")
plt.show()
ravdess_path = "data/ravdess"

# STEP 2 EXTRACT FEARTURES
# Mel-Frequency Cepstral Coefficients (MFCCs) capture info that helps id the emtion

# find the MFCCs of a single file
y, sr = librosa.load("data/ravdess/Actor_02/03-01-04-01-02-02-02.wav")
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# heatmap load example
plt2.figure(figsize = (10,4))
librosa.display.specshow(mfccs, x_axis='time', sr=sr)
plt2.colorbar(format='%+2.0f dB')
plt2.title("MFCCs")
plt2.xlabel("Time")
plt2.ylabel("MFCC Coefficients")
plt2.show()

# extract the MFCCs from the whole data set
mfcc_features = []

for actor in os.listdir(ravdess_path):
    actor_path = os.path.join(ravdess_path, actor)
    if os.path.isdir(actor_path):
        for file in os.listdir(actor_path):
            if file.endswith(".wav"):
                file_path = os.path.join(actor_path, file)
                y, sr = librosa.load(file_path, sr=None)
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                mfcc_features.append(np.mean(mfccs, axis=1))  # Store the average of MFCCs

np.save("mfcc_features.npy", mfcc_features)
print(f"Saved {len(mfcc_features)} feature vectors")

emotion_map ={
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

features = []
labels = []
ravdess_path = "data/ravdess"


# Loop through all audio files in the dataset
for actor in os.listdir(ravdess_path):
    actor_path = os.path.join(ravdess_path, actor)
    if os.path.isdir(actor_path):  # Check if it's a folder
        for file in os.listdir(actor_path):
            if file.endswith(".wav"):  # Process only .wav files
                # Extract emotion code from the filename
                emotion_code = file.split("-")[2]
                emotion = emotion_map[emotion_code]

                # Load and preprocess the audio file
                file_path = os.path.join(actor_path, file)
                y, sr = librosa.load(file_path, sr=None)
                y = librosa.util.normalize(y)

                # Extract MFCCs
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                mfcc_mean = np.mean(mfccs, axis=1)  # Take the mean across time frames

                # Append features and labels
                features.append(mfcc_mean)
                labels.append(emotion)

# Convert to NumPy arrays
features = np.array(features)
labels = np.array(labels)

# Save to .npy files
np.save("features.npy", features)
np.save("labels.npy", labels)

print("Features and labels saved!")
print(f"Total Samples: {len(features)}")
