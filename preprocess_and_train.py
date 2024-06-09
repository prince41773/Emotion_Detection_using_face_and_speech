import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.io import wavfile
import librosa
import noisereduce as nr
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import soundfile as sf

# Define the emotions dictionary
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Observed emotions
observed_emotions = ['calm', 'happy', 'fearful', 'disgust']

def getListOfFiles(dirName):
    listOfFile = os.listdir(dirName)
    allFiles = []
    for entry in listOfFile:
        fullPath = os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
    return allFiles

# Function to apply envelope and mask the signal
def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask

# Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
    with sf.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
    return result

# Load the data and extract features for each sound file
def load_data():
    x, y, filenames = [], [], []
    for file in glob.glob(r'./clean_speech/*.wav'):
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
        filenames.append(file_name)
    return np.array(x), np.array(y), np.array(filenames)

print("Loading and cleaning audio files...")
dirName = './speech-emotion-recognition-ravdess-data'
listOfFiles = getListOfFiles(dirName)
for file in tqdm(listOfFiles):
    signal, rate = librosa.load(file, sr=16000)
    mask = envelope(signal, rate, 0.0005)
    clean_file_name = os.path.join('./clean_speech/', os.path.basename(file))
    wavfile.write(filename=clean_file_name, rate=rate, data=signal[mask])

print("Extracting features and splitting data...")
x, y, filenames = load_data()
x_train, x_test, y_train, y_test, filenames_train, filenames_test = train_test_split(x, y, filenames, test_size=0.25, random_state=9)
print(f'Training data shape: {np.shape(x_train)}, Testing data shape: {np.shape(x_test)}')

# Initialize the Multi Layer Perceptron Classifier
model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)
model.fit(x_train, y_train)

# Save the Model and Data to file
with open("Emotion_Voice_Detection_Model.pkl", 'wb') as file:
    pickle.dump(model, file)
with open("Emotion_Voice_Data.pkl", 'wb') as file:
    pickle.dump((x_train, x_test, y_train, y_test, filenames_train, filenames_test), file)

# Load the Model back from file
with open("Emotion_Voice_Detection_Model.pkl", 'rb') as file:
    Emotion_Voice_Detection_Model = pickle.load(file)

# Predicting
y_pred = Emotion_Voice_Detection_Model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Store the Prediction probabilities into CSV file
y_pred_df = pd.DataFrame(y_pred, columns=['predictions'])
y_pred_df['file_names'] = filenames_test
y_pred_df.to_csv('predictionfinal.csv', index=False)
