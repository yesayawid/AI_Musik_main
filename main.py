import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def extract_features(file_path, mfcc=True, chroma=True, mel=True):
    try:
        audio, sample_rate = librosa.load(file_path, sr=None)   
        features = []
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
            features.append(mfccs)
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate).T,axis=0)
            features.append(chroma)
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T,axis=0)
            features.append(mel)
        return np.concatenate(features)
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None


data_disco       = "Dataset/Disco"
data_pop         = "Dataset/Pop"
data_reggae      = "Dataset/Reggae"
# data_rock      = "Dataset/Rock"
# data_blues     = "Dataset/Blues"
# data_classical = "Dataset/Classical"
# data_country   = "Dataset/Country"
# data_hiphop-   = "Dataset/Hiphop"
# data_jazz-      = "Dataset/Jazz"
# data_metal     = "Dataset/Metal"


features = []
labels = []

# Memproses data Disco
for file in os.listdir(data_disco):
    file_path = os.path.join(data_disco, file)
    feature = extract_features(file_path)
    if feature is not None:
        features.append(feature)
        labels.append(0)

# Memproses data Pop
for file in os.listdir(data_pop):
    file_path = os.path.join(data_pop, file)
    feature = extract_features(file_path)
    if feature is not None:
        features.append(feature)
        labels.append(1)

# Memproses data Rock
for file in os.listdir(data_reggae):
    file_path = os.path.join(data_reggae, file)
    feature = extract_features(file_path)
    if feature is not None:
        features.append(feature)
        labels.append(2)

# # Memproses data Disco
# for file in os.listdir(data_disco):
#     file_path = os.path.join(data_disco, file)
#     feature = extract_features(file_path)
#     if feature is not None:
#         features.append(feature)
#         labels.append(3)

# # Memproses data Hiphop
# for file in os.listdir(data_hiphop):
#     file_path = os.path.join(data_hiphop, file)
#     feature = extract_features(file_path)
#     if feature is not None:
#         features.append(feature)
#         labels.append(4)

# # Memproses data Jazz
# for file in os.listdir(data_jazz):
#     file_path = os.path.join(data_jazz, file)
#     feature = extract_features(file_path)
#     if feature is not None:
#         features.append(feature)
#         labels.append(5)

# # Memproses data Metal
# for file in os.listdir(data_metal):
#     file_path = os.path.join(data_metal, file)
#     feature = extract_features(file_path)
#     if feature is not None:
#         features.append(feature)
#         labels.append(6)

# # Memproses data Pop
# for file in os.listdir(data_pop):
#     file_path = os.path.join(data_pop, file)
#     feature = extract_features(file_path)
#     if feature is not None:
#         features.append(feature)
#         labels.append(7)

# # Memproses data Reggae
# for file in os.listdir(data_reggae):
#     file_path = os.path.join(data_reggae, file)
#     feature = extract_features(file_path)
#     if feature is not None:
#         features.append(feature)
#         labels.append(8)

# # Memproses data Rock
# for file in os.listdir(data_rock):
#     file_path = os.path.join(data_rock, file)
#     feature = extract_features(file_path)
#     if feature is not None:
#         features.append(feature)
#         labels.append(9)

features = np.array(features)
labels = np.array(labels)

X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Reshape((X_train.shape[1], 1)),
    tf.keras.layers.Conv1D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Conv1D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Conv1D(256, 3, activation='relu'),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')  # 10 classes: 0-9
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=150, batch_size=32)

loss, accuracy = model.evaluate(X_val, y_val)
print("Validation Accuracy:", accuracy)

model.summary()

model_save_path = "Model"
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
model.save(os.path.join(model_save_path, "audio.h5"))
 