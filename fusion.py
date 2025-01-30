import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import joblib

# Définition des dossiers
IMG_DIR = "images/"
SPECTROGRAM_DIR = "spectrograms/"
LABELS = {"pub1": 1, "prog1": 0}  # Associe chaque type à un label

X = []  # Features
y = []  # Labels

# Parcourir les vidéos (pub1, prog1, etc.)
for video_name in ["pub1", "prog1"]:
    img_folder = os.path.join(IMG_DIR, video_name)
    spec_path = os.path.join(SPECTROGRAM_DIR, f"{video_name}.png")

    # Charger le spectrogramme (audio)
    if os.path.exists(spec_path):
        spec = cv2.imread(spec_path, cv2.IMREAD_GRAYSCALE)
        spec = cv2.resize(spec, (32, 32)).flatten()
    else:
        spec = np.zeros((32 * 32,))  # Si pas de spectrogramme, on met des zéros

    # Charger les images vidéo associées
    for img_file in sorted(os.listdir(img_folder)):  # Trier par ordre croissant
        img_path = os.path.join(img_folder, img_file)

        # Charger l'image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128)).flatten()

        # Fusionner image et spectrogramme
        feature_vector = np.concatenate((img, spec))
        X.append(feature_vector)
        y.append(LABELS[video_name])

# Convertir en numpy array
X = np.array(X)
y = np.array(y)

# Séparer en données d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Taille du dataset : {X.shape}")
print(f"Entraînement : {X_train.shape}, Test : {X_test.shape}")

joblib.dump((X_train, X_test, y_train, y_test), "dataset.pkl")
print("✅ Dataset sauvegardé dans dataset.pkl")
