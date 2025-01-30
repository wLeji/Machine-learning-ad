import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import cv2
import subprocess

# 📌 Dossiers source et destination
DATASET_DIR = "dataset"
AUDIO_DIR = "audio"
IMAGES_DIR = "images"
SPECTROGRAMS_DIR = "spectrograms"

# 📌 Vérifier et créer les dossiers nécessaires
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(SPECTROGRAMS_DIR, exist_ok=True)

for category in ["pubs", "programs", "test"]:
    os.makedirs(os.path.join(AUDIO_DIR, category), exist_ok=True)
    os.makedirs(os.path.join(IMAGES_DIR, category), exist_ok=True)

### **1️⃣ Extraction de l’audio depuis les vidéos**
def extract_audio(video_path, audio_path):
    """ Extrait l'audio d'une vidéo et l'enregistre en format WAV """
    command = [
        "ffmpeg",
        "-i", video_path,  # Entrée : Vidéo
        "-y",  # Écraser le fichier s'il existe déjà
        "-vn",  # Pas de vidéo
        "-acodec", "pcm_s16le", "-ar", "22050", "-ac", "1",  # Format WAV standard
        audio_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"🎵 Audio extrait : {audio_path}")

print("🚀 Extraction des audios en cours...")

for category in ["pubs", "programs", "test"]:  # Ajout de "test"
    for video_file in os.listdir(f"{DATASET_DIR}/{category}"):
        if video_file.endswith(".mp4"):
            video_path = f"{DATASET_DIR}/{category}/{video_file}"
            audio_path = f"{AUDIO_DIR}/{category}/{video_file.replace('.mp4', '.wav')}"
            extract_audio(video_path, audio_path)

print("✅ Tous les fichiers audio ont été extraits.")

### **2️⃣ Extraction des images depuis les vidéos**
def extract_frames(video_path, output_folder):
    """ Extrait des images d'une vidéo à raison de 1 image par seconde """
    os.makedirs(output_folder, exist_ok=True)
    command = [
        "ffmpeg",
        "-i", video_path,  # Entrée : Vidéo
        "-vf", "fps=1",  # 1 image par seconde
        f"{output_folder}/frame_%04d.jpg"
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"🖼️ Images extraites : {output_folder}")

print("🚀 Extraction des images en cours...")

for category in ["pubs", "programs", "test"]:  # Ajout de "test"
    for video_file in os.listdir(f"{DATASET_DIR}/{category}"):
        if video_file.endswith(".mp4"):
            video_path = f"{DATASET_DIR}/{category}/{video_file}"
            images_folder = f"{IMAGES_DIR}/{video_file.replace('.mp4', '')}"
            extract_frames(video_path, images_folder)

print("✅ Toutes les images ont été extraites.")

### **3️⃣ Génération des spectrogrammes**
def generate_spectrogram(audio_path, output_path):
    """ Génère un spectrogramme à partir d'un fichier audio """
    y, sr = librosa.load(audio_path, sr=22050)

    # Calcul du spectrogramme
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_db = librosa.power_to_db(S, ref=np.max)

    # Sauvegarde du spectrogramme
    plt.figure(figsize=(5, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    plt.axis("off")  # Enlever les axes pour meilleure lisibilité
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

print("🚀 Génération des spectrogrammes en cours...")

for category in ["pubs", "programs", "test"]:  # Ajout de "test"
    for audio_file in os.listdir(f"{AUDIO_DIR}/{category}"):
        if audio_file.endswith(".wav"):
            audio_path = f"{AUDIO_DIR}/{category}/{audio_file}"
            spec_path = f"{SPECTROGRAMS_DIR}/{audio_file.replace('.wav', '.png')}"
            generate_spectrogram(audio_path, spec_path)

print("✅ Tous les spectrogrammes ont été générés.")
