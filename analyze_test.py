import joblib
import os
import numpy as np
import cv2

# 📌 Charger le modèle entraîné
model = joblib.load("pub_detection_model.pkl")
print("✅ Modèle chargé depuis pub_detection_model.pkl")

# 📌 Vérifier les fichiers nécessaires
spec_output = "spectrograms/test.png"
img_folder = "images/test/"
if not os.path.exists(spec_output) or not os.path.exists(img_folder):
    print("⚠️ Fichiers de test introuvables. Exécute `generate.py` avant ce script.")
    exit()

# 📌 Charger le spectrogramme de test
spec = cv2.imread(spec_output, cv2.IMREAD_GRAYSCALE)
spec = cv2.resize(spec, (32, 32)).flatten()

# 📌 Prédire image par image
results = []
for img_file in sorted(os.listdir(img_folder)):
    img_path = os.path.join(img_folder, img_file)

    # Charger l'image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128)).flatten()

    # Fusionner image et spectrogramme
    feature_vector = np.concatenate((img, spec)).reshape(1, -1)

    # Prédiction
    prediction = model.predict(feature_vector)[0]
    results.append(prediction)

# 📌 Calcul des statistiques
total_images = len(results)
pub_count = sum(results)
prog_count = total_images - pub_count
pub_percentage = (pub_count / total_images) * 100
prog_percentage = (prog_count / total_images) * 100

# 📌 Affichage des résultats
print("\n📡 Résumé des prédictions sur test.mp4 :")
print(f"🔸 {pub_percentage:.2f}% des images classées comme Publicité")
print(f"🔹 {prog_percentage:.2f}% des images classées comme Programme Normal")

# 📌 Afficher un échantillon des prédictions image par image
print("\n📸 Prédictions image par image :")
for i, img_file in enumerate(sorted(os.listdir(img_folder))[:10]):  # Afficher 10 images max
    print(f"🖼️ {img_file} → 🧠 Prédit : {'Publicité' if results[i] > 0.5 else 'Programme Normal'}")
