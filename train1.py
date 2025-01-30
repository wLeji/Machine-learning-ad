import joblib
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight

# 📌 Charger les données d'entraînement (sans test.mp4)
X_train, X_test, y_train, y_test = joblib.load("dataset.pkl")
print(f"📊 Données chargées : Entraînement = {X_train.shape}, Test = {X_test.shape}")

# 📌 Calcul du nombre d'exemples par classe
pubs_count = np.sum(y_train)  # Nombre de pubs (y = 1)
programs_count = len(y_train) - pubs_count  # Nombre de programmes (y = 0)

print(f"📊 Nombre d'images Publicité : {pubs_count}")
print(f"📊 Nombre d'images Programme : {programs_count}")

# 📌 Ajustement automatique des poids des classes
class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
weights_dict = {0: class_weights[0], 1: class_weights[1]}

print(f"⚖️ Poids appliqués - Programme: {weights_dict[0]:.2f}, Publicité: {weights_dict[1]:.2f}")

# 📌 Initialisation du modèle avec poids ajustés
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=weights_dict)

# 📌 Entraînement du modèle
model.fit(X_train, y_train)

# 📌 Prédiction et évaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Précision du modèle sur le test set : {accuracy:.2f}")

# 📌 Sauvegarder le modèle entraîné
joblib.dump(model, "pub_detection_model.pkl")
print("✅ Modèle sauvegardé sous pub_detection_model.pkl")

# 📌 Vérification sur quelques exemples du test set
sample_indexes = np.random.choice(len(X_test), 5, replace=False)
sample_X = X_test[sample_indexes]
sample_y = y_test[sample_indexes]
predictions = model.predict(sample_X)

print("\n🎯 Vérification des prédictions (test set) :")
for i in range(len(sample_X)):
    print(f"✅ Réel : {'Publicité' if sample_y[i] == 1 else 'Programme'} | 🧠 Prédit : {'Publicité' if predictions[i] == 1 else 'Programme'}")
