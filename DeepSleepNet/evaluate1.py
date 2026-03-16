import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)

# -------------------------------
# Définition du modèle DeepFeatureNet
# -------------------------------
class DeepFeatureNet(Model):
    def __init__(self, input_shape, num_classes):
        super(DeepFeatureNet, self).__init__()

        self.conv1 = Conv1D(filters=64, kernel_size=50, strides=6, activation='relu')
        self.bn1 = BatchNormalization()
        self.pool1 = MaxPooling1D(pool_size=8, strides=8)

        self.conv2 = Conv1D(filters=128, kernel_size=8, strides=1, activation='relu')
        self.bn2 = BatchNormalization()
        self.pool2 = MaxPooling1D(pool_size=4, strides=4)

        self.conv3 = Conv1D(filters=128, kernel_size=8, strides=1, activation='relu')
        self.bn3 = BatchNormalization()
        self.pool3 = MaxPooling1D(pool_size=4, strides=4)

        self.flatten = Flatten()
        self.fc1 = Dense(512, activation='relu')
        self.drop1 = Dropout(0.5)
        self.fc2 = Dense(256, activation='relu')
        self.drop2 = Dropout(0.5)
        self.output_layer = Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.pool3(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.drop1(x, training=training)
        x = self.fc2(x)
        x = self.drop2(x, training=training)
        return self.output_layer(x)

# -------------------------------
# Chargement des données
# -------------------------------
def load_data(data_dir):
    X_list, y_list = [], []
    for file in os.listdir(data_dir):
        if file.endswith(".npz"):
            data = np.load(os.path.join(data_dir, file))
            X_list.append(data["x"])
            y_list.append(data["y"])
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    if X.ndim == 2:
        X = X[..., np.newaxis]

    return X, y

# -------------------------------
# Paramètres
# -------------------------------
MODEL_PATH = "/content/drive/MyDrive/Old_Deep_Sleep/outputs_pretrain/best_model.h5"
DATA_DIR = "/content/drive/MyDrive/DeepSleepNet/PolySomnoGraphy/data/Sleep-EDF/merged_npz_all"

LABELS = ["W", "N1", "N2", "N3", "REM"]

# -------------------------------
# Chargement des données
# -------------------------------
print("[INFO] Chargement des données...")
X_test, y_test = load_data(DATA_DIR)

num_classes = len(np.unique(y_test))
input_shape = X_test.shape[1:]

print(f"[INFO] Samples : {X_test.shape[0]}")
print(f"[INFO] Input shape : {input_shape}")

# -------------------------------
# Construction du modèle
# -------------------------------
print("[INFO] Construction du modèle...")
model = DeepFeatureNet(input_shape=input_shape, num_classes=num_classes)
model(tf.zeros((1, *input_shape)))
model.load_weights(MODEL_PATH)
print("[INFO] Poids chargés")

# -------------------------------
# Prédictions
# -------------------------------
print("[INFO] Prédiction...")
y_pred_prob = model.predict(X_test, batch_size=64)
y_pred = np.argmax(y_pred_prob, axis=1)

# -------------------------------
# MATRICE DE CONFUSION
# -------------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=LABELS,
    yticklabels=LABELS,
    cbar=True
)
plt.xlabel("Prédiction")
plt.ylabel("Vérité terrain")
plt.title("Matrice de confusion – DeepFeatureNet")
plt.tight_layout()
plt.show()

# -------------------------------
# MÉTRIQUES
# -------------------------------
acc = accuracy_score(y_test, y_pred)

prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
    y_test, y_pred, average="macro"
)
prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
    y_test, y_pred, average="weighted"
)

# -------------------------------
# TABLEAU DES PERFORMANCES GLOBALES
# -------------------------------
global_metrics = pd.DataFrame({
    "Mesure": ["Macro moyenne", "Moyenne pondérée", "Accuracy globale"],
    "Précision": [prec_macro, prec_weighted, np.nan],
    "Rappel": [rec_macro, rec_weighted, np.nan],
    "F1-score": [f1_macro, f1_weighted, acc]
})

print("\n===== TABLEAU DES PERFORMANCES GLOBALES =====\n")
print(global_metrics.round(4))

# -------------------------------
# TABLEAU DES PERFORMANCES PAR CLASSE
# -------------------------------
prec, rec, f1, support = precision_recall_fscore_support(
    y_test, y_pred, average=None
)

per_class_metrics = pd.DataFrame({
    "Classe": LABELS,
    "Précision": prec,
    "Rappel": rec,
    "F1-score": f1,
    "Support": support
})

print("\n===== TABLEAU DES PERFORMANCES PAR CLASSE =====\n")
print(per_class_metrics.round(4))



