import os
import numpy as np
from sklearn.model_selection import train_test_split

def load_npz_data(data_dir):
    """Charge tous les fichiers .npz d'un dossier et concatène x et y."""
    X_list, y_list = [], []
    print("[INFO] Chargement des données...")

    for file in os.listdir(data_dir):
        if file.endswith(".npz"):
            path = os.path.join(data_dir, file)
            data = np.load(path)

            # Correction : certaines clés sont en minuscules
            if "x" in data and "y" in data:
                X_list.append(data["x"])
                y_list.append(data["y"])
            else:
                print(f"[AVERTISSEMENT] Fichier {file} ne contient pas 'x' et 'y' — ignoré")

    # Vérification
    if len(X_list) == 0:
        raise ValueError(f"Aucun fichier .npz valide trouvé dans {data_dir}")

    # Concaténation
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    print(f"[INFO] Données chargées : X={X.shape}, y={y.shape}")
    return X, y


def get_train_test_data(data_dir, test_size=0.2, random_state=42):
    """Retourne les ensembles d'entraînement et de test"""
    X, y = load_npz_data(data_dir)
    print("[INFO] Division train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[INFO] Entraînement : {X_train.shape}, Test : {X_test.shape}")
    return X_train, X_test, y_train, y_test













