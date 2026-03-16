import os
import numpy as np
import argparse
from deepsleep.model import DeepFeatureNet

def main():
    parser = argparse.ArgumentParser(description="Prédiction avec DeepFeatureNet")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Dossier contenant les fichiers .npz de test")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Chemin vers le fichier des poids du modèle (.h5)")
    parser.add_argument("--num_classes", type=int, default=5,
                        help="Nombre de classes pour la classification")
    args = parser.parse_args()

    # Chargement des fichiers .npz
    npz_files = [f"{args.data_dir}/{f}" for f in sorted(os.listdir(args.data_dir)) if f.endswith(".npz")]
    X_list, y_list = [], []
    for file in npz_files:
        data = np.load(file)
        X_list.append(data['x'])
        y_list.append(data['y'])

    X_test = np.concatenate(X_list, axis=0)
    y_test = np.concatenate(y_list, axis=0)
    print(f"[INFO] Chargé {len(npz_files)} fichiers: X={X_test.shape}, y={y_test.shape}")

    # Instancier le modèle
    model = DeepFeatureNet(input_shape=(X_test.shape[1],), num_classes=args.num_classes)
    
    # Charger les poids
    model.load_weights(args.model_path)
    print("[INFO] Poids chargés avec succès !")

    # Prédictions
    y_pred = np.argmax(model.predict(X_test, batch_size=64), axis=1)

    # Sauvegarde
    np.save("y_test.npy", y_test)
    np.save("y_pred.npy", y_pred)
    print("[INFO] Prédictions sauvegardées : y_test.npy et y_pred.npy")

if __name__ == "__main__":
    main()


