import argparse
import os
from deepsleep.trainer import DeepFeatureNetTrainer

def main():
    parser = argparse.ArgumentParser(description="DeepFeatureNet Training")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Chemin vers le dossier contenant les fichiers .npz"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Chemin de sortie pour sauvegarder les modèles et logs"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
        help="Nombre d'époques d'entraînement"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Taille du batch pour l'entraînement"
    )

    args = parser.parse_args()

    print("[INFO] Initialisation de l'entraîneur DeepFeatureNet...")
    trainer = DeepFeatureNetTrainer(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )

    print("[INFO] Début de l'entraînement...")
    trainer.train(epochs=args.epochs, batch_size=args.batch_size)
    print("[INFO] Entraînement terminé.")

if __name__ == "__main__":
    main()



