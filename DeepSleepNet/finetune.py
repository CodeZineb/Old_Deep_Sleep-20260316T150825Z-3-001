import argparse
import os
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from deepsleep.data_loader import get_train_test_data
from deepsleep.model import DeepFeatureNet


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning du modèle DeepFeatureNet")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Chemin vers le dossier contenant les fichiers .npz pour le fine-tuning"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Dossier de sortie pour sauvegarder les modèles et logs"
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        required=True,
        help="Chemin vers le modèle pré-entraîné (.h5)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Nombre d'époques pour le fine-tuning"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Taille du batch"
    )

    args = parser.parse_args()

    print("[INFO] Chargement des données de fine-tuning...")
    X_train, X_test, y_train, y_test = get_train_test_data(args.data_dir)

    # One-hot encoding
    num_classes = len(np.unique(y_train))
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    print(f"[INFO] Données : {X_train.shape[0]} train / {X_test.shape[0]} test")
    print(f"[INFO] Nombre de classes : {num_classes}")

    # Ajouter la dimension canal
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    input_shape = (X_train.shape[1], 1)

    # --- Initialisation du modèle ---
    print("[INFO] Initialisation du modèle DeepFeatureNet pour fine-tuning...")
    model = DeepFeatureNet(input_shape=input_shape, num_classes=num_classes)

    # 🔧 Construction explicite du modèle AVANT le chargement des poids
    model.build(input_shape=(None, X_train.shape[1], 1))

    # Charger les poids du pré-entraînement
    print(f"[INFO] Chargement du modèle pré-entraîné depuis {args.pretrained_model}...")
    # Construire le modèle avant de charger les poids
    dummy_input = np.zeros((1, 3000, 1))  # adapte à ton input_shape
    model(dummy_input)

    model.load_weights(args.pretrained_model)
    print("[INFO] Poids du modèle chargés avec succès !")

    # Compiler le modèle
    model.compile(
        optimizer=Adam(learning_rate=1e-5),  # LR plus bas pour fine-tuning
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # --- Callbacks ---
    checkpoint_path = os.path.join(args.output_dir, "best_model_finetuned.h5")
    os.makedirs(args.output_dir, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
    ]

    print("\n[INFO] Démarrage du fine-tuning...\n")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    print(f"\n[INFO] Fine-tuning terminé. Meilleur modèle sauvegardé : {checkpoint_path}")


if __name__ == "__main__":
    main()


