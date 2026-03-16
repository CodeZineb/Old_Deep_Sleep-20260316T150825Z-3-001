import os
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from deepsleep.data_loader import get_train_test_data
from deepsleep.model import DeepFeatureNet

class DeepFeatureNetTrainer:
    """
    Classe d'entraînement pour le modèle DeepFeatureNet (pré-entraînement).
    """

    def __init__(self, data_dir, output_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        print("[INFO] Chargement des données...")
        self.X_train, self.X_test, self.y_train, self.y_test = get_train_test_data(data_dir)

        # Mise en forme des labels en one-hot
        num_classes = len(np.unique(self.y_train))
        self.y_train = to_categorical(self.y_train, num_classes)
        self.y_test = to_categorical(self.y_test, num_classes)

        print(f"[INFO] Données chargées : {self.X_train.shape[0]} train / {self.X_test.shape[0]} test")
        print(f"[INFO] Nombre de classes : {num_classes}")

        # Adapter la forme d'entrée (ajouter la dimension canal)
        input_shape = (self.X_train.shape[1], 1)
        self.X_train = np.expand_dims(self.X_train, axis=-1)
        self.X_test = np.expand_dims(self.X_test, axis=-1)

        # Initialisation du modèle
        self.model = DeepFeatureNet(input_shape=input_shape, num_classes=num_classes)
        self.model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

    def train(self, epochs=80, batch_size=64):
        """
        Entraîne le modèle DeepFeatureNet et sauvegarde le meilleur modèle.
        """
        checkpoint_path = os.path.join(self.output_dir, "best_model.h5")
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

        print("\n[INFO] Démarrage de l'entraînement du modèle DeepFeatureNet...\n")

        history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_test, self.y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        print(f"\n[INFO] Entraînement terminé. Meilleur modèle sauvegardé : {checkpoint_path}")
        return history











