import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import argparse

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Affiche la matrice de confusion.
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap=cmap,
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Résumé des performances du modèle")
    parser.add_argument("--y_test", type=str, default="y_test.npy",
                        help="Chemin vers y_test.npy")
    parser.add_argument("--y_pred", type=str, default="y_pred.npy",
                        help="Chemin vers y_pred.npy")
    args = parser.parse_args()

    # Charger les labels
    y_test = np.load(args.y_test)
    y_pred = np.load(args.y_pred)

    # Liste des classes (adapter si nécessaire)
    classes = [0, 1, 2, 3, 4]  # Exemple pour 5 classes

    # Classification report
    print("[INFO] Classification Report :")
    report = classification_report(y_test, y_pred, target_names=[str(c) for c in classes])
    print(report)

    # Matrice de confusion
    print("[INFO] Matrice de confusion :")
    plot_confusion_matrix(y_test, y_pred, classes, normalize=True, title="Matrice de confusion normalisée")

if __name__ == "__main__":
    main()




