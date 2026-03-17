# DeepSleepNet

![DeepSleepNet](DeepSleepNet/figure/DeepSleepNet.png)
# Classification des stades du sommeil avec DeepFeatureNet (DeepSleepNet)

## Présentation

Ce projet implémente un système de **classification automatique des stades du sommeil à partir de signaux EEG** en utilisant une architecture de deep learning basée sur **DeepFeatureNet**, composant principal du modèle **DeepSleepNet**.

L'objectif est de classifier les segments EEG en **5 stades de sommeil** :

* **W** : Éveil
* **N1** : Sommeil léger (stade 1)
* **N2** : Sommeil léger (stade 2)
* **N3** : Sommeil profond
* **REM** : Sommeil paradoxal

Le système utilise des **réseaux de neurones convolutifs (CNN)** pour extraire automatiquement les caractéristiques importantes des signaux EEG.

---

# Dataset

Les expériences sont réalisées sur le **dataset Sleep-EDF**.

Chaque signal EEG est découpé en **epochs de 30 secondes**, chacune associée à un label correspondant au stade de sommeil.

Les données sont stockées sous forme de fichiers **NPZ** contenant :

* `x` : les signaux EEG
* `y` : les labels des stades de sommeil

Dossier des données :

```id="pco9is"
merged_npz_all/
```

---

# Préparation des données

Les enregistrements EEG sont fusionnés et convertis en fichiers NPZ grâce au script :

```id="tuxy1e"
merge.py
```

Ce script permet :

* de charger les signaux EEG
* de segmenter les données en epochs
* de sauvegarder les données dans un format adapté à l'entraînement.

---

# Architecture du modèle

Le modèle **DeepFeatureNet** est une architecture CNN composée de plusieurs couches convolutives permettant d’extraire des caractéristiques temporelles des signaux EEG.

Structure du modèle :

### Bloc 1

* Conv1D (64 filtres, kernel size = 50, stride = 6)
* Batch Normalization
* MaxPooling

### Bloc 2

* Conv1D (128 filtres)
* Batch Normalization
* MaxPooling

### Bloc 3

* Conv1D (128 filtres)
* Batch Normalization
* MaxPooling

### Classification

* Flatten
* Dense (512 neurones)
* Dropout (0.5)
* Dense (256 neurones)
* Dropout (0.5)
* Dense (Softmax – 5 classes)

Cette architecture permet d'extraire des **représentations profondes des signaux EEG** pour améliorer la classification des stades du sommeil.

---

# Installation

Bibliothèques nécessaires :

```id="p0m8a0"
tensorflow
numpy
matplotlib
seaborn
pandas
scikit-learn
```

Installation :

```bash id="52p4m8"
pip install tensorflow numpy matplotlib seaborn pandas scikit-learn
```

---

# Entraînement du modèle

L'entraînement est réalisé avec le script :

```id="s6rxhm"
train.py
```

Commande utilisée :

```bash id="w53ruy"
python train.py \
  --data_dir /content/drive/MyDrive/DeepSleepNet/PolySomnoGraphy/data/Sleep-EDF/merged_npz_all \
  --output_dir /content/drive/MyDrive/Old_Deep_Sleep/outputs_pretrain \
  --epochs 80 \
  --batch_size 64
```

Paramètres principaux :

| Paramètre         | Valeur |
| ----------------- | ------ |
| Epochs            | 80     |
| Batch size        | 64     |
| Nombre de classes | 5      |

Les meilleurs poids du modèle sont sauvegardés dans :

```id="ls82ed"
outputs_pretrain/
```

---

# Évaluation du modèle

L'évaluation du modèle est réalisée avec le script :

```id="tjy91c"
evaluate1.py
```

Commande :

```bash id="aygqk2"
python evaluate1.py
```

Ce script permet de :

* charger le modèle entraîné
* générer les prédictions
* calculer les métriques de classification.

---

# Métriques utilisées

Les métriques suivantes sont calculées :

* Accuracy
* Precision
* Recall
* F1-score

Deux tableaux de performances sont générés :

### Performances globales

* Macro moyenne
* Moyenne pondérée
* Accuracy globale

### Performances par classe

* précision par stade
* rappel par stade
* F1-score par stade

---

# Matrice de confusion

Une **matrice de confusion** est générée afin d’analyser les performances du modèle pour chaque stade de sommeil :

* W
* N1
* N2
* N3
* REM

Cette matrice permet d'identifier les erreurs de classification entre les différents stades.

---

# Visualisation des performances

Les performances d'entraînement sont visualisées avec le script :

```id="ukqk7e"
plot.py
```

Commande :

```bash id="yy1mfd"
python plot.py
```

Ce script génère :

### Courbe de loss

* training loss
* validation loss

### Courbe d'accuracy

* training accuracy
* validation accuracy

Les figures sont sauvegardées dans :

```id="1rt29s"
plots_deepsleepnet/
 ├── loss_curve.png
 └── accuracy_curve.png
```

Ces courbes permettent d’analyser l’évolution de l’apprentissage du modèle.

---

# Structure du projet

```id="eewh92"
DeepSleepNet_Project/

│
├── merge.py
├── train.py
├── evaluate1.py
├── plot.py
│
├── merged_npz_all/
│
├── outputs_pretrain/
│
└── plots_deepsleepnet/
```

---

# Résultats

Le projet produit :

* les prédictions du modèle
* la matrice de confusion
* les métriques de classification
* les courbes Accuracy et Loss

Ces résultats permettent d’évaluer les performances du modèle CNN pour la classification des stades du sommeil à partir des signaux EEG.

---

