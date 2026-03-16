#!/bin/bash

# Chemins
DATA_DIR="/content/drive/MyDrive/Old_Deep_Sleep/data"
PRETRAINED_MODEL="/content/drive/MyDrive/Old_Deep_Sleep/outputs_pretrain/best_model.h5"
OUTPUT_DIR="/content/drive/MyDrive/Old_Deep_Sleep/outputs_finetune"

# Nombre de folds
N_SPLITS=5

# Créer le dossier de sortie s'il n'existe pas
mkdir -p "$OUTPUT_DIR"

# Boucle sur chaque fold
for FOLD in $(seq 0 $((N_SPLITS-1)))
do
    echo "--------------------------------------"
    echo "Fine-tuning Fold $FOLD..."
    python3 finetune.py \
        --data_dir "$DATA_DIR" \
        --pretrained_model "$PRETRAINED_MODEL" \
        --n_splits $N_SPLITS
    echo "Fold $FOLD terminé."
done

echo "--------------------------------------"
echo "Fine-tuning terminé pour tous les folds."



