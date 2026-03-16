#!/bin/bash

# Arguments
data=$1           # Répertoire contenant les données
output=$2         # Dossier de sortie
start=$3          # Premier fold
end=$4            # Dernier fold
gpu=$5            # ID GPU
epochs_pretrain=$6  # Nombre d'époques pour le pré-entraînement
epochs_finetune=$7  # Nombre d'époques pour le fine-tuning
batch_size=$8       # Taille de batch

if [[ -n "$data" && -n "$start" && -n "$end" && -n "$gpu" ]]; then
    for i in $(eval echo {$start..$end})
    do
        echo "=============================================="
        echo " Début du pré-entraînement pour le fold $i"
        echo "=============================================="

        CUDA_VISIBLE_DEVICES=$gpu python train.py \
            --data_dir=$data \
            --output_dir=${output}/fold_${i}/pretrain \
            --epochs=$epochs_pretrain \
            --batch_size=$batch_size \
            --mode pretrain

        echo " Pré-entraînement terminé pour le fold $i"
        echo "----------------------------------------------"
        echo " Début du fine-tuning pour le fold $i"
        echo "----------------------------------------------"

        pretrained_model=${output}/fold_${i}/pretrain/best_model.h5

        CUDA_VISIBLE_DEVICES=$gpu python train.py \
            --data_dir=$data \
            --output_dir=${output}/fold_${i}/finetune \
            --epochs=$epochs_finetune \
            --batch_size=$batch_size \
            --pretrained_model=$pretrained_model \
            --mode finetune

        echo " Fine-tuning terminé pour le fold $i"
        echo "----------------------------------------------"
    done
else
    echo " Argument error!"
    echo "Usage: bash batch_train.sh <data_dir> <output_dir> <start_fold> <end_fold> <gpu> <epochs_pretrain> <epochs_finetune> <batch_size>"
fi

