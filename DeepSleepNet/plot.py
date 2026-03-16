import os
import matplotlib.pyplot as plt

# =========
# Données extraites de ton log (Epoch 1 → 18)
# Modèle renommé: DeepSleepNet (au lieu de DeepFeatureNet)
# =========
epochs = list(range(1, 19))

train_acc = [
    0.5802, 0.7199, 0.7439, 0.7573, 0.7632, 0.7694,
    0.7756, 0.7801, 0.7859, 0.7909, 0.7919, 0.8002,
    0.8049, 0.8100, 0.8151, 0.8222, 0.8223, 0.8267
]

train_loss = [
    1.0965, 0.7213, 0.6573, 0.6305, 0.6130, 0.5903,
    0.5800, 0.5629, 0.5516, 0.5388, 0.5311, 0.5129,
    0.5044, 0.4801, 0.4681, 0.4532, 0.4478, 0.4364
]

val_acc = [
    0.6978, 0.7316, 0.7262, 0.6791, 0.6915, 0.7128,
    0.7351, 0.7491, 0.6828, 0.6829, 0.7034, 0.7054,
    0.7105, 0.7318, 0.7227, 0.6795, 0.7216, 0.7422
]

val_loss = [
    0.7914, 0.6969, 0.7072, 0.7994, 0.7964, 0.7112,
    0.6758, 0.6433, 0.8105, 0.7863, 0.7470, 0.7333,
    0.7337, 0.6854, 0.7208, 0.8313, 0.7370, 0.6785
]

# =========
# Paramètres d'annotation (d'après le log)
# =========
best_epoch = 8  # "Restoring model weights from the end of the best epoch: 8."

best_i = epochs.index(best_epoch)

# =========
# Dossier de sortie
# =========
out_dir = "plots_deepsleepnet"
os.makedirs(out_dir, exist_ok=True)

# =========
# Plot Loss
# =========
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_loss, label="train_loss")
plt.plot(epochs, val_loss, label="val_loss")

plt.scatter([best_epoch], [val_loss[best_i]], s=60)
plt.text(best_epoch + 0.2, val_loss[best_i], f"best epoch {best_epoch}", va="center")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("DeepSleepNet - Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=200)
plt.show()

# =========
# Plot Accuracy
# =========
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_acc, label="train_accuracy")
plt.plot(epochs, val_acc, label="val_accuracy")

plt.scatter([best_epoch], [val_acc[best_i]], s=60)
plt.text(best_epoch + 0.2, val_acc[best_i], f"best epoch {best_epoch}", va="center")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("DeepSleepNet - Accuracy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "accuracy_curve.png"), dpi=200)
plt.show()

# =========
# Résumé best epoch
# =========
print(f"Best epoch (from log) = {best_epoch}")
print(f"val_accuracy at best epoch = {val_acc[best_i]:.4f}")
print(f"val_loss at best epoch = {val_loss[best_i]:.4f}")
print(f"Saved figures to: {os.path.abspath(out_dir)}")
print("Files:", "loss_curve.png", "accuracy_curve.png")
