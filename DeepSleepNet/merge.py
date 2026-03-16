# === chemins ===
BASE = "/content/drive/MyDrive/DeepSleepNet/PolySomnoGraphy/data/Sleep-EDF"
OUT  = BASE + "/merged_npz_all"

import os, glob, re, numpy as np
os.makedirs(OUT, exist_ok=True)

def subj_key(path):
    m = re.search(r'(SC|ST)\d{4}(E0|J0)', os.path.basename(path))
    return m.group(0) if m else None

# Récolte des fichiers
fpz = {}
for d in ["cassette/EEG_Fpz-Cz", "telemetry/EEG_Fpz-Cz"]:
    fpz.update({subj_key(p): p for p in glob.glob(os.path.join(BASE, d, "*.npz")) if subj_key(p)})

pz = {}
for d in ["cassette/EEG_Pz-Oz", "telemetry/EEG_Pz-Oz"]:
    pz.update({subj_key(p): p for p in glob.glob(os.path.join(BASE, d, "*.npz")) if subj_key(p)})

# Trouver les paires correspondantes
pairs = sorted(set(fpz) & set(pz))
print(f"Total sujets trouvés: {len(pairs)}")

# Fusion des paires en moyenne
w = 0
for k in pairs:
    with np.load(fpz[k]) as A, np.load(pz[k]) as B:
        if A["x"].shape != B["x"].shape or A["y"].shape != B["y"].shape:
            print("[SKIP] shape mismatch", k, A["x"].shape, B["x"].shape); continue
        if int(A["fs"]) != int(B["fs"]):
            print("[SKIP] fs mismatch", k, A["fs"], B["fs"]); continue

        x = (A["x"].astype(np.float32) + B["x"].astype(np.float32)) / 2.0
        y = A["y"].astype(np.int32)
        fs = int(A["fs"])

    out_path = os.path.join(OUT, k + ".npz")
    np.savez(out_path, x=x, y=y, fs=fs)
    w += 1

print(f"{w} fichiers générés dans: {OUT}")
