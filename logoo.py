import os
import numpy as np
import pandas as pd
from Bio import SeqIO
import logomaker
import matplotlib.pyplot as plt

# ---------- SETTINGS ----------
DATA_DIR = "../data/"
OUTPUT_LOGO_DIR = "../output/logos/"
OUTPUT_MATRIX_DIR = "../output/matrices/"
os.makedirs(OUTPUT_LOGO_DIR, exist_ok=True)
os.makedirs(OUTPUT_MATRIX_DIR, exist_ok=True)

FILES = {
    "species1_groupA": "species1_groupA.fasta",
    "species1_groupB": "species1_groupB.fasta",
    "species2_groupA": "species2_groupA.fasta",
    "species2_groupB": "species2_groupB.fasta",
}

# ---------- FUNCTION: LOAD FASTA ----------
def load_sequences(filepath):
    return [str(rec.seq) for rec in SeqIO.parse(filepath, "fasta")]

# ---------- FUNCTION: BUILD PWM ----------
def build_pwm(sequences, pseudocount=1):
    bases = "ACGT"
    length = len(sequences[0])
    counts = pd.DataFrame(0, index=bases, columns=range(length))

    for seq in sequences:
        for i, base in enumerate(seq):
            counts.loc[base, i] += 1

    pwm = (counts + pseudocount) / (counts.sum(axis=0) + 4*pseudocount)
    return pwm

# ---------- FUNCTION: SAVE LOGO ----------
def save_logo(pwm, name):
    logo = logomaker.Logo(pwm.T, shade_below=.5, fade_below=.5)
    plt.title(name)
    plt.savefig(f"{OUTPUT_LOGO_DIR}{name}.png", dpi=300)
    plt.close()

# ---------- MAIN ----------
pwms = {}
for label, fname in FILES.items():
    seqs = load_sequences(os.path.join(DATA_DIR, fname))
    pwm = build_pwm(seqs)
    pwms[label] = pwm
    save_logo(pwm, label)

# ---------- COMPARE PWMs (Euclidean Distance) ----------
labels = list(pwms.keys())
matrix = np.zeros((len(labels), len(labels)))

for i, a in enumerate(labels):
    for j, b in enumerate(labels):
        matrix[i, j] = np.linalg.norm(pwms[a] - pwms[b])

distance_df = pd.DataFrame(matrix, index=labels, columns=labels)
distance_df.to_csv(f"{OUTPUT_MATRIX_DIR}pwm_distance_matrix.csv")

print("✅ Analysis complete.")
print("Logos saved in output/logos/")
print("Distance matrix saved in output/matrices/")
