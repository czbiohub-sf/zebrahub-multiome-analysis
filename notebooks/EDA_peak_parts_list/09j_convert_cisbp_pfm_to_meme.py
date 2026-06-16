# %% Convert CIS-BP v2 Danio rerio PFM → MEME v4 format
#
# Input:
#   /hpc/mydata/yang-joon.kim/genomes/danRer11/CisBP_ver2_Danio_rerio.pfm
#   /hpc/mydata/yang-joon.kim/genomes/danRer11/CisBP_ver2_Danio_rerio.motif2factors.txt
# Output:
#   /hpc/scratch/group.data.science/yang-joon.kim/peak-parts-list-motifs/motif_dbs/cisbpv2_danrer.meme
#
# MEME line per motif: `MOTIF <motif_id> <primary_TF>`
# — primary TF pulled from motif2factors (first factor per motif).
# Motifs with no factor mapping keep the motif_id as their TF name.
#
# Run: /hpc/user_apps/data.science/conda_envs/single-cell-base/bin/python 09j_convert_cisbp_pfm_to_meme.py

import os, sys, time

PFM_PATH      = "/hpc/mydata/yang-joon.kim/genomes/danRer11/CisBP_ver2_Danio_rerio.pfm"
M2F_PATH      = "/hpc/mydata/yang-joon.kim/genomes/danRer11/CisBP_ver2_Danio_rerio.motif2factors.txt"
OUT_DIR       = "/hpc/scratch/group.data.science/yang-joon.kim/peak-parts-list-motifs/motif_dbs"
OUT_MEME      = f"{OUT_DIR}/cisbpv2_danrer.meme"

os.makedirs(OUT_DIR, exist_ok=True)

print("=== CIS-BP v2 PFM → MEME converter ===")
print(f"Start: {time.strftime('%c')}")

# --- Parse motif2factors.txt → motif_id -> first TF name ---
motif_to_tf = {}
with open(M2F_PATH) as f:
    header = f.readline()  # skip header line
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 2:
            continue
        motif_id, factor = parts[0], parts[1]
        if not factor or factor == ".":
            continue
        # Keep the first TF encountered for each motif (alphabetical order is not guaranteed)
        if motif_id not in motif_to_tf:
            motif_to_tf[motif_id] = factor

print(f"Parsed {len(motif_to_tf)} motif→TF mappings from motif2factors.txt")

# --- Parse PFM: list of (motif_id, matrix) tuples ---
motifs = []
cur_id = None
cur_rows = []
with open(PFM_PATH) as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        if line.startswith(">"):
            if cur_id is not None:
                motifs.append((cur_id, cur_rows))
            cur_id = line[1:].strip()
            cur_rows = []
        else:
            vals = line.split("\t")
            if len(vals) == 4:
                cur_rows.append([float(v) for v in vals])
    if cur_id is not None:
        motifs.append((cur_id, cur_rows))

print(f"Parsed {len(motifs)} motifs from PFM")

# --- Emit MEME v4 ---
n_mapped = 0
n_unmapped = 0
with open(OUT_MEME, "w") as out:
    out.write("MEME version 4\n\n")
    out.write("ALPHABET= ACGT\n\n")
    out.write("strands: + -\n\n")
    out.write("Background letter frequencies\n")
    out.write("A 0.25 C 0.25 G 0.25 T 0.25\n\n")

    for motif_id, rows in motifs:
        if not rows:
            continue
        w = len(rows)
        tf = motif_to_tf.get(motif_id, "")
        if tf:
            out.write(f"MOTIF {motif_id} {tf}\n")
            n_mapped += 1
        else:
            # Fallback: motif_id as both accession and placeholder TF name
            out.write(f"MOTIF {motif_id} {motif_id}\n")
            n_unmapped += 1
        out.write(f"letter-probability matrix: alength= 4 w= {w} nsites= 1 E= 0\n")
        for row in rows:
            # Re-normalize rows to exactly 1.0 (guard against floating-point drift)
            s = sum(row)
            if s <= 0:
                row = [0.25] * 4
            else:
                row = [v / s for v in row]
            out.write("  " + "  ".join(f"{v:.6f}" for v in row) + "\n")
        out.write("\n")

print(f"\nWrote {OUT_MEME}")
print(f"  Total motifs:     {len(motifs)}")
print(f"  With TF mapping:  {n_mapped}")
print(f"  Without mapping:  {n_unmapped}")
print(f"  Unique TFs:       {len(set(motif_to_tf.values()))}")
print(f"End: {time.strftime('%c')}")
