# %%
import scanpy as sc
from tqdm.auto import tqdm
from chromadata import seq_qc
from chromadata.seq_dataset import SEQ_DS
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

# %%
all_samples = {
    key: seq_qc.unpack_seq_dataset(SEQ_DS[key])
    for key in tqdm(SEQ_DS.keys(), desc="Unpacking samples")
}
passed_samples = {}
# %%
args_122 = {
    "min_genes": 400,
    "mito_keys": ["mt-", "NC_"],
    "mito_percent": 10,
    "title_str": "TDR122",
}
passed_samples["TDR122"] = seq_qc.preprocess_stack(all_samples["TDR122"], args_122)

# %%
args_123 = {
    "min_genes": 300,
    "mito_keys": ["mt-", "NC_"],
    "mito_percent": 10,
    "title_str": "TDR123",
}
passed_samples["TDR123"] = seq_qc.preprocess_stack(all_samples["TDR123"], args_123)
# %%
args_130 = {
    "min_genes": 400,
    "mito_keys": ["mt-", "NC_"],
    "mito_percent": 10,
    "title_str": "TDR130",
}
passed_samples["TDR130"] = seq_qc.preprocess_stack(all_samples["TDR130"], args_130)
# %%
args_131 = {
    "min_genes": 100,
    "mito_keys": ["mt-", "NC_"],
    "mito_percent": 10,
    "title_str": "TDR131",
}
passed_samples["TDR131"] = seq_qc.preprocess_stack(all_samples["TDR131"], args_131)
# %%
args_132 = {
    "min_genes": 300,
    "mito_keys": ["mt-", "NC_"],
    "mito_percent": 10,
    "title_str": "TDR132",
}
passed_samples["TDR132"] = seq_qc.preprocess_stack(all_samples["TDR132"], args_132)
# %%
args_133 = {
    "min_genes": 150,
    "mito_keys": ["mt-", "NC_"],
    "mito_percent": 10,
    "title_str": "TDR133",
}
passed_samples["TDR133"] = seq_qc.preprocess_stack(all_samples["TDR133"], args_133)
# %%
args_134 = {
    "min_genes": 400,
    "mito_keys": ["mt-", "NC_"],
    "mito_percent": 10,
    "title_str": "TDR134",
}
passed_samples["TDR134"] = seq_qc.preprocess_stack(all_samples["TDR134"], args_134)
# %%
args_135 = {
    "min_genes": 300,
    "mito_keys": ["mt-", "NC_"],
    "mito_percent": 10,
    "title_str": "TDR135",
}
passed_samples["TDR135"] = seq_qc.preprocess_stack(all_samples["TDR135"], args_135)
# %%
args_136 = {
    "min_genes": 300,
    "mito_keys": ["mt-", "NC_"],
    "mito_percent": 10,
    "title_str": "TDR136",
}
passed_samples["TDR136"] = seq_qc.preprocess_stack(all_samples["TDR136"], args_136)
# %%
args_137 = {
    "min_genes": 400,
    "mito_keys": ["mt-", "NC_"],
    "mito_percent": 10,
    "title_str": "TDR137",
}
passed_samples["TDR137"] = seq_qc.preprocess_stack(all_samples["TDR137"], args_137)
# %%
args_138 = {
    "min_genes": 300,
    "mito_keys": ["mt-", "NC_"],
    "mito_percent": 10,
    "title_str": "TDR138",
}
passed_samples["TDR138"] = seq_qc.preprocess_stack(all_samples["TDR138"], args_138)
# %%
args_139 = {
    "min_genes": 300,
    "mito_keys": ["mt-", "NC_"],
    "mito_percent": 10,
    "title_str": "TDR139",
}
passed_samples["TDR139"] = seq_qc.preprocess_stack(all_samples["TDR139"], args_139)
# %%
args_140 = {
    "min_genes": 300,
    "mito_keys": ["mt-", "NC_"],
    "mito_percent": 10,
    "title_str": "TDR140",
}
passed_samples["TDR140"] = seq_qc.preprocess_stack(all_samples["TDR140"], args_140)

# %%
for key in tqdm(passed_samples, desc="filtering dropout genes"):
    sc.pp.filter_genes(passed_samples[key], min_counts=10)
# %%
# Run scrublet (not recommended without GPU)
for key in tqdm(
    passed_samples.keys(),
    desc="running scrublet, this may take a while if no GPU is available",
):
    passed_samples[key] = seq_qc.run_scrublet(passed_samples[key])
# %%
for key in tqdm(passed_samples.keys(), desc="concatenating replicates + normalizing"):
    passed_samples[key] = seq_qc.normalize_adata(passed_samples[key])
# %%
for key in tqdm(passed_samples, desc="postprocessing samples"):
    passed_samples[key] = seq_qc.postprocess_stack(passed_samples[key])
# %%
for key in tqdm(passed_samples, desc="saving samples"):
    passed_samples[key].write(f"outputs/uniform/{key}.h5ad")

# %%
