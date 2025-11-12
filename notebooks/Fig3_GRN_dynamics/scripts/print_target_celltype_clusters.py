#!/usr/bin/env python
"""Print detailed info about clusters in target developmental celltypes"""

import pandas as pd

figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/sub_GRNs_reg_programs/"
ranking_csv = figpath + "cluster_ranking_temporal_dynamics.csv"

df_ranked = pd.read_csv(ranking_csv)

target_celltypes = ['PSM', 'NMPs', 'tail_bud', 'neural_posterior', 'spinal_cord',
                   'somite', 'neural_floor_plate', 'notochord']

df_target = df_ranked[df_ranked['celltype'].isin(target_celltypes)]
df_target = df_target.sort_values('dynamics_score', ascending=False)

print("="*100)
print("CLUSTERS IN DEVELOPMENTAL CELLTYPES (sorted by dynamics score)")
print("="*100)

for idx, row in df_target.iterrows():
    print(f"\n{'='*100}")
    print(f"Rank in all data: #{list(df_ranked.index).index(idx) + 1}")
    print(f"Cluster: {row['cluster_id']} | Cell type: {row['celltype']} | Dynamics score: {row['dynamics_score']:.3f}")
    print(f"Network: {row['n_total_nodes']} nodes, {row['n_total_edges']} edges | {row['n_developmental_tfs']} dev TFs")
    print(f"Developmental TFs: {', '.join(eval(row['developmental_tfs_list']))}")
    print(f"Edge counts per timepoint: {row['n_timepoints_with_edges']}/{row['n_timepoints']} timepoints with ≥10 edges")

print(f"\n{'='*100}")
print("SUMMARY BY CELLTYPE:")
print(f"{'='*100}")

for celltype in target_celltypes:
    ct_data = df_target[df_target['celltype'] == celltype]
    if len(ct_data) > 0:
        all_tfs = set()
        for tfs in ct_data['developmental_tfs_list']:
            all_tfs.update(eval(tfs))
        print(f"\n{celltype.upper()}: {len(ct_data)} clusters")
        print(f"  Score range: {ct_data['dynamics_score'].min():.3f} - {ct_data['dynamics_score'].max():.3f}")
        print(f"  Total edges range: {ct_data['n_total_edges'].min():.0f} - {ct_data['n_total_edges'].max():.0f}")
        print(f"  Developmental TFs present: {', '.join(sorted(all_tfs))}")
        print(f"  Top cluster: {ct_data.iloc[0]['cluster_id']}")
