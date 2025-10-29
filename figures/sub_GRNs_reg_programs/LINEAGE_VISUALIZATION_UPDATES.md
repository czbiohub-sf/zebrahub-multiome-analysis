# Lineage Dynamics Visualization Updates

## Changes Made

### 1. Unified Coordinate System
**Previous approach:** Each subGRN had its own independent layout, making it difficult to compare nodes across different cell types.

**New approach:** 
- Compute a **single master layout** using ALL nodes and edges from BOTH neural and mesodermal lineages
- All 8 subGRNs (3 neural + 5 mesoderm) use the SAME coordinate system
- Each node appears in the SAME position across all panels
- This makes lineage divergence patterns much clearer visually

**Technical implementation:**
```python
# Collect ALL nodes and edges from BOTH lineages
all_nodes = set()
all_edges = set()
for subgrn in list(neural_subgrns.values()) + list(mesoderm_subgrns.values()):
    all_nodes.update(subgrn['source'])
    all_nodes.update(subgrn['target'])
    for _, row in subgrn.iterrows():
        all_edges.add((row['source'], row['target']))

# Create master graph and compute unified layout
master_G = nx.DiGraph()
master_G.add_edges_from(all_edges)
master_pos = nx.spring_layout(master_G, k=1.8, scale=1.8, iterations=100, seed=42)

# Use master_pos for ALL panels
```

### 2. Consistent Color Scheme
**Previous approach:** Colored nodes based on lineage-specific TF lists (neural TFs blue, mesoderm TFs purple, etc.)

**New approach:** Match the color scheme from temporal dynamics analysis:
- **lightcoral (reddish-pink):** TF-only nodes (nodes that act as transcription factors but are never targets)
- **lightblue:** Target-only nodes (genes that are regulated but never act as TFs)
- **plum (light purple):** TF & Target nodes (dual role - both regulate and are regulated)

**Benefits:**
- Consistent with previous temporal analysis figures
- Biologically meaningful classification
- Easy to identify regulatory roles at a glance
- Unified color language across all figures in manuscript

### 3. Node Classification
Nodes are now classified based on their **functional role across ALL lineages**, not per-celltype:

```python
# Collect sources and targets from ALL subGRNs
all_sources = set()  # All nodes that act as TFs
all_targets = set()  # All nodes that are targets

for subgrn in list(neural_subgrns.values()) + list(mesoderm_subgrns.values()):
    all_sources.update(subgrn['source'])
    all_targets.update(subgrn['target'])

# Classify globally
tf_only_nodes = all_sources - all_targets
target_only_nodes = all_targets - all_sources
tf_target_nodes = all_sources & all_targets
```

This means:
- A node keeps the same color across all panels
- Color reflects its "global" role across the entire lineage comparison
- Makes it easy to track individual nodes across cell types

## Visualization Features

### Layout Structure
```
┌─────────────────────────────────────────────────────────┐
│  NEURAL LINEAGE (top row)                                │
│  neural_posterior | spinal_cord | NMPs                   │
├─────────────────────────────────────────────────────────┤
│  MESODERM LINEAGE (bottom row)                           │
│  NMPs | tail_bud | PSM | somites | fast_muscle          │
└─────────────────────────────────────────────────────────┘
```

### Visual Elements
- **Node size:** 
  - TF-only: 300 (largest)
  - TF & Target: 250 (medium)
  - Target-only: 200 (smallest)
- **Edge thickness:** Proportional to regulatory strength (|coef_mean| × 3)
- **Edge color:** Gray with 50% opacity
- **Edge style:** Directed arrows with slight arc (arc3,rad=0.1)
- **Labels:** All nodes labeled (font size 8)

### Legend
Updated legend shows:
- lightcoral: "TF only"
- plum: "TF & Target"
- lightblue: "Target only"

## Advantages of New Approach

### 1. Easier Pattern Recognition
With unified coordinates, you can immediately see:
- Which nodes appear/disappear across lineages
- Which regulatory connections strengthen/weaken
- Lineage-specific vs shared regulatory programs

### 2. Direct Visual Comparison
Same node position across panels means:
- No mental remapping needed
- Direct visual comparison of network topology
- Clear visualization of divergence patterns

### 3. Consistent with Previous Figures
Using the same color scheme as temporal dynamics:
- Unified visual language across manuscript
- Readers can apply same interpretation rules
- Professional consistency

## Example Interpretations

### Cluster 3_2 (hindbrain, neural-biased)
- **Neural lineage:** Network weakens (29→26→10 edges)
- **Mesoderm lineage:** Network strengthens (10→17 edges)
- **Key insight:** Regulatory program is progressively activated in mesoderm while being shut down in neural cells

### Cluster 22_16 (muscle, mesoderm-biased)
- **Neural lineage:** Network strengthens (10→17→18 edges)
- **Mesoderm lineage:** Network weakens (18→35→29→20→14 edges)
- **Key insight:** Early activation in mesoderm followed by progressive refinement/pruning

### Color Patterns
- **Many lightcoral nodes:** TF-heavy network (regulatory cascade)
- **Many lightblue nodes:** Target-rich network (coordinated regulation)
- **Many plum nodes:** Hierarchical network (regulatory feedback loops)

## Files Updated

1. **visualize_lineage_dynamics.py:**
   - Removed lineage-specific TF coloring
   - Implemented unified coordinate system
   - Updated to use TF/Target/Dual classification
   - Added detailed progress messages

2. **Generated figures:** All 10 PNG and PDF files regenerated with:
   - Unified layouts (25-47 nodes per master graph)
   - Consistent colors across panels
   - Improved visual clarity

## Technical Notes

- Master layout computed with spring_layout (k=1.8, scale=1.8, iterations=100, seed=42)
- Fixed seed ensures reproducibility
- All 8 (or 7) subGRNs contribute to master layout equally
- Edge weights preserved from original GRN (coef_mean column)

## Comparison to Initial Version

| Feature | Initial Version | Updated Version |
|---------|----------------|-----------------|
| Coordinate system | Independent per panel | Unified across all panels |
| Node colors | Lineage-specific TFs | Functional role (TF/Target/Dual) |
| Color consistency | New color scheme | Matches temporal dynamics |
| Visual comparison | Difficult | Easy and direct |
| Interpretability | Requires mental mapping | Immediate visual patterns |

## Next Steps

1. **Review visualizations** to confirm divergence patterns are clear
2. **Select top 5 candidates** for manuscript main figures
3. **Write figure captions** explaining lineage divergence patterns
4. **Literature validation** of observed regulatory programs
