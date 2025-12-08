# LiteMind Peak Cluster Analysis

LLM-based biological interpretation of chromatin accessibility peak clusters using the LiteMind framework.

## Overview

This module provides automated biological interpretation of peak UMAP clusters from the zebrahub-multiome atlas. It uses large language models (via OpenAI or Anthropic) to:

- Analyze pseudobulk chromatin accessibility patterns
- Interpret gene associations and motif enrichments
- Query biological databases (Ensembl, ZFIN, PubMed, Alliance, etc.)
- Generate structured markdown reports with citations
- Optional peer review and revision workflow

## Quick Start

### 1. Install Dependencies

```bash
pip install litemind==2025.7.26 openai==1.96.1 backoff requests-cache ratelimit arbol tabulate
```

### 2. Set API Key

```bash
# Option 1: OpenAI (default)
export OPENAI_API_KEY="sk-your-key-here"

# Option 2: Anthropic
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
```

### 3. Run Analysis

```bash
# Analyze specific coarse clusters
python scripts/litemind_peak_analysis/main.py --coarse-clusters 0,1,2

# Analyze all coarse clusters
python scripts/litemind_peak_analysis/main.py --all-coarse

# Analyze specific fine clusters
python scripts/litemind_peak_analysis/main.py --fine-clusters 0_0,0_1

# Use Anthropic instead of OpenAI
python scripts/litemind_peak_analysis/main.py --coarse-clusters 0 --api anthropic

# Skip review/revision (faster but lower quality)
python scripts/litemind_peak_analysis/main.py --coarse-clusters 0 --no-review
```

## Features

### Core Analysis

- **Coarse Cluster Analysis**: High-level regulatory programs (broad patterns)
- **Fine Cluster Analysis**: Detailed sub-programs within coarse clusters
- **Data Integration**: Pseudobulk profiles, gene associations, motif enrichments, peak statistics

### Quality Control

- **Analysis Review**: Critic agent evaluates analysis for accuracy and completeness
- **Analysis Revision**: Original agent revises based on review feedback
- **Citation Validation**: Detects and prevents hallucinated references

### Biological Tools

- **Database APIs**:
  - Ensembl: Gene metadata lookup
  - ZFIN: Zebrafish-specific gene information
  - PubMed: Literature retrieval
  - Alliance: Expression patterns and disease associations
  - JASPAR: Transcription factor motif database
  - GO: Gene Ontology enrichment

- **Web Search**: Contextual literature search for validation

## Module Structure

```
scripts/litemind_peak_analysis/
├── __init__.py              # Module initialization
├── config.py                # Configuration (paths, API keys, options)
├── main.py                  # Entry point script
│
├── core/                    # Core analysis logic
│   ├── data.py              # Data loading and processing
│   ├── prompts.py           # LLM prompt templates
│   ├── coarse_cluster_analysis.py
│   ├── fine_cluster_analysis.py
│   ├── analysis_review.py
│   ├── analysis_revision.py
│   └── deep_research.py     # Optional deep research
│
├── bio_services/            # Database API wrappers
│   ├── core_http.py         # HTTP client with caching/retry
│   ├── alliance.py
│   ├── ensembl.py
│   ├── pubmed.py
│   ├── jaspar.py
│   ├── zfin.py
│   └── ...
│
└── utils/                   # Utility functions
    ├── markdown.py          # DataFrame to markdown conversion
    └── citations.py         # Citation validation
```

## Configuration

### Environment Variables

```bash
# Required (choose one):
export OPENAI_API_KEY="sk-..."          # For OpenAI API
export ANTHROPIC_API_KEY="sk-ant-..."  # For Anthropic API

# Optional:
export LITEMIND_DEFAULT_API="openai"           # or "anthropic"
export LITEMIND_DEFAULT_MODEL="gpt-4"
export LITEMIND_DATA_DIR="/path/to/data"      # Override data directory
export LITEMIND_DO_REVIEW="true"              # Enable review/revision
export LITEMIND_MAX_WEB_SEARCHES="256"        # Max web searches per analysis
export LITEMIND_SEARCH_CONTEXT_SIZE="high"    # "low", "medium", or "high"
```

### Data Directory

By default, data is loaded from the external litemind subrepo:
```
/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/external/litemind_peak_cluster_queries/src/alpha/project/regulome/data/
```

Override with environment variable:
```bash
export LITEMIND_DATA_DIR="/path/to/your/data"
```

### Required Data Files (99 MB total)

The analysis requires 18 CSV files in the data directory:

**Cluster Statistics:**
- `coarse_cluster_statistics.csv` - Peak counts, genomic annotations
- `fine_cluster_statistics.csv`

**Pseudobulk Profiles:**
- `leiden_by_pseudobulk.csv` - Coarse cluster accessibility
- `leiden_fine_by_pseudobulk.csv` - Fine cluster accessibility
- `num_cells_per_pseudobulk_group.csv` - Cell counts

**Gene Associations:**
- `leiden_by_assoc_genes.csv` - Coarse cluster genes
- `leiden_fine_by_assoc_genes.csv` - Fine cluster genes
- `peaks_assoc_genes_filtered.csv` - Peak-level associations

**Motif Enrichments:**
- `leiden_by_motifs_maelstrom.csv` - Coarse cluster motifs
- `leiden_fine_by_motifs_maelstrom.csv` - Fine cluster motifs
- `info_cisBP_v2_danio_rerio_motif_factors_consensus.csv` - Motif-TF mappings
- (and 7 additional related files)

## Workflow

### Coarse Cluster Analysis

1. **Data Loading**: Loads pseudobulk profiles, gene associations, motif enrichments
2. **Prompt Construction**: Formats data as markdown tables with context
3. **LLM Analysis**: Agent analyzes using database tools and web search
4. **Review** (optional): Critic evaluates analysis accuracy
5. **Revision** (optional): Original agent revises based on critique
6. **Output**: Saves markdown report with structured sections

### Fine Cluster Analysis

Same as coarse but includes parent coarse cluster context to understand sub-programs.

## Output Format

Analysis results are saved as markdown files in the output directory:

```
results/
├── coarse_cluster_analysis/
│   ├── coarse_cluster_analysis_0.md
│   ├── coarse_cluster_analysis_1.md
│   └── ...
├── cluster_analysis_review/
│   ├── coarse_cluster_analysis_0_review.md
│   └── ...
└── coarse_cluster_analysis_revision/
    ├── coarse_cluster_analysis_0_revision.md
    └── ...
```

Each analysis includes:
- **Cluster Name** and **Label**
- **Cluster Overview**
- **Temporal Dynamics** - When accessibility peaks
- **Cell Type Specificity** - Which cell types show strongest signal
- **Overlapping/Correlated Genes** - Associated genes
- **Motif Analysis** - Enriched TF binding motifs
- **Key Transcription Factors** - Top TFs with roles
- **Regulatory Program** - Inferred biological processes
- **Biological Interpretation** - Detailed prose narrative with citations
- **Concise Summary** - High-level takeaways
- **References** - All cited sources

## Performance

- **Time**: ~5-15 minutes per cluster (with review/revision)
- **Cost**: ~$0.50-2.00 per cluster (OpenAI GPT-4)
- **Caching**: HTTP responses cached for 24 hours
- **Rate Limiting**: 5 requests/second to external APIs

## Troubleshooting

### API Key Not Found

```
❌ Configuration Error:
  - OPENAI_API_KEY environment variable not set
```

**Solution**: Set your API key:
```bash
export OPENAI_API_KEY="sk-your-key-here"
```

### Data Directory Not Found

```
❌ Configuration Error:
  - Data directory not found: /path/to/data
```

**Solution**: Either:
1. Ensure external subrepo is at expected location
2. Set `LITEMIND_DATA_DIR` to correct path

### Import Errors

```
ModuleNotFoundError: No module named 'litemind'
```

**Solution**: Install dependencies:
```bash
pip install litemind==2025.7.26 openai requests-cache backoff ratelimit arbol tabulate
```

### Permission Errors with Cache

```
PermissionError: [Errno 13] Permission denied: '.cache/litebio.sqlite'
```

**Solution**: Ensure `.cache/` directory is writable:
```bash
chmod 755 scripts/litemind_peak_analysis/.cache
```

## Advanced Usage

### Custom Prompts

Modify prompt templates in `core/prompts.py` to customize:
- Analysis section structure
- Instructions for LLM
- Citation format
- Biological focus areas

### Additional Bio Services

Add new database tools in `bio_services/`:
1. Create new module (e.g., `my_database.py`)
2. Implement query function with docstring
3. Wrap with `FunctionTool` in `main.py`
4. Add to toolset

### Deep Research (Experimental)

Enable deep research mode for comprehensive literature review:
```bash
export LITEMIND_DO_DEEP_RESEARCH="true"
python scripts/litemind_peak_analysis/main.py --coarse-clusters 0
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `litemind` | 2025.7.26 | LLM agent framework |
| `openai` | 1.96.1 | OpenAI API client |
| `backoff` | ≥2.2.1 | Retry logic |
| `requests-cache` | ≥1.2.1 | HTTP caching |
| `ratelimit` | ≥2.2.1 | Rate limiting |
| `arbol` | latest | Logging |
| `tabulate` | latest | Markdown tables |
| `pandas` | ≥2.3.0 | Data manipulation |
| `numpy` | ≥2.3.1 | Numerical operations |

## Citations

If you use this module in your research, please cite:

- **Zebrahub-Multiome**: [Preprint](https://www.biorxiv.org/content/10.1101/2024.10.18.618987v1)
- **LiteMind**: [GitHub](https://github.com/royerlab/litemind)

## License

See main repository LICENSE file.

## Contact

For questions or issues, please open an issue in the main zebrahub-multiome-analysis repository.
