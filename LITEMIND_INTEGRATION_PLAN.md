# LiteMind Peak Cluster Analysis - Integration Plan

**Date:** 2025-12-07
**Source Repository:** `/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/external/litemind_peak_cluster_queries`
**Target Repository:** `/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis`

---

## Executive Summary

This document outlines the strategy for extracting and integrating the LiteMind-based peak cluster annotation system from the external subrepo into the main zebrahub-multiome-analysis repository. The system uses LLM APIs (OpenAI/Anthropic) to generate biological interpretations of chromatin accessibility clusters through structured queries to biological databases and literature.

**Total Code to Extract:** ~1,500 lines across 32 Python files
**Data Size:** 99 MB of input CSV files
**Key Technology:** LiteMind framework (custom alternative to LangChain)

---

## Subrepo Analysis

### Repository Structure

```
external/litemind_peak_cluster_queries/
├── src/alpha/
│   ├── project/regulome/          # Main analysis logic (1,235 lines)
│   │   ├── main.py                # Orchestration script
│   │   ├── data.py                # Data loading and processing (403 lines)
│   │   ├── prompts.py             # LLM prompt templates (194 lines)
│   │   ├── coarse_cluster_analysis.py    # Coarse cluster analysis (135 lines)
│   │   ├── fine_cluster_analysis.py      # Fine cluster analysis (172 lines)
│   │   ├── analysis_review.py     # Quality control review (54 lines)
│   │   ├── analysis_revision.py   # Revision workflow (66 lines)
│   │   ├── deep_research.py       # Deep research (86 lines, optional)
│   │   └── data/                  # 99 MB of CSV data files
│   │
│   ├── bio_services/              # Database API wrappers (283 lines)
│   │   ├── core_http.py           # HTTP client with caching/retry (34 lines)
│   │   ├── alliance.py            # Alliance of Genome Resources
│   │   ├── ensembl.py             # Ensembl gene lookup
│   │   ├── pubmed.py              # PubMed record fetching
│   │   ├── jaspar.py              # JASPAR motif database
│   │   ├── zfin.py                # ZFIN zebrafish database
│   │   ├── go.py                  # Gene Ontology enrichment
│   │   ├── uniprot.py             # UniProt pathways
│   │   └── epmc.py                # Europe PMC search
│   │
│   └── utils/                     # Utility functions (62 lines)
│       ├── markdown.py            # DataFrame to markdown conversion (42 lines)
│       └── broken_citations.py    # Citation validation (20 lines)
│
├── pyproject.toml                 # Dependencies and project config
├── uv.lock                        # Dependency lock file
└── README.md                      # Original documentation
```

### Module Breakdown

#### 1. Core Analysis Modules (`src/alpha/project/regulome/`)

**Purpose:** Orchestrates LLM-based analysis of peak clusters using structured prompts and biological database queries.

##### `data.py` (403 lines) - **ESSENTIAL**
- `load_coarse_cluster_data()` - Loads all data for coarse cluster analysis
- `load_fine_cluster_data()` - Loads all data for fine cluster analysis
- `process_cluster_data()` - Prepares data for a single cluster for LLM analysis
- `coarse_peaks_details()` - Processes peak-gene associations for coarse clusters
- `fine_peaks_details()` - Processes peak-gene associations for fine clusters

**Key Features:**
- Loads 7 types of data: peak stats, pseudobulk profiles, gene associations, motif enrichments
- Processes gene lists to reduce token count for LLM
- Sorts and filters data for optimal LLM context

##### `prompts.py` (194 lines) - **ESSENTIAL**
- `project_background` - Context about the zebrafish multiome atlas
- `coarse_cluster_analysis_request` - Structured prompt template for coarse clusters
- `fine_cluster_analysis_request` - Structured prompt template for fine clusters
- `expert_system_prompt` - System prompt for analysis agent
- `critic_system_prompt` - System prompt for review agent
- `deep_research_system_prompt` - System prompt for deep research agent

**Key Features:**
- Highly structured prompt templates with exact section requirements
- Instructions for biological interpretation and literature citation
- Format validation requirements

##### `coarse_cluster_analysis.py` (135 lines) - **ESSENTIAL**
- `CoarseClusterAnalysis` class (extends `litemind.workflow.task.Task`)
- `get_coarse_cluster_id_list()` - Static method to get all cluster IDs
- `validate_result()` - Checks for broken citations and minimum word count
- `build_message()` - Constructs LLM prompt with data tables

**Key Features:**
- Integrates with LiteMind agent framework
- Formats data as markdown tables for LLM consumption
- Validates output quality

##### `fine_cluster_analysis.py` (172 lines) - **ESSENTIAL**
- `FineClusterAnalysis` class (extends `litemind.workflow.task.Task`)
- Similar structure to coarse analysis but includes parent cluster context
- `post_process_result_before_saving_pdf()` - Formats cluster IDs for output

**Key Features:**
- Takes coarse cluster analysis as dependency
- Contextualizes fine cluster within broader regulatory program
- Compares fine vs coarse cluster characteristics

##### `analysis_review.py` (54 lines) - **MEDIUM PRIORITY**
- `ClusterAnalysisReview` class
- Implements quality control review workflow
- Uses critic agent to evaluate analysis accuracy and rigor

**Key Features:**
- Validates factual accuracy
- Checks for overstated claims
- Identifies missing alternative interpretations

##### `analysis_revision.py` (66 lines) - **MEDIUM PRIORITY**
- `ClusterAnalysisRevision` class
- Revises analysis based on review feedback
- Continues conversation with original agent

**Key Features:**
- Incorporates critic feedback
- Maintains format consistency
- No mention of revision in final output

##### `deep_research.py` (86 lines) - **LOW PRIORITY / OPTIONAL**
- `DeepResearch` class
- Uses specialized deep research model (e.g., o3-deep-research)
- Generates self-contained research prompts

**Key Features:**
- Searches for corroborating/contradicting evidence
- Assesses novelty of findings
- Optional enhancement step

##### `main.py` (133 lines) - **ORCHESTRATION**
- Entry point for running full analysis pipeline
- Initializes APIs (OpenAI/Anthropic)
- Configures tools (web search, database lookups)
- Loops through coarse and fine clusters

**Key Features:**
- Configurable workflow (enable/disable review, deep research)
- Tool set configuration for LiteMind agents
- Results saved to `results/` folder

#### 2. Biological Database APIs (`src/alpha/bio_services/`)

**Purpose:** Provides LiteMind-compatible function tools for biological database queries.

##### `core_http.py` (34 lines) - **ESSENTIAL**
- `_safe_get()` - HTTP client with exponential backoff and rate limiting
- `fetch_json()` - Main API query function
- Uses `requests_cache` with 24-hour expiration

**Key Features:**
- 5 requests/second rate limit (polite)
- Automatic retry on timeout/connection errors
- SQLite-backed response cache

##### `alliance.py` - Alliance of Genome Resources
- `fetch_alliance_expression_summary()` - Gene expression patterns
- `fetch_alliance_gene_disease()` - Disease associations

##### `ensembl.py` - Ensembl Database
- `lookup_ensembl_gene()` - Gene metadata lookup
- Returns symbol, description, coordinates, biotype, ZFIN ID

##### `pubmed.py` - PubMed Records
- `fetch_pubmed_record()` - Retrieve article metadata by PMID
- `get_pubmed_record_full()` - Full article details

##### `jaspar.py` - JASPAR Motif Database
- `fetch_jaspar_motif_info()` - Motif information
- `jaspar_motif_info()` - Detailed motif data by release

##### `zfin.py` - ZFIN Zebrafish Database
- `fetch_zfin_gene_aliases()` - Gene name aliases

##### `go.py` - Gene Ontology
- `get_go_enrichment()` - GO term enrichment analysis

##### `uniprot.py` - UniProt Database
- `get_pathways_for_gene()` - Pathway annotations

##### `epmc.py` - Europe PMC
- `search_epmc()` - Literature search

#### 3. Utilities (`src/alpha/utils/`)

##### `markdown.py` (42 lines) - **ESSENTIAL**
- `table_to_markdown()` - Converts pandas DataFrame to markdown table
- `quote_text()` - Formats text as markdown quote block

**Key Features:**
- Uses `tabulate` library for clean formatting
- Handles empty DataFrames gracefully
- Fallback to `to_string()` if tabulate fails

##### `broken_citations.py` (20 lines) - **ESSENTIAL**
- `has_broken_citations()` - Validates LLM output for citation placeholders
- Detects patterns like `[turn1search0]`, `[oaicite:0]`, `::contentReference`

**Key Features:**
- Pre-compiled regex patterns
- Prevents hallucinated citations in output

#### 4. Data Files (`data/` - 99 MB)

**18 CSV files containing:**

1. **Cluster Statistics:**
   - `coarse_cluster_statistics.csv` (2.5 KB) - Peak counts, genomic annotations
   - `fine_cluster_statistics.csv` (27 KB)
   - `coarse_chrom_distribution.csv` (4.6 KB)
   - `fine_chrom_distribution.csv` (51 KB)

2. **Pseudobulk Profiles:**
   - `leiden_by_pseudobulk.csv` (127 KB) - Coarse cluster accessibility by cell type × stage
   - `leiden_fine_by_pseudobulk.csv` (1.4 MB)
   - `leiden_coarse_by_pseudobulk_mean_sem.csv` (266 KB)
   - `leiden_fine_by_pseudobulk_mean_sem.csv` (2.8 MB)
   - `num_cells_per_pseudobulk_group.csv` (5.2 KB)

3. **Gene Associations:**
   - `leiden_by_assoc_genes.csv` (4.2 MB) - Genes overlapping/correlated with coarse clusters
   - `leiden_fine_by_assoc_genes.csv` (44 MB) - Fine cluster gene associations
   - `peaks_assoc_genes_filtered.csv` (23 MB) - Peak-level gene associations

4. **Motif Enrichments:**
   - `leiden_by_motifs_maelstrom.csv` (80 KB) - Coarse cluster motif z-scores
   - `leiden_fine_by_motifs_maelstrom.csv` (23 MB)
   - `leiden_fine_motifs_maelstrom.csv` (896 KB)
   - `info_cisBP_v2_danio_rerio_motif_factors.csv` (38 KB) - Motif-TF mappings
   - `info_cisBP_v2_danio_rerio_motif_factors_consensus.csv` (38 KB)
   - `info_cisBP_v2_danio_rerio_motif_factors_fine_clusters_consensus.csv` (37 KB)

---

## Integration Strategy Comparison

### Option A: Standalone Module (⭐ RECOMMENDED)

**Target Location:** `scripts/litemind_peak_analysis/`

#### Advantages
- ✅ Clean separation from existing codebase
- ✅ Easy to maintain and update independently
- ✅ Clear dependency management (can use separate conda env)
- ✅ Can be used as a standalone package
- ✅ No conflicts with existing peak_umap_utils
- ✅ Easy to document and test

#### Disadvantages
- ⚠️ Slight duplication if data processing overlaps with peak_umap_utils
- ⚠️ Requires additional environment setup

#### Proposed Structure
```
scripts/litemind_peak_analysis/
├── __init__.py
├── README.md                      # Module-specific documentation
├── main.py                        # Entry point script
├── config.py                      # Configuration (API keys, paths, options)
│
├── core/                          # Core analysis logic
│   ├── __init__.py
│   ├── data.py                    # Data loading and processing
│   ├── prompts.py                 # LLM prompt templates
│   ├── coarse_analysis.py         # Coarse cluster analysis
│   ├── fine_analysis.py           # Fine cluster analysis
│   ├── review.py                  # Analysis review
│   ├── revision.py                # Analysis revision
│   └── deep_research.py           # Deep research (optional)
│
├── bio_services/                  # Database API wrappers
│   ├── __init__.py
│   ├── core_http.py               # HTTP client with caching/retry
│   ├── alliance.py                # Alliance of Genome Resources
│   ├── ensembl.py                 # Ensembl gene lookup
│   ├── pubmed.py                  # PubMed records
│   ├── jaspar.py                  # JASPAR motifs
│   ├── zfin.py                    # ZFIN zebrafish DB
│   ├── go.py                      # Gene Ontology
│   ├── uniprot.py                 # UniProt pathways
│   └── epmc.py                    # Europe PMC
│
├── utils/                         # Utility functions
│   ├── __init__.py
│   ├── markdown.py                # DataFrame to markdown
│   └── citations.py               # Citation validation
│
└── data/                          # Data files (99 MB)
    ├── README.md                  # Data file descriptions
    └── (18 CSV files)
```

---

### Option B: Integrated into peak_umap_utils

**Target Location:** `scripts/peak_umap_utils/litemind_analysis/`

#### Advantages
- ✅ Tighter integration with existing peak analysis
- ✅ Shared utilities (color palettes, data loading patterns)
- ✅ Single import path for all peak analysis

#### Disadvantages
- ❌ Mixes LLM-based analysis with statistical analysis
- ❌ Different dependency requirements (litemind, OpenAI API)
- ❌ Complicates peak_umap_utils module scope
- ❌ May require all users to install LLM dependencies even if not using

---

### Option C: Notebook-Based Integration

**Target Location:** `notebooks/Fig_peak_umap/litemind_analysis/`

#### Advantages
- ✅ Direct integration with peak UMAP notebooks
- ✅ Easy to run interactively
- ✅ Minimal code organization needed

#### Disadvantages
- ❌ Less reusable across projects
- ❌ Harder to maintain and version control
- ❌ Not suitable for batch processing
- ❌ Duplicates code if used in multiple notebooks

---

## Recommended Implementation Plan

### **RECOMMENDATION: Option A - Standalone Module**

---

## Phase-by-Phase Implementation

### Phase 1: Create Module Structure (1 hour)

**Actions:**
1. Create directory structure:
   ```bash
   mkdir -p scripts/litemind_peak_analysis/{core,bio_services,utils,data}
   ```

2. Create `__init__.py` files for all modules

3. Create initial `README.md` for the module

4. Create `config.py` for configuration management:
   ```python
   # config.py
   import os
   from pathlib import Path

   # API Configuration
   OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
   ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

   # Data paths
   MODULE_DIR = Path(__file__).parent
   DATA_DIR = MODULE_DIR / "data"
   RESULTS_DIR = MODULE_DIR / "results"

   # LiteMind configuration
   DEFAULT_API = "openai"  # or "anthropic"
   DEFAULT_MODEL = "gpt-4"
   MAX_WEB_SEARCHES = 256
   SEARCH_CONTEXT_SIZE = "high"

   # Analysis options
   DO_REVIEW = True
   DO_DEEP_RESEARCH = False
   SAVE_PDF = True
   ```

**Deliverables:**
- Directory structure created
- All `__init__.py` files in place
- Basic README and config files

---

### Phase 2: Copy and Adapt Core Modules (3 hours)

**Actions:**

#### 2.1 Copy Core Analysis Files
```bash
# From external subrepo to main repo
cp external/.../src/alpha/project/regulome/data.py \
   scripts/litemind_peak_analysis/core/data.py

cp external/.../src/alpha/project/regulome/prompts.py \
   scripts/litemind_peak_analysis/core/prompts.py

cp external/.../src/alpha/project/regulome/coarse_cluster_analysis.py \
   scripts/litemind_peak_analysis/core/coarse_analysis.py

cp external/.../src/alpha/project/regulome/fine_cluster_analysis.py \
   scripts/litemind_peak_analysis/core/fine_analysis.py

cp external/.../src/alpha/project/regulome/analysis_review.py \
   scripts/litemind_peak_analysis/core/review.py

cp external/.../src/alpha/project/regulome/analysis_revision.py \
   scripts/litemind_peak_analysis/core/revision.py

cp external/.../src/alpha/project/regulome/deep_research.py \
   scripts/litemind_peak_analysis/core/deep_research.py
```

#### 2.2 Update Import Paths
Replace all `from alpha.` imports with new module paths:
```python
# OLD:
from alpha.project.regulome.data import load_coarse_cluster_data
from alpha.bio_services.alliance import fetch_alliance_expression_summary
from alpha.utils.markdown import table_to_markdown

# NEW:
from scripts.litemind_peak_analysis.core.data import load_coarse_cluster_data
from scripts.litemind_peak_analysis.bio_services.alliance import fetch_alliance_expression_summary
from scripts.litemind_peak_analysis.utils.markdown import table_to_markdown
```

#### 2.3 Update Data Path References
Replace hardcoded data paths with configurable paths:
```python
# OLD:
df = pd.read_csv("data/leiden_by_assoc_genes.csv", index_col=0)

# NEW:
from scripts.litemind_peak_analysis.config import DATA_DIR
df = pd.read_csv(DATA_DIR / "leiden_by_assoc_genes.csv", index_col=0)
```

#### 2.4 Remove External Dependencies
Update imports that reference `__trash` or other non-existent modules:
```python
# data.py line 7 - REMOVE:
from __trash.module_litemind_query import get_table_info, convert_clusters_genes_to_lists

# REPLACE WITH:
# Either reimplement these functions or remove if unused
```

**Deliverables:**
- All core analysis modules copied and adapted
- Import paths updated
- Data paths made configurable
- No broken imports

---

### Phase 3: Copy Biological Service APIs (1 hour)

**Actions:**

#### 3.1 Copy Bio Service Files
```bash
cp external/.../src/alpha/bio_services/*.py \
   scripts/litemind_peak_analysis/bio_services/
```

#### 3.2 Update Import Paths
```python
# OLD:
from src.alpha.bio_services.core_http import fetch_json

# NEW:
from scripts.litemind_peak_analysis.bio_services.core_http import fetch_json
```

#### 3.3 Update Cache Configuration
In `core_http.py`, update cache file location:
```python
# OLD:
requests_cache.install_cache("litebio", expire_after=86400)

# NEW:
from scripts.litemind_peak_analysis.config import MODULE_DIR
cache_path = MODULE_DIR / ".cache" / "litebio"
requests_cache.install_cache(str(cache_path), expire_after=86400)
```

**Deliverables:**
- All bio service modules copied
- Import paths updated
- Cache location configured
- HTTP client ready to use

---

### Phase 4: Copy Utility Functions (30 minutes)

**Actions:**

#### 4.1 Copy Utility Files
```bash
cp external/.../src/alpha/utils/markdown.py \
   scripts/litemind_peak_analysis/utils/markdown.py

cp external/.../src/alpha/utils/broken_citations.py \
   scripts/litemind_peak_analysis/utils/citations.py
```

#### 4.2 No Modifications Needed
These files are self-contained and don't require path updates.

**Deliverables:**
- Utility functions ready to use

---

### Phase 5: Data Integration (1 hour)

**Decision Point:** Choose data integration approach:

#### Option 5A: Copy Data Files (RECOMMENDED for portability)
```bash
cp external/.../src/alpha/project/regulome/data/*.csv \
   scripts/litemind_peak_analysis/data/
```
**Pros:** Self-contained, no external dependencies
**Cons:** 99 MB additional storage

#### Option 5B: Symlink to External Data
```bash
ln -s /hpc/projects/.../external/litemind_peak_cluster_queries/src/alpha/project/regulome/data \
      scripts/litemind_peak_analysis/data
```
**Pros:** No duplication, always synced
**Cons:** Breaks if external repo moves

#### Option 5C: Configure External Path
In `config.py`:
```python
DATA_DIR = Path("/hpc/projects/.../external/litemind_peak_cluster_queries/src/alpha/project/regulome/data")
```
**Pros:** No duplication, explicit
**Cons:** Hardcoded path, not portable

**Deliverables:**
- Data files accessible to analysis modules
- Document data source and update procedure in README

---

### Phase 6: Create Entry Point (2 hours)

**Actions:**

#### 6.1 Create `main.py`
```python
#!/usr/bin/env python3
"""
LiteMind Peak Cluster Analysis - Main Entry Point

Usage:
    python scripts/litemind_peak_analysis/main.py --help
    python scripts/litemind_peak_analysis/main.py --coarse-clusters 0,1,2
    python scripts/litemind_peak_analysis/main.py --fine-clusters 0_0,0_1
"""

import argparse
from pathlib import Path
from litemind import OpenAIApi, AnthropicApi
from litemind.agent.tools.builtin_tools.web_search_tool import BuiltinWebSearchTool
from litemind.agent.tools.function_tool import FunctionTool
from litemind.agent.tools.toolset import ToolSet

from scripts.litemind_peak_analysis import config
from scripts.litemind_peak_analysis.core.coarse_analysis import CoarseClusterAnalysis
from scripts.litemind_peak_analysis.core.fine_analysis import FineClusterAnalysis
from scripts.litemind_peak_analysis.core.review import ClusterAnalysisReview
from scripts.litemind_peak_analysis.core.revision import ClusterAnalysisRevision
from scripts.litemind_peak_analysis.bio_services.alliance import (
    fetch_alliance_expression_summary,
    fetch_alliance_gene_disease
)
from scripts.litemind_peak_analysis.bio_services.ensembl import lookup_ensembl_gene
from scripts.litemind_peak_analysis.bio_services.pubmed import fetch_pubmed_record


def setup_api():
    """Initialize LiteMind API based on config."""
    if config.DEFAULT_API == "openai":
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        return OpenAIApi()
    elif config.DEFAULT_API == "anthropic":
        if not config.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        return AnthropicApi()
    else:
        raise ValueError(f"Unknown API: {config.DEFAULT_API}")


def setup_toolset():
    """Create toolset with web search and bio service tools."""
    toolset = ToolSet()

    # Add web search
    web_search = BuiltinWebSearchTool(
        max_web_searches=config.MAX_WEB_SEARCHES,
        search_context_size=config.SEARCH_CONTEXT_SIZE
    )
    toolset.add_tool(web_search)

    # Add bio service tools
    toolset.add_tool(FunctionTool(fetch_pubmed_record))
    toolset.add_tool(FunctionTool(lookup_ensembl_gene))
    toolset.add_tool(FunctionTool(fetch_alliance_expression_summary))
    toolset.add_tool(FunctionTool(fetch_alliance_gene_disease))

    return toolset


def run_coarse_cluster_analysis(cluster_id, api, toolset, output_dir):
    """Run analysis for a single coarse cluster."""
    print(f"\n{'='*60}")
    print(f"Analyzing Coarse Cluster {cluster_id}")
    print(f"{'='*60}\n")

    # Run initial analysis
    analysis = CoarseClusterAnalysis(
        coarse_cluster_id=cluster_id,
        api=api,
        toolset=toolset,
        folder=output_dir
    )
    analysis()

    # Optional: Run review
    if config.DO_REVIEW:
        review = ClusterAnalysisReview(
            analysis_task=analysis,
            toolset=toolset,
            api=api,
            folder=output_dir
        )
        review()

        # Revision based on review
        from scripts.litemind_peak_analysis.core.prompts import coarse_cluster_analysis_request
        revision = ClusterAnalysisRevision(
            analysis_task=analysis,
            review_task=review,
            analysis_format_instructions=coarse_cluster_analysis_request,
            folder=output_dir
        )
        revision()

        return revision

    return analysis


def main():
    parser = argparse.ArgumentParser(
        description="Run LiteMind-based peak cluster analysis"
    )
    parser.add_argument(
        "--coarse-clusters",
        type=str,
        help="Comma-separated list of coarse cluster IDs (e.g., '0,1,2') or 'all'"
    )
    parser.add_argument(
        "--fine-clusters",
        type=str,
        help="Comma-separated list of fine cluster IDs (e.g., '0_0,0_1,1_0') or 'all'"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for analysis results"
    )
    parser.add_argument(
        "--api",
        type=str,
        choices=["openai", "anthropic"],
        default=config.DEFAULT_API,
        help="LLM API to use"
    )
    parser.add_argument(
        "--no-review",
        action="store_true",
        help="Skip review and revision steps"
    )

    args = parser.parse_args()

    # Override config with CLI args
    if args.api:
        config.DEFAULT_API = args.api
    if args.no_review:
        config.DO_REVIEW = False

    # Setup
    api = setup_api()
    toolset = setup_toolset()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run coarse cluster analyses
    if args.coarse_clusters:
        if args.coarse_clusters == "all":
            cluster_ids = CoarseClusterAnalysis.get_coarse_cluster_id_list()
        else:
            cluster_ids = [int(x.strip()) for x in args.coarse_clusters.split(",")]

        for cluster_id in cluster_ids:
            run_coarse_cluster_analysis(cluster_id, api, toolset, output_dir)

    # Run fine cluster analyses
    if args.fine_clusters:
        if args.fine_clusters == "all":
            cluster_ids = FineClusterAnalysis.get_fine_cluster_id_list()
        else:
            cluster_ids = [x.strip() for x in args.fine_clusters.split(",")]

        for cluster_id in cluster_ids:
            # TODO: Implement fine cluster analysis
            print(f"Fine cluster {cluster_id} analysis not yet implemented")

    print(f"\nAnalysis complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
```

#### 6.2 Make Executable
```bash
chmod +x scripts/litemind_peak_analysis/main.py
```

**Deliverables:**
- Functional CLI entry point
- Help documentation
- API configuration
- Toolset setup

---

### Phase 7: Environment and Dependencies (1 hour)

**Actions:**

#### 7.1 Update CLAUDE.md
Add new section:
```markdown
## LiteMind Peak Cluster Analysis

The `scripts/litemind_peak_analysis/` module provides LLM-based biological interpretation
of chromatin accessibility peak clusters.

### Setup

1. **Install Dependencies:**
   ```bash
   pip install litemind==2025.7.26 openai==1.96.1 backoff requests-cache ratelimit arbol tabulate
   ```

2. **Set API Key:**
   ```bash
   export OPENAI_API_KEY="your-key-here"
   # OR
   export ANTHROPIC_API_KEY="your-key-here"
   ```

3. **Run Analysis:**
   ```bash
   python scripts/litemind_peak_analysis/main.py --coarse-clusters 0,1,2
   ```

### Features

- Automated biological interpretation of peak clusters
- Database queries (Ensembl, ZFIN, PubMed, Alliance, etc.)
- Web search for literature context
- Optional peer review and revision workflow
- Structured markdown output with citations

### Dependencies

- `litemind`: LLM agent framework
- `openai`: OpenAI API client
- `requests-cache`: HTTP caching
- `backoff`: Retry logic
- `arbol`: Logging
```

#### 7.2 Create Module README
Create `scripts/litemind_peak_analysis/README.md` with:
- Module overview
- Quick start guide
- Usage examples
- Configuration options
- Data file descriptions
- API requirements
- Troubleshooting

#### 7.3 Optional: Create Separate Conda Environment
```yaml
# environments/litemind_env.yml
name: litemind_env
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.13
  - pandas>=2.3.0
  - numpy>=2.3.1
  - pip
  - pip:
    - litemind==2025.7.26
    - openai==1.96.1
    - backoff>=2.2.1
    - requests-cache>=1.2.1
    - ratelimit>=2.2.1
    - arbol
    - tabulate
```

**Deliverables:**
- Documentation updated
- Environment setup documented
- Optional conda environment spec

---

### Phase 8: Integration Testing (2 hours)

**Test Cases:**

#### 8.1 Test Data Loading
```python
from scripts.litemind_peak_analysis.core.data import load_coarse_cluster_data

# Should load all 7 data structures without errors
data = load_coarse_cluster_data()
assert len(data) == 9  # 9 return values
print("✅ Data loading successful")
```

#### 8.2 Test Coarse Cluster Analysis
```python
from scripts.litemind_peak_analysis.main import run_coarse_cluster_analysis
from litemind import OpenAIApi
from scripts.litemind_peak_analysis.main import setup_toolset

api = OpenAIApi()
toolset = setup_toolset()
result = run_coarse_cluster_analysis(0, api, toolset, "test_results")
print("✅ Coarse cluster analysis successful")
```

#### 8.3 Test Bio Services
```python
from scripts.litemind_peak_analysis.bio_services.ensembl import lookup_ensembl_gene

result = lookup_ensembl_gene("ENSDARG00000000001", "danio_rerio")
assert result is not None
print("✅ Ensembl API working")
```

#### 8.4 Test Citation Validation
```python
from scripts.litemind_peak_analysis.utils.citations import has_broken_citations

good_text = "This is valid text with a [proper link](https://example.com)"
bad_text = "This has a [oaicite:0] placeholder"

assert not has_broken_citations(good_text)
assert has_broken_citations(bad_text)
print("✅ Citation validation working")
```

**Deliverables:**
- All tests passing
- Sample outputs generated
- No import errors

---

### Phase 9: Documentation (1 hour)

**Create Comprehensive Documentation:**

#### 9.1 Module README (`scripts/litemind_peak_analysis/README.md`)
- Overview and purpose
- Installation instructions
- Quick start guide
- CLI usage examples
- Configuration options
- Data file descriptions
- Troubleshooting

#### 9.2 Data README (`scripts/litemind_peak_analysis/data/README.md`)
- Description of each CSV file
- Data generation process
- Column definitions
- File sizes
- Update procedure

#### 9.3 Update Main CLAUDE.md
- Add LiteMind module section
- Document API key requirements
- Link to module README
- Note environment considerations

#### 9.4 Add Docstrings
Ensure all functions have complete docstrings:
```python
def process_cluster_data(cluster_id, ...):
    """
    Process data for a single cluster and prepare for LLM analysis.

    Parameters
    ----------
    cluster_id : int or str
        ID of the cluster to process
    df_peak_stats : pd.DataFrame
        Peak statistics per cluster
    ...

    Returns
    -------
    tuple
        (peak_stats, groups, genes_text, overlap, corr, anticorr, motifs)

    Raises
    ------
    KeyError
        If cluster_id not found in required DataFrames
    """
```

**Deliverables:**
- Complete documentation
- All functions documented
- Examples provided

---

## Code Modifications Summary

### Critical Changes Required

#### 1. Import Path Updates (ALL FILES)
```python
# Pattern to find:
from alpha\.

# Replace with:
from scripts.litemind_peak_analysis.
```

#### 2. Data Path Updates (`core/data.py`)
```python
# Lines to update:
# Line 16, 22, 26, 30, 34, 36, 63, 67, 71, 75, 82, 85

# OLD:
df = pd.read_csv("data/file.csv", index_col=0)

# NEW:
from scripts.litemind_peak_analysis.config import DATA_DIR
df = pd.read_csv(DATA_DIR / "file.csv", index_col=0)
```

#### 3. Remove Broken Import (`core/data.py` line 7)
```python
# REMOVE:
from __trash.module_litemind_query import get_table_info, convert_clusters_genes_to_lists

# ACTION: Check if convert_clusters_genes_to_lists is used
# If yes: Reimplement function
# If no: Remove import
```

#### 4. Cache Path Update (`bio_services/core_http.py` line 6)
```python
# OLD:
requests_cache.install_cache("litebio", expire_after=86400)

# NEW:
from scripts.litemind_peak_analysis.config import MODULE_DIR
cache_path = MODULE_DIR / ".cache" / "litebio"
MODULE_DIR.joinpath(".cache").mkdir(exist_ok=True)
requests_cache.install_cache(str(cache_path), expire_after=86400)
```

#### 5. Relative Import Fixes (`bio_services/*.py`)
```python
# OLD:
from src.alpha.bio_services.core_http import fetch_json

# NEW:
from scripts.litemind_peak_analysis.bio_services.core_http import fetch_json
```

#### 6. Database Filename Update (`main.py`)
```python
# If litebio.sqlite is hardcoded anywhere, update to:
cache_db = MODULE_DIR / ".cache" / "litebio.sqlite"
```

---

## Dependencies to Add

### Required Python Packages

| Package | Version | Purpose |
|---------|---------|---------|
| `litemind` | ==2025.7.26 | LLM agent framework (alternative to LangChain) |
| `openai` | ==1.96.1 | OpenAI API client |
| `backoff` | >=2.2.1 | Exponential backoff for retries |
| `requests-cache` | >=1.2.1 | HTTP response caching |
| `ratelimit` | >=2.2.1 | Rate limiting for API calls |
| `arbol` | latest | Logging and output formatting |
| `tabulate` | latest | Markdown table formatting |

### Optional Packages

| Package | Purpose |
|---------|---------|
| `anthropic` | Anthropic API client (if using Claude instead of GPT) |
| `md2pdf` | PDF generation from markdown |
| `weasyprint` | Alternative PDF generation |

### Environment Variables Required

```bash
# Required (choose one):
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Optional:
export LITEMIND_DEFAULT_API="openai"  # or "anthropic"
export LITEMIND_DEFAULT_MODEL="gpt-4"
```

### Installation Commands

```bash
# Using pip:
pip install litemind==2025.7.26 openai==1.96.1 backoff requests-cache ratelimit arbol tabulate

# Using uv (from pyproject.toml):
uv add litemind==2025.7.26 openai==1.96.1 backoff requests-cache ratelimit arbol tabulate
```

---

## Potential Issues and Solutions

### Issue 1: Missing `convert_clusters_genes_to_lists` Function

**Problem:** `data.py` line 7 imports from `__trash.module_litemind_query`

**Solution:**
1. Check if function is actually used (line 39, 88)
2. If used, search external repo for original implementation
3. Reimplement in `core/data.py` or remove if unused

**Investigation needed:**
```bash
grep -n "convert_clusters_genes_to_lists" external/.../src/alpha/project/regulome/data.py
```

---

### Issue 2: Cache File Permissions

**Problem:** SQLite cache might have permission issues in shared environment

**Solution:**
```python
# In core_http.py:
cache_dir = MODULE_DIR / ".cache"
cache_dir.mkdir(exist_ok=True, mode=0o755)
cache_file = cache_dir / "litebio.sqlite"
requests_cache.install_cache(str(cache_file), expire_after=86400)
```

---

### Issue 3: API Rate Limits

**Problem:** Multiple API calls might hit rate limits

**Solution:**
- Current implementation has 5 req/s limit (good)
- Ensure backoff decorator is working
- Document expected API costs in README

---

### Issue 4: Data File Size

**Problem:** 99 MB of CSV files is substantial

**Solution Options:**
1. **Copy all files** (99 MB, but self-contained)
2. **Compress files** (e.g., gzip CSVs, ~10-20 MB)
3. **Regenerate from source** (document pipeline)
4. **Symlink** (0 MB, but fragile)

**Recommendation:** Copy all files for portability

---

### Issue 5: LiteMind Version Compatibility

**Problem:** `litemind==2025.7.26` is very recent, may have breaking changes

**Solution:**
- Test thoroughly with specified version
- Pin version in requirements
- Document any version-specific issues

---

## Testing Checklist

### Pre-Integration Tests
- [ ] External repo code runs without errors
- [ ] All data files are accessible
- [ ] API keys are valid
- [ ] Dependencies install correctly

### Post-Integration Tests
- [ ] All imports resolve correctly
- [ ] Data loading works (9 DataFrames load)
- [ ] Configuration is read correctly
- [ ] HTTP caching works (check `.cache/` directory)
- [ ] API calls succeed (test with one cluster)
- [ ] Output is generated in expected format
- [ ] Markdown formatting is correct
- [ ] Citation validation works
- [ ] CLI help text displays correctly

### Integration Tests
- [ ] Run coarse cluster 0 analysis end-to-end
- [ ] Verify results saved to correct location
- [ ] Check PDF generation (if enabled)
- [ ] Verify review workflow (if enabled)
- [ ] Test with both OpenAI and Anthropic APIs
- [ ] Run with different command-line options

### Edge Case Tests
- [ ] Handle missing cluster ID gracefully
- [ ] Handle API errors (wrong key, rate limit)
- [ ] Handle network errors (timeout, connection)
- [ ] Handle empty DataFrames
- [ ] Handle malformed data files

---

## Success Criteria

### Functional Requirements
✅ All coarse cluster analyses can be run
✅ All fine cluster analyses can be run
✅ Database API calls work correctly
✅ Results are saved in markdown format
✅ Citations are validated
✅ No broken imports

### Quality Requirements
✅ Code follows repository conventions
✅ All functions have docstrings
✅ Module is documented in CLAUDE.md
✅ README provides clear instructions
✅ Tests pass consistently

### Performance Requirements
✅ Analysis completes within reasonable time (~5-10 min per cluster)
✅ HTTP caching reduces redundant API calls
✅ Rate limiting prevents API blocks

---

## Timeline Estimate

| Phase | Duration | Cumulative |
|-------|----------|-----------|
| Phase 1: Module Structure | 1 hour | 1 hour |
| Phase 2: Core Modules | 3 hours | 4 hours |
| Phase 3: Bio Services | 1 hour | 5 hours |
| Phase 4: Utilities | 0.5 hours | 5.5 hours |
| Phase 5: Data Integration | 1 hour | 6.5 hours |
| Phase 6: Entry Point | 2 hours | 8.5 hours |
| Phase 7: Environment/Deps | 1 hour | 9.5 hours |
| Phase 8: Testing | 2 hours | 11.5 hours |
| Phase 9: Documentation | 1 hour | 12.5 hours |
| **Total** | **~12-13 hours** | - |

**Suggested Schedule:** 2-3 working days with thorough testing

---

## Risk Assessment

### High Risk
- ⚠️ **Missing function implementations** (e.g., `convert_clusters_genes_to_lists`)
  - **Mitigation:** Thorough code review before copying

### Medium Risk
- ⚠️ **API compatibility issues** between litemind versions
  - **Mitigation:** Pin exact version, test thoroughly
- ⚠️ **Data file updates** needed in future
  - **Mitigation:** Document data generation pipeline

### Low Risk
- ⚠️ **Import path conflicts** with existing code
  - **Mitigation:** Use explicit `scripts.litemind_peak_analysis` prefix
- ⚠️ **Cache file permissions** on shared system
  - **Mitigation:** Use user-specific cache directory

---

## Future Enhancements

### Phase 10+ (Post-Integration)

1. **Batch Processing**
   - Parallel cluster analysis
   - Progress tracking
   - Resume capability

2. **Results Visualization**
   - Generate summary figures
   - Compare cluster analyses
   - Export to publication format

3. **Integration with Peak UMAP Notebooks**
   - Import analysis results into notebooks
   - Cross-reference with statistical analysis
   - Automated figure annotation

4. **Custom Prompt Templates**
   - User-defined analysis sections
   - Domain-specific question sets
   - Multi-species support

5. **Performance Optimization**
   - Cache LLM responses
   - Reduce token usage
   - Optimize data serialization

---

## References

- **External Subrepo:** `/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/external/litemind_peak_cluster_queries`
- **LiteMind Documentation:** https://github.com/royerlab/litemind
- **OpenAI API:** https://platform.openai.com/docs/
- **Anthropic API:** https://docs.anthropic.com/
- **Original README:** See external subrepo for original documentation

---

## Approval and Sign-Off

**Plan Author:** Claude Code
**Date Created:** 2025-12-07
**Status:** Awaiting Review

**Review Checklist:**
- [ ] All essential files identified
- [ ] Integration strategy is sound
- [ ] Data handling approach is acceptable
- [ ] Timeline is realistic
- [ ] Dependencies are manageable
- [ ] Risks are identified and mitigated

**Next Steps After Approval:**
1. Confirm Option A (Standalone Module) is preferred
2. Choose data integration approach (copy vs symlink)
3. Proceed with Phase 1 implementation
4. Review progress after each phase
