# LiteMind Peak Analysis - Installation Guide

## Quick Start

### 1. Verify Module Structure

```bash
cd /hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis
ls -la scripts/litemind_peak_analysis/
```

You should see:
- `config.py` - Configuration file
- `main.py` - Entry point (executable)
- `core/` - Core analysis modules
- `bio_services/` - Database API wrappers
- `utils/` - Utility functions
- `.cache/` - HTTP cache directory
- `README.md` - Full documentation

### 2. Install Dependencies

```bash
pip install litemind==2025.7.26 openai==1.96.1 backoff>=2.2.1 requests-cache>=1.2.1 ratelimit>=2.2.1 arbol tabulate pandas>=2.3.0 numpy>=2.3.1
```

Or use a requirements file:

```bash
cat > litemind_requirements.txt <<EOF
litemind==2025.7.26
openai==1.96.1
backoff>=2.2.1
requests-cache>=1.2.1
ratelimit>=2.2.1
arbol
tabulate
pandas>=2.3.0
numpy>=2.3.1
EOF

pip install -r litemind_requirements.txt
```

### 3. Set API Key

```bash
# OpenAI (default)
export OPENAI_API_KEY="sk-your-key-here"

# OR Anthropic
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
```

Add to your `~/.bashrc` or `~/.bash_profile` to make permanent:
```bash
echo 'export OPENAI_API_KEY="sk-your-key-here"' >> ~/.bashrc
source ~/.bashrc
```

### 4. Verify Data Directory

```bash
# Check if data directory exists
ls /hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/external/litemind_peak_cluster_queries/src/alpha/project/regulome/data/

# If using custom location, set environment variable:
export LITEMIND_DATA_DIR="/path/to/your/data"
```

### 5. Test Installation

```bash
# Test help message
python scripts/litemind_peak_analysis/main.py --help

# Expected output:
# usage: main.py [-h] [--coarse-clusters COARSE_CLUSTERS] [--all-coarse] ...
```

### 6. Run First Analysis

```bash
# Analyze a single cluster (will take ~5-15 minutes)
python scripts/litemind_peak_analysis/main.py --coarse-clusters 0

# Results will be in:
# scripts/litemind_peak_analysis/results/coarse_cluster_analysis/coarse_cluster_analysis_0.md
```

## Troubleshooting

### "No module named 'litemind'"

**Problem**: Dependencies not installed

**Solution**:
```bash
pip install litemind==2025.7.26 openai
```

### "OPENAI_API_KEY environment variable not set"

**Problem**: API key not configured

**Solution**:
```bash
export OPENAI_API_KEY="your-key-here"
```

### "Data directory not found"

**Problem**: External data directory not accessible

**Solution**:
```bash
# Option 1: Check external repo location
ls /hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/external/litemind_peak_cluster_queries/src/alpha/project/regulome/data/

# Option 2: Set custom data directory
export LITEMIND_DATA_DIR="/path/to/data"
```

### "Permission denied: .cache/litebio.sqlite"

**Problem**: Cache directory not writable

**Solution**:
```bash
chmod 755 scripts/litemind_peak_analysis/.cache
```

## Module Files Created

The implementation created 25 Python files:

**Configuration:**
- `config.py` - Paths, API keys, settings
- `__init__.py` - Module initialization

**Core Analysis (7 files):**
- `core/data.py` - Data loading and processing
- `core/prompts.py` - LLM prompt templates
- `core/coarse_cluster_analysis.py` - Coarse cluster analysis task
- `core/fine_cluster_analysis.py` - Fine cluster analysis task
- `core/analysis_review.py` - Quality control review
- `core/analysis_revision.py` - Revision workflow
- `core/deep_research.py` - Deep research (optional)

**Bio Services (11 files):**
- `bio_services/core_http.py` - HTTP client with caching
- `bio_services/alliance.py` - Alliance Genome Resources API
- `bio_services/ensembl.py` - Ensembl gene lookup
- `bio_services/pubmed.py` - PubMed records
- `bio_services/jaspar.py` - JASPAR motif database
- `bio_services/zfin.py` - ZFIN zebrafish database
- `bio_services/go.py` - Gene Ontology
- `bio_services/uniprot.py` - UniProt pathways
- `bio_services/epmc.py` - Europe PMC search
- `bio_services/bio_services.py` - Base module
- `bio_services/__init__.py`

**Utils (3 files):**
- `utils/markdown.py` - DataFrame to markdown conversion
- `utils/citations.py` - Citation validation
- `utils/__init__.py`

**Entry Point:**
- `main.py` - Command-line interface (executable)

**Documentation:**
- `README.md` - Full module documentation
- `INSTALLATION.md` - This file

## Key Modifications from Original

1. **Import Paths**: Updated from `alpha.*` to `scripts.litemind_peak_analysis.*`
2. **Data Paths**: Use `config.DATA_DIR` instead of hardcoded "data/"
3. **Cache Location**: HTTP cache stored in module `.cache/` directory
4. **Missing Function**: Implemented `convert_clusters_genes_to_lists()` locally
5. **Configuration**: Centralized in `config.py` with environment variable support

## Next Steps

1. **Install dependencies** (see step 2 above)
2. **Set API key** (see step 3 above)
3. **Read full documentation**: `scripts/litemind_peak_analysis/README.md`
4. **Run first analysis**: Start with one coarse cluster
5. **Review output**: Check generated markdown reports

## Resources

- **Module README**: `scripts/litemind_peak_analysis/README.md`
- **Main CLAUDE.md**: Updated with LiteMind section
- **Integration Plan**: `LITEMIND_INTEGRATION_PLAN.md` (repository root)
- **LiteMind Framework**: https://github.com/royerlab/litemind
- **OpenAI API**: https://platform.openai.com/docs/

## Success Criteria

✅ All 25 Python files created
✅ Valid Python syntax (no syntax errors)
✅ Import paths updated
✅ Data paths configured
✅ HTTP caching configured
✅ Documentation complete
✅ Main entry point functional
✅ CLAUDE.md updated

**Status**: Implementation complete! Ready for dependency installation and testing.
