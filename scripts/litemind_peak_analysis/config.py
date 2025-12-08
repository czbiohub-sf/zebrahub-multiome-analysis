"""
Configuration for LiteMind Peak Analysis Module

Set environment variables:
    export OPENAI_API_KEY="sk-..."
    export ANTHROPIC_API_KEY="sk-ant-..."
"""

import os
from pathlib import Path

# ============================================================================
# Paths
# ============================================================================

MODULE_DIR = Path(__file__).parent
CACHE_DIR = MODULE_DIR / ".cache"
CACHE_DIR.mkdir(exist_ok=True, mode=0o755)

# Data directory - points to external litemind subrepo data
# Users can override this by setting LITEMIND_DATA_DIR environment variable
DEFAULT_DATA_DIR = Path("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/external/litemind_peak_cluster_queries/src/alpha/project/regulome/data")
DATA_DIR = Path(os.environ.get("LITEMIND_DATA_DIR", DEFAULT_DATA_DIR))

# Results output directory
RESULTS_DIR = MODULE_DIR / "results"

# ============================================================================
# API Configuration
# ============================================================================

# API Keys (required)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# Default API provider
DEFAULT_API = os.environ.get("LITEMIND_DEFAULT_API", "openai")  # or "anthropic"

# Default model
DEFAULT_MODEL = os.environ.get("LITEMIND_DEFAULT_MODEL", "gpt-4")

# ============================================================================
# LiteMind Tool Configuration
# ============================================================================

# Web search configuration
MAX_WEB_SEARCHES = int(os.environ.get("LITEMIND_MAX_WEB_SEARCHES", "256"))
SEARCH_CONTEXT_SIZE = os.environ.get("LITEMIND_SEARCH_CONTEXT_SIZE", "high")  # low, medium, high

# ============================================================================
# Analysis Workflow Options
# ============================================================================

# Enable/disable analysis review and revision steps
DO_REVIEW = os.environ.get("LITEMIND_DO_REVIEW", "true").lower() == "true"

# Enable/disable deep research (uses specialized deep research models)
DO_DEEP_RESEARCH = os.environ.get("LITEMIND_DO_DEEP_RESEARCH", "false").lower() == "true"

# Save results as PDF
SAVE_PDF = os.environ.get("LITEMIND_SAVE_PDF", "true").lower() == "true"

# ============================================================================
# Validation
# ============================================================================

def validate_config():
    """Validate configuration and raise helpful errors if misconfigured."""
    errors = []

    if not DATA_DIR.exists():
        errors.append(f"Data directory not found: {DATA_DIR}")
        errors.append("Set LITEMIND_DATA_DIR environment variable to the correct path")

    if DEFAULT_API == "openai" and not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY environment variable not set")
        errors.append("Either set OPENAI_API_KEY or change DEFAULT_API to 'anthropic'")

    if DEFAULT_API == "anthropic" and not ANTHROPIC_API_KEY:
        errors.append("ANTHROPIC_API_KEY environment variable not set")
        errors.append("Either set ANTHROPIC_API_KEY or change DEFAULT_API to 'openai'")

    if DEFAULT_API not in ["openai", "anthropic"]:
        errors.append(f"Invalid DEFAULT_API: {DEFAULT_API}")
        errors.append("Must be 'openai' or 'anthropic'")

    if errors:
        error_msg = "\n❌ Configuration Error:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(error_msg)

# Validate on import (but allow ImportError to be caught)
try:
    validate_config()
except ValueError as e:
    import warnings
    warnings.warn(str(e), UserWarning)
