"""Publication-quality matplotlib/seaborn style — single source of truth.

Import and call once at the top of any figure script (after importing
matplotlib/seaborn) and BEFORE drawing:

    from pub_fig_style import apply as apply_pub_style
    apply_pub_style()

Why this exact order matters (do not reorder):
  1. reset rcParams to defaults  — clears stale settings from prior imports
  2. pdf/ps.fonttype = 42         — TrueType embedding → editable text strings
                                    in Illustrator (the real lever for editable
                                    text; survives sns.set)
  3. sns.set(...)                 — applies paper/whitegrid theme; NOTE this
                                    RESETS both font.family (→ sans-serif) and
                                    savefig.dpi (→ ~72), so both must be set AFTER
  4. font.family = 'Arial'        — set AFTER sns.set() or it gets clobbered back
                                    to sans-serif (this was a latent bug in the
                                    old inline blocks — Arial never actually took
                                    effect because it was set before sns.set)
  5. savefig.dpi = 300            — likewise re-set AFTER sns.set()

Verification: `pdffonts file.pdf` should show `TrueType ArialMT` (fonttype=42).
Arial is confirmed available in the single-cell-base env; if absent, matplotlib
warns and falls back to DejaVu Sans (still editable TrueType via fonttype=42).

Reference: feedback_figure_standards.md (user memory); pattern from
notebooks/Fig2_ATAC_RNA_correlation_metacells/Fig2_metacell_RNA_ATAC_dynamics_v2.py
"""

import matplotlib as _mpl
import seaborn as _sns


def apply(dpi: int = 300, context: str = "paper", style: str = "whitegrid",
          font_family: str = "Arial"):
    """Apply the canonical publication figure style.

    Parameters
    ----------
    dpi : int
        savefig DPI for raster (PNG) exports. Default 300.
    context : str
        seaborn context ('paper', 'notebook', 'talk', 'poster'). Default 'paper'.
    style : str
        seaborn style. Default 'whitegrid'.
    font_family : str
        Font family. Default 'Arial' (forces TrueType embedding on SLURM).
    """
    _mpl.rcParams.update(_mpl.rcParamsDefault)   # 1. reset
    _mpl.rcParams["pdf.fonttype"] = 42           # 2. editable text in Illustrator
    _mpl.rcParams["ps.fonttype"]  = 42
    _sns.set(style=style, context=context)       # 3. seaborn (resets font + dpi)
    _mpl.rcParams["font.family"] = font_family   # 4. Arial AFTER sns.set (or clobbered)
    _mpl.rcParams["savefig.dpi"]  = dpi          # 5. DPI AFTER sns.set (or clobbered)
