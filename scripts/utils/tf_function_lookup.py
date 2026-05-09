"""Curated activator / repressor / bifunctional classification for TF
families in JASPAR2024 motif labels.

Caveat
------
TF function is context-dependent. Many TFs activate with one cofactor and
repress with another (e.g., LEF1/TCF: activator with β-catenin, repressor
with Groucho/TLE). The labels below reflect the *most common* canonical
role reported in vertebrate development literature; they are useful as a
first-pass annotation but should NOT be taken as ground truth for any
particular cellular context.

Categories
----------
  activator      — predominantly activate transcription
  repressor      — predominantly repress transcription
  bifunctional   — equally well-known in both modes; cofactor-dependent
  insulator      — primarily insulator / boundary element function
  unknown        — no consensus / not curated here
"""
import re

# Family-prefix tables. Bidirectional prefix matching: a TF (or any
# component of a JASPAR2024 dimer label like "PAX_PHOX2") is matched if
# either it starts with a prefix or the prefix starts with it (≥3 chars).

ACTIVATOR_PREFIXES = {
    # Tissue-defining TFs in development — generally activators
    "PAX",       # PAX2/5/8 — classic activators
    "SOX",       # SOX family — pioneer activators
    "GATA",      # GATA1/2/3/4/5/6 — activators
    "MYOD", "MYF", "MYOG",       # muscle bHLH activators
    "NEUROG", "NEUROD",          # neural bHLH activators
    "ASCL",                       # neural bHLH
    "OTX", "GBX",                 # MHB boundary — activators
    "EN1", "EN2",                 # engrailed — activators
    "LMX",                        # LIM-homeobox activators
    "DBX", "DMBX", "EMX",         # homeobox activators
    "FOXA",                       # pioneer FOX activator
    "FOXC", "FOXD", "FOXG", "FOXM",
    "FOXO", "FOXP", "FOXR",
    "FOX",                        # FOX family — generally activator
    "HOX", "HXA", "HXB", "HXC", "HXD",  # HOX family activators
    "MEIS", "PBX",                # HOX cofactor activators
    "TBR", "TBX5", "TBX6",         # T-box activators
    "TBXT",                       # Brachyury — activator
    "TFAP2", "AP2",                # AP2 — activator
    "POU2F", "POU3F", "POU4F", "POU5F",  # POU class activators
    "OCT",                         # POU/OCT
    "MYC",                         # MYC — activator
    "MAX",                         # MAX — partner of MYC, activator with MYC
    "ETV1", "ETV4", "ETV5",        # ETS activators
    "ELK", "FLI",                  # ETS activators (ELK1, FLI1)
    "ELF",                         # ETS activators
    "EWSR1-FLI1",                  # fusion activator
    "JUN", "FOS", "ATF",           # AP-1 family activators
    "CREB",                        # CREB activator
    "SP1", "SP3", "SP4",           # SP family activators
    "KLF4", "KLF5",                # KLF activators (KLF4 in pluripotency)
    "PHOX",                        # PHOX2 activator
    "NKX2",                        # most NKX2 are activators
    "NKX6",
    "ZIC",                         # ZIC1/2/3 activators
    "IRX",                         # IRX activators
    "LHX",                         # LIM-homeobox
    "MSX",                         # MSX (mostly activator)
    "BARHL", "BARX",               # BARH activators
    "DLX",                         # DLX activators
    "RFX",                         # RFX activators
    "EBF",                         # EBF activators
    "RBPJ",                        # Notch effector — activator with NICD
    "TEAD",                        # TEAD/YAP activator
    "MEF2",                        # MEF2 activator
    "RUNX",                        # context, but mostly activator
    "PAX_PHOX2", "POU2F_POU3F",   # JASPAR dimers
    "MEIS_PBX", "EBF_TFAP2",
    "GATA_TFCP2",
    "BRACHYURY",
}

REPRESSOR_PREFIXES = {
    "REST",                        # RE1-silencing TF — neural-specific repressor outside neurons
    "HES",                         # Notch repressors (HES1, HES7)
    "HEY",                         # Notch repressors
    "GFI",                         # GFI1, GFI1B — repressors
    "SNAI",                        # Snail/Slug — repressors of E-cadherin
    "ZEB",                         # ZEB1/2 — repressors
    "TBX2", "TBX3",                # T-box repressors
    "TBX_TBX2", "MGA_TBX",         # T-box repressor dimer labels
    "BCL6",                        # BCL6 — repressor
    "BACH",                        # BACH1/2 — repressors
    "MAFB", "MAFK",                # small MAFs — often repressive
    "ID1", "ID2", "ID3", "ID4",    # bHLH inhibitors (sequester E-proteins)
    "TLE",                         # Groucho/TLE — corepressor
    "CTBP",                        # CtBP — corepressor
    "GLIS",                        # GLIS — repressor
    "PRDM1",                       # BLIMP1 — repressor
    "PRDM6",                       # PRDM6 — repressor
}

INSULATOR_PREFIXES = {
    "CTCF",                        # canonical insulator
    "CTCF_INSM",                   # JASPAR dimer
    "INSM",                        # INSM — silencer
    "ZNF143",                      # boundary
}

# KRAB/zinc-finger family — predominantly repressive in vertebrates
# (KRAB-ZFs recruit TRIM28/SETDB1 → H3K9me3 silencing of TEs and
# developmental genes). Generic "ZNF" hits in FIMO are usually background
# but biologically lean repressive when real.
KRAB_ZF_PREFIXES = {
    "ZNF",  "ZN", "ZBT", "ZSCAN", "ZKSCAN", "ZFP", "ZFX",
    "PRDM", "PRDM_ZNF", "ZBTB",
    "RBPJ_ZNF",                    # JASPAR dimer
    "RREB",                        # RREB1 — zinc-finger
    "MTF1",                        # MTF1 — zinc-finger
    "KLF_ZNF", "KLF_SP",           # JASPAR dimers (often repressive in dev)
    "MAF_ZNF",
    "EGR_ZBTB",
    "GLI_ZBTB",
    "FEZF_ZFP",
    "PLAG_ZNF",
    "THAP_ZSCAN",
    "ZBTB_ZNF",
    "ZIC_ZNF",
}

BIFUNCTIONAL_PREFIXES = {
    # Context- or cofactor-dependent — both modes are well documented
    "TCF",        # TCF/LEF: activator with β-catenin, repressor without
    "LEF1",
    "LEF1_TCF",
    "LEF",
    "NR1",        # nuclear receptors — context-dependent
    "NR2", "NR3", "NR4", "NR5",
    "RAR", "RXR", "THR", "ESR",
    "PPAR",
    "NR1H_THR",   # JASPAR dimer
    "NR2F_RXR",
    "ESR_NR1I",
    "NFKB", "REL",   # NF-κB family — mostly activator but RELB/p50 can repress
    "STAT",          # STAT — context-dependent
    "IRF_STAT",
    "SMAD",          # TGF-β SMADs — both modes
    "E2F",           # E2F1-3 activators, E2F4-6 repressors
    "E2F_TFDP",
    "NRF1",          # context-dependent
    "NFY",           # NFY — context-dependent
    "NFYA", "NFYB", "NFYC",
    "GLI",           # GLI1/2/3 — bifunctional (Hh-dependent)
    "OLIG",          # OLIG context-dependent
    "BHLH",          # generic bHLH dimer label — varies
    "BHLHE_OLIG",
    "CUX",           # CUX1 — context (mostly repressor in some contexts)
    "ONECUT",
    "CUX_ONECUT",
    "HNF",           # HNF — context (mostly activator but some HNF1 splice forms)
    "HNF_NR2E", "HNF_POU4F",
    "TFCP2",         # TFCP2 — context
    "DRGX",
    "DRGX_ETV",
    "MAF",           # MAF family — context
    "MAF_NRL",
    "ATF_CEBP",
    "CEBP",          # CEBP — generally activator but repressive isoforms exist
    "CDX",
    "CDX_HOXA",
    "MGA",
    "BCL_RUNX",
    "BCL_STAT",
    "GRHL",          # Grainyhead — context
    "GRHL_TFCP2", "GRHL_SCRT",
    "MSANTD",
    "MSANTD_NKX2",
    "FOXI", "FOXJ", "FOXH",
    "FOXI_FOXO", "FOXO_FOXP",
    "POU2F_POU3F",   # already in activator but can be either
}


def _name_match(parts: list, prefix_set: set, min_len: int = 3) -> bool:
    """True if any component of `parts` matches any prefix bidirectionally."""
    for p in parts:
        if len(p) < 2:
            continue
        for t in prefix_set:
            if p == t:
                return True
            if len(p) >= min_len and t.startswith(p):
                return True
            if len(t) >= min_len and p.startswith(t):
                return True
    return False


def classify_function(tf_name: str) -> str:
    """Classify a TF (or JASPAR2024 dimer label) into one of:
       'activator', 'repressor', 'bifunctional', 'insulator',
       'krab_zf' (zinc-finger; usually repressive but flagged separately),
       'unknown'.
    """
    if not tf_name:
        return "unknown"
    up = tf_name.upper()
    parts = re.split(r"[\W_]+", up)
    parts = [p for p in parts if p]

    # Check insulator first (most specific)
    if _name_match(parts, INSULATOR_PREFIXES):
        return "insulator"
    # Repressors next (specific list of well-known ones)
    if _name_match(parts, REPRESSOR_PREFIXES):
        return "repressor"
    # Activators (broad list of canonical developmental TFs)
    if _name_match(parts, ACTIVATOR_PREFIXES):
        return "activator"
    # Bifunctional
    if _name_match(parts, BIFUNCTIONAL_PREFIXES):
        return "bifunctional"
    # KRAB-ZF / generic zinc-finger background — usually repressive in dev
    if _name_match(parts, KRAB_ZF_PREFIXES):
        return "krab_zf"
    return "unknown"


# Color palette for plotting
FUNCTION_COLORS = {
    "activator":    "#2ca02c",    # green
    "repressor":    "#d62728",    # red
    "bifunctional": "#ff7f0e",    # orange
    "insulator":    "#9467bd",    # purple
    "krab_zf":      "#8c564b",    # brown (zinc-finger background)
    "unknown":      "#bcbcbc",    # gray
}

FUNCTION_ORDER = ["activator", "bifunctional", "repressor",
                  "insulator", "krab_zf", "unknown"]
